import rospy
import math
import time
import numpy as np
import cv2
import copy
import tf
import random
import roslaunch
import pickle

from geometry_msgs.msg import Twist, PoseStamped, Quaternion, PoseWithCovarianceStamped
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry, Path
from rosgraph_msgs.msg import Clock
from actionlib_msgs.msg import GoalID
from move_base_msgs.msg import MoveBaseGoal
from actionlib_msgs.msg import GoalStatusArray
from gazebo_msgs.msg import ModelState, ModelStates

class GazeboWorld():
    def __init__(self, table, robot_name, beam_num=100, object_detection=False, rviz=False):
        # initiliaze

        # uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        # roslaunch.configure_logging(uuid)
        # if rviz:
        #     self.launch = roslaunch.parent.ROSLaunchParent(uuid, ['./launch/rviz.launch'])
        # else:
        #     self.launch = roslaunch.parent.ROSLaunchParent(uuid, ['./launch/navigation_single.launch'])
        # self.launch.start()

        rospy.sleep(2.)

        rospy.init_node(robot_name+'-GazeboWorld', anonymous=False)

        #------------Params--------------------
        self.stop_counter = 0
        self.state_call_back_flag = False
        self.state_call_back_cnt = 0
        self.dist_threshold = 0.3
        self.delta_theta = np.pi
        self.U_tm1 = np.array([0., 0.])
        self.PID_X_tm1 = np.array([0., 0.])
        self.PID_X_t = np.array([0., 0.])
        self.PID_X_buff = []
        self.last_time = time.time()
        self.curr_time = time.time()
        self.target_point = [0., 0.]
        self.object_list = ['bookshelf', 'cabinet', 'person', 'cardboard_box', 'Construction Cone', 'fire_hydrant']
        self.model_states_data = None
        self.robot_name = robot_name
        if robot_name == 'robot1':
            self.y_pos = 0.
        else:
            self.y_pos = 1.05
        self.depth_image_size = [160, 128]
        self.rgb_image_size = [128, 84]
        self.bridge = CvBridge()

        self.object_poses = []
        self.object_names = []

        self.self_speed = [0.0, 0.0]
        
        self.start_time = time.time()
        self.max_steps = 10000
        self.sim_time = Clock().clock
        self.state_cnt = 0

        self.scan = None
        self.beam_num = beam_num
        self.laser_cb_num = 0

        self.move_base_goal = PoseStamped()
        self.next_near_goal = None
        self.cmd = None
        self.plan = None
        self.plan_num = 0
        self.published_status = 'PENDING'
        self.status_vect = ['PENDING', 'ACTIVE', 'PREEMPTED', 'SUCCEEDED', 'ABORTED'
                            'REJECTED', 'PREEMPTING', 'RECALLING', 'RECALLED', 'LOST']

        self.robot_size = 0.5
        self.target_size = 0.55
        self.target_theta_range = np.pi/3

        self.table = copy.deepcopy(table)

        #-----------Default Robot State-----------------------
        self.default_state = ModelState()
        self.default_state.model_name = robot_name  
        self.default_state.pose.position.x = 16.5
        self.default_state.pose.position.y = 16.5
        self.default_state.pose.position.z = self.y_pos
        self.default_state.pose.orientation.x = 0.0
        self.default_state.pose.orientation.y = 0.0
        self.default_state.pose.orientation.z = 0.0
        self.default_state.pose.orientation.w = 1.0
        self.default_state.twist.linear.x = 0.
        self.default_state.twist.linear.y = 0.
        self.default_state.twist.linear.z = 0.
        self.default_state.twist.angular.x = 0.
        self.default_state.twist.angular.y = 0.
        self.default_state.twist.angular.z = 0.
        self.default_state.reference_frame = 'world'

        #-----------Publisher and Subscriber-------------
        self.cmd_vel = rospy.Publisher(robot_name+'/mobile_base/commands/velocity', Twist, queue_size = 10)
        self.set_state = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size = 10)
        self.goal_pub = rospy.Publisher(robot_name+'/move_base_simple/goal', PoseStamped, queue_size=5)
        self.goal_cancel_pub = rospy.Publisher(robot_name+'/move_base/cancel', GoalID, queue_size=5)
        self.path_pub = rospy.Publisher(robot_name+'/pred_path', Path, queue_size=5)
        self.resized_depth_img = rospy.Publisher(robot_name+'/camera/depth/image_resized',Image, queue_size = 10)
        self.resized_rgb_img = rospy.Publisher(robot_name+'/camera/rgb/image_resized',Image, queue_size = 10)
        self.pose_GT_pub = rospy.Publisher(robot_name+'/base_pose_ground_truth',Odometry, queue_size = 10)
        self.init_pose_pub = rospy.Publisher(robot_name+'/initialpose',PoseWithCovarianceStamped, queue_size = 10)
        self.dynamic_path_pub = rospy.Publisher(robot_name+'/dynamic_path', Path, queue_size=5)


        self.object_state_sub = rospy.Subscriber('gazebo/model_states', ModelStates, self.ModelStateCallBack)
        self.laser_sub = rospy.Subscriber('mybot/laser/scan', LaserScan, self.LaserScanCallBack)
        self.odom_sub = rospy.Subscriber(robot_name+'/odom', Odometry, self.OdometryCallBack)
        self.sim_clock = rospy.Subscriber('clock', Clock, self.SimClockCallBack)
        # self.depth_image_sub = rospy.Subscriber(robot_name+'/camera/depth/image_raw', Image, self.DepthImageCallBack)
        self.rgb_image_sub = rospy.Subscriber(robot_name+'/camera/rgb/image_raw', Image, self.RGBImageCallBack)
        self.global_plan = rospy.Subscriber(robot_name+'/move_base/TrajectoryPlannerROS/global_plan', Path, self.GlobalPlanCallBack)
        self.whole_plan = rospy.Subscriber(robot_name+'/move_base/NavfnROS/plan', Path, self.WholePlanCallBack)
        self.cmd_sub = rospy.Subscriber(robot_name+'/move_base/cmd_vel', Twist, self.CmdCallBack)
        self.move_base_status_sub = rospy.Subscriber(robot_name+'/move_base/status', GoalStatusArray, self.MoveBaseStatusCallBack)

        rospy.on_shutdown(self.shutdown)

    # def ModelStateCallBack(self, data):
    #     start_time = time.time()

    #     if self.state_call_back_flag:
    #         self.state_call_back_cnt += 1
    #     if self.state_call_back_cnt == 3:
    #         self.state_call_back_flag = False
    #         self.state_call_back_cnt = 0
    #     robot_name = copy.deepcopy(self.robot_name)
    #     if robot_name in data.name:
    #         idx = data.name.index(robot_name)
    #         # if data.name[idx] == "mobile_base":
    #         quaternion = (data.pose[idx].orientation.x,
    #                       data.pose[idx].orientation.y,
    #                       data.pose[idx].orientation.z,
    #                       data.pose[idx].orientation.w)
    #         euler = tf.transformations.euler_from_quaternion(quaternion)
    #         self.robot_pose = data.pose[idx]
    #         self.state_GT = [data.pose[idx].position.x, data.pose[idx].position.y, copy.deepcopy(euler[2])]
    #         v_x = data.twist[idx].linear.x
    #         v_y = data.twist[idx].linear.y
    #         v = np.sqrt(v_x**2 + v_y**2)
    #         self.speed_GT = [v, data.twist[idx].angular.z]

    #         odom_GT = Odometry()
    #         odom_GT.header.stamp = self.sim_time
    #         odom_GT.header.seq = self.state_cnt
    #         odom_GT.header.frame_id = robot_name+'_tf/odom'
    #         odom_GT.child_frame_id = ''
    #         odom_GT.pose.pose = data.pose[idx]
    #         odom_GT.twist.twist = data.twist[idx]
    #         self.pose_GT_pub.publish(odom_GT)

    #         # self.last_time = copy.deepcopy(self.curr_time)
    #         # self.curr_time = time.time()
    #         # print self.curr_time - self.last_time

    #         # init_pose = PoseWithCovarianceStamped()

    #         self.state_cnt += 1

    #     if 'robot1' in data.name:
    #         idx1 = data.name.index('robot1')
    #         robot1_pose = data.pose[idx1]
    #         self.robot1_position = [robot1_pose.position.x, robot1_pose.position.y]    

    #     if self.state_call_back_flag:
    #         self.model_states_data = copy.deepcopy(data)

    #     used_time =  time.time() - start_time
    #     if used_time > 0.004:
    #         print used_time

    def ModelStateCallBack(self, data):
        start_time = time.time()
        if self.robot_name in data.name:
            idx = data.name.index(self.robot_name)
            quaternion = (data.pose[idx].orientation.x,
                          data.pose[idx].orientation.y,
                          data.pose[idx].orientation.z,
                          data.pose[idx].orientation.w)
            euler = tf.transformations.euler_from_quaternion(quaternion)
            self.robot_pose = data.pose[idx]
            self.state_GT = [data.pose[idx].position.x, data.pose[idx].position.y, copy.deepcopy(euler[2])]
            v_x = data.twist[idx].linear.x
            v_y = data.twist[idx].linear.y
            v = np.sqrt(v_x**2 + v_y**2)
            self.speed_GT = [v, data.twist[idx].angular.z]

            odom_GT = Odometry()
            odom_GT.header.stamp = self.sim_time
            odom_GT.header.seq = self.state_cnt
            odom_GT.header.frame_id = self.robot_name+'_tf/odom'
            odom_GT.child_frame_id = ''
            odom_GT.pose.pose = data.pose[idx]
            odom_GT.twist.twist = data.twist[idx]
            self.pose_GT_pub.publish(odom_GT)

        if self.state_call_back_flag:
            self.model_states_data = copy.deepcopy(data)
            self.state_call_back_flag = False

        used_time =  time.time() - start_time
        

    def RGBImageCallBack(self, img):
        self.rgb_image = img

    def DepthImageCallBack(self, img):
        self.depth_image = img

    def LaserScanCallBack(self, scan):
        self.scan_param = [scan.angle_min, scan.angle_max, scan.angle_increment, scan.time_increment,
                           scan.scan_time, scan.range_min, scan. range_max]
        self.scan = np.array(scan.ranges)
        self.laser_cb_num += 1

    def OdometryCallBack(self, odometry):
        Quaternions = odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
        self.state = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, Euler[2]]
        self.speed = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]

    def SimClockCallBack(self, clock):
        self.sim_time = clock.clock

    def GlobalPlanCallBack(self, path):
        point_num = len(path.poses)
        self.next_near_goal = path.poses[-1].pose.position
        Quaternions = path.poses[-1].pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
        self.next_near_goal_theta = Euler[2]

    def CmdCallBack(self, cmd):
        self.cmd = cmd
        self.action = [cmd.linear.x, cmd.angular.z]

    def WholePlanCallBack(self, path):
        self.plan = []
        for t in xrange(len(path.poses)):
            self.plan.append([path.poses[t].pose.position.x, path.poses[t].pose.position.y])
        self.plan_num += 1

    def MoveBaseStatusCallBack(self, data):
        if len(data.status_list) > 0:
            self.published_status = self.status_vect[data.status_list[0].status]
        else:
            self.published_status = 'PENDING'

    def GetLaserObservation(self):
        scan = copy.deepcopy(self.scan)
        scan[np.isnan(scan)] = 5.6
        scan[np.isinf(scan)] = 5.6
        return scan/5.6 - 0.5

    def GetDepthImageObservation(self):
        # ros image to cv2 image

        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.depth_image, "32FC1")
        except Exception as e:
            raise e

        cv_img = np.array(cv_img, dtype=np.float32)
        # resize
        dim = (self.depth_image_size[0], self.depth_image_size[1])
        cv_img = cv2.resize(cv_img, dim, interpolation = cv2.INTER_AREA)

        cv_img[np.isnan(cv_img)] = 0.
        cv_img[cv_img < 0.4] = 0.
        cv_img/=(10./255.)

        # # inpainting
        # mask = copy.deepcopy(cv_img)
        # mask[mask == 0.] = 1.
        # mask[mask != 1.] = 0.
        # mask = np.uint8(mask)
        # cv_img = cv2.inpaint(np.uint8(cv_img), mask, 3, cv2.INPAINT_TELEA)

        cv_img = np.array(cv_img, dtype=np.float32)
        cv_img*=(10./255.)

        # cv2 image to ros image and publish
        try:
            resized_img = self.bridge.cv2_to_imgmsg(cv_img, "passthrough")
        except Exception as e:
            raise e
        self.resized_depth_img.publish(resized_img)
        return(cv_img/5.)

    def GetRGBImageObservation(self):
        # ros image to cv2 image
        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.rgb_image, "bgr8")
        except Exception as e:
            raise e
        # resize
        dim = (self.rgb_image_size[0], self.rgb_image_size[1])
        cv_resized_img = cv2.resize(cv_img, dim, interpolation = cv2.INTER_AREA)
        # cv2 image to ros image and publish
        try:
            resized_img = self.bridge.cv2_to_imgmsg(cv_resized_img, "bgr8")
        except Exception as e:
            raise e
        self.resized_rgb_img.publish(resized_img)
        return(cv_resized_img)

    def GetSelfState(self):
        return copy.deepcopy(self.state)

    def GetSelfStateGT(self):
        return copy.deepcopy(self.state_GT)

    def GetSelfSpeedGT(self):
        return copy.deepcopy(self.speed_GT)

    def GetSelfSpeed(self):
        return copy.deepcopy(self.speed)

    def GetSimTime(self):
        return copy.deepcopy(self.sim_time)

    def GetNextNearGoal(self, theta_flag=False):
        if theta_flag:
            return copy.deepcopy([self.next_near_goal.x, self.next_near_goal.y, self.next_near_goal_theta])
        else:
            return copy.deepcopy([self.next_near_goal.x, self.next_near_goal.y])

    def GetLocalPoint(self, vector, self_pose=None):
        if self_pose is None:
            [x, y, theta] =  self.GetSelfStateGT()
        else:
            [x, y, theta] = self_pose
        [target_x, target_y] = vector[:2]
        local_x = (target_x - x) * np.cos(theta) + (target_y - y) * np.sin(theta)
        local_y = -(target_x - x) * np.sin(theta) + (target_y - y) * np.cos(theta)
        return [local_x, local_y]

    def GetGlobalPoint(self, vector):
        [x, y, theta] =  self.GetSelfStateGT()
        [target_x, target_y] = vector[:2]
        global_x = target_x * np.cos(theta) - target_y * np.sin(theta) + x
        global_y = target_x * np.sin(theta) + target_y * np.cos(theta) + y
        return [global_x, global_y]

    def GetAction(self):
        return copy.deepcopy(self.action)

    def GetPlan(self):
        # print self.plan
        return copy.deepcopy(self.plan)

    def GetMoveBaseStatus(self):
        return copy.deepcopy(self.published_status)

    def GetModelStates(self):
        self.state_call_back_flag = True
        while self.state_call_back_flag and not rospy.is_shutdown():
            pass

    def SetObjectPose(self, name, pose, once=False):
        object_state = copy.deepcopy(self.default_state)
        object_state.model_name = name
        object_state.pose.position.x = pose[0]
        object_state.pose.position.y = pose[1]
        object_state.pose.position.z = pose[2]
        quaternion = tf.transformations.quaternion_from_euler(0., 0., pose[3])
        object_state.pose.orientation.x = quaternion[0]
        object_state.pose.orientation.y = quaternion[1]
        object_state.pose.orientation.z = quaternion[2]
        object_state.pose.orientation.w = quaternion[3]

        self.set_state.publish(object_state)
        if not once:
            start_time = time.time()
            while time.time() - start_time < 0.5 and not rospy.is_shutdown():
            # for i in xrange(0,2):
                self.set_state.publish(object_state)
                # rospy.sleep(0.1)
        print 'Set '+name

    def ResetWorld(self):
        self.self_speed = [0.0, 0.0]
        self.start_time = time.time()
        self.delta_theta = np.pi
        self.U_tm1 = np.array([0., 0.])
        self.PID_X_tm1 = np.array([0., 0.])
        self.PID_X_t = np.array([0., 0.])
        self.PID_X_buff = []

        rospy.sleep(0.5)


    def ResetModelsPose(self, path):
        print '-----------------------Reseting models-------------------------------'
        origin_model_states = pickle.load(open(path, "rb"))
        for obj_name, obj_pose in zip(origin_model_states.name, origin_model_states.pose):
            origin_position = obj_pose.position
            curr_position = self.model_states_data.pose[self.model_states_data.name.index(obj_name)].position
            if np.linalg.norm([origin_position.x-curr_position.x, 
                               origin_position.y-curr_position.y]) > 0.5 and obj_name is not self.robot_name:
                quaternion = (obj_pose.orientation.x,
                              obj_pose.orientation.y,
                              obj_pose.orientation.z,
                              obj_pose.orientation.w)
                euler = tf.transformations.euler_from_quaternion(quaternion)
                theta = euler[2]
                self.SetObjectPose(obj_name, [origin_position.x, origin_position.y, origin_position.z, theta])
        print '-----------------------Reseting finished-----------------------------'
                

    def Control(self):
        self.cmd_vel.publish(self.cmd)

    def SelfControl(self, action, action_range=[10., 10.]):

        if action[0] < 0.:
            action[0] = 0.
        if action[0] > action_range[0]:
            action[0] = action_range[0]
        if action[1] < -action_range[1]:
            action[1] = -action_range[1]
        if action[1] > action_range[1]:
            action[1] = action_range[1]

        move_cmd = Twist()
        move_cmd.linear.x = action[0]
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = action[1]
        self.cmd_vel.publish(move_cmd)
        return action

    def shutdown(self):
        # stop turtlebot
        rospy.loginfo("Stop Moving")
        self.cmd_vel.publish(Twist())

        # if self.launch != None:
        #     self.launch.shutdown()

        rospy.sleep(1)

    def GetRewardAndTerminate(self, t, delta=None):
        terminate = False
        reset = False
        laser_scan = self.GetLaserObservation()
        laser_min = np.amin(laser_scan)
        [x, y, theta] =  self.GetSelfStateGT()
        [v, w] = self.GetSelfSpeedGT()
        self.pre_distance = copy.deepcopy(self.distance)
        self.distance = np.sqrt((self.target_point[0] - x)**2 + (self.target_point[1] - y)**2)
        result = 0

        if laser_min < 0.25 / 5.6 - 0.5:
            self.stop_counter += 1
        else:
            self.stop_counter = 0

        if t == 0:
            self.movement_counter = 0
        else:
            if np.linalg.norm([x-self.last_pose[0], y-self.last_pose[1], theta-self.last_pose[2]]) < 0.01 and t > 20:
                self.movement_counter += 1
            else:
                self.movement_counter = 0.
        self.last_pose = np.array([x, y, theta])

        if delta is None:
            if t == 0:
                delta = 0.
            else:
                delta = self.pre_distance - self.distance
        reward = delta * np.cos(w) - 0.01

        if self.distance < self.target_size:
            terminate = True
            result = 1
            print 'reach the goal'
            reward = 1.
        else:
            if self.stop_counter == 5:
                terminate = True
                print 'crash'
                result = 2
                reward = -1.
            if t >= 150:
                result = 2
                print 'time out'
            if self.movement_counter >= 10:
                terminate = True
                print 'stuck'
                result = 2
                reward = -1.
                self.movement_counter = 0

        return terminate, result, reward

    def Global2Local(self, path, pose):
        x = pose[0]
        y = pose[1]
        theta = pose[2]
        local_path = copy.deepcopy(path)
        for t in xrange(0, len(path)):
            local_path[t][0] = (path[t][0] - x) * np.cos(theta) + (path[t][1] - y) * np.sin(theta)
            local_path[t][1] = -(path[t][0] - x) * np.sin(theta) + (path[t][1] - y) * np.cos(theta)
        return local_path

    def GoalPublish(self, pose):
        x = pose[0]
        y = pose[1]
        yaw = pose[2]

        self.move_base_goal.header.frame_id = "map"
        self.move_base_goal.header.stamp = rospy.Time()
        self.move_base_goal.pose.position.x = x
        self.move_base_goal.pose.position.y = y
        self.move_base_goal.pose.position.z = self.y_pos
        quaternion = tf.transformations.quaternion_from_euler(0., 0., yaw)
        self.move_base_goal.pose.orientation.x = quaternion[0]
        self.move_base_goal.pose.orientation.y = quaternion[1]
        self.move_base_goal.pose.orientation.z = quaternion[2]
        self.move_base_goal.pose.orientation.w = quaternion[3]

        print "Sending goal"
        self.PublishTopic(self.goal_pub, self.move_base_goal, 0.)

    def PublishTopic(self, publisher, content, delay=0.):
        publisher.publish(content)
        if delay != 0.:
            rospy.sleep(delay)

    def PathPublish(self, position):
        my_path = Path()
        my_path.header.frame_id = 'map'

        init_pose = PoseStamped()
        init_pose.pose = self.robot_pose
        my_path.poses.append(init_pose)

        goal_pose = copy.deepcopy(init_pose)
        [x, y] = self.GetGlobalPoint(position)
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        my_path.poses.append(goal_pose)

        self.path_pub.publish(my_path)

    def LongPathPublish(self, path):
        my_path = Path()
        my_path.header.frame_id = 'map'

        init_pose = PoseStamped()
        init_pose.pose = self.robot_pose
        my_path.poses.append(init_pose)

        for position in path:
            pose = copy.deepcopy(init_pose)
            [x, y] = position[:2]
            pose.pose.position.x = x
            pose.pose.position.y = y
            my_path.poses.append(pose)

        self.dynamic_path_pub.publish(my_path)

    def GoalCancel(self):
        self.PublishTopic(self.goal_cancel_pub, GoalID(), 0.)
        self.plan = []
        self.plan_num = 0

    def WaitPlan(self):
        plan_time_start = time.time()
        no_plan_flag = False

        while (self.plan_num < 2 or self.next_near_goal is None) \
                and not rospy.is_shutdown():
            if time.time() - plan_time_start > 3.:
                no_plan_flag = True
                break

        if no_plan_flag:
            print 'no available plan'
            self.GoalCancel()
            rospy.sleep(2.)
            return False

        print 'plan recieved'

        plan = self.GetPlan()
        
        if plan:
            print 'plan length', len(plan)
            if len(plan) == 0 :
                self.GoalCancel()
                rospy.sleep(2.)
                return False
        else:
            print 'plan is None'
            self.GoalCancel()
            rospy.sleep(2.)
            return False

        return plan

    def PIDController(self, target_point=None, target_theta=None):
        self.PID_X_tm1 = copy.deepcopy(self.PID_X_t)
        if target_point is None:
            point = self.GetLocalPoint(self.target_point)
        else:
            point = target_point

        delta_x = point[0]
        delta_y = point[1]

        if target_theta is None:
            theta = np.arctan2(delta_y, delta_x)
        else:
            theta = self.wrap2pi(target_theta - self.state_GT[2])

        X_t = [delta_x, theta]

        self.PID_X_buff.append(X_t)
        self.PID_X_t = copy.deepcopy(np.array(X_t))
        
        if len(self.PID_X_buff) > 5:
            self.PID_X_buff = self.PID_X_buff[1:]

        PID_X_sum = np.sum(self.PID_X_buff, axis=0)

        PID_X_sum[0] = np.amin([PID_X_sum[0], 5.])
        PID_X_sum[0] = np.amax([PID_X_sum[0], -5.])
        PID_X_sum[1] = np.amin([PID_X_sum[1], np.pi])
        PID_X_sum[1] = np.amax([PID_X_sum[1], -np.pi])

        err_p = self.PID_X_t
        err_i = PID_X_sum
        err_d = self.PID_X_t - self.PID_X_tm1

        P = np.array([.6, .8])
        I = np.array([.0, .0])
        D = np.array([.0, .8])

        U_t = err_p * P + err_i * I + err_d * D

        return U_t      
        

    def Controller(self, target_point, target_theta, stage, acc_limits=[0.1, np.pi/6], action_bound=[0.3, np.pi/6]):        
        U_t = self.PIDController(target_point=target_point, target_theta=target_theta)

        if target_theta is None:
            self.delta_theta = np.pi
        else:
            self.delta_theta = self.wrap2pi(target_theta - self.state_GT[2])    

        # extra rules

        if stage == 0:
            U_t[0] = 0.

        # acc restrict
        U_t[0] = np.amin([U_t[0], self.U_tm1[0] + acc_limits[0]])
        U_t[1] = np.amin([U_t[1], self.U_tm1[1] + acc_limits[1]])
        U_t[1] = np.amax([U_t[1], self.U_tm1[1] - acc_limits[1]])

        # velocity limits
        U_t[0] = np.amin([U_t[0], action_bound[0]])
        U_t[0] = np.amax([U_t[0], 0])
        U_t[1] = np.amin([U_t[1], action_bound[1]])
        U_t[1] = np.amax([U_t[1], -action_bound[1]])

        self.U_tm1 = U_t

        return U_t

    def GetObjectDists(self, self_pose=None):
        object_pose_dict = {}
        object_dist_dict = {}
        model_names = self.model_states_data.name
        for object_name in self.object_list:
            dists = []
            poses = []
            for model_idx, model_name in enumerate(model_names):
                if object_name in model_name:
                    pose = self.model_states_data.pose[model_idx]
                    model_posistion = [pose.position.x, pose.position.y]
                    rela_pose = self.GetLocalPoint(model_posistion, self_pose)
                    poses.append(rela_pose)
                    dists.append(np.linalg.norm(rela_pose))
            min_idx = np.argmin(dists)
            min_dist = dists[min_idx]
            nearest_pose = poses[min_idx]
            if min_dist < 2.5:
                object_pose_dict[object_name] = nearest_pose
                object_dist_dict[object_name] = min_dist
            else:
                object_pose_dict[object_name] = 0
                object_dist_dict[object_name] = 0

        return object_pose_dict, object_dist_dict

    def wrap2pi(self, ang):
        while ang > np.pi:
            ang -= (np.pi * 2)
        while ang <= -np.pi:
            ang += (np.pi * 2)
        return ang

    def SwitchRoom(self, switch_action, grid_size=1.0):
        print '---------------------Switching rooms-----------------------------'
        for action in switch_action:
            room0_origin = (np.asarray(action[:2]) * 6.0 + 2.5) * grid_size
            room1_origin = (np.asarray(action[2:4]) * 6.0 + 2.5) * grid_size
            rotate_theta = self.wrap2pi(action[4] * np.pi/2)
            room0_model = []
            room1_model = []
            for model_name, model_pose in zip(self.model_states_data.name, self.model_states_data.pose):
                x = model_pose.position.x
                y = model_pose.position.y
                if model_name is self.robot_name:
                    continue
                if room0_origin[1] - 2.5 * grid_size < x < room0_origin[1] + 2.5 * grid_size and\
                    room0_origin[0] - 2.5 * grid_size < y < room0_origin[0] + 2.5 * grid_size:
                    room0_model.append([model_name, model_pose])
                elif room1_origin[1] - 2.5 * grid_size < x < room1_origin[1] + 2.5 * grid_size and\
                    room1_origin[0] - 2.5 * grid_size < y < room1_origin[0] + 2.5 * grid_size:
                    room1_model.append([model_name, model_pose])

            self.MoveRoomObjects(room0_model, room0_origin, room1_origin.tolist()+[rotate_theta])
            self.MoveRoomObjects(room1_model, room1_origin, room0_origin.tolist()+[-rotate_theta])
        print '----------------------Switching finished---------------------------'


    def MoveRoomObjects(self, models, origin_pose, target_pose):
        for model_name, model_pose in models:
            x = model_pose.position.x
            y = model_pose.position.y
            z = model_pose.position.z
            quaternion = (model_pose.orientation.x,
                          model_pose.orientation.y,
                          model_pose.orientation.z,
                          model_pose.orientation.w)
            euler = tf.transformations.euler_from_quaternion(quaternion)
            theta = euler[2]

            vect = [x - origin_pose[1], y - origin_pose[0]]
            rot_x = vect[0] * np.cos(target_pose[2]) + vect[1] * np.sin(target_pose[2])
            rot_y = -vect[0] * np.sin(target_pose[2]) + vect[1] * np.cos(target_pose[2])
            rot_theta = self.wrap2pi(theta - target_pose[2])

            shift_x = rot_x + target_pose[1]
            shift_y = rot_y + target_pose[0]
            shift_z = z
            # print 'theta: %.2f, rot_theta: %.2f' %(theta, rot_theta)
            self.SetObjectPose(model_name, [shift_x, shift_y, shift_z, rot_theta])  


# env = StageWorld(10)
# print env.GetLaserObservation()
