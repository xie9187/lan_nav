import numpy as np
import cv2
import copy
# import tensorflow as tf
import math
import os
import sys
import time
import rospy
import random
import matplotlib.pyplot as plt
import math 
import csv
import png
import socket
import random
import pickle

from GazeboWorld import GazeboWorld
from GazeboRoomDataGenerator import GridWorld

CWD = os.getcwd()
RANDOM_SEED = 1234


def FileProcess():
    # --------------change the map_server launch file-----------
    with open('./config/map.yaml', 'r') as launch_file:
        launch_data = launch_file.readlines()
        launch_data[0] = 'image: ' + CWD + '/world/map.png\n'
    with open('./config/map.yaml', 'w') as launch_file:
        launch_file.writelines(launch_data)     

    time.sleep(1.)
    print "file processed"


def LogData(Data, image_save, num, path):
    name = ['laser', 'action', 'cmd', 'cmd_next', 'obj_pose']
    for x in xrange(len(name)):
        file = open(path+'/'+str(num)+'_'+name[x]+'.csv', 'w')
        writer = csv.writer(file, delimiter=',', quotechar='|')
        for row in Data[x]:
            if not isinstance(row, list):
                row = [row]
            writer.writerow(row)

    image_path = os.path.join(path, str(num)+'_image')
    try:
        os.stat(image_path)
    except:
        os.makedirs(image_path)
    for idx, image in enumerate(image_save):
        cv2.imwrite(os.path.join(image_path, str(idx))+'.jpg', image)

def main(robot_name, rviz, data_path):
    # np.random.seed(RANDOM_SEED)
    robot_name = 'robot1'
    # tf.set_random_seed(RANDOM_SEED)
    world = GridWorld()
    switch_action = False
    world.map, switch_action = world.RandomSwitchRoom()
    world.GetAugMap()
    world.CreateTable()
    cv2.imwrite('./world/map.png', np.flipud(1-world.map)*255)

    FileProcess()

    env = GazeboWorld(world.table, robot_name, rviz=rviz)
    print "Env initialized"
    time.sleep(2.)

    rate = rospy.Rate(5.)

    pickle_path = os.path.join(CWD, 'world/model_states_data.p')
    
    env.GetModelStates()
    # pickle.dump( env.model_states_data, open(pickle_path, "wb"))
    # assert False

    env.GetModelStates()
    env.ResetWorld()
    env.ResetModelsPose(pickle_path)
    if switch_action:
        env.SwitchRoom(switch_action)

    env.state_call_back_flag = True
    env.target_theta_range = 1.
    time.sleep(2.)

 
    episode = 0
    T = 0
    room_position = np.array([-1, -1])
    while not rospy.is_shutdown():
        print ''
        # set robot in a room
        init_pose, init_map_pose, room_position = world.RandomInitPoseInRoom(room_position)
        # generate a path to other room
        target_pose, target_map_pose, _ = world.RandomInitPoseInRoom(room_position)
        # get a long path
        map_plan, real_route = world.GetPath(init_map_pose+target_map_pose)
        # init the direction according to the path
        init_pose[2] = np.arctan2(real_route[5][1] - real_route[0][1], real_route[5][0] - real_route[0][0])
        env.SetObjectPose(robot_name, [init_pose[0], init_pose[1], 0., init_pose[2]], once=True)
        time.sleep(1)
        dynamic_route = copy.deepcopy(real_route)
        env.LongPathPublish(real_route)
        time.sleep(1.)

        if len(map_plan) == 0:
            print 'no path'
            time.sleep(1.)
            continue

        pose = env.GetSelfStateGT()
        target_table_point_list, cmd_list, check_point_list = world.GetCommandPlan(pose, real_route)

        print 'target_table_point_list:', target_table_point_list
        print 'cmd_list:', cmd_list

        if len(target_table_point_list) == 0:
            print 'no check point'
            continue

        table_goal = target_table_point_list[0]
        goal = world.Table2RealPosition(table_goal)
        env.target_point = goal
        env.distance = np.sqrt(np.linalg.norm([pose[0]-goal[0], pose[1]-goal[1]]))

        total_reward = 0
        laser = env.GetLaserObservation()
        laser_stack = np.stack([laser, laser, laser], axis=-1)
        prev_action = [0., 0.]
        epi_q = []
        loop_time = []
        t = 0
        terminal = False
        result = 0
        cmd_idx = 0
        laser_save = []
        action_save = []
        cmd_save = []
        cmd_next_save = []
        goal_pose_save = []
        image_save = []
        in_flag = False
        file_num = len([f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))])
        while not rospy.is_shutdown():
            start_time = time.time()

            terminal, result, reward = env.GetRewardAndTerminate(t)
            total_reward += reward

            if t == 0:
                print 'current cmd:', cmd_list[cmd_idx]
            else:
                # make real cmd
                if cmd_idx <= 1:
                    real_cmd = 6 + cmd_list[1]
                    real_next_cmd = cmd_list[2]
                else:
                    real_cmd = cmd_list[cmd_idx]
                    real_next_cmd = cmd_list[cmd_idx+1]

                table_x = int(pose[0]/world.grid_size/world.p2r)
                table_y = int(pose[1]/world.grid_size/world.p2r)
                if [table_y, table_x] in check_point_list:
                    real_cmd = 0
                    real_next_cmd = 0
                    in_flag = True
                if in_flag:
                    in_flag = False
                    real_cmd = 0
                    real_next_cmd = 0

                real_cmd = [real_cmd]
                real_next_cmd = [real_next_cmd]

                # log data
                laser_save.append(laser.tolist())
                action_save.append(action.tolist())
                cmd_save.append(real_cmd)
                cmd_next_save.append(real_next_cmd)
                goal_pose_save.append(local_goal_a)
                image_save.append(rgb_image)
                
            if result > 1:
                break
            elif result == 1:
                if cmd_list[cmd_idx] == 5:
                    print 'Finish!!!!!!!'
                    Data = [laser_save, action_save, cmd_save, cmd_next_save, goal_pose_save]
                    print "save sequence "+str(file_num/len(Data))
                    LogData(Data, image_save, str(file_num/len(Data)), data_path)
                    laser_save, action_save, cmd_save, cmd_next_save, goal_pose_save, image_save = [], [], [], [], [], []
                    break
                cmd_idx += 1
                t = 0
                table_goal = target_table_point_list[cmd_idx]
                goal = world.Table2RealPosition(table_goal)
                env.target_point = goal
                env.distance = np.sqrt(np.linalg.norm([pose[0]-goal[0], pose[1]-goal[1]]))
                continue

            local_goal = env.GetLocalPoint(goal)
            env.PathPublish(local_goal)

            cmd = [cmd_list[cmd_idx]]
            cmd_next = [cmd_list[cmd_idx+1]]

            laser = env.GetLaserObservation()
            laser_stack = np.stack([laser, laser_stack[:, 0], laser_stack[:, 1]], axis=-1)
            rgb_image = env.GetRGBImageObservation()

            if cmd_list[cmd_idx] == 5:
                local_goal_a = local_goal
            else:
                local_goal_a = [0., 0.]

            # get action
            pose = env.GetSelfStateGT()
            try:
                near_goal, dynamic_route = world.GetNextNearGoal(dynamic_route, pose)
            except:
                pass
            local_near_goal = env.GetLocalPoint(near_goal)
            action = env.Controller(local_near_goal, None, 1)

            env.SelfControl(action, [0.3, np.pi/6])


            # print '{:.4f}'.format(time.time() - start_time)

            t += 1
            T += 1
            loop_time.append(time.time() - start_time)

            rate.sleep()

        print 'Episode:{:} | Steps:{:} | Reward:{:.2f} | T:{:}'.format(episode, 
                                                                       t, 
                                                                       total_reward, 
                                                                       T)
        episode += 1

if __name__ == '__main__':
    args = sys.argv
    robot_name = args[1]
    rviz = False

    machine_id = socket.gethostname()
    data_path = os.path.join(CWD[:-7], 'lan_nav_data/room_zero')
    data_path = os.path.join(data_path, machine_id+'_'+robot_name)

    try:
        os.stat(data_path)
    except:
        os.makedirs(data_path)

    main(robot_name, rviz, data_path)