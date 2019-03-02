import numpy as np
import cv2
import copy
import tensorflow as tf
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

from model.ddpg_cmd2a import DDPG as DDPG_basic
from data_generation.GazeboWorld import GazeboWorld
from data_generation.GazeboRoomDataGenerator import GridWorld
from utils.ou_noise import OUNoise
from utils.model_utils import variable_summaries

CWD = os.getcwd()
RANDOM_SEED = 1234

tf_flags = tf.app.flags

# network param
tf_flags.DEFINE_float('a_learning_rate', 1e-4, 'Actor learning rate.')
tf_flags.DEFINE_float('c_learning_rate', 1e-3, 'Critic learning rate.')
tf_flags.DEFINE_integer('batch_size', 32, 'Batch size to use during training.')
tf_flags.DEFINE_integer('n_hidden', 256, 'Size of each model layer.')
tf_flags.DEFINE_integer('n_cmd_type', 6, 'NUmber of command type.')
tf_flags.DEFINE_integer('n_layers', 1, 'Number of rnn layers in the model.')
tf_flags.DEFINE_integer('max_steps', 10, 'Max number of steps in an episode.')
tf_flags.DEFINE_integer('dim_action', 2, 'Dimension of action.')
tf_flags.DEFINE_integer('dim_laser_b', 666, 'Laser beam number.')
tf_flags.DEFINE_integer('dim_laser_c', 3, 'Laser channel.')
tf_flags.DEFINE_integer('dim_goal', 2, 'Dimension of goal.')
tf_flags.DEFINE_integer('dim_emb', 64, 'Dimension of embedding.')
tf_flags.DEFINE_integer('dim_cmd', 1, 'Dimension of command.')
tf_flags.DEFINE_float('a_linear_range', 0.3, 'Range of the linear speed')
tf_flags.DEFINE_float('a_angular_range', np.pi/6, 'Range of the angular speed')
tf_flags.DEFINE_float('tau', 0.01, 'Target network update rate')
tf_flags.DEFINE_boolean('actor_training', True, 'Whether to train actor')
tf_flags.DEFINE_boolean('prioritize', False, 'Whether to use prioritized replay buffer')


# training param
tf_flags.DEFINE_integer('total_steps', 500000, 'Total training steps.')
tf_flags.DEFINE_string('model_dir', os.path.join(CWD[:-7], 'lan_nav_data/saved_network/'), 'saved model directory.')
tf_flags.DEFINE_string('model_name', 'ddpg_cmd2a_demo', 'Name of the model.')
tf_flags.DEFINE_integer('steps_per_checkpoint', 10000, 'How many training steps to do per checkpoint.')
tf_flags.DEFINE_integer('buffer_size', 100000, 'The size of Buffer')
tf_flags.DEFINE_float('gamma', 0.99, 'reward discount')
tf_flags.DEFINE_boolean('load_model', False, 'Whether to load model')
tf_flags.DEFINE_integer('noise_stop_episode', 1000000, 'episode to stop add exploration noise')
tf_flags.DEFINE_float('init_epsilon', 0.01, 'true action rate')
tf_flags.DEFINE_float('init_noise', 1., 'init noise rate')
tf_flags.DEFINE_boolean('testing', False, 'testing')
# noise param
tf_flags.DEFINE_float('mu', 0., 'mu')
tf_flags.DEFINE_float('theta', 0.15, 'theta')
tf_flags.DEFINE_float('sigma', 0.3, 'sigma')
# ros param
tf_flags.DEFINE_boolean('rviz', False, 'rviz')

flags = tf_flags.FLAGS

def FileProcess():
    # --------------change the map_server launch file-----------
    with open('./config/map.yaml', 'r') as launch_file:
        launch_data = launch_file.readlines()
        launch_data[0] = 'image: ' + CWD + '/world/map.png\n'
    with open('./config/map.yaml', 'w') as launch_file:
        launch_file.writelines(launch_data)     

    time.sleep(1.)
    print "file processed"

def main(robot_name, rviz):
    # np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)
    world = GridWorld()
    switch_action = False
    # world.map, switch_action = world.RandomSwitchRoom()
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
    env.ResetWorld()
    # env.ResetModelsPose(pickle_path)
    if switch_action:
        env.SwitchRoom(switch_action)

    env.state_call_back_flag = True
    env.target_theta_range = 1.
    time.sleep(2.)

    exploration_noise = OUNoise(action_dimension=flags.dim_action, 
                                mu=flags.mu, theta=flags.theta, sigma=flags.sigma)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:

        agent = DDPG_basic(flags, sess)
        print 'DDPG_basic'

        trainable_var = tf.trainable_variables()
        sess.run(tf.global_variables_initializer())
        
        model_dir = os.path.join(flags.model_dir, flags.model_name)
        if not os.path.exists(model_dir): 
            os.makedirs(model_dir)

        # summary
        if not flags.testing:
            print "  [*] printing trainable variables"
            for idx, v in enumerate(trainable_var):
                print "  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name)
                with tf.name_scope(v.name.replace(':0', '')):
                    variable_summaries(v)

            reward_ph = tf.placeholder(tf.float32, [], name='reward')
            q_ph = tf.placeholder(tf.float32, [], name='q_pred')
            noise_ph = tf.placeholder(tf.float32, [], name='noise')
            epsilon_ph = tf.placeholder(tf.float32, [], name='epsilon')

            tf.summary.scalar('reward', reward_ph)
            tf.summary.scalar('q_estimate', q_ph)
            tf.summary.scalar('noise', noise_ph)
            tf.summary.scalar('epsilon', epsilon_ph)
            merged = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(model_dir, sess.graph)

        # model saver
        saver = tf.train.Saver(trainable_var, max_to_keep=5)

        # load model
        if flags.load_model:
            checkpoint = tf.train.get_checkpoint_state(model_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print "Network model loaded: ", checkpoint.model_checkpoint_path
            else:
                print 'No model is found'

        episode = 0
        T = 0
        epsilon = flags.init_epsilon
        noise = flags.init_noise 
        # start training


        room_position = np.array([-1, -1])
        while T < flags.total_steps and not rospy.is_shutdown():
            print ''
            # set robot in a room
            init_pose, init_map_pose, room_position = world.RandomInitPoseInRoom()
            env.SetObjectPose(robot_name, [init_pose[0], init_pose[1], 0., init_pose[2]])
            time.sleep(2.)
            # generate a path to other room
            target_pose, target_map_pose, _ = world.RandomInitPoseInRoom(room_position)
            # get a long path
            map_plan, real_route = world.GetPath(init_map_pose+target_map_pose)

            dynamic_route = copy.deepcopy(real_route)
            env.LongPathPublish(real_route)

            if len(map_plan) == 0:
                print 'no path'
                continue

            pose = env.GetSelfStateGT()
            target_table_point_list, cmd_list, _ = world.GetCommandPlan(pose, real_route)

            # print 'target_table_point_list:', target_table_point_list
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

            if np.random.rand() <= epsilon:
                true_flag = True
                print '-------using true actions------'
            else:
                true_flag = False

            while not rospy.is_shutdown():
                start_time = time.time()
                if t == 0:
                    print 'current cmd:', cmd_list[cmd_idx]
                
                terminal, result, reward = env.GetRewardAndTerminate(t)
                total_reward += reward

                if t > 0:
                    agent.Add2Mem((laser_stack,
                                   cmd,
                                   cmd_next,
                                   local_goal, 
                                   prev_action, 
                                   action, 
                                   reward, 
                                   terminal))

                if result > 1:
                    break
                elif result == 1:
                    if cmd_list[cmd_idx] == 5:
                        print 'Finish!!!!!!!'
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
                

                if not flags.testing:
                    if noise > 0:
                        noise -= flags.init_noise/flags.noise_stop_episode
                        epsilon -= flags.init_epsilon/flags.noise_stop_episode

                time1 = time.time() - start_time

                if not true_flag :
                    # predict action
                    if t > 0:
                        prev_action = action
                    if cmd_list[cmd_idx] == 5:
                        local_goal_a = local_goal
                    else:
                        local_goal_a = [0., 0.]
                    action = agent.ActorPredict([laser_stack], [cmd], [cmd_next], [local_goal_a], [prev_action])[0]

                    if not flags.testing:
                        if T < flags.noise_stop_episode:
                            action += (exploration_noise.noise() * np.asarray(agent.action_range) * noise )
                else:
                    # get true action
                    pose = env.GetSelfStateGT()
                    near_goal, dynamic_route = world.GetNextNearGoal(dynamic_route, pose)

                    local_near_goal = env.GetLocalPoint(near_goal)
                    action = env.Controller(local_near_goal, None, 1)

                env.SelfControl(action, agent.action_range)

                time2 = time.time() - start_time - time1

                if (T + 1) % flags.steps_per_checkpoint == 0 and not flags.testing:
                    saver.save(sess, os.path.join(model_dir, 'network') , global_step=T)

                if T > agent.batch_size and not flags.testing:
                    q = agent.Train()
                    epi_q.append(np.amax(q))

                t += 1
                T += 1
                loop_time.append(time.time() - start_time)

                time3 = time.time() - start_time - time1 - time2
                used_time = time.time() - start_time
                # if used_time > 0.025:
                #     print '{:.4f} | {:.4f} | {:.4f} | {:.4f}'.format(time1, time2, time3, used_time)
                rate.sleep()

            if T > agent.batch_size and not flags.testing:
                if not true_flag:
                    summary = sess.run(merged, feed_dict={reward_ph: total_reward,
                                                          q_ph: np.amax(q),
                                                          noise_ph: noise,
                                                          epsilon_ph: epsilon
                                                          })
                    summary_writer.add_summary(summary, T)
                print 'Episode:{:} | Steps:{:} | Reward:{:.2f} | T:{:} | Q:{:.2f} | Loop Time:{:.3f} '.format(
                                                                                          episode, 
                                                                                          t, 
                                                                                          total_reward, 
                                                                                          T, 
                                                                                          np.amax(q),
                                                                                          np.mean(loop_time))

            else: 
                print 'Episode:{:} | Steps:{:} | Reward:{:.2f} | T:{:}'.format(episode, 
                                                                               t, 
                                                                               total_reward, 
                                                                               T)
            episode += 1

if __name__ == '__main__':
    robot_name = 'robot1'
    rviz = flags.rviz

    main(robot_name, rviz)