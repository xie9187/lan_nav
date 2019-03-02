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

from model.rdpg_BPTT import RDPG_BPTT
from data_generation.GazeboWorld import GazeboWorld
from data_generation.GazeboRoomDataGenerator import GridWorld
from utils.ou_noise import OUNoise
from utils.model_utils import variable_summaries

CWD = os.getcwd()
RANDOM_SEED = 1234

flag = tf.app.flags

# network param
flag.DEFINE_integer('batch_size', 64, 'Batch size to use during training.')
flag.DEFINE_float('a_learning_rate', 1e-4, 'Actor learning rate.')
flag.DEFINE_float('c_learning_rate', 1e-3, 'Critic learning rate.')
flag.DEFINE_integer('max_step', 150, 'max step.')
flag.DEFINE_integer('n_hidden', 256, 'Size of each model layer.')
flag.DEFINE_integer('n_layers', 1, 'Number of layers in the model.')
flag.DEFINE_integer('n_cmd_type', 6, 'number of cmd class.')
flag.DEFINE_integer('dim_emb', 64, 'embedding dimension.')
flag.DEFINE_integer('dim_laser_b', 666, 'number of laser beam.')
flag.DEFINE_integer('dim_laser_c', 3, 'number of laser channel.')
flag.DEFINE_integer('dim_goal', 2, 'dimension of goal.')
flag.DEFINE_integer('dim_cmd', 1, 'dimension of cmd.')
flag.DEFINE_integer('dim_action', 2, 'dimension of action.')
flag.DEFINE_float('loss_w0', 1., 'loss weight0')
flag.DEFINE_float('loss_w1', 1., 'loss weight1')
flag.DEFINE_boolean('encoder_training', True, 'whether to train encoder')
flag.DEFINE_float('keep_prob', 0.8, 'Drop out parameter.')
flag.DEFINE_float('a_linear_range', 0.3, 'linear action range: 0 ~ 0.4')
flag.DEFINE_float('a_angular_range', np.pi/6, 'angular action range: -np.pi/4 ~ np.pi/4')
flag.DEFINE_integer('gpu_num', 1, 'Number of GPUs')
flag.DEFINE_float('tau', 0.01, 'Target network update rate')
flag.DEFINE_boolean('demo_flag', False, 'Whether to use demonstrations on action')

# training param
flag.DEFINE_integer('total_steps', 500000, 'Total training steps.')
flag.DEFINE_string('model_dir', os.path.join(CWD[:-7], 'lan_nav_data/saved_network/'), 'saved model directory.')
flag.DEFINE_string('model_name', 'test', 'Name of the model.')
flag.DEFINE_integer('steps_per_checkpoint', 10000, 'How many training steps to do per checkpoint.')
flag.DEFINE_integer('buffer_size', 10000, 'The size of Buffer')
flag.DEFINE_float('gamma', 0.99, 'reward discount')
flag.DEFINE_boolean('load_model', False, 'Whether to load model')
flag.DEFINE_integer('noise_stop_episode', 1000000, 'episode to stop add exploration noise')
flag.DEFINE_float('init_epsilon', 0.01, 'true action rate')
flag.DEFINE_float('init_noise', 1., 'true action rate')
flag.DEFINE_boolean('testing', False, 'testing')
flag.DEFINE_string('init_model_name', 'empty', 'model to initailize')

# noise param
flag.DEFINE_float('mu', 0., 'mu')
flag.DEFINE_float('theta', 0.15, 'theta')
flag.DEFINE_float('sigma', 0.3, 'sigma')
# ros param
flag.DEFINE_boolean('rviz', False, 'rviz')

flags = flag.FLAGS

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

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        agent = RDPG_BPTT(flags, sess)

        trainable_var = tf.trainable_variables()
        sess.run(tf.global_variables_initializer())
        
        model_dir = os.path.join(flags.model_dir, flags.model_name)
        if not os.path.exists(model_dir): 
            os.makedirs(model_dir)
        init_dir = os.path.join(flags.model_dir, flags.init_model_name)

        # summary
        if not flags.testing:
            print "  [*] printing trainable variables"
            for idx, v in enumerate(trainable_var):
                print "  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name)
                # with tf.name_scope(v.name.replace(':0', '')):
                #     variable_summaries(v)

            reward_ph = tf.placeholder(tf.float32, [], name='reward')
            q_ph = tf.placeholder(tf.float32, [], name='q_pred')
            noise_ph = tf.placeholder(tf.float32, [], name='noise')
            epsilon_ph = tf.placeholder(tf.float32, [], name='epsilon')
            err_h_ph = tf.placeholder(tf.float32, [], name='err_h')
            err_c_ph = tf.placeholder(tf.float32, [], name='err_c')

            tf.summary.scalar('reward', reward_ph)
            tf.summary.scalar('q_estimate', q_ph)
            tf.summary.scalar('noise', noise_ph)
            tf.summary.scalar('epsilon', epsilon_ph)
            tf.summary.scalar('err_h', err_h_ph)
            tf.summary.scalar('err_c', err_c_ph)
            merged = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(model_dir, sess.graph)

        part_var = []
        for idx, v in enumerate(trainable_var):
            if 'actor/online/encoder' in  v.name or 'actor/online/controller' in v.name:
                part_var.append(v)
        saver = tf.train.Saver(trainable_var, max_to_keep=5)
        part_saver = tf.train.Saver(part_var, max_to_keep=5)

        # load model
        if 'empty' in flags.init_model_name:
            checkpoint = tf.train.get_checkpoint_state(model_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print("Network model loaded: ", checkpoint.model_checkpoint_path)
            else:
                print('No model is found')
        else:
            checkpoint = tf.train.get_checkpoint_state(init_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                part_saver.restore(sess, checkpoint.model_checkpoint_path)
                print("Network model loaded: ", checkpoint.model_checkpoint_path)
            else:
                print('No model is found')


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
            time.sleep(1.)
            # generate a path to other room
            target_pose, target_map_pose, _ = world.RandomInitPoseInRoom(room_position)
            # get a long path
            map_plan, real_route = world.GetPath(init_map_pose+target_map_pose)

            dynamic_route = copy.deepcopy(real_route)
            env.LongPathPublish(real_route)
            time.sleep(1.)

            if len(map_plan) == 0:
                print 'no path'
                continue

            pose = env.GetSelfStateGT()
            target_table_point_list, cmd_list, check_point_list = world.GetCommandPlan(pose, real_route)

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
            action = [0., 0.]
            epi_q = []
            loop_time = []
            t = 0
            terminal = False
            result = 0
            cmd_idx = 0
            status = [0]
            prev_action = [0., 0.]
            err_h_epi, err_c_epi = 0., 0.
            status_cnt = 5
            if np.random.rand() <= epsilon:
                true_flag = True
                print '-------using true actions------'
            else:
                true_flag = False
            seq = []
            prev_state = (np.zeros((1, agent.n_hidden)), np.zeros((1, agent.n_hidden)))
            while not rospy.is_shutdown():
                start_time = time.time()
                if t == 0:
                    print 'current cmd:', cmd_list[cmd_idx]
                
                terminal, result, reward = env.GetRewardAndTerminate(t)
                total_reward += reward

                if t > 0:
                    status = [1] if result == 1 else [0]
                    seq.append((laser_stack,
                                   cmd,
                                   cmd_next,
                                   cmd_skip,
                                   prev_action,
                                   local_goal,
                                   local_goal_a,
                                   action, 
                                   reward,
                                   status,
                                   true_action))
                    prev_state = copy.deepcopy(state)
                    prev_action = copy.deepcopy(action)
                if result > 1:
                    break
                elif result == 1:
                    if cmd_list[cmd_idx] == 5:
                        print 'Finish!!!!!!!'
                        break
                    cmd_idx += 1
                    t = 0
                    status_cnt = 0
                    table_goal = target_table_point_list[cmd_idx]
                    goal = world.Table2RealPosition(table_goal)
                    env.target_point = goal
                    env.distance = np.sqrt(np.linalg.norm([pose[0]-goal[0], pose[1]-goal[1]]))

                    continue

                local_goal = env.GetLocalPoint(goal)
                env.PathPublish(local_goal)

                cmd = [cmd_list[cmd_idx]]
                cmd_next = [cmd_list[cmd_idx]]
                cmd_skip = [cmd_list[cmd_idx+1]]

                if status_cnt < 5:
                    cmd = [0]
                    cmd_skip = [0]
                else:
                    cmd_next = [0]
                status_cnt += 1

                # print '{}, {}, {}'.format(cmd[0], cmd_next[0], cmd_skip[0])

                laser = env.GetLaserObservation()
                laser_stack = np.stack([laser, laser_stack[:, 0], laser_stack[:, 1]], axis=-1)        

                if not flags.testing:
                    if noise > 0:
                        noise -= flags.init_noise/flags.noise_stop_episode
                        epsilon -= flags.init_epsilon/flags.noise_stop_episode

                time1 = time.time() - start_time
                
               
                # predict action
                if t > 0:
                    prev_action = action
                if cmd_list[cmd_idx] == 5:
                    local_goal_a = local_goal
                else:
                    local_goal_a = [0., 0.]
                # input_laser, input_cmd, input_cmd_next, prev_status, prev_action, input_goal, prev_state
                action, state = agent.ActorPredict([laser_stack], 
                                                    [cmd], 
                                                    [cmd_next], 
                                                    [action], 
                                                    [local_goal_a],
                                                    prev_state)

                if not flags.testing:
                    if T < flags.noise_stop_episode:
                        action += (exploration_noise.noise() * np.asarray(agent.action_range) * noise )
            
                # get true action
                pose = env.GetSelfStateGT()
                near_goal, dynamic_route = world.GetNextNearGoal(dynamic_route, pose)

                local_near_goal = env.GetLocalPoint(near_goal)
                true_action = env.Controller(local_near_goal, None, 1)

                if true_flag :
                    action = true_action

                env.SelfControl(action, agent.action_range)

                time2 = time.time() - start_time - time1


                if (T + 1) % flags.steps_per_checkpoint == 0 and not flags.testing:
                    saver.save(sess, os.path.join(model_dir, 'network') , global_step=T+1)

                t += 1
                T += 1
                loop_time.append(time.time() - start_time)
                time3 = time.time() - start_time - time1 - time2

                used_time = time.time() - start_time

                if used_time > 0.04:
                    print '{:.4f} | {:.4f} | {:.4f} | {:.4f}'.format(time1, time2, time3, used_time)
                rate.sleep()
            agent.Add2Mem(seq)
            if episode > agent.batch_size and not flags.testing:
                q = agent.Train()

            if episode > agent.batch_size and not flags.testing:
                if not true_flag:
                    summary = sess.run(merged, feed_dict={reward_ph: total_reward,
                                                          q_ph: np.amax(q),
                                                          noise_ph: noise,
                                                          epsilon_ph: epsilon,
                                                          err_h_ph: 0.,
                                                          err_c_ph: 0.,
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
                print 'Episode:{:} | Steps:{:} | Reward:{:.2f} | T:{:} |  Loop Time:{:.3f} '.format(episode, 
                                                                                                    t, 
                                                                                                    total_reward, 
                                                                                                    T,
                                                                                                    np.mean(loop_time))
            episode += 1

if __name__ == '__main__':
    robot_name = 'robot1'
    rviz = flags.rviz

    main(robot_name, rviz)