import numpy as np
import cv2
import tensorflow as tf
import math
import os
import random
import time
import random
import rospy
import pickle
import copy
import model.model_status as mode_basic
import utils.data_utils_tri_cmd as data_utils
import csv

from utils.model_utils import variable_summaries
from data_generation.GazeboWorld import GazeboWorld
from data_generation.GeneratorNoMoveBase import GridWorld, FileProcess
from utils.ou_noise import OUNoise

CWD = os.getcwd()
RANDOM_SEED = 1234

flag = tf.app.flags

# network param
flag.DEFINE_integer('batch_size', 64, 'Batch size to use during training.')
flag.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flag.DEFINE_integer('max_step', 700, 'max step.')
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
flag.DEFINE_boolean('encoder_training', False, 'whether to train encoder')
flag.DEFINE_float('keep_prob', 0.8, 'Drop out parameter.')
flag.DEFINE_float('a_linear_range', 0.3, 'linear action range: 0 ~ 0.4')
flag.DEFINE_float('a_angular_range', np.pi/6, 'angular action range: -np.pi/4 ~ np.pi/4')
flag.DEFINE_integer('gpu_num', 1, 'Number of GPUs')
flag.DEFINE_float('noise_track_rate', 0.1, 'The rate of tracking true error')
flag.DEFINE_boolean('single_lstm', False, 'Whether to use lstm for status prediction')

# training param
flag.DEFINE_string('data_dir',  os.path.join(CWD[:-7], 'lan_nav_data/room_new/validation_meta'),
                    'Data directory')
flag.DEFINE_string('vali_dir',  os.path.join(CWD[:-7], 'lan_nav_data/room_new/validation_meta'),
                    'Data directory')
flag.DEFINE_string('model_dir', os.path.join(CWD[:-7], 'lan_nav_data/saved_network'), 'saved model directory.')
flag.DEFINE_string('model_name', 'test_model', 'Training directory.')
flag.DEFINE_integer('max_epoch', 200, 'max epochs.')
flag.DEFINE_boolean('save_model', True, 'save model.')
flag.DEFINE_boolean('load_model', False, 'load model.')
flag.DEFINE_string('init_model_name', 'empty', 'model to initailize')
flag.DEFINE_boolean('test', False, 'whether to test.')

# ros param
flag.DEFINE_boolean('rviz', False, 'testing')
# noise param
flag.DEFINE_float('mu', 0., 'mu')
flag.DEFINE_float('theta', 0.15, 'theta')
flag.DEFINE_float('sigma', 0.3, 'sigma')

flags = flag.FLAGS

def LogData(pose, cmd_list, num, path):
    file_pose = open(path+'/'+str(num)+'pose.csv', 'w')
    writer = csv.writer(file_pose, delimiter=',', quotechar='|')
    for row in pose:
        if not isinstance(row, list):
            row = [row]
        writer.writerow(row)
    file_pose.close()

    file_cmd = open(path+'/'+str(num)+'cmd.csv', 'w')
    writer = csv.writer(file_cmd, delimiter=',', quotechar='|')
    for row in cmd_list:
        if not isinstance(row, list):
            row = [row]
        writer.writerow(row)
    file_cmd.close()

def testing(sess, model):  
    # --------------------load model-------------------------
    model_dir = os.path.join(flags.model_dir, flags.model_name)
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir)
    init_dir = os.path.join(flags.model_dir, flags.init_model_name)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    all_var = tf.global_variables()
    load_var = all_var[:21]
    trainable_var = tf.trainable_variables()
    part_var = []
    for idx, v in enumerate(load_var):
        print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))
        if 'encoder' in  v.name or 'controller' in v.name:
            part_var.append(v)
    saver = tf.train.Saver(trainable_var, max_to_keep=5)
    part_saver = tf.train.Saver(load_var, max_to_keep=1)

    # load model
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        part_saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Network model loaded: ", checkpoint.model_checkpoint_path)
    else:
        print('No model is found')

    # ----------------------init env-------------------------
    # np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)
    world = GridWorld()
    switch_action = False
    # world.map, switch_action = world.RandomSwitchRoom()
    world.GetAugMap()
    world.CreateTable()
    cv2.imwrite('./world/map.png', np.flipud(1-world.map)*255)

    FileProcess()

    robot_name = 'robot1'
    rviz = False
    env = GazeboWorld(world.table, robot_name, rviz=rviz)
    print "Env initialized"
    time.sleep(2.)

    rate = rospy.Rate(5.)
    

    pickle_path = os.path.join(CWD, 'world/model_states_data.p')

    env.GetModelStates()
    env.ResetWorld()
    env.ResetModelsPose(pickle_path)
    if switch_action:
        env.SwitchRoom(switch_action)

    env.state_call_back_flag = True
    env.target_theta_range = 1.
    time.sleep(2.)


    init_pose = world.RandomInitPose()
    env.target_point = init_pose

    episode = 0
    T = 0

    # ------------------start to test------------------------
    room_position = np.array([-1, -1])
    SR_cnt = 0
    TIME_list = []
    POSE_list = []
    while not rospy.is_shutdown():
        print ''
        time.sleep(1.)
        # set robot in a room
        init_pose, init_map_pose, room_position = world.RandomInitPoseInRoom(room_position)
        env.SetObjectPose(robot_name, [init_pose[0], init_pose[1], 0., init_pose[2]])
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
        target_table_point_list, cmd_list, checkpoint_list = world.GetCommandPlan(pose, real_route)
        cmd_list += [0]
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
        prob = 0.
        used_action = [0., 0.]
        true_action = [0., 0.]
        logits = [0., 0.]
        cnt = 5
        status = 0
        prev_status = 0
        exploration_noise = OUNoise(action_dimension=flags.dim_action, 
                                    mu=flags.mu, theta=flags.theta, sigma=flags.sigma)
        epi_start_time = time.time()
        positions = []
        CMD_list = []
        while not rospy.is_shutdown():
            start_time = time.time()
            if t == 0:
                print 'current cmd:', cmd_list[cmd_idx]
            prev_result = result
            terminal, result, reward = env.GetRewardAndTerminate(t)
            total_reward += reward
            # if t > 0:
            #     print  '{}, {}, | {:.3f}, {:.3f} | {}, {}, {}'.format(result, status, logits[0], logits[1], cmd[0], cmd_next[0], cmd_skip[0])  
            true_status = 0
            if result > 1:
                break
            # elif result == 1:
            #     true_status = 1
            #     if cmd_list[cmd_idx] == 5:
            #         print 'Finish!!!!!!!'
            #         break
            #     cmd_idx += 1
            #     t = 0
            #     table_goal = target_table_point_list[cmd_idx]
            #     goal = world.Table2RealPosition(table_goal)
            #     env.target_point = goal
            #     env.distance = np.sqrt(np.linalg.norm([pose[0]-goal[0], pose[1]-goal[1]]))
            #     continue



            local_goal = env.GetLocalPoint(goal)
            env.PathPublish(local_goal)

            if cnt < 5:
                cnt += 1
                cmd = [0]
                cmd_next = [cmd_list[cmd_idx]]
                cmd_skip = [0]
            else:
                cmd = [cmd_list[cmd_idx]]
                cmd_next = [0]
                cmd_skip = [cmd_list[cmd_idx+1]]
            CMD_list.append(cmd[0])
            curr_status = [cmd[0] * flags.n_cmd_type + cmd_next[0]]
            next_status = [cmd_next[0] * flags.n_cmd_type + cmd_skip[0]]
            prev_laser = laser 
            laser = env.GetLaserObservation()
            if np.random.rand():
                pass
            laser_stack = np.stack([laser, laser_stack[:, 0], laser_stack[:, 1]], axis=-1)

            # predict action
            if cmd_list[cmd_idx] == 5:
                local_goal_a = local_goal
            else:
                local_goal_a = [0., 0.]

            prev_status = copy.deepcopy(status)
            status, action, logits, _ = model.Predict([laser_stack], [curr_status], [next_status], [local_goal_a], t, 
                                            [[used_action[0], used_action[1]]])
            action += (exploration_noise.noise() * np.asarray(model.action_range) * 0.1)
            if prev_status == 0 and status == 1 and cnt >= 5:
                if cmd_list[cmd_idx] == 5:
                    print 'Finish!!!!!!!'
                    break
                cmd_idx += 1
                t = 0
                cnt = 0
                table_goal = target_table_point_list[cmd_idx]
                goal = world.Table2RealPosition(table_goal)
                env.target_point = goal
                env.distance = np.sqrt(np.linalg.norm([pose[0]-goal[0], pose[1]-goal[1]]))
                continue


            # get action
            pose = env.GetSelfStateGT()
            near_goal, dynamic_route = world.GetNextNearGoal(dynamic_route, pose)

            local_near_goal = env.GetLocalPoint(near_goal)
            true_action = env.Controller(local_near_goal, None, 1)
            positions.append(pose[:2])
            # print action - np.asarray(true_action), action
            if cmd_list[cmd_idx] == 0:
                used_action = true_action
            else:
                used_action = action

            env.SelfControl(used_action, [flags.a_linear_range, flags.a_angular_range])

            t += 1
            T += 1
            loop_time.append(time.time() - start_time)
            rate.sleep()


        print 'Episode:{:} | Steps:{:} | Reward:{:.2f} | T:{:}'.format(episode, 
                                                                       t, 
                                                                       total_reward, 
                                                                       T)
        episode += 1
        if cmd[0] == 5:
            SR_cnt += 1.0
            TIME_list.append(time.time() - epi_start_time)
            # LogData(positions, CMD_list, int(SR_cnt), os.path.join(CWD, 'experiments/positions'))
            print 'SR: {:.3f} | AVG TIME: {:.3f}, | MAX TIME: {:.3f}'.format(SR_cnt/episode, np.mean(TIME_list), np.amax(TIME_list))


def offline_testing(sess, model):
    # --------------------load model-------------------------
    model_dir = os.path.join(flags.model_dir, flags.model_name)
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir)
    init_dir = os.path.join(flags.model_dir, flags.init_model_name)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    trainable_var = tf.trainable_variables()
    part_var = []
    for idx, v in enumerate(trainable_var):
        print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))
        if 'encoder' in  v.name or 'controller' in v.name:
            part_var.append(v)
    saver = tf.train.Saver(max_to_keep=5)
    part_saver = tf.train.Saver(part_var, max_to_keep=1)

    # load model
    if flags.load_model:
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Network model loaded: ", checkpoint.model_checkpoint_path)
        else:
            print('No model is found')
    if 'empty' not in flags.init_model_name:
        checkpoint = tf.train.get_checkpoint_state(init_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Network model loaded: ", checkpoint.model_checkpoint_path)
        else:
            print('No model is found')

    train_path = flags.data_dir
    train_file_list = os.listdir(train_path)
    n = 0
    meta_folder_name = train_file_list[0]
    data, batch_num = data_utils.read_meta_file_insert_zeros(os.path.join(train_path, meta_folder_name), 
                                                flags.max_step, 
                                                flags.batch_size)

    #'laser', 'action', 'cmd', 'cmd_next', 'cmd_skip', 'obj_pose', 'status', 'length'
    seq = 1
    laser = data[0][seq][0]
    action = data[1][seq][0]
    cmd = data[2][seq][0]
    cmd_next = data[3][seq][0]
    cmd_skip = data[4][seq][0]
    obj_pose = data[5][seq][0]
    status = data[6][seq][0]
    length = data[7][seq][0]

    # stack laser
    laser_expand = np.expand_dims(laser, axis=2) # l, beam, 1
    laser_hist_1 = np.concatenate([np.expand_dims(laser_expand[0], axis=0), laser_expand[0:-1]], axis=0)
    laser_hist_2 = np.concatenate([np.expand_dims(laser_expand[0], axis=0), np.expand_dims(laser_expand[0], axis=0), laser_expand[0:-2]], axis=0)
    laser_stack = np.concatenate([laser_expand, laser_hist_1, laser_hist_2], axis=2)
    # print laser_expand[0:10, 0:10]
    # print '----------------------------'
    # print laser_stack[0:10, 0:10]

    # prev action
    action_0 = np.zeros([1, 2])
    action_0_tm1 = action[:-1]
    prev_action = np.concatenate([action_0, action_0_tm1], axis=0)

    # make status
    status_curr = cmd * flags.n_cmd_type + cmd_next
    status_next = cmd_next * flags.n_cmd_type + cmd_skip


    for t in xrange(int(length)):
        laser_stack_t = laser_stack[t]
        status_t = status_curr[t]
        status_next_t = status_next[t]
        obj_pose_t = obj_pose[t]
        prev_action_t = prev_action[t]
        status_pred, _, logits, test_input = model.Predict([laser_stack_t], [status_t], [status_next_t], [obj_pose_t], t, 
                                        [prev_action_t])
        print status[t], status_pred, logits

    # logits, status_pred, laser_stack_tf, train_input, train_prev_actoion = sess.run([model.training_logits, 
    #                                                 model.training_pred, 
    #                                                 model.laser_stack,
    #                                                 model.training_input,
    #                                                 model.prev_action], feed_dict={
    #     model.input_placeholder[0]: [laser],
    #     model.input_placeholder[1]: [action],
    #     model.input_placeholder[2]: [cmd],
    #     model.input_placeholder[3]: [cmd_next],
    #     model.input_placeholder[4]: [cmd_skip],
    #     model.input_placeholder[5]: [obj_pose],
    #     model.input_placeholder[6]: [status],
    #     model.input_placeholder[7]: [length],
    #     model.keep_prob: 1.
    #     })
    # status = np.reshape(status, [-1]).astype(np.int32)
    # print '----------------------------'
    # print laser_stack_tf[0, 0:10, 0:10]
    # print prev_action[1]
    # print '--------------------'
    # print train_prev_actoion[1]
    # print '--------------------'
    # print train_input[1] - test_input[0]


def training(sess, model):
    train_path = flags.data_dir
    validation_path = flags.vali_dir

    model_dir = os.path.join(flags.model_dir, flags.model_name)
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir)
    init_dir = os.path.join(flags.model_dir, flags.init_model_name)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    all_var = tf.global_variables()
    load_var = all_var[:21]
    trainable_var = tf.trainable_variables()
    part_var = []
    for idx, v in enumerate(load_var):
        print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))
        if 'encoder' in  v.name or 'controller' in v.name:
            part_var.append(v)

    saver = tf.train.Saver( max_to_keep=5)
    part_saver = tf.train.Saver(load_var, max_to_keep=1)

    summary_writer = tf.summary.FileWriter(model_dir, sess.graph)

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

    print('data loaded')
    start_time = time.time()
    for epoch in range(flags.max_epoch):
        loss_sum = 0.0
        loss_status_sum = 0.0
        loss_action_sum = 0.0

        train_file_list = os.listdir(train_path)
        n = 0
        for meta_folder_name  in train_file_list:
            fetch_start_time = time.time()
            data, batch_num = data_utils.read_meta_file_insert_zeros(os.path.join(train_path, meta_folder_name), 
                                                        flags.max_step, 
                                                        flags.batch_size)
            # print 'meta file fetch time: {:.3f}'.format(time.time() - fetch_start_time)
            for batch_id in xrange(batch_num):
                step_start_time = time.time()
                input_feed = {}
                for name_id, inp in enumerate(model.input_placeholder):
                    input_feed[inp.name] = data[name_id][batch_id]
                input_feed[model.keep_prob.name] = flags.keep_prob

                (_, 
                 loss, 
                 loss_status, 
                 loss_action,
                 ) = sess.run((model.optim, 
                               model.objective, 
                               model.loss_status, 
                               model.loss_action,
                               ), input_feed)
                # print 'step train time: {:.3f}'.format(time.time() - step_start_time)
                loss_sum += loss
                loss_status_sum += loss_status
                loss_action_sum += loss_action
                n += 1
            data = []
            input_feed = {}

        total_batch = n
        print 'Train batches: ' + str(total_batch)

        # -----------------------------------------------------
        # validation
        vali_loss_sum = 0.0
        vali_loss_status_sum = 0.0
        vali_loss_action_sum = 0.0

        validation_file_list = os.listdir(validation_path)
        n = 0
        for meta_folder_name  in validation_file_list:
            data, batch_num = data_utils.read_meta_file_insert_zeros(os.path.join(validation_path, meta_folder_name), 
                                                        flags.max_step, 
                                                        flags.batch_size)
            for batch_id in xrange(batch_num):
                step_start_time = time.time()
                input_feed = {}
                input_feed[model.keep_prob.name] = 1.
                for name_id, inp in enumerate(model.input_placeholder):
                    input_feed[inp.name] = data[name_id][batch_id]
                
                (loss, 
                 loss_status, 
                 loss_action) = sess.run((model.objective, 
                                          model.loss_status, 
                                          model.loss_action, 
                                          ), input_feed)
                vali_loss_sum += loss
                vali_loss_status_sum += loss_status
                vali_loss_action_sum += loss_action
                n += 1


        total_vali_batch = copy.deepcopy(batch_num)
        print 'Vali batches: ' + str(total_vali_batch)

        total_train = total_batch * flags.batch_size
        total_validation = total_vali_batch * flags.batch_size
        info_train = '| Epoch:{:3d}'.format(epoch) + \
              '|| TRAINING  ' + \
              '| Loss:{:3.5f}'.format(loss_sum/(total_train)) + \
              '| Status:{:3.5f}'.format(loss_status_sum/(total_train)) + \
              '| Action:{:3.5f}'.format(loss_action_sum/(total_train))
        info_veli = '\n| Time: {:2.1f}'.format((time.time() - start_time)/3600.) + \
              '|| VALIDATION'  + \
              '| Loss:{:3.5f}'.format(vali_loss_sum/(total_validation)) + \
              '| Status:{:3.5f}'.format(vali_loss_status_sum/(total_validation)) + \
              '| Action:{:3.5f}'.format(vali_loss_action_sum/(total_validation))

        print info_train + info_veli

        if flags.save_model and (epoch+1)%5 == 0:
            saver.save(sess, os.path.join(model_dir, 'network') , global_step=epoch)

        summary = tf.Summary()
        summary.value.add(tag='Training/loss', simple_value=float(loss_sum/total_train))
        summary.value.add(tag='Training/status', simple_value=float(loss_status_sum/total_train))
        summary.value.add(tag='Training/action', simple_value=float(loss_action_sum/total_train))
        summary.value.add(tag='Validation/loss', simple_value=float(vali_loss_sum/total_validation))
        summary.value.add(tag='Validation/status', simple_value=float(vali_loss_status_sum/total_validation))
        summary.value.add(tag='Validation/action', simple_value=float(vali_loss_action_sum/total_validation))
        
        summary_writer.add_summary(summary, epoch)
        summary_writer.flush()

def main():
    config = tf.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = mode_basic.Nav(sess=sess,
                               batch_size=flags.batch_size,
                               learning_rate=flags.learning_rate,
                               max_step=flags.max_step,
                               n_layers=flags.n_layers,
                               n_hidden=flags.n_hidden,
                               n_cmd_type=flags.n_cmd_type,
                               dim_emb=flags.dim_emb,
                               dim_laser=[flags.dim_laser_b, flags.dim_laser_c],
                               dim_goal=flags.dim_goal,
                               dim_cmd=flags.dim_cmd,
                               dim_action=flags.dim_action,
                               action_range=[flags.a_linear_range, flags.a_angular_range],
                               loss_weight=[flags.loss_w0, flags.loss_w1],
                               encoder_training=flags.encoder_training,
                               keep_prob=flags.keep_prob,
                               gpu_num=flags.gpu_num,
                               noise_track_rate=flags.noise_track_rate,
                               single_lstm=flags.single_lstm
                               )
        if not flags.test:
            training(sess, model)
        else:
            testing(sess, model)
        # offline_testing(sess, model)

if __name__ == '__main__':
    main()  