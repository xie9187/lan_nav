import numpy as np
import tensorflow as tf
import math
import os
import random
import csv
import model.model_parallel as mode_basic
import matplotlib.pyplot as plt
from model.ddpg import DDPG
from model.rdpg import RDPG
from model.rdpg_BPTT import RDPG_BPTT

CWD = os.getcwd()


def supervised_training_test():
    flag = tf.app.flags
    # network param
    flag.DEFINE_integer('batch_size', 2, 'Batch size to use during training.')
    flag.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
    flag.DEFINE_integer('max_step', 25, 'max step.')
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
    flag.DEFINE_float('loss_w2', 1., 'loss weight2')
    flag.DEFINE_boolean('encoder_training', True, 'whether to train encoder')
    flag.DEFINE_float('keep_prob', 0.8, 'Drop out parameter.')

    flags = flag.FLAGS

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
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
                           loss_weight=[flags.loss_w0,flags.loss_w1,flags.loss_w2],
                           encoder_training=flags.encoder_training,
                           keep_prob=flags.keep_prob
                           )

    init = tf.global_variables_initializer()
    sess.run(init)

    # fake data
    input_laser = np.random.rand(flags.batch_size*flags.max_step, model.dim_laser[0], model.dim_laser[1])
    input_cmd = np.array([[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5],
                          [5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1]])
    input_cmd = np.expand_dims(input_cmd, axis=2)
    input_cmd = np.reshape(input_cmd, [-1, flags.dim_cmd])

    input_cmd_next = np.array([[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5],
                               [5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1]])
    input_cmd_next = np.expand_dims(input_cmd_next, axis=2)
    input_cmd_next = np.reshape(input_cmd_next, [-1, flags.dim_cmd])

    input_obj_goal = np.random.rand(flags.batch_size*flags.max_step, flags.dim_goal)

    label_action = np.random.rand(flags.batch_size*flags.max_step, flags.dim_action)

    keep_prob = 0.8
    length = [10, 20]

    out = model.Train(input_laser,
                    input_cmd,
                    input_cmd_next,
                    input_obj_goal,
                    label_action,
                    length,
                    keep_prob
                    )
    objective, loss_goal, loss_status, loss_action, label_status, _ = out
    print 'Train test'
    print label_status


    # fake data
    input_laser = np.random.rand(1, model.dim_laser[0], model.dim_laser[1])
    input_cmd = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
    input_cmd_next = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
    input_obj_goal = np.random.rand(1, flags.dim_goal)

    print 'Predict test'
    for t in xrange(0, 21):
        out = model.Predict(input_laser,
                            np.asarray([[input_cmd[t]]]),
                            np.asarray([[input_cmd_next[t]]]),
                            input_obj_goal,
                            t)
        pred_goal, pred_status, pred_action, state = out
        print pred_status


def ddpg_test():
    CWD = os.getcwd()

    tf_flags = tf.app.flags

    # network param
    tf_flags.DEFINE_float('a_learning_rate', 1e-4, 'Actor learning rate.')
    tf_flags.DEFINE_float('c_learning_rate', 1e-3, 'Critic learning rate.')
    tf_flags.DEFINE_integer('batch_size', 32, 'Batch size to use during training.')
    tf_flags.DEFINE_integer('n_hidden', 256, 'Size of each model layer.')
    tf_flags.DEFINE_integer('n_layers', 1, 'Number of rnn layers in the model.')
    tf_flags.DEFINE_integer('max_steps', 10, 'Max number of steps in an episode.')
    tf_flags.DEFINE_integer('dim_action', 2, 'Dimension of action.')
    tf_flags.DEFINE_integer('dim_laser_b', 666, 'Laser beam number.')
    tf_flags.DEFINE_integer('dim_laser_c', 3, 'Laser channel.')
    tf_flags.DEFINE_integer('dim_goal', 2, 'Dimension of goal.')
    tf_flags.DEFINE_integer('dim_emb', 64, 'Dimension of embedding.')
    tf_flags.DEFINE_float('a_linear_range', 0.4, 'Range of the linear speed')
    tf_flags.DEFINE_float('a_angular_range', np.pi/4, 'Range of the angular speed')
    tf_flags.DEFINE_float('tau', 0.01, 'Target network update rate')
    tf_flags.DEFINE_boolean('actor_training', True, 'Whether to train actor')

    # training param
    tf_flags.DEFINE_integer('total_steps', 1000000, 'Total training steps.')
    tf_flags.DEFINE_string('model_dir', os.path.join(CWD, 'saved_network'), 'saved model directory.')
    tf_flags.DEFINE_string('model_name', 'model', 'Name of the model.')
    tf_flags.DEFINE_integer('steps_per_checkpoint', 10000, 'How many training steps to do per checkpoint.')
    tf_flags.DEFINE_integer('buffer_size', 1000, 'The size of Buffer')
    tf_flags.DEFINE_float('gamma', 0.99, 'reward discount')

    # noise param
    tf_flags.DEFINE_float('mu', 0., 'mu')
    tf_flags.DEFINE_float('theta', 0.15, 'theta')
    tf_flags.DEFINE_float('sigma', 0.3, 'sigma')

    flags = tf_flags.FLAGS


    with tf.Session() as sess:
        agent = DDPG(flags, sess)

        trainable_var = tf.trainable_variables()
        print "  [*] printing trainable variables"
        for idx, v in enumerate(trainable_var):
            print "  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name)

        sess.run(tf.global_variables_initializer())
        q_estimation = []
        T = 0
        for episode in xrange(1, 200):
            print episode
            q_list = []
            t = 0
            term = False
            while not term:
                if t == agent.max_steps - 1:
                    term = True
                else:
                    term = False

                if t > 0:
                    agent.Add2Mem((state_t, action, reward, terminal, state_t1))
                    state_t = state_t1

                laser_stack = np.ones([flags.dim_laser_b, flags.dim_laser_c])*t/np.float(agent.max_steps) 
                save_local_goal = [0., 0.]
                local_final_goal = [0., 0.]
                prev_action = [0., 0.]
                action = [0., 0.]
                reward = 1./agent.max_steps 
                terminal = term

                state_t1 = np.concatenate([np.reshape(laser_stack, [-1]),
                                          np.reshape(save_local_goal, [-1]),
                                          np.reshape(local_final_goal, [-1]),
                                          np.reshape(prev_action, [-1])], axis=0)
                if t == 0:
                    state_t = state_t1


                if T > agent.batch_size:
                    q = agent.Train()
                    q_list.append(np.amax(q))

                t += 1
                T += 1
            if T > agent.batch_size:
                q_estimation.append(np.amax(q_list))


        plt.plot(q_estimation, label='q_max')

        plt.legend()
        plt.show()


def read_file(file_name):
    file = open(file_name, 'r')
    if 'status' in file_name:
        file_reader = csv.reader(file, delimiter=',')
        curr_seq = []
        for row in file_reader:
            if 'True' in row[0]:
                curr_seq.append([1])
            else:
                curr_seq.append([0])
    else:
        file_reader = csv.reader(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
        curr_seq = []
        for row in file_reader:
            curr_seq.append(row)

    file.close()
    return curr_seq

def rdpg_test():
    flag = tf.app.flags

    # network param
    flag.DEFINE_integer('batch_size', 64, 'Batch size to use during training.')
    flag.DEFINE_float('a_learning_rate', 1e-4, 'Actor learning rate.')
    flag.DEFINE_float('c_learning_rate', 1e-3, 'Critic learning rate.')
    flag.DEFINE_integer('max_step', 10, 'max step.')
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
    flag.DEFINE_float('a_linear_range', 0.4, 'linear action range: 0 ~ 0.4')
    flag.DEFINE_float('a_angular_range', np.pi/4, 'angular action range: -np.pi/4 ~ np.pi/4')
    flag.DEFINE_integer('gpu_num', 1, 'Number of GPUs')
    flag.DEFINE_float('tau', 0.01, 'Target network update rate')

    # training param
    flag.DEFINE_integer('total_steps', 1000000, 'Total training steps.')
    flag.DEFINE_string('model_dir', os.path.join(CWD, 'saved_network'), 'saved model directory.')
    flag.DEFINE_string('model_name', 'model', 'Name of the model.')
    flag.DEFINE_integer('steps_per_checkpoint', 10000, 'How many training steps to do per checkpoint.')
    flag.DEFINE_integer('buffer_size', 10000, 'The size of Buffer')
    flag.DEFINE_float('gamma', 0.99, 'reward discount')

    # noise param
    flag.DEFINE_float('mu', 0., 'mu')
    flag.DEFINE_float('theta', 0.15, 'theta')
    flag.DEFINE_float('sigma', 0.3, 'sigma')

    flags = flag.FLAGS

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        agent = RDPG(flags, sess)

        trainable_var = tf.trainable_variables()
        print "  [*] printing trainable variables"
        for idx, v in enumerate(trainable_var):
            print "  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name)

        sess.run(tf.global_variables_initializer())
        q_estimation = []
        T = 0
        for episode in xrange(1, 200):
            print episode
            q_epi = []
            for t in xrange(0, agent.max_step):
                # (laser, cmd, cmd_next, prev_status, prev_action, obj_goal, prev_state, action, r, terminate)
                if t < agent.max_step-1:
                    term = False
                else:
                    term = True
                sample = (np.ones([flags.dim_laser_b, flags.dim_laser_c])*t/np.float(agent.max_step), 
                            [0], 
                            [0], 
                            [0],
                            [0., 0.], 
                            [0., 0.],
                            (np.zeros((1, agent.n_hidden)), np.zeros((1, agent.n_hidden))),
                            [0., 0.],
                            1./agent.max_step,
                            term)
                agent.Add2Mem(sample)
                T += 1
                q = agent.critic.PredictOnline([np.ones([flags.dim_laser_b, flags.dim_laser_c])*t/np.float(agent.max_step)], 
                                                [[0]], 
                                                [[0]], 
                                                [[0]],
                                                [[0., 0.]], 
                                                [[0., 0.]],
                                                [[0., 0.]])
                q_epi.append(q[0]) 

                if T > agent.batch_size:
                    agent.Train()
            q_estimation.append(q_epi)
            agent.UpdateState()
        q_estimation = np.asarray(q_estimation)
        # print q_estimation
        for t in xrange(agent.max_step):
            plt.plot(q_estimation[:, t], label='step{}'.format(t))

        plt.legend()
        plt.show()

def rdpg_BPTT_test():
    flag = tf.app.flags

    # network param
    flag.DEFINE_integer('batch_size', 32, 'Batch size to use during training.')
    flag.DEFINE_float('a_learning_rate', 1e-4, 'Actor learning rate.')
    flag.DEFINE_float('c_learning_rate', 1e-3, 'Critic learning rate.')
    flag.DEFINE_integer('max_step', 10, 'max step.')
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
    flag.DEFINE_float('a_linear_range', 0.4, 'linear action range: 0 ~ 0.4')
    flag.DEFINE_float('a_angular_range', np.pi/4, 'angular action range: -np.pi/4 ~ np.pi/4')
    flag.DEFINE_integer('gpu_num', 1, 'Number of GPUs')
    flag.DEFINE_float('tau', 0.01, 'Target network update rate')
    flag.DEFINE_boolean('demo_flag', False, ' ')

    # training param
    flag.DEFINE_integer('total_steps', 1000000, 'Total training steps.')
    flag.DEFINE_string('model_dir', os.path.join(CWD, 'saved_network'), 'saved model directory.')
    flag.DEFINE_string('model_name', 'model', 'Name of the model.')
    flag.DEFINE_integer('steps_per_checkpoint', 10000, 'How many training steps to do per checkpoint.')
    flag.DEFINE_integer('buffer_size', 10000, 'The size of Buffer')
    flag.DEFINE_float('gamma', 0.99, 'reward discount')

    # noise param
    flag.DEFINE_float('mu', 0., 'mu')
    flag.DEFINE_float('theta', 0.15, 'theta')
    flag.DEFINE_float('sigma', 0.3, 'sigma')

    flags = flag.FLAGS
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        agent = RDPG_BPTT(flags, sess)

        trainable_var = tf.trainable_variables()
        print "  [*] printing trainable variables"
        for idx, v in enumerate(trainable_var):
            print "  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name)

        sess.run(tf.global_variables_initializer())
        q_estimation = []
        T = 0
        for episode in xrange(1, 2000):
            print episode
            q_epi = []
            seq = []
            for t in xrange(0, agent.max_step):
                                       # seqs of (laser_stack, cmd, cmd_next, cmd_skip, 
                                       #          prev_action, goal, a_goal, action, 
                                       #          r, status, action_label)
                if t < agent.max_step-1:
                    term = False
                else:
                    term = True
                sample = (np.ones([flags.dim_laser_b, flags.dim_laser_c])*t/np.float(agent.max_step), 
                            [0], 
                            [0], 
                            [0],
                            [0., 0.], 
                            [0., 0.],
                            [0., 0.],
                            [0., 0.],
                            1./agent.max_step,
                            int(term),
                            [0., 0.])
                seq.append(sample)
                T += 1
            agent.Add2Mem(seq)
            if episode > agent.batch_size:
                q = agent.Train()
                q_estimation.append(q[:agent.max_step])
            else:
                q_estimation.append(np.zeros([agent.max_step, 1]))
        q_estimation = np.hstack(q_estimation)
        # print q_estimation
        for t in xrange(agent.max_step):
            plt.plot(q_estimation[t], label='step{}'.format(t))

        plt.legend()
        plt.show()

  

def main():
    # supervised_training_test()

    # ddpg_test()

    # data = read_file(os.path.join(CWD[:-7], 'lan_nav_data/room_new/linhai-Intel-Z370_robot1/0_status.csv'))
    # print data
    # data = read_file(os.path.join(CWD[:-7], 'lan_nav_data/room_new/linhai-Intel-Z370_robot1/0_cmd.csv'))
    # print data

    # rdpg_test()
    rdpg_BPTT_test()

if __name__ == '__main__':
    main()  