import tensorflow as tf
import numpy as np
import os
import copy
import time
import utils.model_utils as model_utils
import matplotlib.pyplot as plt
from tensorflow.python.ops.rnn_cell import LSTMStateTuple

class Actor(object):
    def __init__(self,
                 sess,
                 dim_laser,
                 dim_action,
                 dim_cmd,
                 dim_goal,
                 dim_emb,
                 n_hidden,
                 n_cmd_type,
                 max_steps,
                 learning_rate,
                 batch_size,
                 action_range,
                 tau,
                 actor_training
                 ):

        self.sess = sess
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.action_range = action_range
        self.dim_action = dim_action
        self.dim_laser = dim_laser
        self.dim_cmd = dim_cmd
        self.dim_goal=dim_goal
        self.dim_emb = dim_emb
        self.n_cmd_type = n_cmd_type
        self.tau = tau
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.actor_training = actor_training

        with tf.variable_scope('actor'):
            self.input_laser = tf.placeholder(tf.float32, [None, dim_laser[0], dim_laser[1]], name='input_laser')        
            self.input_prev_action = tf.placeholder(tf.float32, [None, dim_action], name='input_prev_action')
            self.input_cmd = tf.placeholder(tf.int64, shape=[None, dim_cmd], name='input_cmd')
            self.input_cmd_next = tf.placeholder(tf.int64, shape=[None, dim_cmd], name='input_cmd_next')
            self.input_goal = tf.placeholder(tf.float32, [None, dim_goal], name='input_goal')

            with tf.variable_scope('online'):
                self.a_online = self.Model(self.actor_training)
            self.network_params = tf.trainable_variables()

            with tf.variable_scope('target'):
                self.a_target = self.Model(self.actor_training)
            self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + \
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))] 

        # This gradient will be provided by the critic network
        self.a_gradient = tf.placeholder(tf.float32, [None, self.dim_action]) # b, 2

        # Combine the gradients here
        self.gradients = tf.gradients(self.a_online, self.network_params, -self.a_gradient)

        # Optimization Op by applying gradient, variable pairs
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def Model(self, training):     
        with tf.variable_scope('encoder'):
            conv1 = model_utils.Conv1D(self.input_laser, 2, 5, 4, scope='conv1', trainable=training)
            conv2 = model_utils.Conv1D(conv1, 4, 5, 4, scope='conv2', trainable=training)
            conv3 = model_utils.Conv1D(conv2, 8, 5, 4, scope='conv3', trainable=training)
            shape = conv3.get_shape().as_list()
            vector_laser = tf.reshape(conv3, (-1, shape[1]*shape[2]))

            embedding_w_goal = tf.get_variable('embedding_w_goal', [self.dim_action, self.dim_emb])
            embedding_b_goal = tf.get_variable('embedding_b_goal', [self.dim_emb])

            embedding_w_action = tf.get_variable('embedding_w_action', [self.dim_action, self.dim_emb], trainable=training)
            embedding_b_action = tf.get_variable('embedding_b_action', [self.dim_emb], trainable=training)
            
            embedding_cmd = tf.get_variable('cmd_embedding', [self.n_cmd_type, self.dim_emb]) 

        with tf.variable_scope('controller'):
            vector_goal = tf.matmul(self.input_goal, embedding_w_goal)+ embedding_b_goal
            vector_prev_action = tf.matmul(self.input_prev_action, embedding_w_action)+ embedding_b_action
            vector_cmd = tf.reshape(tf.nn.embedding_lookup(embedding_cmd, self.input_cmd), [-1, self.dim_emb])
            vector_cmd_next = tf.reshape(tf.nn.embedding_lookup(embedding_cmd, self.input_cmd_next), [-1, self.dim_emb])

            controller_input = tf.concat([vector_laser, vector_cmd, vector_cmd_next, vector_goal, vector_prev_action], axis=1)

            shape = controller_input.get_shape().as_list()
            w_hidden = tf.get_variable('w_hidden', [shape[1], self.n_hidden], 
                                initializer=tf.contrib.layers.xavier_initializer(), trainable=training)
            b_hidden = tf.get_variable('b_hidden', [self.n_hidden], 
                                initializer=tf.contrib.layers.xavier_initializer(), trainable=training)
            w_action_linear = tf.get_variable('w_action_linear', [self.n_hidden, self.dim_action/2], 
                                    initializer=tf.contrib.layers.xavier_initializer(), trainable=training)
            b_action_linear = tf.get_variable('b_action_linear', [self.dim_action/2], 
                                    initializer=tf.contrib.layers.xavier_initializer(), trainable=training)
            w_action_angular = tf.get_variable('w_action_angular', [self.n_hidden, self.dim_action/2], 
                                    initializer=tf.contrib.layers.xavier_initializer(), trainable=training)
            b_action_angular = tf.get_variable('b_action_angular', [self.dim_action/2], 
                                    initializer=tf.contrib.layers.xavier_initializer(), trainable=training)
            
            hidden = tf.nn.leaky_relu(tf.matmul(controller_input, w_hidden) + b_hidden) # b*l. n
            a_linear = tf.nn.sigmoid(tf.matmul(hidden, w_action_linear) + b_action_linear) * self.action_range[0]
            a_angular = tf.nn.tanh(tf.matmul(hidden, w_action_angular) + b_action_angular) * self.action_range[1]
            pred_action = tf.concat([a_linear, a_angular], axis=1)

        return pred_action

    def Train(self, input_laser, input_cmd, input_cmd_next, input_goal, input_prev_action, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.input_laser: input_laser,
            self.input_cmd: input_cmd,
            self.input_goal: input_goal,
            self.input_cmd_next: input_cmd_next,
            self.input_prev_action: input_prev_action,
            self.a_gradient: a_gradient
            })

    def PredictTarget(self, input_laser, input_cmd, input_cmd_next, input_goal, input_prev_action):
        return self.sess.run(self.a_target, feed_dict={
            self.input_laser: input_laser,
            self.input_cmd: input_cmd,
            self.input_goal: input_goal,
            self.input_cmd_next: input_cmd_next,
            self.input_prev_action: input_prev_action
            })

    def PredictOnline(self, input_laser, input_cmd, input_cmd_next, input_goal, input_prev_action):
        return self.sess.run(self.a_online, feed_dict={
            self.input_laser: input_laser,
            self.input_cmd: input_cmd,
            self.input_goal: input_goal,
            self.input_cmd_next: input_cmd_next,
            self.input_prev_action: input_prev_action
            })

    def UpdateTarget(self):
        self.sess.run(self.update_target_network_params)

    def TrainableVarNum(self):
        return self.num_trainable_vars



class Critic(object):
    def __init__(self,
                 sess,
                 dim_laser,
                 dim_cmd,
                 dim_action,
                 dim_goal,
                 dim_emb,
                 n_cmd_type,
                 n_hidden,
                 max_steps,
                 learning_rate,
                 batch_size,
                 num_actor_vars,
                 tau
                 ):

        self.sess = sess
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.tau = tau
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.dim_action = dim_action
        self.dim_laser = dim_laser
        self.dim_goal = dim_goal
        self.dim_emb = dim_emb
        self.dim_cmd = dim_cmd
        self.n_cmd_type = n_cmd_type

        with tf.variable_scope('critic'):
            self.input_laser = tf.placeholder(tf.float32, [None, dim_laser[0], dim_laser[1]], name='input_laser')
            self.input_cmd = tf.placeholder(tf.int64, shape=[None, dim_cmd], name='input_cmd')
            self.input_cmd_next = tf.placeholder(tf.int64, shape=[None, dim_cmd], name='input_cmd_next')
            self.input_action = tf.placeholder(tf.float32, [None, dim_action], name='input_action') 
            self.input_goal = tf.placeholder(tf.float32, [None, dim_goal], name='input_goal')
            self.input_prev_action = tf.placeholder(tf.float32, [None, dim_action], name='input_prev_action')

            with tf.variable_scope('online'):
                self.q_online = self.Model()
            self.network_params = tf.trainable_variables()[num_actor_vars:]

            with tf.variable_scope('target'):
                self.q_target = self.Model()
            self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        self.predicted_q = tf.placeholder(tf.float32, [self.batch_size, 1], name='predicted_q')
        self.square_diff = tf.pow(self.predicted_q - self.q_online, 2) # b, l, 1

        self.loss = tf.reduce_mean(self.square_diff)

        self.gradient = tf.gradients(self.loss, self.network_params)
        self.opt = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.opt.apply_gradients(zip(self.gradient, self.network_params))

        self.action_grads = tf.gradients(self.q_online, self.input_action)

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + \
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))] 


    def Model(self):
        with tf.variable_scope('encoder'):
            conv1 = model_utils.Conv1D(self.input_laser, 2, 5, 4, scope='conv1')
            conv2 = model_utils.Conv1D(conv1, 4, 5, 4, scope='conv2')
            conv3 = model_utils.Conv1D(conv2, 8, 5, 4, scope='conv3')
            shape = conv3.get_shape().as_list()
            vector_laser = tf.reshape(conv3, (-1, shape[1]*shape[2]))

            embedding_w_goal = tf.get_variable('embedding_w_goal', [self.dim_action, self.dim_emb])
            embedding_b_goal = tf.get_variable('embedding_b_goal', [self.dim_emb])

            embedding_w_action = tf.get_variable('embedding_w_action', [self.dim_action, self.dim_emb])
            embedding_b_action = tf.get_variable('embedding_b_action', [self.dim_emb])

            embedding_cmd = tf.get_variable('cmd_embedding', [self.n_cmd_type, self.dim_emb]) 

        with tf.variable_scope('controller'):
            vector_goal = tf.matmul(self.input_goal, embedding_w_goal) + embedding_b_goal           
            vector_prev_action = tf.matmul(self.input_prev_action, embedding_w_action)+embedding_b_action
            vector_action = tf.matmul(self.input_action, embedding_w_action)+embedding_b_action
            vector_cmd = tf.reshape(tf.nn.embedding_lookup(embedding_cmd, self.input_cmd), [-1, self.dim_emb])
            vector_cmd_next = tf.reshape(tf.nn.embedding_lookup(embedding_cmd, self.input_cmd_next), [-1, self.dim_emb])

            inputs = tf.concat([vector_laser, vector_cmd, vector_cmd_next, vector_goal, vector_prev_action, vector_action], axis=1)

            shape = inputs.get_shape().as_list()
            w_hidden = tf.get_variable('w_hidden', [shape[1], self.n_hidden], 
                                initializer=tf.contrib.layers.xavier_initializer())
            b_hidden = tf.get_variable('b_hidden', [self.n_hidden], 
                                initializer=tf.contrib.layers.xavier_initializer())
            w_q = tf.get_variable('w_q', [self.n_hidden, 1], initializer=tf.initializers.random_uniform(-0.003, 0.003))
            b_q = tf.get_variable('b_q', [1], initializer=tf.initializers.random_uniform(-0.003, 0.003))
    

        hidden = tf.nn.leaky_relu(tf.matmul(inputs, w_hidden)) + b_hidden

        q = tf.matmul(hidden, w_q) + b_q

        return q

    def Train(self, input_laser, input_cmd, input_cmd_next, input_goal, input_action, input_prev_action, predicted_q):
        return self.sess.run([self.q_online, self.optimize], feed_dict={
            self.input_laser: input_laser,
            self.input_cmd: input_cmd,
            self.input_cmd_next: input_cmd_next,
            self.input_goal: input_goal,
            self.input_action: input_action,
            self.input_prev_action: input_prev_action,
            self.predicted_q: predicted_q
            })

    def PredictOnline(self, input_laser, input_cmd, input_cmd_next, input_goal, input_action, input_prev_action):
        return self.sess.run(self.q_online, feed_dict={
            self.input_laser: input_laser,
            self.input_cmd: input_cmd,
            self.input_cmd_next: input_cmd_next,
            self.input_goal: input_goal,
            self.input_action: input_action,
            self.input_prev_action: input_prev_action
            })

    def PredictTarget(self, input_laser, input_cmd, input_cmd_next, input_goal, input_action, input_prev_action):
        return self.sess.run(self.q_target, feed_dict={
            self.input_laser: input_laser,
            self.input_cmd: input_cmd,
            self.input_cmd_next: input_cmd_next,
            self.input_goal: input_goal,
            self.input_action: input_action,
            self.input_prev_action: input_prev_action
            })

    def ActionGradients(self, input_laser, input_cmd, input_cmd_next, input_goal, input_action, input_prev_action):
        return self.sess.run(self.action_grads, feed_dict={
            self.input_laser: input_laser,
            self.input_cmd: input_cmd,
            self.input_cmd_next: input_cmd_next,
            self.input_goal: input_goal,
            self.input_action: input_action,
            self.input_prev_action: input_prev_action
            })

    def UpdateTarget(self):
        self.sess.run(self.update_target_network_params)



class DDPG(object):
    """docstring for DDPG"""
    def __init__(self, flags, sess):
        self.dim_laser = [flags.dim_laser_b, flags.dim_laser_c]
        self.dim_goal = flags.dim_goal
        self.dim_action = flags.dim_action
        self.dim_emb = flags.dim_emb
        self.dim_cmd = flags.dim_cmd
        self.n_hidden = flags.n_hidden
        self.n_cmd_type = flags.n_cmd_type
        self.a_learning_rate = flags.a_learning_rate
        self.c_learning_rate = flags.c_learning_rate
        self.batch_size = flags.batch_size
        self.max_steps = flags.max_steps
        self.tau = flags.tau
        self.action_range = [flags.a_linear_range, flags.a_angular_range]
        self.buffer_size = flags.buffer_size
        self.gamma = flags.gamma
        self.actor_training = flags.actor_training

        self.actor = Actor(sess=sess,
                           dim_laser=self.dim_laser,
                           dim_cmd=self.dim_cmd,
                           dim_action=self.dim_action,
                           dim_goal=self.dim_goal,
                           dim_emb=self.dim_emb,
                           n_cmd_type=self.n_cmd_type,
                           n_hidden=self.n_hidden,
                           max_steps=self.max_steps,
                           learning_rate=self.a_learning_rate,
                           batch_size=self.batch_size,
                           action_range=self.action_range,
                           tau=self.tau,
                           actor_training=self.actor_training)

        self.critic = Critic(sess=sess,
                             dim_laser=self.dim_laser,
                             dim_cmd=self.dim_cmd,
                             dim_action = self.dim_action,
                             dim_goal=self.dim_goal,
                             dim_emb=self.dim_emb,
                             n_cmd_type=self.n_cmd_type,
                             n_hidden=self.n_hidden,
                             max_steps=self.max_steps,
                             learning_rate=self.c_learning_rate,
                             batch_size=self.batch_size,
                             num_actor_vars=len(self.actor.network_params)+len(self.actor.target_network_params),
                             tau=self.tau)
        self.memory = []
        self.load_mem_flag = False
        
    def ActorPredict(self, input_laser, input_cmd, input_cmd_next, input_goal, input_prev_action):
        a = self.actor.PredictOnline(input_laser, input_cmd, input_cmd_next, input_goal, input_prev_action)
        return a

    def Add2Mem(self, sample):
        self.memory.append(sample) # (l, cmd, cmd_next, g, prev_a, a, r, t)
        if len(self.memory) > self.buffer_size:
            self.memory.pop(0)

    def SampleBatch(self):
        if len(self.memory) >= self.batch_size:

            indices = np.random.randint(1, len(self.memory)-1, size=self.batch_size)
            laser_t_batch = np.empty((self.batch_size, self.dim_laser[0], self.dim_laser[1]), dtype=np.float32)
            cmd_t_batch = np.empty((self.batch_size, self.dim_cmd), dtype=np.int64)
            cmd_next_t_batch = np.empty((self.batch_size, self.dim_cmd), dtype=np.int64)
            goal_t_batch = np.empty((self.batch_size, self.dim_goal), dtype=np.float32)
            goal_a_t_batch = np.empty((self.batch_size, self.dim_goal), dtype=np.float32)
            prev_action_t_batch = np.empty((self.batch_size, self.dim_action), dtype=np.float32)
            action_batch = np.empty((self.batch_size, self.dim_action), dtype=np.float32)
            reward_batch = np.empty((self.batch_size,), dtype=np.float32)
            terminate_batch = np.empty((self.batch_size,), dtype=bool)
            laser_t1_batch = np.empty((self.batch_size, self.dim_laser[0], self.dim_laser[1]), dtype=np.float32)
            cmd_t1_batch = np.empty((self.batch_size, self.dim_cmd), dtype=np.int64)
            cmd_next_t1_batch = np.empty((self.batch_size, self.dim_cmd), dtype=np.int64)
            goal_t1_batch = np.empty((self.batch_size, self.dim_goal), dtype=np.float32)
            goal_a_t1_batch = np.empty((self.batch_size, self.dim_goal), dtype=np.float32)
            prev_action_t1_batch = np.empty((self.batch_size, self.dim_action), dtype=np.float32)

            for i, idx in enumerate(indices):
                laser_t_batch[i] = self.memory[idx][0]
                cmd_t_batch[i] = self.memory[idx][1]
                cmd_next_t_batch[i] = self.memory[idx][2]
                goal_t_batch[i] = self.memory[idx][3]
                prev_action_t_batch[i] = self.memory[idx][4]
                action_batch[i] = self.memory[idx][5]
                reward_batch[i] = self.memory[idx][6]
                terminate_batch[i] = self.memory[idx][7]
                laser_t1_batch[i] = self.memory[idx+1][0]
                cmd_t1_batch[i] = self.memory[idx][1]
                cmd_next_t1_batch[i] = self.memory[idx][2]
                goal_t1_batch[i] = self.memory[idx+1][3]
                prev_action_t1_batch[i] = self.memory[idx+1][4]

                if cmd_t_batch[i] == 5:
                    goal_a_t_batch[i] = self.memory[idx][3]
                else:
                    goal_a_t_batch[i] = [0., 0.]
                if cmd_t1_batch[i] == 5:
                    goal_a_t1_batch[i] = self.memory[idx+1][3]
                else:
                    goal_a_t1_batch[i] = [0., 0.]

            return [laser_t_batch, cmd_t_batch, cmd_next_t_batch, goal_t_batch, goal_a_t_batch, prev_action_t_batch, 
                    action_batch, reward_batch, terminate_batch, 
                    laser_t1_batch, cmd_t1_batch, cmd_next_t1_batch, goal_t1_batch, goal_a_t1_batch, prev_action_t1_batch]
        else:
            print 'sample sequences are not enough'
            return None


    def Train(self):
        start_time = time.time()

        batch = self.SampleBatch()

        sample_time =  time.time() - start_time

        if batch is None:
            return 0.
        else:
            (laser_t_batch, cmd_t_batch, cmd_next_t_batch, goal_t_batch, goal_a_t_batch, prev_action_t_batch, 
             action_batch, reward_batch, terminate_batch, 
             laser_t1_batch, cmd_t1_batch, cmd_next_t1_batch, goal_t1_batch, goal_a_t1_batch, prev_action_t1_batch) = batch

            #compute target y
            target_a_t1_pred = self.actor.PredictTarget(input_laser=laser_t1_batch,
                                                        input_cmd=cmd_t1_batch,
                                                        input_cmd_next=cmd_next_t1_batch,
                                                        input_goal=goal_a_t1_batch,
                                                        input_prev_action=prev_action_t1_batch) # b, 2

            target_q_pred = self.critic.PredictTarget(input_laser=laser_t1_batch,
                                                      input_cmd=cmd_t1_batch,
                                                      input_cmd_next=cmd_next_t1_batch,
                                                      input_goal=goal_t1_batch, 
                                                      input_prev_action=prev_action_t1_batch, 
                                                      input_action=target_a_t1_pred) # b, 1
            y = []
            for i in xrange(self.batch_size):
                if terminate_batch[i]:
                    y.append(reward_batch[i])
                else:
                    y.append(reward_batch[i] + self.gamma * target_q_pred[i, 0])

            y = np.expand_dims(np.stack(y), axis=1)

            y_time = time.time() - start_time - sample_time

            # critic update
            q, _ = self.critic.Train(input_laser=laser_t_batch, 
                                     input_cmd=cmd_t_batch,
                                     input_cmd_next=cmd_next_t_batch,
                                     input_goal=goal_t_batch, 
                                     input_prev_action=prev_action_t_batch, 
                                     input_action=action_batch, 
                                     predicted_q=y)

            # actions for a_gradients from critic
            actions = self.actor.PredictOnline(input_laser=laser_t_batch, 
                                               input_cmd=cmd_t_batch,
                                               input_cmd_next=cmd_next_t_batch,
                                               input_goal=goal_a_t_batch,
                                               input_prev_action=prev_action_t_batch)

            # a_gradients
            a_gradients = self.critic.ActionGradients(input_laser=laser_t_batch, 
                                                      input_cmd=cmd_t_batch,
                                                      input_cmd_next=cmd_next_t_batch,
                                                      input_goal=goal_t_batch, 
                                                      input_prev_action=prev_action_t_batch, 
                                                      input_action=actions)                                                      

            # actor update
            self.actor.Train(input_laser=laser_t_batch, 
                             input_cmd=cmd_t_batch, 
                             input_cmd_next=cmd_next_t_batch,
                             input_goal=goal_a_t_batch,
                             input_prev_action=prev_action_t_batch, 
                             a_gradient=a_gradients[0])

            train_time = time.time() - start_time - sample_time - y_time

            # target networks update
            self.critic.UpdateTarget()
            self.actor.UpdateTarget()

            target_time = time.time() - start_time - sample_time - y_time - train_time

            # print 'sample_time:{:.3f}, y_time:{:.3f}, train_time:{:.3f}, target_time:{:.3f}'.format(sample_time,
            #                                                                                         y_time,
            #                                                                                         train_time,
            #                                                                                         target_time)
            
            return q