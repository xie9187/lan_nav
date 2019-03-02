import tensorflow as tf
import utils.model_utils as model_utils
import numpy as np
import copy
from tensorflow.python.ops.rnn_cell import LSTMStateTuple


class Critic(object):
    def __init__(self,
                 sess,
                 batch_size,
                 max_step,
                 n_layers,
                 n_hidden,
                 n_cmd_type,
                 num_actor_vars,
                 dim_emb=64,
                 dim_laser=[666, 3],
                 dim_goal=2,
                 dim_cmd=1,
                 dim_action=2,
                 gpu_num=1,
                 tau=0.1,
                 learning_rate=1e-3
                 ):
        self.sess = sess
        self.max_step = max_step
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_cmd_type = n_cmd_type
        self.dim_emb = dim_emb
        self.dim_laser = dim_laser
        self.dim_goal = dim_goal
        self.dim_cmd = dim_cmd
        self.dim_action = dim_action
        self.gpu_num = gpu_num
        self.tau = tau
        self.num_actor_vars = num_actor_vars
        self.learning_rate = learning_rate

        with tf.variable_scope('critic'):

            # training input
            self.input_laser = tf.placeholder(tf.float32, shape=[None, dim_laser[0], dim_laser[1]], name='input_laser')
            self.input_cmd = tf.placeholder(tf.int64, shape=[None, dim_cmd], name='input_cmd')
            self.input_cmd_next = tf.placeholder(tf.int64, shape=[None, dim_cmd], name='input_cmd_next')
            self.prev_action = tf.placeholder(tf.float32, shape=[None, dim_action], name='prev_action')
            self.input_obj_goal = tf.placeholder(tf.float32, shape=[None, dim_goal], name='input_obj_goal')
            self.input_action = tf.placeholder(tf.float32, shape=[None, dim_action], name='input_action')
            self.length = tf.placeholder(tf.int32, [self.batch_size], name='length') # b

            # build model with multi-gpu parallely
            inputs = [self.input_laser, 
                      self.input_cmd, 
                      self.input_cmd_next,
                      self.prev_action,
                      self.input_obj_goal,
                      self.input_action]

            with tf.variable_scope('online'):
                self.pred_q  = self.Model(inputs)
            self.network_params = tf.trainable_variables()[num_actor_vars:]

            with tf.variable_scope('target'):
                self.q_target = self.Model(inputs)
            self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        self.y = tf.placeholder(tf.float32, [self.batch_size, self.max_step, 1], name='y')
        self.mask = tf.expand_dims(tf.sequence_mask(self.length, maxlen=self.max_step, dtype=tf.float32), axis=2) # b, l, 1
        self.square_diff = tf.pow((self.y - tf.reshape(self.pred_q, (self.batch_size, self.max_step, 1)))*self.mask, 2) # b, l, 1

        self.loss_t = tf.reduce_sum(self.square_diff, reduction_indices=1) / tf.cast(self.length, tf.float32)# b, 1
        self.loss_n = tf.reduce_sum(self.loss_t, reduction_indices=0) / self.batch_size # 1

        self.gradient = tf.gradients(self.loss_n, self.network_params)
        self.opt = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.opt.apply_gradients(zip(self.gradient, self.network_params))

        mask_reshape = tf.reshape(self.mask, (self.batch_size*self.max_step, 1)) # b*l, 1
        self.a_gradient_mask = tf.tile(mask_reshape, [1, 2]) # b*l, 2
        self.action_grads = tf.gradients(self.pred_q, self.input_action) * self.a_gradient_mask

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + \
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))] 


    def Model(self, inputs):
        laser, cmd, cmd_next, prev_action, obj_goal, action = inputs
        with tf.variable_scope('encoder'):

            embedding_w_goal = tf.get_variable('embedding_w_goal', [self.dim_action, self.dim_emb])
            embedding_b_goal = tf.get_variable('embedding_b_goal', [self.dim_emb])
            embedding_status = tf.get_variable('embedding_status', [self.n_cmd_type**2, self.dim_emb])
            embedding_w_action = tf.get_variable('embedding_w_action', [self.dim_action, self.dim_emb])
            embedding_b_action = tf.get_variable('embedding_b_action', [self.dim_emb])
            embedding_w_status = tf.get_variable('embedding_w_status', [self.dim_cmd, self.dim_emb])
            embedding_b_status = tf.get_variable('embedding_b_status', [self.dim_emb])

            conv1 = model_utils.Conv1D(self.input_laser, 2, 5, 4, scope='conv1')
            conv2 = model_utils.Conv1D(conv1, 4, 5, 4, scope='conv2')
            conv3 = model_utils.Conv1D(conv2, 8, 5, 4, scope='conv3')
            shape = conv3.get_shape().as_list()
            vector_laser = tf.reshape(conv3, (-1, shape[1]*shape[2]))
            curr_status = cmd * self.n_cmd_type + cmd_next
            vector_curr_status = tf.reshape(tf.nn.embedding_lookup(embedding_status, curr_status), (-1, self.dim_emb))
            vector_prev_action = tf.matmul(prev_action, embedding_w_action) + embedding_b_action
            vector_obj_goal = tf.matmul(obj_goal, embedding_w_goal) + embedding_b_goal
            vector_action = tf.matmul(action, embedding_w_action) + embedding_b_action

            input_vector = tf.concat([vector_laser, 
                                      vector_curr_status,
                                      vector_prev_action,
                                      vector_obj_goal,
                                      vector_action], 
                                      axis=1)

        with tf.variable_scope('q'):
            rnn_cell = model_utils._lstm_cell(self.n_hidden, self.n_layers, name='rnn/basic_lstm_cell')
            w_q = tf.get_variable('w_q', [self.n_hidden, 1], initializer=tf.initializers.random_uniform(-0.003, 0.003))
            b_q = tf.get_variable('b_q', [1], initializer=tf.initializers.random_uniform(-0.003, 0.003))
            
            shape = input_vector.get_shape().as_list()
            input_vector_reshape = tf.reshape(input_vector, [self.batch_size, self.max_step, shape[1]])

            rnn_output, _ = tf.nn.dynamic_rnn(rnn_cell, 
                                                input_vector_reshape, 
                                                sequence_length=self.length,
                                                dtype=tf.float32) # b, l, h
            rnn_output_reshape = tf.reshape(rnn_output, [-1, self.n_hidden]) # b*l, h
            q = tf.matmul(rnn_output_reshape, w_q) + b_q                  

        return q


    def Train(self, laser, cmd, cmd_next, prev_action, obj_goal, action, y, length):
        return self.sess.run([self.pred_q, self.optimize], feed_dict={
            self.input_laser: laser,
            self.input_cmd: cmd,
            self.input_cmd_next: cmd_next,
            self.prev_action: prev_action,
            self.input_obj_goal: obj_goal,
            self.input_action: action,
            self.y: y,
            self.length: length
            })



    def PredictTarget(self, laser, cmd, cmd_next, prev_action, obj_goal, action, length):
        return self.sess.run(self.q_target, feed_dict={
            self.input_laser: laser,
            self.input_cmd: cmd,
            self.input_cmd_next: cmd_next,
            self.prev_action: prev_action,
            self.input_obj_goal: obj_goal,
            self.input_action: action,
            self.length: length
            })


    def PredictOnline(self, laser, cmd, cmd_next, prev_action, obj_goal, action, length):
        return self.sess.run(self.pred_q, feed_dict={
            self.input_laser: laser,
            self.input_cmd: cmd,
            self.input_cmd_next: cmd_next,
            self.prev_action: prev_action,
            self.input_obj_goal: obj_goal,
            self.input_action: action,
            self.length: length
            })

    def ActionGradients(self, laser, cmd, cmd_next, prev_action, obj_goal, action, length):
        return self.sess.run(self.action_grads, feed_dict={
            self.input_laser: laser,
            self.input_cmd: cmd,
            self.input_cmd_next: cmd_next,
            self.prev_action: prev_action,
            self.input_obj_goal: obj_goal,
            self.input_action: action,
            self.length: length
            })


    def UpdateTarget(self):
        self.sess.run(self.update_target_network_params)


    def TrainableVarNum(self):
        return self.num_trainable_vars











            
 