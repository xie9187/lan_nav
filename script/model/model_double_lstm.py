import tensorflow as tf
import utils.model_utils as model_utils
import numpy as np
import copy
from tensorflow.python.ops.rnn_cell import LSTMStateTuple


class Nav(object):
    def __init__(self,
                 sess,
                 batch_size,
                 learning_rate,
                 max_step,
                 n_layers,
                 n_hidden,
                 n_cmd_type,
                 dim_emb=64,
                 dim_laser=[666, 3],
                 dim_goal=2,
                 dim_cmd=1,
                 dim_action=2,
                 action_range=[0.4, np.pi/4],
                 loss_weight=[1.,1.],
                 encoder_training=True,
                 keep_prob=0.8,
                 gpu_num=1,
                 noise_track_rate=0.1,
                 single_lstm=False
                 ):
        self.sess = sess
        self.max_step = max_step
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_cmd_type = n_cmd_type
        self.dim_emb = dim_emb
        self.dim_laser = dim_laser
        self.dim_goal = dim_goal
        self.dim_cmd = dim_cmd
        self.dim_action = dim_action
        self.loss_weight = loss_weight
        self.encoder_training = encoder_training
        self.action_range = action_range
        self.gpu_num = gpu_num
        self.noise_track_rate = noise_track_rate
        self.single_lstm = single_lstm

        # training input
        self.input_placeholder = (
            tf.placeholder(tf.float32, shape=[None, max_step*dim_laser[0]], name='laser'),
            tf.placeholder(tf.float32, shape=[None, max_step*dim_action], name='action'),
            tf.placeholder(tf.int64, shape=[None, max_step*dim_cmd], name='cmd'),
            tf.placeholder(tf.int64, shape=[None, max_step*dim_cmd], name='cmd_next'),
            tf.placeholder(tf.float32, shape=[None, max_step*dim_goal], name='obj_goal'),
            tf.placeholder(tf.int64, shape=[None, max_step*dim_cmd], name='status'),
            tf.placeholder(tf.int64, shape=[None, ], name='lengths')
            )

        (laser, action, cmd, cmd_next, obj_goal, status, length) = self.input_placeholder

        # testing inpit
        self.test_laser = tf.placeholder(tf.float32, [None, dim_laser[0], dim_laser[1]], name='test_laser')
        self.test_cmd = tf.placeholder(tf.int32, [None, dim_cmd], name='test_cmd')
        self.test_cmd_next = tf.placeholder(tf.int32, [None, dim_cmd], name='test_cmd_next')
        self.test_obj_goal = tf.placeholder(tf.float32, [None, dim_goal], name='test_obj_goal')
        self.test_prev_status = tf.placeholder(tf.int32, [None, dim_cmd], name='test_prev_status')
        self.test_prev_action = tf.placeholder(tf.float32, [None, dim_action], name='test_prev_action')

        # error param
        self.a_err_mean_param = tf.get_variable('a_err_mean_param', shape=[dim_action], initializer=tf.constant_initializer([0., 0.]))
        self.a_err_var_param = tf.get_variable('var_a_err_param', shape=[dim_action], initializer=tf.constant_initializer([1e-5, 1e-5]))


        self.keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')

        # build model with multi-gpu parallely
        inputs = [laser, cmd, cmd_next, obj_goal, status, action, length]
        inputs_splits = []
        for var in inputs:
            inputs_splits.append(tf.split(var, self.gpu_num, axis=0))
        laser_splits, cmd_splits, cmd_next_splits, obj_goal_splits, status_splits, action_splits, length_splits = inputs_splits
        objective_splits = []
        loss_status_splits = []
        loss_action_splits = []
        a_err_mean_splits = []
        a_err_var_splits = []
        for i in range(self.gpu_num):
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
                with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                    objective, loss_status, loss_action, a_err_mean, a_err_var = self.Model(laser_splits[i],
                                                                                             cmd_splits[i],
                                                                                             cmd_next_splits[i],
                                                                                             obj_goal_splits[i],
                                                                                             status_splits[i],
                                                                                             action_splits[i],
                                                                                             length_splits[i])
                    objective_splits.append(objective)
                    loss_status_splits.append(loss_status)
                    loss_action_splits.append(loss_action)
                    a_err_mean_splits.append(a_err_mean)
                    a_err_var_splits.append(a_err_var)

        self.objective = tf.add_n(objective_splits)
        self.loss_status = tf.add_n(loss_status_splits)
        self.loss_action = tf.add_n(loss_action_splits)
        self.a_err_mean = tf.add_n(a_err_mean_splits) / self.gpu_num
        self.a_err_var = tf.add_n(a_err_var_splits) / self.gpu_num

        # error param update op
        self.err_param_update = [self.a_err_mean_param.assign(self.a_err_mean * self.noise_track_rate
                                                              + self.a_err_mean_param * (1. - self.noise_track_rate)), 
                                 self.a_err_var_param.assign(self.a_err_var * self.noise_track_rate
                                                             + self.a_err_var_param * (1. - self.noise_track_rate))]

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.optim = optimizer.minimize(self.objective, colocate_gradients_with_ops=True)


    def Model(self, laser, cmd, cmd_next, obj_goal, status, action, length):
        # reshape to (batch_size, max_step, dim)
        batch_size = self.batch_size / self.gpu_num

        laser = tf.reshape(laser, [batch_size, self.max_step, self.dim_laser[0], 1])
        cmd = tf.reshape(cmd, [batch_size, self.max_step, self.dim_cmd])
        cmd_next = tf.reshape(cmd_next, [batch_size, self.max_step, self.dim_cmd])
        obj_goal = tf.reshape(obj_goal, [batch_size, self.max_step, self.dim_goal])
        status = tf.reshape(status, [batch_size, self.max_step, self.dim_cmd])
        action = tf.reshape(action, [batch_size, self.max_step, self.dim_action])

        # stack laser
        laser_t = laser

        laser_0_to_m1 = tf.slice(laser, [0, 0, 0, 0], [batch_size, self.max_step-1, self.dim_laser[0], 1])
        laser_0 = tf.slice(laser, [0, 0, 0, 0], [batch_size, 1, self.dim_laser[0], 1])
        laser_tm1 = tf.concat([laser_0, laser_0_to_m1], axis=1)

        laser_0_to_m2 = tf.slice(laser, [0, 0, 0, 0], [batch_size, self.max_step-2, self.dim_laser[0], 1])
        laser_tm2 = tf.concat([laser_0, laser_0, laser_0_to_m2], axis=1)

        laser_stack = tf.concat([laser_t, laser_tm1, laser_tm2], axis=2)

        # reshape to (batch_size*max_step, dim)
        input_laser = tf.reshape(laser_stack, [batch_size * self.max_step, self.dim_laser[0], self.dim_laser[1]])
        input_cmd = tf.reshape(cmd, [batch_size * self.max_step, self.dim_cmd])
        input_cmd_next = tf.reshape(cmd_next ,[batch_size * self.max_step, self.dim_cmd])
        input_obj_goal = tf.reshape(obj_goal ,[batch_size * self.max_step, self.dim_goal])

        label_action = tf.reshape(action ,[batch_size * self.max_step, self.dim_action])
        label_status = tf.reshape(status ,[batch_size * self.max_step, self.dim_cmd])

        mask = tf.reshape(tf.sequence_mask(length, maxlen=self.max_step, dtype=tf.float32), [-1]) # b*l


        # construct network
        with tf.variable_scope('actor/online'):
            training_input, testing_input = self.Encoder(input_laser, 
                                                         input_cmd, 
                                                         input_cmd_next, 
                                                         input_obj_goal, 
                                                         label_action, 
                                                         label_status)

            (loss_status,
             self.pred_status_test,
             self.state1,
             self.prev_state1) = self.Planner(training_input, 
                                              testing_input, 
                                              label_status, 
                                              length, 
                                              mask)
            if self.single_lstm:
                (loss_action, 
                self.pred_action_test,
                mean,
                variance) = self.ControllerDense(training_input, 
                                                 testing_input, 
                                                 label_action,
                                                 length,
                                                 mask)
            else:
                (loss_action, 
                self.pred_action_test,
                self.state2,
                self.prev_state2,
                mean,
                variance) = self.ControllerLSTM(training_input, 
                                                testing_input, 
                                                label_action,
                                                length,
                                                mask)

        # optimizer
        objective = loss_status * self.loss_weight[0] + loss_action * self.loss_weight[1]

        return objective, loss_status, loss_action, mean, variance


    def Encoder(self, input_laser, input_cmd, input_cmd_next, input_obj_goal, label_action, label_status):
        with tf.variable_scope('encoder'):
            embedding_w_goal = tf.get_variable('embedding_w_goal', [self.dim_action, self.dim_emb])
            embedding_b_goal = tf.get_variable('embedding_b_goal', [self.dim_emb])
            embedding_cmd = tf.get_variable('cmd_embedding', [self.n_cmd_type, self.dim_emb])
            embedding_status = tf.get_variable('status_embedding', [2, self.dim_emb])
            embedding_w_action = tf.get_variable('embedding_w_action', [self.dim_action, self.dim_emb])
            embedding_b_action = tf.get_variable('embedding_b_action', [self.dim_emb])
            embedding_w_status = tf.get_variable('embedding_w_status', [self.dim_cmd, self.dim_emb])
            embedding_b_status = tf.get_variable('embedding_b_status', [self.dim_emb])
            
            batch_size = self.batch_size / self.gpu_num
            # training input
            conv1 = model_utils.Conv1D(input_laser, 2, 5, 4, scope='conv1', trainable=self.encoder_training)
            conv2 = model_utils.Conv1D(conv1, 4, 5, 4, scope='conv2', trainable=self.encoder_training)
            conv3 = model_utils.Conv1D(conv2, 8, 5, 4, scope='conv3', trainable=self.encoder_training)
            shape = conv3.get_shape().as_list()
            vector_laser = tf.reshape(conv3, (-1, shape[1]*shape[2]))

            vector_cmd = tf.reshape(tf.nn.embedding_lookup(embedding_cmd, input_cmd), (-1, self.dim_emb))
            vector_cmd_next = tf.reshape(tf.nn.embedding_lookup(embedding_cmd, input_cmd_next), (-1, self.dim_emb))

            label_status_reshape = tf.reshape(label_status, [-1, self.max_step, self.dim_cmd]) #  b, l, 1
            label_status_0_to_m1 = tf.slice(label_status_reshape, [0, 0, 0], [batch_size, self.max_step-1, self.dim_cmd])
            prev_status_0 = tf.tile(tf.zeros([1, 1, self.dim_cmd], dtype=tf.int64), [batch_size, 1, 1])
            prev_status = tf.reshape(tf.concat([prev_status_0, label_status_0_to_m1], axis=1), [-1, self.dim_cmd])
            vector_prev_status = tf.reshape(tf.nn.embedding_lookup(embedding_status, prev_status), (-1, self.dim_emb))

            linear_err_mean, angular_err_mean = tf.split(self.a_err_mean_param, 2)
            linear_err_var, angular_err_var = tf.split(self.a_err_var_param, 2)

            a_linear_noise = tf.random_normal([batch_size * self.max_step, 1], linear_err_mean, linear_err_var)
            a_angular_noise = tf.random_normal([batch_size * self.max_step, 1], angular_err_mean, angular_err_var)
            a_noise = tf.concat([a_linear_noise, a_angular_noise], axis=1)
            clip_min = tf.tile(tf.constant([[0., -self.action_range[1]]]), 
                                           [batch_size * self.max_step, 1])
            clip_max = tf.tile(tf.constant([[self.action_range[0], self.action_range[1]]]), 
                                           [batch_size * self.max_step, 1])
            # noisy_action = tf.clip_by_value(a_noise + label_action, clip_min, clip_max)
            noisy_action = tf.clip_by_value(label_action, clip_min, clip_max)

            noisy_action_reshape = tf.reshape(noisy_action, [batch_size, self.max_step, self.dim_action]) #  b, l, 2
            noisy_action_0_to_m1 = tf.slice(noisy_action_reshape, [0, 0, 0], [batch_size, self.max_step-1, self.dim_action])
            prev_action_0 = tf.tile(tf.zeros([1, 1, self.dim_action]), [batch_size, 1, 1])
            prev_action = tf.reshape(tf.concat([prev_action_0, noisy_action_0_to_m1], axis=1), [-1, self.dim_action])
            vector_prev_action = tf.matmul(prev_action, embedding_w_action)+embedding_b_action

            vector_obj_goal = tf.matmul(input_obj_goal, embedding_w_goal) + embedding_b_goal

            training_input = tf.concat([vector_laser, 
                                        vector_cmd, 
                                        vector_cmd_next, 
                                        vector_prev_status,
                                        vector_prev_action,
                                        vector_obj_goal], 
                                        axis=1)

        # testing input
            conv1 = model_utils.Conv1D(self.test_laser, 2, 5, 4, scope='conv1', trainable=self.encoder_training, reuse=True)
            conv2 = model_utils.Conv1D(conv1, 4, 5, 4, scope='conv2', trainable=self.encoder_training, reuse=True)
            conv3 = model_utils.Conv1D(conv2, 8, 5, 4, scope='conv3', trainable=self.encoder_training, reuse=True)
            shape = conv3.get_shape().as_list()
            vector_laser_test = tf.reshape(conv3, (-1, shape[1]*shape[2]))  

            vector_cmd_test = tf.reshape(tf.nn.embedding_lookup(embedding_cmd, self.test_cmd), (-1, self.dim_emb))
            vector_cmd_next_test = tf.reshape(tf.nn.embedding_lookup(embedding_cmd, self.test_cmd_next), (-1, self.dim_emb))
            vector_prev_status_test = tf.reshape(tf.nn.embedding_lookup(embedding_status, self.test_prev_status), (-1, self.dim_emb))
            vector_prev_action_test = tf.matmul(self.test_prev_action, embedding_w_action) + embedding_b_action
            vector_obj_goal_test = tf.matmul(self.test_obj_goal, embedding_w_goal) + embedding_b_goal

            testing_input = tf.concat([vector_laser_test, 
                                       vector_cmd_test, 
                                       vector_cmd_next_test, 
                                       vector_prev_status_test,
                                       vector_prev_action_test, 
                                       vector_obj_goal_test], 
                                       axis=1)

        return training_input, testing_input


    def Planner(self, training_input, testing_input, label_status, length, mask):
        with tf.variable_scope('planner'):
            batch_size = self.batch_size / self.gpu_num

            rnn_cell = model_utils._lstm_cell(self.n_hidden, self.n_layers)

            w_status = tf.get_variable('w_status', [self.n_hidden, 2], initializer=tf.contrib.layers.xavier_initializer())
            b_status = tf.get_variable('b_status', [2], initializer=tf.contrib.layers.xavier_initializer())

            # training
            training_input_dropout = tf.nn.dropout(training_input, self.keep_prob)  # b*l, h
            shape = training_input_dropout.get_shape().as_list()
            training_input_reshape = tf.reshape(training_input_dropout, [batch_size, self.max_step, shape[1]]) # b, l, h
            rnn_output, _ = tf.nn.dynamic_rnn(rnn_cell, 
                                              training_input_reshape, 
                                              sequence_length=length,
                                              dtype=tf.float32) # b, l, h
            rnn_output_dropout = tf.nn.dropout(rnn_output, self.keep_prob)
            rnn_output_reshape = tf.reshape(rnn_output_dropout, [-1, self.n_hidden]) # b*l, h
            logits = tf.reshape(tf.matmul(rnn_output_reshape, w_status), [-1, 2]) + b_status # b*l, n
                    
            label_status_reshape = tf.reshape(label_status, [-1])
            loss_status = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_status_reshape, logits=logits)       

            loss_status_scalar = tf.reduce_sum(loss_status * mask)

            # testing
            prev_state = []
            for l in xrange(self.n_layers):
                prev_state.append(
                    LSTMStateTuple(tf.placeholder(tf.float32, shape=[None, self.n_hidden], name='initial_state{0}.c'.format(l)),
                                   tf.placeholder(tf.float32, shape=[None, self.n_hidden], name='initial_state{0}.h'.format(l))))
            if self.n_layers == 1:
                prev_state = prev_state[0]

            rnn_output_test, state = rnn_cell(testing_input, prev_state) # b*l, h 
            prob = tf.reshape(tf.nn.softmax(tf.matmul(rnn_output_test, w_status) + b_status), [-1, 2])
            # pred_status_test = tf.argmax(prob, axis=1)
            return loss_status_scalar, prob, state, prev_state


    def ControllerLSTM(self, training_input, testing_input, label_action, length, mask):
        with tf.variable_scope('controller'):
            batch_size = self.batch_size / self.gpu_num
            
            rnn_cell = model_utils._lstm_cell(self.n_hidden, self.n_layers)

            w_action_linear = tf.get_variable('w_action_linear', [self.n_hidden, self.dim_action/2], initializer=tf.contrib.layers.xavier_initializer())
            b_action_linear = tf.get_variable('b_action_linear', [self.dim_action/2], initializer=tf.contrib.layers.xavier_initializer())
            w_action_angular = tf.get_variable('w_action_angular', [self.n_hidden, self.dim_action/2], initializer=tf.contrib.layers.xavier_initializer())
            b_action_angular = tf.get_variable('b_action_angular', [self.dim_action/2], initializer=tf.contrib.layers.xavier_initializer())
            

            training_input_dropout = tf.nn.dropout(training_input, self.keep_prob)  # b*l, h
            shape = training_input_dropout.get_shape().as_list()
            training_input_reshape = tf.reshape(training_input_dropout, [batch_size, self.max_step, shape[1]]) # b, l, h
            rnn_output, _ = tf.nn.dynamic_rnn(rnn_cell, 
                                              training_input_reshape, 
                                              sequence_length=length,
                                              dtype=tf.float32) # b, l, h
            rnn_output_dropout = tf.nn.dropout(rnn_output, self.keep_prob)
            rnn_output_reshape = tf.reshape(rnn_output_dropout, [-1, self.n_hidden]) # b*l, h


            a_linear = tf.nn.sigmoid(tf.matmul(rnn_output_reshape, w_action_linear) + b_action_linear) * self.action_range[0]
            a_angular = tf.nn.tanh(tf.matmul(rnn_output_reshape, w_action_angular) + b_action_angular) * self.action_range[1] 
            pred_action = tf.concat([a_linear, a_angular], axis=1)

            # calculate the mean and variance of the masked error
            mask_reshape = tf.reshape(mask, [batch_size * self.max_step, 1]) # b*l, 1
            mask_tile = tf.tile(mask_reshape, [1, 2]) # b*l, 2
            masked_error = (pred_action - label_action) * mask_tile # b*l, 2
            mean = tf.reduce_sum(masked_error, axis=0) / tf.cast(tf.reduce_sum(length), tf.float32) # 2
            mean_expand = tf.expand_dims(mean, axis=0) # 1, 2
            mean_tile = tf.tile(mean_expand, [batch_size * self.max_step, 1]) # b*l, 2
            variance = tf.square(tf.reduce_sum((masked_error - mean_tile) * mask_tile, axis=0)) / tf.cast(tf.reduce_sum(length), tf.float32)

            loss_action = tf.losses.mean_squared_error(labels=label_action, predictions=pred_action, reduction=tf.losses.Reduction.NONE)
            loss_action = tf.reduce_sum(loss_action, axis=1)
            loss_action_scalar = tf.reduce_sum(loss_action * mask)

            # testing
            prev_state = []
            for l in xrange(self.n_layers):
                prev_state.append(
                    LSTMStateTuple(tf.placeholder(tf.float32, shape=[None, self.n_hidden], name='initial_state{0}.c'.format(l)),
                                   tf.placeholder(tf.float32, shape=[None, self.n_hidden], name='initial_state{0}.h'.format(l))))
            if self.n_layers == 1:
                prev_state = prev_state[0]

            rnn_output_test, state = rnn_cell(testing_input, prev_state) # b*l, h 
            a_linear_test = tf.nn.sigmoid(tf.matmul(rnn_output_test, w_action_linear) + b_action_linear) * self.action_range[0]
            a_angular_test = tf.nn.tanh(tf.matmul(rnn_output_test, w_action_angular) + b_action_angular) * self.action_range[1]
            pred_action_test = tf.concat([a_linear_test, a_angular_test], axis=1)

            return loss_action_scalar, pred_action_test, state, prev_state, mean, variance

    def ControllerDense(self, training_input, testing_input, label_action, length, mask):
        with tf.variable_scope('controller'):
            batch_size = self.batch_size / self.gpu_num
            shape = training_input.get_shape().as_list()

            w_hidden = tf.get_variable('w_hidden', [shape[1], self.n_hidden], initializer=tf.contrib.layers.xavier_initializer())
            b_hidden = tf.get_variable('b_hidden', [self.n_hidden], initializer=tf.contrib.layers.xavier_initializer())
            w_action_linear = tf.get_variable('w_action_linear', [self.n_hidden, self.dim_action/2], initializer=tf.contrib.layers.xavier_initializer())
            b_action_linear = tf.get_variable('b_action_linear', [self.dim_action/2], initializer=tf.contrib.layers.xavier_initializer())
            w_action_angular = tf.get_variable('w_action_angular', [self.n_hidden, self.dim_action/2], initializer=tf.contrib.layers.xavier_initializer())
            b_action_angular = tf.get_variable('b_action_angular', [self.dim_action/2], initializer=tf.contrib.layers.xavier_initializer())

            training_input_dropout = tf.nn.dropout(training_input, self.keep_prob)  # b*l, h
            shape = training_input_dropout.get_shape().as_list()
            hidden = tf.nn.leaky_relu(tf.matmul(training_input_dropout, w_hidden) + b_hidden) # b*l. n
            # a_linear = tf.nn.sigmoid(tf.matmul(hidden, w_action_linear) + b_action_linear) * self.action_range[0]
            # a_angular = tf.nn.tanh(tf.matmul(hidden, w_action_angular) + b_action_angular) * self.action_range[1]
            a_linear = tf.nn.sigmoid(tf.matmul(rnn_output, w_action_linear) + b_action_linear) * self.action_range[0]
            a_angular = tf.nn.tanh(tf.matmul(rnn_output, w_action_angular) + b_action_angular) * self.action_range[1]
            pred_action = tf.concat([a_linear, a_angular], axis=1)

            # calculate the mean and variance of the masked error
            mask_reshape = tf.reshape(mask, [batch_size * self.max_step, 1]) # b*l, 2
            mask_tile = tf.tile(mask_reshape, [1, 2]) # b*l, 2
            masked_error = (pred_action - label_action) * mask_tile # b*l, 2
            mean = tf.reduce_sum(masked_error, axis=0) / tf.cast(tf.reduce_sum(length), tf.float32) # 2
            mean_expand = tf.expand_dims(mean, axis=0) # 1, 2
            mean_tile = tf.tile(mean_expand, [batch_size * self.max_step, 1]) # b*l, 2
            variance = tf.square(tf.reduce_sum((masked_error - mean_tile) * mask_tile, axis=0)) / tf.cast(tf.reduce_sum(length), tf.float32)

            loss_action = tf.losses.mean_squared_error(labels=label_action, predictions=pred_action, reduction=tf.losses.Reduction.NONE)
            loss_action = tf.reduce_sum(loss_action, axis=1)
            loss_action_scalar = tf.reduce_sum(loss_action * mask)

            # testing
            hidden_test = tf.nn.leaky_relu(tf.matmul(testing_input, w_hidden) + b_hidden)
            # a_linear_test = tf.nn.sigmoid(tf.matmul(hidden_test, w_action_linear) + b_action_linear)
            # a_angular_test = tf.nn.tanh(tf.matmul(hidden_test, w_action_angular) + b_action_angular)
            a_linear_test = tf.matmul(hidden_test, w_action_linear) + b_action_linear
            a_angular_test = tf.matmul(hidden_test, w_action_angular) + b_action_angular
            pred_action_test = tf.concat([a_linear_test, a_angular_test], axis=1)

            return loss_action_scalar, pred_action_test, mean, variance


    def Predict(self,
                test_laser, 
                test_cmd,
                test_cmd_next,
                test_obj_goal,
                t,
                test_prev_action=None,
                test_prev_status=None
                ):
        if not self.single_lstm:
            if t == 0:
                prev_state1 = (np.zeros([1, self.n_hidden]), np.zeros([1, self.n_hidden]))
                prev_state2 = (np.zeros([1, self.n_hidden]), np.zeros([1, self.n_hidden]))
                if not test_prev_action:
                    test_prev_action = np.zeros([1, self.dim_action])
                if not test_prev_status:
                    test_prev_status = np.zeros([1, self.dim_cmd])
            else:
                prev_state1 = copy.deepcopy(self.lstm_state1)
                prev_state2 = copy.deepcopy(self.lstm_state2)
                if not test_prev_action:
                    test_prev_action = copy.deepcopy(self.action)
                if not test_prev_status:
                    test_prev_status = copy.deepcopy(self.status)

            (status_prob, 
             self.action, 
             self.lstm_state1,
             self.lstm_state2) = self.sess.run([self.pred_status_test,
                                                self.pred_action_test,
                                                self.state1,
                                                self.state2],
                                                feed_dict={self.test_laser: test_laser,
                                                           self.test_cmd: test_cmd,
                                                           self.test_cmd_next: test_cmd_next,
                                                           self.test_obj_goal: test_obj_goal,
                                                           self.test_prev_status: test_prev_status,
                                                           self.test_prev_action: test_prev_action,
                                                           self.keep_prob: 1.,
                                                           self.prev_state1: prev_state1,
                                                           self.prev_state2: prev_state2
                                                           })
        else:
            if t == 0:
                prev_state1 = (np.zeros([1, self.n_hidden]), np.zeros([1, self.n_hidden]))
                if not test_prev_action:
                    test_prev_action = np.zeros([1, self.dim_action])
                if not test_prev_status:
                    test_prev_status = np.zeros([1, self.dim_cmd])
            else:
                prev_state1 = copy.deepcopy(self.lstm_state1)
                if not test_prev_action:
                    test_prev_action = copy.deepcopy(self.action)
                if not test_prev_status:
                    test_prev_status = copy.deepcopy(self.status)

            (status_prob, 
             self.action, 
             self.lstm_state1) = self.sess.run([self.pred_status_test,
                                                self.pred_action_test,
                                                self.state1],
                                                feed_dict={self.test_laser: test_laser,
                                                           self.test_cmd: test_cmd,
                                                           self.test_cmd_next: test_cmd_next,
                                                           self.test_obj_goal: test_obj_goal,
                                                           self.test_prev_status: test_prev_status,
                                                           self.test_prev_action: test_prev_action,
                                                           self.keep_prob: 1.,
                                                           self.prev_state1: prev_state1
                                                           })
        if status_prob[0][1] > 0.5:
            self.status = [[1.]]
        else:
            self.status = [[0.]]
        return self.status[0][0], self.action[0], status_prob[0][1]













            
 