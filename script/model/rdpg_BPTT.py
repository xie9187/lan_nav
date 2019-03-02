import tensorflow as tf
import numpy as np
import os
import copy
import time
import utils.model_utils as model_utils
import matplotlib.pyplot as plt
from rdpg_actor_BPTT import Actor
from rdpg_critic_BPTT import Critic
from tensorflow.python.ops.rnn_cell import LSTMStateTuple


class RDPG_BPTT(object):
    """docstring for DDPG"""
    def __init__(self, flags, sess):
        self.dim_laser = [flags.dim_laser_b, flags.dim_laser_c]
        self.dim_goal = flags.dim_goal
        self.dim_action = flags.dim_action
        self.dim_emb = flags.dim_emb
        self.dim_cmd = flags.dim_cmd
        self.n_hidden = flags.n_hidden
        self.n_cmd_type = flags.n_cmd_type
        self.n_layers = flags.n_layers
        self.a_learning_rate = flags.a_learning_rate
        self.c_learning_rate = flags.c_learning_rate
        self.batch_size = flags.batch_size
        self.max_step = flags.max_step
        self.tau = flags.tau
        self.action_range = [flags.a_linear_range, flags.a_angular_range]
        self.buffer_size = flags.buffer_size
        self.gamma = flags.gamma
        self.demo_flag = flags.demo_flag

        self.actor = Actor(sess=sess,
                           dim_laser=self.dim_laser,
                           dim_cmd=self.dim_cmd,
                           dim_action=self.dim_action,
                           dim_goal=self.dim_goal,
                           dim_emb=self.dim_emb,
                           n_cmd_type=self.n_cmd_type,
                           n_hidden=self.n_hidden,
                           n_layers=self.n_layers,
                           max_step=self.max_step,
                           batch_size=self.batch_size,
                           action_range=self.action_range,
                           tau=self.tau,
                           gpu_num=1,
                           demo_flag=self.demo_flag)

        self.critic = Critic(sess=sess,
                             dim_laser=self.dim_laser,
                             dim_cmd=self.dim_cmd,
                             dim_action = self.dim_action,
                             dim_goal=self.dim_goal,
                             dim_emb=self.dim_emb,
                             n_cmd_type=self.n_cmd_type,
                             n_hidden=self.n_hidden,
                             n_layers=self.n_layers,
                             max_step=self.max_step,
                             batch_size=self.batch_size,
                             num_actor_vars=len(self.actor.network_params)+len(self.actor.target_network_params),
                             tau=self.tau,
                             gpu_num=1)
        self.memory = []
        
    def ActorPredict(self, 
                     input_laser, 
                     input_cmd, 
                     input_cmd_next, 
                     prev_action, 
                     input_goal, 
                     prev_state):

        a, state = self.actor.PredictStep(input_laser, 
                                         input_cmd, 
                                         input_cmd_next, 
                                         prev_action, 
                                         input_goal, 
                                         prev_state)
        return a[0], state

    def Add2Mem(self, sample):
        if len(sample) <= self.max_step:
            self.memory.append(sample) # seqs of (laser_stack, cmd, cmd_next, cmd_skip, 
                                       #          prev_action, goal, a_goal, action, 
                                       #          r, status, action_label)
        if len(self.memory) > self.buffer_size:
            self.memory.pop(0)

    def SampleBatch(self):
        if len(self.memory) >= self.batch_size:

            indices = np.random.randint(0, len(self.memory), size=self.batch_size)
            laser_batch = np.zeros([self.batch_size, self.max_step, self.dim_laser[0], self.dim_laser[1]])
            cmd_batch = np.zeros([self.batch_size, self.max_step, self.dim_cmd])
            cmd_next_batch = np.zeros([self.batch_size, self.max_step, self.dim_cmd])
            cmd_skip_batch = np.zeros([self.batch_size, self.max_step, self.dim_cmd])
            prev_action_batch = np.zeros([self.batch_size, self.max_step, self.dim_action])
            goal_batch = np.zeros([self.batch_size, self.max_step, self.dim_goal])
            goal_a_batch = np.zeros([self.batch_size, self.max_step, self.dim_goal])
            action_batch = np.zeros([self.batch_size, self.max_step, self.dim_action])

            r_batch = np.zeros([self.batch_size, self.max_step])
            status_batch = np.zeros([self.batch_size, self.max_step, self.dim_cmd])
            action_label_batch = np.zeros([self.batch_size, self.max_step, self.dim_action])
            length_batch = []

            for x, idx in enumerate(indices):
                sampled_seq = self.memory[idx]
                seq_len = len(sampled_seq)
                for t in xrange(0, seq_len):
                    laser_batch[x, t, :, :] = sampled_seq[t][0]
                    cmd_batch[x, t, :] = sampled_seq[t][1]
                    cmd_next_batch[x, t, :] = sampled_seq[t][2]
                    cmd_skip_batch[x, t, :] = sampled_seq[t][3]
                    prev_action_batch[x, t:, ] = sampled_seq[t][4]
                    goal_batch[x, t, :] = sampled_seq[t][5]
                    goal_a_batch[x, t, :] = sampled_seq[t][6]
                    action_batch[x, t, :] = sampled_seq[t][7]
                    r_batch[x, t] = sampled_seq[t][8]
                    status_batch[x, t, :] = sampled_seq[t][9]
                    action_label_batch[x, t, :] = sampled_seq[t][10]

                length_batch.append(seq_len)   

            laser_batch = np.reshape(laser_batch, [self.batch_size * self.max_step, self.dim_laser[0], self.dim_laser[1]])
            cmd_batch = np.reshape(cmd_batch, [self.batch_size * self.max_step, self.dim_cmd])
            cmd_next_batch = np.reshape(cmd_next_batch, [self.batch_size * self.max_step, self.dim_cmd])
            cmd_skip_batch = np.reshape(cmd_skip_batch, [self.batch_size * self.max_step, self.dim_cmd])
            prev_action_batch = np.reshape(prev_action_batch, [self.batch_size * self.max_step, self.dim_action])
            goal_batch = np.reshape(goal_batch, [self.batch_size * self.max_step, self.dim_goal])
            goal_a_batch = np.reshape(goal_a_batch, [self.batch_size * self.max_step, self.dim_goal])
            action_batch = np.reshape(action_batch, [self.batch_size * self.max_step, self.dim_action])
            r_batch = np.reshape(r_batch, [-1])
            status_batch = np.reshape(status_batch, [self.batch_size * self.max_step, self.dim_cmd])
            action_label_batch = np.reshape(action_label_batch, [self.batch_size * self.max_step, self.dim_action])

            return [laser_batch, cmd_batch, cmd_next_batch, cmd_skip_batch, prev_action_batch, goal_batch,
                    goal_a_batch, action_batch, r_batch, status_batch, action_label_batch, length_batch]
        else:
            print 'samples are not enough'
            return None, None


    def Train(self):
        start_time = time.time()

        batch = self.SampleBatch()

        sample_time =  time.time() - start_time

        if not batch:
            return 0.
        else:
            [laser_batch, cmd_batch, cmd_next_batch, cmd_skip_batch, prev_action_batch, goal_batch, 
             goal_a_batch, action_batch, r_batch, status_batch, action_label_batch, length_batch] = batch

            #compute target y
            target_a_pred = self.actor.PredictTarget(laser=laser_batch, 
                                                     cmd=cmd_batch, 
                                                     cmd_next=cmd_next_batch, 
                                                     prev_action=prev_action_batch, 
                                                     obj_goal=goal_batch,
                                                     length=length_batch)

            target_q_pred = self.critic.PredictTarget(laser=laser_batch, 
                                                      cmd=cmd_batch, 
                                                      cmd_next=cmd_next_batch, 
                                                      prev_action=prev_action_batch, 
                                                      obj_goal=goal_batch, 
                                                      action=action_batch,
                                                      length=length_batch)
            y = []
            for i in xrange(self.batch_size):
                y_seq = np.zeros([self.max_step])
                for t in xrange(self.max_step):
                    if t == length_batch[i]-1:
                        y_seq[t] = r_batch[i*self.max_step+t]
                    elif t < length_batch[i]-1:
                        y_seq[t] = r_batch[i*self.max_step+t] + self.gamma * target_q_pred[i*self.max_step+t+1, 0]
                y.append(y_seq)
            y = np.expand_dims(np.stack(y), axis=2)

            y_time = time.time() - start_time - sample_time

            # critic update
            q, _ = self.critic.Train(laser=laser_batch, 
                                     cmd=cmd_batch, 
                                     cmd_next=cmd_next_batch, 
                                     prev_action=prev_action_batch, 
                                     obj_goal=goal_batch, 
                                     action=action_batch, 
                                     y=y,
                                     length=length_batch)

            # actions for a_gradients from critic
            actions = self.actor.PredictOnline(laser=laser_batch, 
                                               cmd=cmd_batch, 
                                               cmd_next=cmd_next_batch, 
                                               prev_action=prev_action_batch, 
                                               obj_goal=goal_a_batch,
                                               length=length_batch)

            # a_gradients
            a_gradients = self.critic.ActionGradients(laser=laser_batch, 
                                                      cmd=cmd_batch, 
                                                      cmd_next=cmd_next_batch, 
                                                      prev_action=prev_action_batch, 
                                                      obj_goal=goal_batch, 
                                                      action=actions,
                                                      length=length_batch)                                                    

            # actor update
            self.actor.Train(laser=laser_batch, 
                             cmd=cmd_batch, 
                             cmd_next=cmd_next_batch, 
                             cmd_skip=cmd_skip_batch,
                             prev_action=prev_action_batch, 
                             obj_goal=goal_a_batch, 
                             a_gradient=a_gradients[0],
                             status_label=status_batch,
                             action_label=action_batch,
                             length=length_batch)

            train_time = time.time() - start_time - sample_time - y_time

            # target networks update
            self.critic.UpdateTarget()
            self.actor.UpdateTarget()


            target_time = time.time() - start_time - sample_time - y_time - train_time

            print 'sample_time:{:.3f}, y_time:{:.3f}, train_time:{:.3f}, target_time:{:.3f}'.format(sample_time,
                                                                                                    y_time,
                                                                                                    train_time,
                                                                                                    target_time)
            
            return q

