import tensorflow as tf
import numpy as np
import os
import copy
import time
import utils.model_utils as model_utils
import matplotlib.pyplot as plt
from rdpg_actor import Actor
from rdpg_critic import Critic
from tensorflow.python.ops.rnn_cell import LSTMStateTuple


class RDPG(object):
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
                    input_cmd_skip, 
                    prev_action, 
                    input_goal, 
                    prev_state,
                    prev_state_2):
        a, state, state_2 = self.actor.PredictOnline(input_laser, 
                                                     input_cmd, 
                                                     input_cmd_next, 
                                                     input_cmd_skip, 
                                                     prev_action, 
                                                     input_goal, 
                                                     prev_state,
                                                     prev_state_2)
        return a[0], state, state_2

    def Add2Mem(self, sample):
        if len(sample) <= self.max_step:
            self.memory.append(sample) # seqs of (laser, cmd, cmd_next, cmd_skip, prev_status, 
                                       #          prev_action, obj_goal, prev_state, action, 
                                       #          r, terminate, status, action_label)
        if len(self.memory) > self.buffer_size:
            self.memory.pop(0)

    def SampleBatch(self):
        if len(self.memory) >= self.batch_size:

            indices = np.random.randint(0, len(self.memory)-1, size=self.batch_size)

            laser_t_batch = np.empty((self.batch_size, self.dim_laser[0], self.dim_laser[1]), dtype=np.float32)
            cmd_t_batch = np.empty((self.batch_size, self.dim_cmd), dtype=np.int64)
            cmd_next_t_batch = np.empty((self.batch_size, self.dim_cmd), dtype=np.int64)
            cmd_skip_t_batch = np.empty((self.batch_size, self.dim_cmd), dtype=np.int64)
            prev_action_t_batch = np.empty((self.batch_size, self.dim_action), dtype=np.float32)
            goal_t_batch = np.empty((self.batch_size, self.dim_goal), dtype=np.float32)
            goal_a_t_batch = np.empty((self.batch_size, self.dim_goal), dtype=np.float32)
            prev_state_t_batch = [np.empty((self.batch_size, self.n_hidden), dtype=np.float32), 
                                  np.empty((self.batch_size, self.n_hidden), dtype=np.float32)]
            prev_state_2_t_batch = [np.empty((self.batch_size, self.n_hidden), dtype=np.float32), 
                                    np.empty((self.batch_size, self.n_hidden), dtype=np.float32)]                                
            action_t_batch = np.empty((self.batch_size, self.dim_action), dtype=np.float32)

            reward_batch = np.empty((self.batch_size), dtype=np.float32)
            terminate_batch = np.empty((self.batch_size), dtype=bool)

            status_batch = np.empty((self.batch_size, 1), dtype=np.int64)
            action_batch = np.empty((self.batch_size, self.dim_action), dtype=np.float32)

            laser_t1_batch = np.empty((self.batch_size, self.dim_laser[0], self.dim_laser[1]), dtype=np.float32)
            cmd_t1_batch = np.empty((self.batch_size, self.dim_cmd), dtype=np.int64)
            cmd_next_t1_batch = np.empty((self.batch_size, self.dim_cmd), dtype=np.int64)
            cmd_skip_t1_batch = np.empty((self.batch_size, self.dim_cmd), dtype=np.int64)
            prev_action_t1_batch = np.empty((self.batch_size, self.dim_action), dtype=np.float32)
            goal_t1_batch = np.empty((self.batch_size, self.dim_goal), dtype=np.float32)
            goal_a_t1_batch = np.empty((self.batch_size, self.dim_goal), dtype=np.float32)
            prev_state_t1_batch = [np.empty((self.batch_size, self.n_hidden), dtype=np.float32), 
                                   np.empty((self.batch_size, self.n_hidden), dtype=np.float32)]
            prev_state_2_t1_batch = [np.empty((self.batch_size, self.n_hidden), dtype=np.float32), 
                                     np.empty((self.batch_size, self.n_hidden), dtype=np.float32)]
            action_t1_batch = np.empty((self.batch_size, self.dim_action), dtype=np.float32)

            for i, idx in enumerate(indices):
                laser_t_batch[i] = self.memory[idx][0]
                cmd_t_batch[i] = self.memory[idx][1]
                cmd_next_t_batch[i] = self.memory[idx][2]
                cmd_skip_t_batch[i] = self.memory[idx][3]
                prev_action_t_batch[i] = self.memory[idx][4]
                goal_t_batch[i] = self.memory[idx][5]
                prev_state_t_batch[0][i] = self.memory[idx][6][0][0]
                prev_state_t_batch[1][i] = self.memory[idx][6][1][0]
                prev_state_2_t_batch[0][i] = self.memory[idx][7][0][0]
                prev_state_2_t_batch[1][i] = self.memory[idx][7][1][0]            
                action_t_batch[i] = self.memory[idx][8]

                reward_batch[i] = self.memory[idx][9]
                terminate_batch[i] = self.memory[idx][10]

                status_batch[i] = self.memory[idx][11]
                action_batch[i] = self.memory[idx][12]

                laser_t1_batch[i] = self.memory[idx+1][0]
                cmd_t1_batch[i] = self.memory[idx+1][1]
                cmd_next_t1_batch[i] = self.memory[idx+1][2]
                # prev_action_t1_batch[i] = self.memory[idx+1][4]
                goal_t1_batch[i] = self.memory[idx+1][5]
                prev_state_t1_batch[0][i] = self.memory[idx+1][6][0][0]
                prev_state_t1_batch[1][i] = self.memory[idx+1][6][1][0]
                action_t1_batch[i] = self.memory[idx+1][8]

                if cmd_t_batch[i] == 5:
                    goal_a_t_batch[i] = self.memory[idx][5]
                else:
                    goal_a_t_batch[i] = [0., 0.]
                if cmd_t1_batch[i] == 5:
                    goal_a_t1_batch[i] = self.memory[idx+1][5]
                else:
                    goal_a_t1_batch[i] = [0., 0.]

            return [laser_t_batch, cmd_t_batch, cmd_next_t_batch, cmd_skip_t_batch, prev_action_t_batch, 
                    goal_t_batch, goal_a_t_batch, prev_state_t_batch, prev_state_2_t_batch, action_t_batch,
                    reward_batch, terminate_batch, status_batch, action_batch,
                    laser_t1_batch, cmd_t1_batch, cmd_next_t1_batch, action_t_batch, 
                    goal_t1_batch, goal_a_t1_batch, prev_state_t1_batch, action_t1_batch], indices
        else:
            print 'samples are not enough'
            return None, None


    def Train(self):
        start_time = time.time()

        batch, indices = self.SampleBatch()

        sample_time =  time.time() - start_time

        if not batch:
            return 0.
        else:
            [laser_t_batch, cmd_t_batch, cmd_next_t_batch, cmd_skip_t_batch, prev_action_t_batch, 
             goal_t_batch, goal_a_t_batch, prev_state_t_batch, prev_state_2_t_batch, action_t_batch,
             reward_batch, terminate_batch, status_batch, action_batch,
             laser_t1_batch, cmd_t1_batch, cmd_next_t1_batch, prev_action_t1_batch, 
             goal_t1_batch, goal_a_t1_batch, prev_state_t1_batch, action_t1_batch] = batch

            #compute target y
            target_a_pred = self.actor.PredictTarget(laser=laser_t1_batch, 
                                                     cmd=cmd_t1_batch, 
                                                     cmd_next=cmd_next_t1_batch, 
                                                     prev_action=prev_action_t1_batch, 
                                                     obj_goal=goal_a_t1_batch, 
                                                     prev_state=prev_state_t1_batch)

            target_q_pred = self.critic.PredictTarget(laser=laser_t1_batch, 
                                                      cmd=cmd_t1_batch, 
                                                      cmd_next=cmd_next_t1_batch, 
                                                      prev_action=prev_action_t1_batch, 
                                                      obj_goal=goal_t1_batch, 
                                                      action=action_t1_batch)
            y = []
            for i in xrange(self.batch_size):
                if terminate_batch[i]:
                    y.append(reward_batch[i])
                else:
                    y.append(reward_batch[i] + self.gamma * target_q_pred[i, 0])

            y = np.expand_dims(np.stack(y), axis=1)

            y_time = time.time() - start_time - sample_time

            # critic update
            q, _ = self.critic.Train(laser=laser_t_batch, 
                                     cmd=cmd_t_batch, 
                                     cmd_next=cmd_next_t_batch, 
                                     prev_action=prev_action_t_batch, 
                                     obj_goal=goal_t_batch, 
                                     action=action_t_batch, 
                                     y=y)

            # actions for a_gradients from critic
            actions, states, states_2 = self.actor.PredictOnline(laser=laser_t_batch, 
                                                                 cmd=cmd_t_batch, 
                                                                 cmd_next=cmd_next_t_batch, 
                                                                 cmd_skip=cmd_skip_t_batch,
                                                                 prev_action=prev_action_t_batch, 
                                                                 obj_goal=goal_a_t_batch, 
                                                                 prev_state=prev_state_t_batch,
                                                                 prev_state_2=prev_state_2_t_batch)
            # memeory states update
            err_h, err_c = self.UpdateState(states, states_2, indices)

            # a_gradients
            a_gradients = self.critic.ActionGradients(laser=laser_t_batch, 
                                                      cmd=cmd_t_batch, 
                                                      cmd_next=cmd_next_t_batch, 
                                                      prev_action=prev_action_t_batch, 
                                                      obj_goal=goal_t_batch, 
                                                      action=actions)                                                    

            # actor update
            self.actor.Train(laser=laser_t_batch, 
                             cmd=cmd_t_batch, 
                             cmd_next=cmd_next_t_batch, 
                             cmd_skip=cmd_skip_t_batch,
                             prev_action=prev_action_t_batch, 
                             obj_goal=goal_a_t_batch, 
                             prev_state=prev_state_t_batch,
                             prev_state_2=prev_state_2_t_batch,
                             a_gradient=a_gradients[0],
                             status_label=status_batch,
                             action_label=action_batch)

            train_time = time.time() - start_time - sample_time - y_time

            # target networks update
            self.critic.UpdateTarget()
            self.actor.UpdateTarget()


            target_time = time.time() - start_time - sample_time - y_time - train_time

            # print 'sample_time:{:.3f}, y_time:{:.3f}, train_time:{:.3f}, target_time:{:.3f}'.format(sample_time,
            #                                                                                         y_time,
            #                                                                                         train_time,
            #                                                                                         target_time)
            
            return q, err_h, err_c

    def UpdateState(self, states, states_2, indices):
        err_h = 0.
        err_c = 0.
        for idx, sample_id in enumerate(indices):
            state_h = states[0][idx]
            state_c = states[1][idx]
            state_2_h = states_2[0][idx]
            state_2_c = states_2[1][idx]
            if not self.memory[sample_id][10]:
                err_h += np.mean(np.fabs(self.memory[sample_id+1][6][0][0] - state_h))
                err_c += np.mean(np.fabs(self.memory[sample_id+1][6][1][0] - state_c))
                self.memory[sample_id+1][6][0][0] = state_h
                self.memory[sample_id+1][6][1][0] = state_c
                self.memory[sample_id+1][7][0][0] = state_2_h
                self.memory[sample_id+1][7][1][0] = state_2_c
        return err_h/len(indices), err_c/len(indices)


