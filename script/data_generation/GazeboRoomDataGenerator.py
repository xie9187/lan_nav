import numpy as np
import cv2
import copy
import os
import sys
import time
import rospy
import matplotlib.pyplot as plt
import math 
import csv
import png
import socket
import random
import pickle

from AStar import pathFind
from GazeboWorld import GazeboWorld

CWD = os.getcwd()

class GridWorld(object):
    """docstring for GridWorld"""
    def __init__(self, grid_size=10, table_size=23, P2R=0.1000):
        self.table_size = table_size
        self.grid_size = grid_size
        self.map_size = grid_size*table_size
        self.path_width = self.grid_size
        self.wall_width = 1
        self.p2r = P2R
        self.null_cmd = 0

        # n = m = table_size
        # directions = 4 # number of possible directions to move on the map
        # dx = [1, 0, -1, 0]
        # dy = [0, 1, 0, -1]

        # self.table_se = np.array([[1, 1, n - 2, m - 2],
        #                           [1, m - 2, n - 2, 1],
        #                           ])
        # self.map_se = (self.table_se + 0.5) * self.grid_size * self.p2r
        # path = ''

        self.Clear()
        self.CreateMap()
        self.CreateTable()

    def Clear(self):
        self.table = np.zeros((self.table_size, self.table_size))
        self.map = np.zeros((self.map_size, self.map_size))

    def CreateTable(self):
        self.table = np.zeros((self.table_size, self.table_size))
        for row in xrange(self.table_size):
            for column in xrange(self.table_size):
                val = self.map[row*self.grid_size+1, column*self.grid_size+1]
                if val == 1.:
                    self.table[row, column] = val

    def DrawVerticalLine(self, y, x, val, cali=0):
        y = np.sort(y).tolist()
        self.map[y[0]*self.grid_size : np.amin([y[1]*self.grid_size+cali, self.map_size]), np.amin([x*self.grid_size+cali, self.map_size-1])] = val

    def DrawHorizontalLine(self, x, y, val, cali=0):
        x = np.sort(x).tolist()
        self.map[np.amin([y*self.grid_size+cali, self.map_size-1]), x[0]*self.grid_size : np.amin([x[1]*self.grid_size+cali, self.map_size])] = val

    def DrawSquare(self, x, y, val):
        self.map[x*self.grid_size+1:np.amin([(x+1)*self.grid_size, self.map_size-1]), y*self.grid_size+1:np.amin([(y+1)*self.grid_size, self.map_size-1])] = val

    def CreateMap(self, room_size=5):
        self.map = np.zeros((self.map_size, self.map_size))
        self.room_size = room_size
        # construct walls
        # boundary
        self.DrawVerticalLine([0, self.table_size], 0, 1)
        self.DrawHorizontalLine([0, self.table_size], 0, 1)
        self.DrawVerticalLine([0, self.table_size], self.table_size, 1)
        self.DrawHorizontalLine([0, self.table_size], self.table_size, 1)
        # internal walls
        i = 0
        j = 0
        skip = [[1, 1], [1, 2], [3, 1], [3, 2]]
        for j in xrange(0, 4):
            for i in xrange(0, 4):
                if [j, i] in skip:
                    continue
                self.DrawVerticalLine([j*(room_size+1), j*(room_size+1)+room_size], i*(room_size+1), 1)
                self.DrawHorizontalLine([i*(room_size+1), i*(room_size+1)+room_size], j*(room_size+1), 1)
                self.DrawVerticalLine([j*(room_size+1), j*(room_size+1)+room_size], i*(room_size+1)+room_size, 1, cali=-1)
                self.DrawHorizontalLine([i*(room_size+1), i*(room_size+1)+room_size], j*(room_size+1)+room_size, 1, cali=-1)

        # doors  [y, x, 0:h/1:v]
        doors = [[[3, 0, 0], [3, 0, 0], [3, 0, 0], [3, 0, 0]], 
                 [[0, 3, 1], [0, 0, 0], [0, 0, 0], [0, -2, 1]],
                 [[0, 3, 1], [3, 0, 0], [3, 0, 0], [0, -2, 1]],
                 [[-2, 0, 0], [0, 0, 0], [0, 0, 0], [-2, 0, 0]]]

        self.doors = doors 
        for j in xrange(len(doors)):
            for i in xrange(len(doors[0])):
                if doors[j][i][0] > 0 or doors[j][i][1] > 0:
                    calib = -1
                else:
                    calib = 0
                if doors[j][i][2] == 1:
                    self.DrawVerticalLine([j*room_size+j+2+doors[j][i][0], j*room_size+j+3+doors[j][i][0]], i*room_size+i+2+doors[j][i][1], 0, cali=calib)
                else:
                    self.DrawHorizontalLine([i*room_size+i+2++doors[j][i][1], i*room_size+i+3+doors[j][i][1]], j*room_size+j+2+doors[j][i][0], 0, cali=calib)
        # funitures
        full1 = [] # h
        full2 = [] # v
        for j in xrange(-2, 3):
            for i in xrange(-2, 3):
                full1.append([j, i])
        for j in xrange(-2, 3):
            for i in xrange(-2, 3):
                full2.append([i, j])
        funitures = [[[[-2, -2], [-1, -2], [-2, -1], [-2, 0], [-1, 0], [-2, 1], [-1, 1], [-2, 2], [-1, 2], [0, 2]], 
                      [[-2, -2], [-1, -2], [-2, -1], [-1, -1], [-2, 0], [-1, 0], [-2, 1], [-2, 2], [-1, 2]], 
                      [[-1, -2], [-2, -1], [-2, 0], [-1, 0], [-2, 1], [-1, 2]], 
                      [[-2, -2], [-2, -1], [-2, 0], [-1, 0], [-2, 1], [-2, 2], [0, 2]]], 
                     [[[-2, -2], [2, -2], [-1, -2], [-2, -1], [-2, 0], [-1, 0], [-2, 1], [2, 0]], 
                      full1, 
                      full1,
                      full2[10:]],
                     [full2[:5]+[[-2, -1], [0, -1], [2, -1], [-2, 0], [-2, 1]],
                      full1[:5]+[[-1, -2], [0, -2], [-1, -1], [-1, 1]], 
                      full1[:5]+[[-1, 0], [-1, 2]], 
                      [[-1, 0], [0, 0], [1, 0], [2, 0], [-1, 1], [0, 1], [1, 1], [2, 1], [-1, 2], [0, 2], [1, 2], [2, 2]]],
                     [full2[:5]+full1[-5:]+[[0, -1], [1, -1], [1, 0]], 
                      full1, 
                      full1, 
                      [[1, -2], [0, 0], [1, 0], [2, 0], [0, 2], [1, 2], [2, 2]]]]
        for Y in xrange(0, 4):
            for X in xrange(0, 4):
                for y in xrange(0, room_size):
                    for x in xrange(0,room_size):
                        for coor in funitures[Y][X]:
                            self.DrawSquare(coor[0]+Y*(room_size+1)+2, coor[1]+X*(room_size+1)+2, 1)
                            
    def GetAugMap(self):
        augment_area = 3
        mid_map = np.zeros([self.map_size, self.map_size])
        self.aug_map = np.zeros([self.map_size, self.map_size])
        for y in xrange(0, self.map_size):
            for x in xrange(0, self.map_size):
                if self.map[y][x] == 1:
                   x_min = np.amax([x-augment_area, 0])
                   x_max = np.amin([x+augment_area+1, self.map_size])
                   y_min = np.amax([y-augment_area, 0])
                   y_max = np.amin([y+augment_area+1, self.map_size])
                   mid_map[y_min:y_max, x_min:x_max]= 1

        augment_area = 1
        self.aug_map = copy.deepcopy(mid_map)
        for y in xrange(0, self.map_size):
            for x in xrange(0, self.map_size):
                table_y = y/self.grid_size
                table_x = x/self.grid_size
                if table_y % (self.room_size + 1)==self.room_size or table_x % (self.room_size + 1)==self.room_size:
                    if mid_map[y][x] == 1:
                       x_min = np.amax([x-augment_area, 0])
                       x_max = np.amin([x+augment_area+1, self.map_size])
                       y_min = np.amax([y-augment_area, 0])
                       y_max = np.amin([y+augment_area+1, self.map_size])
                       self.aug_map[y_min:y_max, x_min:x_max]= 1   

        # fig=plt.figure(figsize=(1, 2))
        # fig.add_subplot(1, 2, 1)
        # plt.imshow(mid_map)
        # fig.add_subplot(1, 2, 2)
        # plt.imshow(self.aug_map)
        # plt.show()
        # assert False

    def RandomInitPose(self):
        space = 1.
        while space == 1.:
            position = np.random.randint(0, 4, size=[2])
            table_goal_x = position[0]*6+2
            table_goal_y = position[1]*6+2
            space = copy.deepcopy(self.table[table_goal_y, table_goal_x])
        real_goal_x = (table_goal_x+0.5)*self.grid_size * self.p2r
        real_goal_y = (table_goal_y+0.5)*self.grid_size * self.p2r
        real_goal_theta = self.wrap2pi(np.random.randint(4) * np.pi/2)

        return [real_goal_x, real_goal_y, real_goal_theta]

    def Table2RealPosition(self, table_position):
        real_y = (table_position[0]+0.5)*self.grid_size * self.p2r
        real_x = (table_position[1]+0.5)*self.grid_size * self.p2r
        return [real_x, real_y]

    def RandomInitPoseInRoom(self, current_room=None):
        space = 1.
        room_sum = 25
        aug_map = 1.

        room_table_list = [[1, 2], [2, 2], [3, 2],
                           [2, 1], [2, 2], [2, 3]]
        while space == 1. or room_sum == 1 or aug_map == 1:
            position = np.random.randint(0, 4, size=[2])
            if (position == current_room).all():
                continue
            # table_goal_x = position[0] * (self.room_size + 1) + np.random.randint(0, self.room_size-1)
            # table_goal_y = position[1] * (self.room_size + 1) + np.random.randint(0, self.room_size-1)
            idx =  np.random.randint(0, len(room_table_list))
            table_goal_x = position[0] * (self.room_size + 1) + room_table_list[idx][0]
            table_goal_y = position[1] * (self.room_size + 1) + room_table_list[idx][1]
            mid_x = position[0] * (self.room_size + 1) + 2
            mid_y = position[1] * (self.room_size + 1) + 2
            space = copy.deepcopy(self.table[table_goal_y, table_goal_x])

            room_sum = np.sum(self.table[mid_y-2:mid_y+3, mid_x-2:mid_x+3])

            map_x = int((table_goal_x+0.5) * self.grid_size)
            map_y = int((table_goal_y+0.5) * self.grid_size)
            aug_map = self.aug_map[map_y][map_x]

        real_x = (table_goal_x+0.5)*self.grid_size * self.p2r
        real_y = (table_goal_y+0.5)*self.grid_size * self.p2r
        real_theta = self.wrap2pi(np.random.rand() * np.pi * 2)

        return [real_x, real_y, real_theta], [map_x, map_y], position

    def GetPath(self, se):
        n = m = self.map_size
        directions = 8 # number of possible directions to move on the map

        if directions == 4:
            dx = [1, 0, -1, 0]
            dy = [0, 1, 0, -1]
        elif directions == 8:
            dx = [1, 1, 0, -1, -1, -1, 0, 1]
            dy = [0, 1, 1, 1, 0, -1, -1, -1]

        [xA, yA, xB, yB] = se
        path = pathFind(copy.deepcopy(self.aug_map), directions, dx, dy, xA, yA, xB, yB, n, m)
        map_route = []
        x = copy.deepcopy(xA)
        y = copy.deepcopy(yA)
        for t in xrange(len(path)):
            x+=dx[int(path[t])]
            y+=dy[int(path[t])]
            map_route.append([x, y])
        if len(map_route) > 0:
            real_route = (np.asarray(map_route, dtype=float) * self.p2r).tolist()
        else:
            real_route = []
        return map_route, real_route

    def GetNextNearGoal(self, path, pose):
        last_point = path[0]
        if np.linalg.norm([pose[0]-last_point[0], pose[1]-last_point[1]]) < 0.5:
            return last_point, path[1:]
        else:
            return last_point, path

    def NextGoal(self, pose, model_states_data):
        table_current_x = int(pose[0] / self.grid_size / self.p2r)
        table_current_y = int(pose[1] / self.grid_size / self.p2r)

        assert self.table[table_current_y, table_current_x] == 0, "current position is not free"

        # curr room
        curr_room_position = np.array([table_current_y/6, table_current_x/6])
        door = self.doors[curr_room_position[0]][curr_room_position[1]]
        if door[2] == 0: # h
            real_door_x = (curr_room_position[1]*6+2.5+door[1])*self.grid_size * self.p2r
            real_door_y = (curr_room_position[0]*6+2+door[0])*self.grid_size * self.p2r
        else: # v
            real_door_x = (curr_room_position[1]*6+2+door[1])*self.grid_size * self.p2r
            real_door_y = (curr_room_position[0]*6+2.5+door[0])*self.grid_size * self.p2r
        theta = np.arctan2(real_door_y-pose[1], real_door_x-pose[0])
        init_goal = [pose[0], pose[1], theta]

        # select room
        space = 1
        room_position = copy.deepcopy(curr_room_position)
        while space == 1 or (room_position == curr_room_position).all():
            room_position = np.random.randint(0, 4, size=[2])
            table_goal_x = room_position[1]*6+2
            table_goal_y = room_position[0]*6+2
            space = self.table[table_goal_y, table_goal_x]

        real_goal_x = (table_goal_x+0.5)*self.grid_size * self.p2r
        real_goal_y = (table_goal_y+0.5)*self.grid_size * self.p2r

        door = self.doors[room_position[0]][room_position[1]]
        if door[2] == 0: # h
            real_door_x = (room_position[1]*6+2.5+door[1])*self.grid_size * self.p2r
            real_door_y = (room_position[0]*6+2+door[0])*self.grid_size * self.p2r
        else: # v
            real_door_x = (room_position[1]*6+2+door[1])*self.grid_size * self.p2r
            real_door_y = (room_position[0]*6+2.5+door[0])*self.grid_size * self.p2r

        # find objects in goal room
        goal_name_class = ['side_table', 'trash_bin', 'human', 'plant']
        room_model = []
        room_origin = room_position * 6.0 + 2.5
        for model_name, model_pose in zip(model_states_data.name, model_states_data.pose):
            x = model_pose.position.x
            y = model_pose.position.y
            for class_name in goal_name_class:
                if class_name in model_name:
                    if room_origin[1] - 2.4 < x < room_origin[1] + 2.4 and\
                        room_origin[0] - 2.4 < y < room_origin[0] + 2.4:
                        room_model.append([model_name, model_pose])

        door_x = real_door_x
        door_y = real_door_y    
        if len(room_model) > 0:
            model_name, model_pose = room_model[np.random.randint(len(room_model))]
            obj_x = model_pose.position.x
            obj_y = model_pose.position.y
            obj_z = model_pose.position.z

            dist = np.linalg.norm([door_x-obj_x, door_y-obj_y])
            theta = np.arctan2(obj_y-door_y, obj_x-door_x)

            final_goal = np.asarray([obj_x, obj_y])+(np.asarray([door_x, door_y])-np.asarray([obj_x, obj_y]))*1.3/dist
            final_goal = final_goal.tolist()+[theta]
        else:
            model_name = None
            obj_x = room_position[1]
            obj_y = room_position[0]
            obj_z = 0.

            dist = np.linalg.norm([door_x-obj_x, door_y-obj_y])
            theta = np.arctan2(obj_y-door_y, obj_x-door_x)

            final_goal = np.asarray([obj_x, obj_y])+(np.asarray([door_x, door_y])-np.asarray([obj_x, obj_y]))*2./dist
            final_goal = final_goal.tolist()+[theta]    

        return init_goal, final_goal, model_name

    def GetTablePlan(self, plan):
        table_plan = []
        plan_buf = []
        [x, y] = plan[0]
        table_x = int(x/self.grid_size/self.p2r)
        table_y = int(y/self.grid_size/self.p2r)
        table_plan.append([table_x, table_y])
        for t in xrange(len(plan)):
            [x, y] = plan[t]
            prev_table_x = table_x
            prev_table_y = table_y 
            table_x = int(x/self.grid_size/self.p2r)
            table_y = int(y/self.grid_size/self.p2r)
            if table_x != prev_table_x or table_y != prev_table_y:
                table_plan.append([table_x, table_y])

        return table_plan

    def Direction(self, theta):
        theta = self.wrap2pi(theta)
        if theta < np.pi/4 and theta >= -np.pi/4:
            direction = 0
        elif theta >= np.pi/4 and theta < np.pi/4*3:
            direction = 1
        elif theta >= np.pi/4*3 and theta < -np.pi/4*3:
            direction = 2
        else:
            direction = 3
        dx = [1, 0, -1, 0]
        dy = [0, 1, 0, -1]
        return [dx[direction], dy[direction]]

    def NumOfDirect(self, x, y):
        num = 0
        if self.table[y-1][x] == 0:
            num += 1
        if self.table[y+1][x] == 0:
            num += 1
        if self.table[y][x+1] == 0:
            num += 1
        if self.table[y][x-1] == 0:
            num += 1
        return num

    def GetCommandFromTablePlan(self, curr_table_pos, table_plan):
        check_point_list = [[5, 2], [5, 5], [5, 8], [5, 11], [5, 14], [5, 17], [5, 20],
                            [8, 5], [8, 17],
                            [11, 5], [11, 11], [11, 17],
                            [14, 5], [14, 17],
                            [17, 2], [17, 5], [17, 8], [17, 11], [17, 14], [17, 17], [17, 20]]

        check_points_on_path = []
        cmd_list = []
        for table_idx, table_point in enumerate(table_plan):
            if [table_point[1], table_point[0]] in check_point_list:
                x_tm1 = table_plan[table_idx-1][0]
                y_tm1 = table_plan[table_idx-1][1]
                x_t = table_plan[table_idx][0]
                y_t = table_plan[table_idx][1]
                x_t1 = table_plan[table_idx+1][0]
                y_t1 = table_plan[table_idx+1][1]
                v1 = np.array([x_t - x_tm1, y_t - y_tm1])
                v2 = np.array([x_t1 - x_t, y_t1 - y_t])
                err = self.wrap2pi(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])) 
                cmd = self.err2cmd(err)
                cmd_list.append(cmd)
                check_points_on_path.append([table_point[1], table_point[0]])

        return check_points_on_path, cmd_list 


    def GetCommandPlan(self, pose, path):
        check_point_list = [[5, 2], [5, 5], [5, 8], [5, 11], [5, 14], [5, 17], [5, 20],
                            [8, 5], [8, 17],
                            [11, 5], [11, 11], [11, 17],
                            [14, 5], [14, 17],
                            [17, 2], [17, 5], [17, 8], [17, 11], [17, 14], [17, 17], [17, 20]]

        table_plan = self.GetTablePlan(path)
        target_points = []
        cmd_list = [3]
        for table_idx, table_point in enumerate(table_plan):
            if [table_point[1], table_point[0]] in check_point_list:
                x_tm1 = table_plan[table_idx-1][0]
                y_tm1 = table_plan[table_idx-1][1]
                x_t = table_plan[table_idx][0]
                y_t = table_plan[table_idx][1]
                x_t1 = table_plan[table_idx+1][0]
                y_t1 = table_plan[table_idx+1][1]
                v1 = np.array([x_t - x_tm1, y_t - y_tm1])
                v2 = np.array([x_t1 - x_t, y_t1 - y_t])
                err = self.wrap2pi(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])) 
                cmd = self.err2cmd(err)
                cmd_list.append(cmd)
                
                if len(cmd_list) == 2:
                    target_points.append([table_point[1], table_point[0]])
                target_points.append([table_plan[table_idx+1][1], table_plan[table_idx+1][0]])

        target_points.append([table_point[1], table_point[0]])

        return target_points, cmd_list+[5, 0], check_point_list


    def RandomSwitchRoom(self):
        # pool_room = np.array([[0, 0, 1], [1, 0, 0], [2, 0, 0], [3, 0, 3],
        #                     [0, 1, 1], [2, 1, 1], [0, 2, 1], [2, 2, 1],
        #                     [0, 3, 1], [1, 3, 2], [2, 3, 2], [3, 3, 3]])
        pool_room = np.array([[0, 0, 1], [0, 1, 1], [0, 2, 1], [0, 3, 1],
                              [1, 0, 0], [1, 1, 0], [1, 2, 3], [1, 3, 2],
                              [2, 0, 0], [2, 1, 1], [2, 2, 1], [2, 3, 2],
                              [3, 0, 3], [3, 1, 3], [3, 2, 2], [3, 3, 3]])

        switch_action = []
        switched_map = copy.deepcopy(self.map)
        # randomly swtich room
        ids = random.sample(range(0, len(pool_room)), 6)
        for x in xrange(0, 3):
            room_ids = [ids[2*x], ids[2*x+1]]
            temp_pool_room = copy.deepcopy(pool_room[room_ids[0]][:2])
            pool_room[room_ids[0]][:2] = pool_room[room_ids[1]][:2]
            pool_room[room_ids[1]][:2] = temp_pool_room
            # y0, x0, y1, x1, 0->1 theta(times of 90 degree)
            theta = pool_room[room_ids[1]][2] - pool_room[room_ids[0]][2]
            switch_action.append(pool_room[room_ids[0]][:2].tolist()\
                                 +pool_room[room_ids[1]][:2].tolist()\
                                 +[theta])
            map_y0 = self.grid_size*6*pool_room[room_ids[0]][0]
            map_x0 = self.grid_size*6*pool_room[room_ids[0]][1]
            map_y1 = self.grid_size*6*pool_room[room_ids[1]][0]
            map_x1 = self.grid_size*6*pool_room[room_ids[1]][1]
            # temp_room = copy.deepcopy(switched_map[map_y0+1:(map_y0+5*self.grid_size), map_x0+1:(map_x0+5*self.grid_size)])
            # switched_map[map_y0+1:(map_y0+5*self.grid_size), map_x0+1:(map_x0+5*self.grid_size)] = \
            #   copy.deepcopy(np.rot90(switched_map[map_y1+1:(map_y1+5*self.grid_size), map_x1+1:(map_x1+5*self.grid_size)], -theta))
            # switched_map[map_y1+1:(map_y1+5*self.grid_size), map_x1+1:(map_x1+5*self.grid_size)] = \
            #   copy.deepcopy(np.rot90(temp_room, theta))
            temp_room = copy.deepcopy(switched_map[map_y0:(map_y0+5*self.grid_size), map_x0:(map_x0+5*self.grid_size)])
            switched_map[map_y0:(map_y0+5*self.grid_size), map_x0:(map_x0+5*self.grid_size)] = \
                copy.deepcopy(np.rot90(switched_map[map_y1:(map_y1+5*self.grid_size), map_x1:(map_x1+5*self.grid_size)], -theta))
            switched_map[map_y1:(map_y1+5*self.grid_size), map_x1:(map_x1+5*self.grid_size)] = \
                copy.deepcopy(np.rot90(temp_room, theta))

        return switched_map, switch_action


    def err2cmd(self, err):
        if err == np.pi/2:
            cmd = 2
        elif err == -np.pi/2:
            cmd = 4
        elif err == 0:
            cmd = 1
        else:
            cmd = 3
        return cmd

    def wrap2pi(self, ang):
        while ang > np.pi:
            ang -= np.pi * 2
        while ang < -np.pi:
            ang += np.pi * 2
        return ang

def FileProcess():
    # --------------change the map_server launch file-----------
    with open('./config/map.yaml', 'r') as launch_file:
        launch_data = launch_file.readlines()
        launch_data[0] = 'image: ' + CWD + '/world/map.png\n'
    with open('./config/map.yaml', 'w') as launch_file:
        launch_file.writelines(launch_data)     

    time.sleep(1.)
    print "file processed"

def LogData(Data, num, path):
    name = ['laser', 'action', 'cmd', 'goal', 'goal_pose', 'cmd_list', 'obj_name']
    for x in xrange(len(name)):
        file = open(path+'/'+str(num)+'_'+name[x]+'.csv', 'w')
        writer = csv.writer(file, delimiter=',', quotechar='|')
        for row in Data[x]:
            if not isinstance(row, list):
                row = [row]
            writer.writerow(row)

def DataGenerate(data_path, robot_name='robot1', rviz=False):
    world = GridWorld()
    world.map, switch_action = world.RandomSwitchRoom()
    print(switch_action)
    world.CreateTable()
    cv2.imwrite('./world/map.png', np.flipud(1-world.map)*255)

    FileProcess()

    env = GazeboWorld(world.table, robot_name, rviz=rviz)
    print "Env initialized"

    rate = rospy.Rate(5.)
    T = 0

    time.sleep(2.)
    pickle_path = os.path.join(CWD, 'world/model_states_data.p')
    # pickle.dump( env.model_states_data, open(pickle_path, "wb"))
    # assert False
    env.ResetWorld()
    env.ResetModelsPose(pickle_path)
    env.SwitchRoom(switch_action)
    env.state_call_back_flag = True
    time.sleep(2.)


    init_pose = world.RandomInitPose()
    env.target_point = init_pose
    
    for x in xrange(10000):
        rospy.sleep(2.)
        env.SetObjectPose(robot_name, [env.target_point[0], env.target_point[1], 0., env.target_point[2]])
        rospy.sleep(4.)
        print ''
        env.plan = None
        pose = env.GetSelfStateGT()
        init_goal, final_goal, obj_name = world.NextGoal(pose, env.model_states_data)
        env.target_point = final_goal
        goal = copy.deepcopy(env.target_point)
        env.distance = np.sqrt(np.linalg.norm([pose[0]-goal[0], pose[1]-goal[1]]))
        env.GoalPublish(goal)
        print 'goal', goal

        j = 0
        terminal = False
        laser_save = []
        action_save = []
        cmd_save = []
        goal_save = []
        goal_pose_save = []

        plan_time_start = time.time()
        no_plan_flag = False
        while (env.plan_num < 2 or env.next_near_goal is None) \
                and not rospy.is_shutdown():
            if time.time() - plan_time_start > 3.:
                no_plan_flag = True
                break

        if no_plan_flag:
            print 'no available plan'
            env.GoalCancel()
            rospy.sleep(2.)
            env.plan_num = 0
            continue

        print 'plan recieved'
        pose = env.GetSelfStateGT()
        plan = env.GetPlan()

        if plan:
            print 'plan length', len(plan)
        else:
            env.GoalCancel()
            rospy.sleep(2.)
            env.plan_num = 0
            continue

        if len(plan) == 0 :
            env.GoalCancel()
            rospy.sleep(2.)
            env.plan_num = 0
            continue

        check_point_list, cmd_list = world.GetCommandPlan(pose, plan)
        print cmd_list

        idx = 0
        file_num = len([f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))])
        stage = 0
        table_current_x = int(pose[0] / world.grid_size / world.p2r)
        table_current_y = int(pose[1] / world.grid_size / world.p2r)
        table_position = [table_current_x, table_current_y]
        cmd_idx = 0
        prev_cmd_idx = 0
        loop_time_buff = []
        while not terminal and not rospy.is_shutdown():
            start_time = time.time()
            rgb = env.GetRGBImageObservation()
            laser = env.GetLaserObservation()
            pose = env.GetSelfStateGT()
            [v, w] = env.GetSelfSpeedGT()
            goal = env.GetNextNearGoal(theta_flag=True)

            

            terminal, result, _ = env.GetRewardAndTerminate(j)
            if result == 1: # crash
                break
            
            init_dist = np.linalg.norm([pose[0] - init_goal[0], pose[1] - init_goal[1], pose[2] - init_goal[2]])
            goal_dist = np.linalg.norm([pose[0] - final_goal[0], pose[1] - final_goal[1]])

            if init_dist < 0.1 and stage == 0:
                stage += 1
                prev_cmd_idx = copy.deepcopy(cmd_idx)
                cmd_idx += 1
            elif goal_dist < env.dist_threshold and stage == 1:
                stage += 1

            if stage == 0:
                curr_goal = init_goal
                curr_goal_theta = init_goal[2]
            elif stage == 2:
                curr_goal = goal
                curr_goal_theta = goal[2]
            else:
                curr_goal = goal
                curr_goal_theta = None

            local_goal = env.GetLocalPoint(curr_goal)
            save_local_goal = env.GetLocalPoint(goal)

            env.PathPublish(save_local_goal)
            action = env.Controller(stage=stage, target_point=local_goal, target_theta=curr_goal_theta)
            env.SelfControl(action)
            
            # command
            table_current_x = int(pose[0] / world.grid_size / world.p2r)
            table_current_y = int(pose[1] / world.grid_size / world.p2r)
            prev_table_position = copy.deepcopy(table_position)
            table_position = [table_current_y, table_current_x]
            if prev_table_position in check_point_list and table_position not in check_point_list:
                cmd_idx += 1

            if table_position in check_point_list:
                cmd = 0
            else:
                cmd = cmd_list[cmd_idx]

            if cmd == 5:
                goal_pose = env.GetLocalPoint(env.target_point) 
            else:
                goal_pose = [0., 0.]

            # log data
            laser_save.append(laser.tolist())
            action_save.append(action.tolist())
            cmd_save.append(cmd)
            goal_save.append(save_local_goal)
            goal_pose_save.append(goal_pose)

            # cv2.imwrite(data_path+'/image/'+str(j)+'.png', rgb)

            if result == 2 and cmd == 5:
                cmd_save[0] = 0
                Data = [laser_save[1:], action_save[1:], cmd_save[:-1], goal_save[1:], goal_pose_save[1:], cmd_list, obj_name]
                print "save sequence "+str(file_num/7)
                LogData(Data, str(file_num/7), data_path)
                laser_save, action_save, cmd_save, goal_save, goal_pose_save = [], [], [], [], []

            j += 1
            T += 1
            rate.sleep()
            loop_time = time.time() - start_time
            loop_time_buff.append(loop_time)
            # print 'loop time: {:.4f} |action: {:.2f}, {:.2f} | stage: {:d} | obj: {:s}'.format(
            #                   time.time() - start_time, action[0], action[1], stage, obj_name)
            if (j+1)%100 == 0:
                print 'loop time mean: {:.4f} | max:{:.4f}'.format(np.mean(loop_time_buff), np.amax(loop_time_buff))
        env.plan_num = 0
        

if __name__ == '__main__':
    args = sys.argv
    robot_name = args[1]
    rviz = args[2]
    print robot_name

    machine_id = socket.gethostname()
    data_path = os.path.join(CWD[:-7], 'lan_nav_data/room')
    data_path = os.path.join(data_path, machine_id+'_'+robot_name)

    try:
        os.stat(data_path)
    except:
        os.makedirs(data_path)
    DataGenerate(data_path, rviz=rviz)

