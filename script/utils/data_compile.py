from __future__ import print_function

import numpy as np
import csv
import random
import os
import copy
import sys
from collections import defaultdict


def read_file(file_name):
    file = open(file_name, 'r')
    print(file_name)
    file_reader = csv.reader(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
    curr_seq = []
    for row in file_reader:
        if type(row[0]) == str:
            new_row = row[0][1:-1].split()
            row = [ float(r) for r in new_row]
        curr_seq.append(row)
    return curr_seq

def save_file(file_name, data):
    file = open(file_name, 'w')
    writer = csv.writer(file, delimiter=',')
    for idx, row in enumerate(data):
        print('save {:}/{:} \r'.format(idx, len(data)), end="\r")
        sys.stdout.flush() 
        if not isinstance(row, list):
            row = [row]
        writer.writerow(row)
    file.close()
    

def main(target_path, source_list):
    names = ['laser', 'action', 'cmd', 'goal', 'goal_pose', 'cmd_list']
    
    Data =  [[], [], [], [], [], []]
    for path in source_list:
        nums = []
        file_list = os.listdir(path)
        print('load from '+path)
        for file_name in file_list:
            if 'laser' in file_name:
                nums.append(int(file_name[:file_name.find('_')]))
        nums = np.sort(nums).tolist()
        print('find '+str(max(nums)-min(nums)+1)+' samples')
        for n in nums:
            print('load {:}/{:} \r'.format(n, len(nums)), end="\r")
            sys.stdout.flush() 
            for name_id, name in enumerate(names): 
                file_name = os.path.join(path, str(n)+'_'+name+'.csv')
                data = np.reshape(np.squeeze(read_file(file_name)), [-1]).tolist()
                if len(read_file(file_name)) > 700:
                    print('seq too long')
                    break
                Data[name_id].append(data)

    for name_id, name in enumerate(names): 
        meta_file_name = os.path.join(target_path, name+'.csv')
        print('start to save '+name)
        save_file(meta_file_name, Data[name_id])

def check_max_step_and_cmd(source_list):
    names = ['cmd', 'cmd_list']
    lengths = [[], []]
    n_len_above_500 = 0
    for path in source_list:
        nums = []
        file_list = os.listdir(path)
        print('load from '+path)
        for file_name in file_list:
            if 'laser' in file_name:
                nums.append(int(file_name[:file_name.find('_')]))
        nums = np.sort(nums).tolist()
        print('find '+str(max(nums)-min(nums)+1)+' samples')
        for n in nums:
            print('load {:}/{:} \r'.format(n, len(nums)), end="\r")
            sys.stdout.flush() 
            for name_id, name in enumerate(names): 
                file_name = os.path.join(path, str(n)+'_'+name+'.csv')
                length = len(read_file(file_name))
                lengths[name_id].append(length)
                if length > 700:
                    n_len_above_500 += 1
    print('max_step:'+str(np.amax(lengths[0]))+', max_cmd:'+str(np.amax(lengths[1]))+', n_l>700:'+str(n_len_above_500))

if __name__ == '__main__':

    CWD = os.getcwd()
    target_path = os.path.join(CWD[:-7], 'lan_nav_data/room/meta_file')
    if not os.path.exists(target_path): 
        os.makedirs(target_path)


    source_list = [os.path.join(CWD[:-7], 'lan_nav_data/room/validation')]
    main(target_path, source_list)
    # check_max_step_and_cmd(source_list)


