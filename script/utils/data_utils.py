from __future__ import print_function

import numpy as np
import csv
import random
import os
import copy
import time
import sys

CWD = os.getcwd()
cwd_idx = CWD.rfind('/')

LASER_DIM = 666
SCALE = 1e5

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


def write_unit_meta_file(source_path_number_list, meta_file_folder, max_steps):
    names = ['cmd', 'laser', 'action', 'cmd_next', 'obj_pose', 'status']
    dims = [1, LASER_DIM, 2, 1, 2, 1]
    # dtypes = [np.float32, np.float16, np.int8, np.int8, np.float16, np.int8]
    dtypes = [np.int8, np.int32, np.int32, np.int8, np.int32, np.int8]
    scales = [1, SCALE, SCALE, 1, SCALE, 1]
    Data = [[], [], [], [], [], [], []]

    for file_id, path_num in enumerate(source_path_number_list):
        print('load {:}/{:} \r'.format(file_id, len(source_path_number_list)), end="\r")
        sys.stdout.flush() 
        
        for name_id, name in enumerate(names): 
            file_name = path_num+'_'+name+'.csv'
            data = (np.reshape(read_file(file_name), [-1]) * scales[name_id]).astype(dtypes[name_id])
            if name is 'cmd' :
                # count the numer of 3
                num_3 = (data == 3).sum()

            max_len = min(len(data), max_steps * dims[name_id])
            data_vector = np.zeros((max_steps * dims[name_id]), dtype=dtypes[name_id])
            data_vector[:max_len-num_3*dims[name_id]] = data[num_3*dims[name_id]:max_len]
            Data[name_id].append(data_vector)
            if name is 'status':
                Data[-1].append(len(data))

    for name_id, name in enumerate(names): 
        meta_file_name = os.path.join(meta_file_folder, name+'.csv')
        print('start to save '+meta_file_name)
        data_save = np.stack(Data[name_id], axis=0)
        data_save.tofile(meta_file_name)
        # save_file(meta_file_name, Data[name_id])

    meta_file_name = os.path.join(meta_file_folder, 'length.csv')
    print('start to save '+meta_file_name)
    data_save = np.stack(Data[-1], axis=0)
    data_save.tofile(meta_file_name)
    # save_file(meta_file_name, Data[-1])


def write_multiple_meta_files(source_list, meta_file_path, max_file_num):
    file_path_number_list = []
    for path in source_list:
        nums = []
        file_list = os.listdir(path)
        print('Loading from '+path)
        for file_name in file_list:
            if 'laser' in file_name:
                nums.append(int(file_name[:file_name.find('_')]))
        nums = np.sort(nums).tolist()
        for num in nums:
            file_path_number_list.append(path+'/'+str(num))
    print('Found {} sequences!!'.format(len(file_path_number_list)))

    start_time = time.time()
    for x in xrange(len(file_path_number_list)/max_file_num): # only create full meta batch
        source_path_number_list = file_path_number_list[max_file_num*x:
                                                        min(max_file_num*(x+1), len(file_path_number_list))]                                                                                  
        meta_file_folder = os.path.join(meta_file_path, str(x))
        if not os.path.exists(meta_file_folder): 
            os.makedirs(meta_file_folder) 
        print('Creating ' + meta_file_folder)
        write_unit_meta_file(source_path_number_list, meta_file_folder, 700)

        used_time = time.time() - start_time
        print('Created {}/{} meta files | time used (min): {:.1f}'.format(x+1, len(file_path_number_list)/max_file_num+1, used_time/60.))


def read_meta_file(meta_file_folder, max_steps, batch_size):
    names = ['laser', 'action', 'cmd', 'cmd_next', 'obj_pose', 'status', 'length']
    dims = [LASER_DIM, 2, 1, 1, 2, 1, 1]
    dtypes = [np.int32, np.int32, np.int8, np.int8, np.int32, np.int8, np.int64]
    scales = [SCALE, SCALE, 1, 1, SCALE, 1, 1]
    start_time = time.time()
    Data = []
    for name_id, name in enumerate(names): 
        file_name = os.path.join(meta_file_folder, name+'.csv')
        file = open(file_name, 'r')
        data = np.fromfile(file, dtype=dtypes[name_id])
        file.close()
        if name is not 'length':
            real_data = np.reshape(np.asarray(data, dtype=np.float32), [-1, max_steps * dims[name_id]])/scales[name_id]
        else:
            real_data = np.reshape(np.asarray(data, dtype=np.float32), [-1])
        if name_id == 0:
            indices = random.sample(range(len(real_data)), len(real_data))
        Data.append(np.split(real_data[indices], len(real_data)/batch_size))
    # print(time.time() - start_time)
    return Data, len(real_data)/batch_size


def read_meta_file_mp(inputs):
    meta_file_folder, max_steps, batch_size = inputs
    names = ['laser', 'action', 'cmd', 'cmd_next', 'obj_pose', 'status', 'length']
    dims = [LASER_DIM, 2, 1, 1, 2, 1]
    dtypes = [np.int32, np.int32, np.int8, np.int8, np.int32, np.int8, np.int64]
    scales = [SCALE, SCALE, 1, 1, SCALE, 1]
    start_time = time.time()
    Data = []
    
    for name_id, name in enumerate(names): 
        file_name = os.path.join(meta_file_folder, name+'.csv')
        file = open(file_name, 'r')
        data = np.fromfile(file, dtype=dtypes[name_id])
        file.close()
        if name is not 'length':
            real_data = np.reshape(np.asarray(data, dtype=np.float32), [-1, max_steps * dims[name_id]])/scales[name_id]
        else:
            real_data = np.reshape(np.asarray(data, dtype=np.float32), [-1])
        if name_id == 0:
            indices = random.sample(range(len(real_data)), len(real_data))
        Data.append(np.split(real_data[indices], len(real_data)/batch_size))
    return (Data, len(real_data)/batch_size)


def read_meta_file_old(meta_file_folder, max_steps, batch_size):
    names = ['laser', 'action', 'cmd', 'cmd_next', 'obj_pose', 'status', 'length']
    dims = [LASER_DIM, 2, 1, 1, 2, 1]
    dtypes = [np.int32, np.int32, np.int8, np.int8, np.int32, np.int8, np.int64]
    scales = [SCALE, SCALE, 1, 1, SCALE, 1]
    start_time = time.time()
    Data = []
    for name_id, name in enumerate(names): 
        file_name = os.path.join(meta_file_folder, name+'.csv')
        data = read_file(file_name)
        if name is not 'length':
            real_data = np.reshape(np.asarray(data, dtype=np.float32), [-1, max_steps * dims[name_id]])/scales[name_id]
        else:
            real_data = np.reshape(np.asarray(data, dtype=np.float32), [-1])
        # print(np.shape(real_data))
        # Data.append(np.split(real_data, len(real_data)/batch_size))
        Data.append(real_data)
    # print(time.time() - start_time)
    return Data, len(real_data)/batch_size


if __name__ == '__main__':
    # source_list = [os.path.join(CWD[:-7], 'lan_nav_data/room_new/linhai-Intel-Z370_robot1'),
    #                os.path.join(CWD[:-7], 'lan_nav_data/room_new/linhai-Intel-Z370_robot2'),
    #                os.path.join(CWD[:-7], 'lan_nav_data/room_new/ymiao-office_robot1_1'),
    #                os.path.join(CWD[:-7], 'lan_nav_data/room_new/ymiao-office_robot2_1'),
    #                os.path.join(CWD[:-7], 'lan_nav_data/room_new/ymiao-office_robot1_2'),
    #                os.path.join(CWD[:-7], 'lan_nav_data/room_new/ymiao-office_robot2_2'),
    #                os.path.join(CWD[:-7], 'lan_nav_data/room_new/ymiao-office_robot1_3'),
    #                os.path.join(CWD[:-7], 'lan_nav_data/room_new/ymiao-office_robot2_3'),
    #                os.path.join(CWD[:-7], 'lan_nav_data/room_new/server-bane_robot1'),
    #                os.path.join(CWD[:-7], 'lan_nav_data/room_new/server-bane_robot2'),
    #                os.path.join(CWD[:-7], 'lan_nav_data/room_new/office_robot1'),
    #                os.path.join(CWD[:-7], 'lan_nav_data/room_new/office_robot2'),
    #                os.path.join(CWD[:-7], 'lan_nav_data/room_new/changhao-Z390-AORUS-PRO_robot1'),
    #                os.path.join(CWD[:-7], 'lan_nav_data/room_new/changhao-Z390-AORUS-PRO_robot2'),
    #                os.path.join(CWD[:-7], 'lan_nav_data/room_new/ymiao-System-Product-Name_robot1'),
    #                os.path.join(CWD[:-7], 'lan_nav_data/room_new/ymiao-System-Product-Name_robot2')
    #                ]

    source_list = [os.path.join(CWD[:-7], 'lan_nav_data/room_new/linhai-Intel-Z370_robot1')]

    # meta_file_path = os.path.join(CWD[:-7], 'lan_nav_data/room_new/training_meta')

    # source_list = [os.path.join(CWD[:-7], 'lan_nav_data/room_new/validation')]
    meta_file_path = os.path.join(CWD[:-7], 'lan_nav_data/room_new/validation_meta')
    # meta_file_path_old = os.path.join(CWD[:-7], 'lan_nav_data/room_new/validation_meta')

    # Data, _ = read_meta_file(os.path.join(meta_file_path, '0'), 700, 64)
    # Data_old, _ = read_meta_file_old(os.path.join(meta_file_path_old, '0'), 700, 64)

    # for a, b in zip(Data, Data_old):
    #     err = a-b
    #     print(np.sum(np.fabs(err)))
    # assert(False)

    if not os.path.exists(meta_file_path): 
        os.makedirs(meta_file_path)
    write_multiple_meta_files(source_list, meta_file_path, 1024)