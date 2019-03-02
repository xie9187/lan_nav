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
    names = ['status', 'cmd', 'cmd_next', 'cmd_skip', 'laser', 'action', 'obj_pose'] #  + seq_len
    dims = [1, 1, 1, 1, LASER_DIM, 2, 2]
    dtypes = [np.int8, np.int8, np.int8, np.int8, np.int32, np.int32, np.int32]
    scales = [1, 1, 1, 1, SCALE, SCALE, SCALE]
    Data = [[], [], [], [], [], [], [], []]
    for file_id, path_num in enumerate(source_path_number_list):
        print('load {:}/{:} \r'.format(file_id, len(source_path_number_list)), end="\r")
        sys.stdout.flush() 
        
        for name_id, name in enumerate(names):
            if name is 'cmd':
                file_name = path_num+'_'+name+'.csv'
                cmd_data = (np.reshape(read_file(file_name), [-1]) * scales[name_id]).astype(dtypes[name_id])
                data = cmd_data

                # cmd_list and cmd_idx_seq
                cmd_x_status = cmd_data * status_data.astype(np.int8)
                non_zeros_indices = np.nonzero(cmd_x_status)
                cmd_list = cmd_x_status[non_zeros_indices].tolist() + [0, 0]
                cmd_idx_seq = np.zeros_like(status_data)
                for idx in xrange(1, len(non_zeros_indices[0])):
                    cmd_idx_seq[non_zeros_indices[0][idx-1]+1:non_zeros_indices[0][idx]+1] = idx
                cmd_next_idx_seq = cmd_idx_seq + 1
                cmd_skip_idx_seq = cmd_idx_seq + 2
                cmd_list = np.asarray(cmd_list)

                # print(cmd_list)
                # print(cmd_data)
                # print(cmd_list[cmd_idx_seq])
                # print(cmd_list[cmd_next_idx_seq])
                # print(cmd_list[cmd_skip_idx_seq])
                # assert False

            elif name is 'cmd_next':
                file_name = path_num+'_'+'cmd_next'+'.csv'
                cmd_next_data = cmd_list[cmd_next_idx_seq]
                data = cmd_next_data
            elif name is 'cmd_skip':  
                file_name = path_num+'_'+name+'.csv'
                cmd_skip_data = cmd_list[cmd_skip_idx_seq]
                data = cmd_skip_data
            else:
                file_name = path_num+'_'+name+'.csv'
                data = (np.reshape(read_file(file_name), [-1]) * scales[name_id]).astype(dtypes[name_id])
                if name is 'status':
                    status_data = data

            max_len = min(len(data), max_steps * dims[name_id])
            data_vector = np.zeros((max_steps * dims[name_id]), dtype=dtypes[name_id])
            data_vector[:max_len] = data[:max_len]
            Data[name_id].append(data_vector)
            if name is 'status': # seq_len
                Data[-1].append(len(data))
        
    
    # print(cmd_data)
    # print(cmd_next_data)
    # print(cmd_skip_data)
    # assert False
    # print(np.stack(Data[6], axis=0))
    # assert False
    names += ['length']
    for name_id, name in enumerate(names): 
        meta_file_name = os.path.join(meta_file_folder, name+'.csv')
        print('start to save '+meta_file_name)
        data_save = np.stack(Data[name_id], axis=0)
        data_save.tofile(meta_file_name)


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
        print('Created {}/{} meta files | time used (min): {:.1f}'.format(x+1, len(file_path_number_list)/max_file_num, used_time/60.))


def read_meta_file(meta_file_folder, max_steps, batch_size):
    names = ['laser', 'action', 'cmd', 'cmd_next', 'cmd_skip', 'obj_pose', 'status', 'length']
    dims = [LASER_DIM, 2, 1, 1, 1, 2, 1, 1]
    dtypes = [np.int32, np.int32, np.int8, np.int8, np.int8, np.int32, np.int8, np.int64]
    scales = [SCALE, SCALE, 1, 1, 1, SCALE, 1, 1]
    start_time = time.time()
    Data = []
    for name_id, name in enumerate(names): 
        file_name = os.path.join(meta_file_folder, name+'.csv')
        file = open(file_name, 'r')
        data = np.fromfile(file, dtype=dtypes[name_id])
        file.close()
        if name is not 'length':
            real_data = np.reshape(np.asarray(data, dtype=np.float32), [-1, max_steps, dims[name_id]])/scales[name_id]
        else:
            real_data = np.reshape(np.asarray(data, dtype=np.float32), [-1])
        if name_id == 0:
            indices = random.sample(range(len(real_data)), len(real_data))
        Data.append(np.split(real_data[indices], len(real_data)/batch_size))
    # print(time.time() - start_time)
    return Data, len(real_data)/batch_size


def read_meta_file_insert_zeros(meta_file_folder, max_steps, batch_size):
    names = ['laser', 'action', 'cmd', 'cmd_next', 'cmd_skip', 'obj_pose', 'status', 'length']
    dims = [LASER_DIM, 2, 1, 1, 1, 2, 1, 1]
    dtypes = [np.int32, np.int32, np.int8, np.int8, np.int8, np.int32, np.int8, np.int64]
    scales = [SCALE, SCALE, 1, 1, 1, SCALE, 1, 1]
    start_time = time.time()
    Data = []
    for name_id, name in enumerate(names): 
        file_name = os.path.join(meta_file_folder, name+'.csv')
        file = open(file_name, 'r')
        data = np.fromfile(file, dtype=dtypes[name_id])
        file.close()
        if name is not 'length':
            real_data = np.reshape(np.asarray(data, dtype=np.float32), [-1, max_steps, dims[name_id]])/scales[name_id]
        else:
            real_data = np.reshape(np.asarray(data, dtype=np.float32), [-1])
        if name_id == 0:
            indices = random.sample(range(len(real_data)), len(real_data))
        Data.append(real_data[indices])
    final_Data = []
    status_Data = Data[6]
    cmd_Data = Data[2]
    cmd_next_Data = Data[3]
    zero_len = 5
    row = 0
    new_cmd_Data = np.zeros_like(cmd_Data)
    new_cmd_next_Data = np.zeros_like(cmd_Data)
    new_cmd_skip_Data = np.zeros_like(cmd_Data)
    for status_data, cmd_data, cmd_next_data in zip(status_Data, cmd_Data, cmd_next_Data):
        non_zeros_indices = np.nonzero(status_data)[0]
        new_cmd_data = copy.deepcopy(cmd_data)
        new_cmd_next_data = np.zeros_like(cmd_data)
        new_cmd_skip_data = copy.deepcopy(cmd_next_data)
        for idx in xrange(len(non_zeros_indices)-1):
            pos = non_zeros_indices[idx]
            new_cmd_data[pos+1:pos+1+zero_len] = 0
            new_cmd_next_data[pos+1:pos+1+zero_len] = new_cmd_data[pos+1+zero_len]
            new_cmd_skip_data[pos+1:pos+1+zero_len] = 0
        # print(np.reshape(new_cmd_data, [-1]))
        # print(np.reshape(new_cmd_next_data, [-1]))
        # print(np.reshape(new_cmd_skip_data, [-1]))
        # assert False
        new_cmd_Data[row, :] = new_cmd_data
        new_cmd_next_Data[row, :] = new_cmd_next_data
        new_cmd_skip_Data[row, :] = new_cmd_skip_data
        row += 1
    final_Data = [np.split(Data[0], len(status_Data)/batch_size),
                  np.split(Data[1], len(status_Data)/batch_size),
                  np.split(new_cmd_Data, len(status_Data)/batch_size),
                  np.split(new_cmd_next_Data, len(status_Data)/batch_size),
                  np.split(new_cmd_skip_Data, len(status_Data)/batch_size),
                  np.split(Data[5], len(status_Data)/batch_size),
                  np.split(Data[6], len(status_Data)/batch_size),
                  np.split(Data[7], len(status_Data)/batch_size)]
    # print(time.time() - start_time)
    return final_Data, len(real_data)/batch_size


def read_old_meta_file(meta_file_folder, max_steps, batch_size):
    names = ['laser', 'action', 'cmd', 'cmd_next', 'obj_pose', 'status', 'length']
    dims = [LASER_DIM, 2, 1, 1, 2, 1, 1]
    dtypes = [np.int32, np.int32, np.int8, np.int8, np.int32, np.int8, np.int64]
    scales = [SCALE, SCALE, 1, 1, SCALE, 1, 1]
    start_time = time.time()
    Data = []
    final_Data = []
    for name_id, name in enumerate(names): 
        file_name = os.path.join(meta_file_folder, name+'.csv')
        file = open(file_name, 'r')
        data = np.fromfile(file, dtype=dtypes[name_id])
        file.close()
        if name is not 'length':
            real_data = np.reshape(np.asarray(data, dtype=np.float32), [-1, max_steps, dims[name_id]])/scales[name_id]
        else:
            real_data = np.reshape(np.asarray(data, dtype=np.float32), [-1])
        if name_id == 0:
            indices = random.sample(range(len(real_data)), len(real_data))
        Data.append(real_data[indices])

    status_Data = Data[5]
    cmd_Data = Data[2]
    for status_data, cmd_data in zip(status_Data, cmd_Data):
        cmd_x_status = cmd_data * status_data.astype(np.int8)
        non_zeros_indices = np.nonzero(cmd_x_status)
        cmd_list = cmd_x_status[non_zeros_indices].tolist() + [0, 0]
        cmd_idx_seq = np.zeros_like(status_data)
        for idx in xrange(1, len(non_zeros_indices[0])):
            cmd_idx_seq[non_zeros_indices[0][idx-1]+1:non_zeros_indices[0][idx]+1] = idx
        cmd_next_idx_seq = cmd_idx_seq + 1
        cmd_skip_idx_seq = cmd_idx_seq + 2
        cmd_list = np.asarray(cmd_list)
        cmd_skip_data = cmd_list[cmd_skip_idx_seq]
    for name_id, name in enumerate(names): 
        final_Data.append(np.split(Data[name_id][indices], len(Data[name_id])/batch_size))
    final_Data.append(np.split(cmd_skip_data), len(cmd_skip_data)/batch_size)
    # print(time.time() - start_time)
    return final_Data, len(real_data)/batch_size

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
    # meta_file_path = os.path.join(CWD[:-7], 'lan_nav_data/room_new/training_meta')
    source_list = [os.path.join(CWD[:-7], 'lan_nav_data/room_new/linhai-Intel-Z370_robot1')]
    # source_list = [os.path.join(CWD[:-7], 'lan_nav_data/room_new/validation')]
    meta_file_path = os.path.join(CWD[:-7], 'lan_nav_data/room_new/validation_meta')

    # Data, _ = read_meta_file(os.path.join(meta_file_path, '0'), 700, 64)
    # Data_old, _ = read_meta_file_old(os.path.join(meta_file_path_old, '0'), 700, 64)

    # # for a, b in zip(Data, Data_old):
    # #     err = a-b
    # #     print(np.sum(np.fabs(err)))
    # assert(False)

    if not os.path.exists(meta_file_path): 
        os.makedirs(meta_file_path)
    write_multiple_meta_files(source_list, meta_file_path, 1024)

    data, lenth = read_meta_file(os.path.join(meta_file_path, '0'), 700, 256)
    num = 100
    print(data[0][0][0][:num])
    print(data[1][0][0][:num])
    print(np.reshape(data[2][0][0][:num], [-1]))
    print(np.reshape(data[3][0][0][:num], [-1]))
    print(np.reshape(data[4][0][0][:num], [-1]))
    print(data[5][0][0][:num])
    print(np.reshape(data[6][0][0][:num], [-1]))
    print(data[7][0][0][:num])
