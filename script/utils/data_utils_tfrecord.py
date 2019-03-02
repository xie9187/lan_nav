from __future__ import print_function

import numpy as np
import csv
import random
import os
import copy
import tarfile
import multiprocessing
import scipy.misc
import tensorflow as tf

CWD = os.getcwd()
cwd_idx = CWD.rfind('/')

laser_dim = 666

def read_file(file_name):
    file = open(file_name, 'r')
    file_reader = csv.reader(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
    curr_seq = []
    for row in file_reader:
        if type(row[0]) == str:
            new_row = row[0][1:-1].split()
            row = [float(r) for r in new_row]
        curr_seq.append(row)
    file.close()
    return curr_seq


def read_raw_files(path_num_list):
    seq = {}
    cmd = []
    seq_len = []
    cmd_len = []
    names = ['laser', 'action', 'cmd', 'goal', 'goal_pose']
    all_names = ['laser', 'action', 'cmd', 'goal', 'goal_pose', 'cmd_list']
    dim = {'laser':laser_dim, 'action':2, 'cmd':1, 'goal':2, 'goal_pose':2}    

    for name in all_names:
        seq[name] = []
        if name in names:
            for path_num in path_num_list:
                file_name = path_num+'_'+name+'.csv'
                data = read_file(file_name)
                reshape_data = np.reshape(data, [-1, dim[name]])
                seq[name].append(reshape_data)
                if name is 'laser':
                    seq_len.append(len(reshape_data))
        elif name == 'cmd_list':
            for path_num in path_num_list:
                file_name = path_num+'_'+name+'.csv'
                data = read_file(file_name)
                reshape_data = np.reshape(data, [-1])
                cmd_data = np.array(reshape_data).astype(float)
                cmd.append(cmd_data)
                cmd_len.append(len(cmd_data)) 

    total_lines = len(cmd)
    seq_trans = [[] for line in xrange(total_lines)]
    for idx in xrange(len(names)):
        name = names[idx]
        for line in xrange(total_lines):
            seq_trans[line].append(seq[name][line])   

    new_seq = []
    new_cmd = []
    new_seq_len = []
    new_cmd_len = []
    for segment, cmd_item , seq_len_item, cmd_len_item in zip(seq_trans, cmd, seq_len, cmd_len):
        cmd_curr = segment[2]
        cmd_list = np.append(cmd_item, [0, 0])

        valid = True
        cmd_next = []
        cmd_skip = []
        idx = 0
        flag = 0
        for c in cmd_curr:                      # [031250]  03333000111100022  -> 10001001000100101
            if int(c[0]) == 0 or int(c[0]) ==5: #           30000111000022200  
                if flag == 0:                   #           01111000222200000
                    idx += 1
                    flag = 1
                #print('cmd: {0}  idx: {1}   flag:  {2}'.format(c[0], idx, flag))
                if idx < len(cmd_list)-1:
                    cmd_next.append([cmd_list[idx]])
                    cmd_skip.append([0])
                else:
                    valid = False
                    break
            else:
                flag = 0
                cmd_next.append([0])
                cmd_skip.append([cmd_list[idx+1]])

        if valid:
            new_seq.append([])
            new_seq[len(new_seq)-1].append(segment[0])
            new_seq[len(new_seq)-1].append(segment[1])
            new_seq[len(new_seq)-1].append(segment[2])
            new_seq[len(new_seq)-1].append(cmd_next)
            new_seq[len(new_seq)-1].append(cmd_skip)
            new_seq[len(new_seq)-1].append(segment[3])
            new_seq[len(new_seq)-1].append(segment[4])
            
            new_cmd.append(cmd_item) 
            new_seq_len.append(seq_len_item) 
            new_cmd_len.append(cmd_len_item)

            # print(segment[2])
            # print(cmd_next)
            # print(cmd_skip)
            # assert(False)

    data = (new_seq, new_cmd, new_seq_len, new_cmd_len)

    return data 

def read_tar_data(source_path):
    seq = {}
    cmd = []
    seq_len = []
    cmd_len = []
    names = ['laser', 'action', 'cmd', 'goal', 'goal_pose']
    dim = {'laser':laser_dim, 'action':2, 'cmd':1, 'goal':2, 'goal_pose':2}

    tar = tarfile.open(source_path)
    for idx, member in enumerate(tar.getmembers()):
        name = member.name.split('.')[0]
        print(name)
        f = tar.extractfile(member)
        seq[name] = []
        if  name in names:
            while True:
                row = f.readline()
                if not row:
                    break
                reshape_row = np.reshape(row.split(','), [-1, dim[name]]).astype(float)
                seq[name].append(reshape_row)
                if idx == 0:
                    seq_len.append(len(reshape_row))
        elif name == 'cmd_list':
            while True:
                row = f.readline()
                if not row:
                    break
                cmd_line = np.array(row.strip().split(',')).astype(float)
                cmd.append(cmd_line)
                cmd_len.append(len(cmd_line)) 


    total_lines = len(cmd)
    seq_trans = [[] for line in xrange(total_lines)]
    for idx in xrange(len(names)):
        name = names[idx]
        for line in xrange(total_lines):
            seq_trans[line].append(seq[name][line])      

    new_seq = []
    new_cmd = []
    new_seq_len = []
    new_cmd_len = []
    for segment, cmd_item , seq_len_item, cmd_len_item in zip(seq_trans, cmd, seq_len, cmd_len):
        cmd_curr = segment[2]
        cmd_list = np.append(cmd_item, [0, 0])

        valid = True
        cmd_next = []
        cmd_skip = []
        idx = 0
        flag = 0
        for c in cmd_curr:                      # [031250]  03333000111100022  -> 10001001000100101
            if int(c[0]) == 0 or int(c[0]) ==5: #           30000111000022200  
                if flag == 0:                   #           01111000222200000
                    idx += 1
                    flag = 1
                #print('cmd: {0}  idx: {1}   flag:  {2}'.format(c[0], idx, flag))
                if idx < len(cmd_list)-1:
                    cmd_next.append([cmd_list[idx]])
                    cmd_skip.append([0])
                else:
                    valid = False
                    break
            else:
                flag = 0
                cmd_next.append([0])
                cmd_skip.append([cmd_list[idx+1]])

        if valid:
            new_seq.append([])
            new_seq[len(new_seq)-1].append(segment[0])
            new_seq[len(new_seq)-1].append(segment[1])
            new_seq[len(new_seq)-1].append(segment[2])
            new_seq[len(new_seq)-1].append(cmd_next)
            new_seq[len(new_seq)-1].append(cmd_skip)
            new_seq[len(new_seq)-1].append(segment[3])
            new_seq[len(new_seq)-1].append(segment[4])
            
            new_cmd.append(cmd_item) 
            new_seq_len.append(seq_len_item) 
            new_cmd_len.append(cmd_len_item)

            # print(segment[2])
            # print(cmd_next)
            # print(cmd_skip)
            # assert(False)



    data = (new_seq, new_cmd, new_seq_len, new_cmd_len)

    return data 


def create_batches(data, batch_size, shuffle=True):
    # create a list of (doc, sent)
    idx_list = range(len(data[0]))

    if shuffle:
        random.shuffle(idx_list)

    # create chunk batches
    chunk_batches = []
    for i in xrange(len(idx_list) / batch_size):
        start = i * batch_size
        end = (i + 1) * batch_size
        chunk_batches.append(idx_list[start:end])

    # the batch of which the length is less than batch_size
    rest = len(idx_list) % batch_size
    if rest > 0:
        chunk_batches.append(idx_list[-rest:] + [-1] * (batch_size - rest))  # -1 as padding

    return chunk_batches



def fetch_batch(data, max_steps, max_cmds, idx_chunk):
    """fetch a batch of data by given idx_chunk."""

    batch_size = len(idx_chunk)
    new_size = [laser_dim, 2, 1, 1, 1, 2, 2]

    batch_inputs = [[] for i in xrange(len(new_size))]
    batch_cmds = []
    batch_lengths = []
    batch_lengths_cmd_list = []

    (seq_set, cmd_set, seq_len_set, cmd_len_set) = data


    for idx in idx_chunk:
    
        if idx == -1:  # batch padding
            segments = [np.zeros([max_steps, new_size[i]]) for i in xrange(len(new_size))]
            cmds = np.zeros([max_cmds])
            batch_lengths.append(0)
            batch_lengths_cmd_list.append(0)    #??
        else:
            curr_segment = seq_set[idx]
            curr_cmd = cmd_set[idx]
            curr_seq_len = seq_len_set[idx]
            curr_cmd_len = cmd_len_set[idx]
            if curr_seq_len > max_steps:    # cut from max_steps
                segments = [curr_segment[n][:max_steps] for n in xrange(len(new_size))]
            else:                       # padding if curr_seq_len < max_steps
                #segments = [curr_segment[n][:curr_seq_len] + (max_steps-curr_seq_len)*[new_size[n]*[0]] for n in xrange(len(new_size))] 
                segments = [np.pad(curr_segment[n][:curr_seq_len], ((0, max_steps-curr_seq_len), (0,0)), 'constant') for n in xrange(len(new_size))]            
            
            cmds = np.append(curr_cmd, (max_cmds-curr_cmd_len)*[0])

            assert(max_cmds > curr_cmd_len)
            batch_lengths.append(curr_seq_len)
            batch_lengths_cmd_list.append(curr_cmd_len) 


        for i in xrange(len(new_size)): # reshape batch inputs
            batch_inputs[i].append(segments[i])   # b, l, 

        batch_cmds.append(cmds)


    batch_inputs = [np.array(batch) for batch in batch_inputs]
    batch_cmds = np.array(batch_cmds)
    batch_lengths = np.array(batch_lengths)
    batch_lengths_cmd_list = np.array(batch_lengths_cmd_list)

    return (batch_inputs, 
            batch_cmds,
            batch_lengths,
            batch_lengths_cmd_list)    


def _int64_feature(value):
    value = value if type(value) == list else value.tolist()
    #print(len(value))
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _bytes_feature(value):
    value = value if type(value) == list else value.tolist()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
def _float_feature(value):
    value = value if type(value) == list else value.tolist()
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def write_tfrecord(source_path_number_list, tfrecord_filename, max_steps, max_cmds, batch_size):

    writer = tf.python_io.TFRecordWriter(tfrecord_filename)

    # train_data = read_tar_data(source_filename)
    train_data = read_raw_files(source_path_number_list)
    train_batches = create_batches(train_data, 1, shuffle=False)

    for batch in train_batches:

        batch_inputs, batch_cmd_list, batch_lengths, batch_lengths_cmd_list = fetch_batch(train_data, max_steps, max_cmds, batch)

        batch_laser, batch_action, batch_cmd,  batch_cmd_next, batch_cmd_skip, batch_goal, batch_goal_pose = batch_inputs
        # print(np.shape(batch_laser), 
        #       np.shape(batch_action), 
        #       np.shape(batch_cmd),
        #       np.shape(batch_cmd_next),
        #       np.shape(batch_cmd_skip),
        #       np.shape(batch_goal),
        #       np.shape(batch_goal_pose))
        # assert(False)

        feature_list = {'train/laser': _float_feature(np.reshape(batch_laser, (-1) )),
                        'train/action': _float_feature(np.reshape(batch_action, (-1))),
                        'train/cmd': _int64_feature(np.reshape(batch_cmd, (-1)).astype(int)),
                        'train/cmd_next': _int64_feature(np.reshape(batch_cmd_next, (-1)).astype(int)),
                        'train/cmd_skip': _int64_feature(np.reshape(batch_cmd_skip, (-1)).astype(int)),
                        'train/goal': _float_feature(np.reshape(batch_goal, (-1))),
                        'train/goal_pose': _float_feature(np.reshape(batch_goal_pose, (-1))),
                        'train/lengths': _int64_feature(np.reshape(batch_lengths, (-1)).astype(int)),
                        }
        # Create an example protocol buffer

        features = tf.train.Features(feature=feature_list)
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())

    print('Data_size:', len(train_batches))


    # add paddings 

    padding_size = (batch_size - len(train_batches)%batch_size)%batch_size

    print('Padding_size:', padding_size)

    for _ in xrange(padding_size):

        feature_list = {'train/laser': _float_feature(np.zeros((max_steps*laser_dim))),
                        'train/action': _float_feature(np.zeros((max_steps*2))),
                        'train/cmd': _int64_feature(np.zeros((max_steps)).astype(int)),
                        'train/cmd_next': _int64_feature(np.zeros((max_steps)).astype(int)),
                        'train/cmd_skip': _int64_feature(np.zeros((max_steps)).astype(int)),
                        'train/goal': _float_feature(np.zeros((max_steps*2))),
                        'train/goal_pose': _float_feature(np.zeros((max_steps*2))),
                        'train/lengths': _int64_feature(np.zeros((1)).astype(int)),
                        }
        # Create an example protocol buffer
        features = tf.train.Features(feature=feature_list)
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())


    writer.close()

def tfrecord_dataset(tfrecord_file_path, max_steps, n_hidden_laser, batch_size):


    def parse_fn(serialized_example):

        feature = {'train/laser': tf.FixedLenFeature([max_steps*n_hidden_laser], tf.float32),
                   'train/action': tf.FixedLenFeature([max_steps*2], tf.float32),
                   'train/cmd':  tf.FixedLenFeature([max_steps], tf.int64),
                   'train/cmd_next':  tf.FixedLenFeature([max_steps], tf.int64),
                   'train/cmd_skip':  tf.FixedLenFeature([max_steps], tf.int64),
                   'train/goal':  tf.FixedLenFeature([max_steps*2], tf.float32),
                   'train/goal_pose':  tf.FixedLenFeature([max_steps*2], tf.float32),
                   'train/lengths': tf.FixedLenFeature([], tf.int64),
                   }
        features = tf.parse_single_example(serialized_example, features=feature)

        return (features['train/laser'], 
                features['train/action'], 
                features['train/cmd'], 
                features['train/cmd_next'], 
                features['train/cmd_skip'],  
                features['train/goal'], 
                features['train/goal_pose'], 
                features['train/lengths'], 
                )


    file_list = os.listdir(tfrecord_file_path)
    print('reading tfrecord file lists:', file_list)
    tfrecord_file_list = []
    for file_name in file_list:
        tfrecord_file_list.append(os.path.join(tfrecord_file_path, file_name))
    dataset = tf.data.TFRecordDataset(tfrecord_file_list)
    dataset = dataset.map(parse_fn)  # Parse the record into tensors.

    return dataset


def test_read(tfrecord_path):
    dataset = tfrecord_dataset(tfrecord_path, 700, laser_dim, 64)
    dataset = dataset.repeat(1)  # Repeat the input indefinitely.
    dataset = dataset.batch(64)

    iterator = dataset.make_initializable_iterator()
    next_inp = iterator.get_next()

    for x in xrange(len(next_inp)):
        print(np.shape(next_inp[x]))

    #laser, action, cmd, cmd_next, cmd_skip, goal, lengths = iterator.get_next()
    
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    sess.run(iterator.initializer)
    n = 0
    while True:
        try:
            next_inp_batch = sess.run(next_inp)

            print(n)

            # print(next_inp_batch)

            n+=1

        except tf.errors.OutOfRangeError:
            break

def write_multiple_tfrecords(source_list, tfrecord_path, max_file_num=1024):
    file_path_number_list = []
    for path in source_list:
        nums = []
        file_list = os.listdir(path)
        print('load from '+path)
        for file_name in file_list:
            if 'laser' in file_name:
                nums.append(int(file_name[:file_name.find('_')]))
        nums = np.sort(nums).tolist()
        for num in nums:
            file_path_number_list.append(path+'/'+str(num))

    for x in xrange(len(file_path_number_list)/max_file_num+1):
        source_path_number_list = file_path_number_list[max_file_num*x:
                                                        min(max_file_num*(x+1), len(file_path_number_list))]                                          
        tfrecord_filename = os.path.join(tfrecord_path, str(x)+'.tfrecords')
        print('Creating '+tfrecord_filename)
        write_tfrecord(source_path_number_list, tfrecord_filename, 700, 20, 64)

if __name__ == '__main__':
    # source_filename = os.path.join(CWD[:-7], 'lan_nav_data/room/meta_file/validation_data.tar.gz')
    # tfrecord_filename = os.path.join(CWD[:-7], 'lan_nav_data/room/meta_file/train.tfrecords')  # address to save the TFRecords file
    # write_tfrecord(source_filename, tfrecord_filename, 700, 20, 64)

    source_list = [os.path.join(CWD[:-7], 'lan_nav_data/room/linhai-Intel-Z370_robot1'),
                   os.path.join(CWD[:-7], 'lan_nav_data/room/linhai-Intel-Z370_robot2')]
    tfrecord_path = os.path.join(CWD[:-7], 'lan_nav_data/room/training_tr')

    # test_read(tfrecord_path)
    # assert(False)

    if not os.path.exists(tfrecord_path): 
        os.makedirs(tfrecord_path)
    write_multiple_tfrecords(source_list, tfrecord_path, 10240)




