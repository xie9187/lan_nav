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
import time

CWD = os.getcwd()
cwd_idx = CWD.rfind('/')

laser_dim = 666

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


def read_raw_files(path_num_list):
    seq = {}
    seq_len = []
    cmd_len = []
    names = ['laser', 'action', 'cmd', 'cmd_next', 'obj_pose', 'status']
    dim = {'laser':laser_dim, 'action':2, 'cmd':1, 'cmd_next':1, 'obj_pose':2, 'status':1}    

    for name in names:
        seq[name] = []
        for path_num in path_num_list:
            file_name = path_num+'_'+name+'.csv'
            data = read_file(file_name)
            reshape_data = np.reshape(data, [-1, dim[name]])
            seq[name].append(reshape_data)
            if name is 'cmd':
                seq_len.append(len(reshape_data))
    total_lines = len(seq[name])
    seq_trans = [[] for line in xrange(total_lines)]
    for idx in xrange(len(names)):
        name = names[idx]
        for line in xrange(total_lines):
            seq_trans[line].append(seq[name][line])

    data = (seq_trans, seq_len)

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

    new_size = [laser_dim, 2, 1, 1, 2, 1]

    batch_inputs = [[] for i in xrange(len(new_size))]
    batch_lengths = []

    (seq_set, seq_len_set) = data


    for idx in idx_chunk:
    
        if idx == -1:  # batch padding
            segments = [np.zeros([max_steps, new_size[i]]) for i in xrange(len(new_size))]
            batch_lengths.append(0)
        else:
            curr_segment = seq_set[idx]
            curr_seq_len = seq_len_set[idx]
            if curr_seq_len > max_steps:    # cut from max_steps
                segments = [curr_segment[n][:max_steps] for n in xrange(len(new_size))]
            else:                       # padding if curr_seq_len < max_steps
                segments = [np.pad(curr_segment[n][:curr_seq_len], ((0, max_steps-curr_seq_len), (0,0)), 'constant') for n in xrange(len(new_size))]            
            
            batch_lengths.append(curr_seq_len)

        for i in xrange(len(new_size)): # reshape batch inputs
            batch_inputs[i].append(segments[i])   # b, l, 

    batch_inputs = [np.array(batch) for batch in batch_inputs]
    batch_lengths = np.array(batch_lengths)

    return (batch_inputs, batch_lengths)    


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

        batch_inputs, batch_lengths = fetch_batch(train_data, max_steps, max_cmds, batch)

        batch_laser, batch_action, batch_cmd,  batch_cmd_next, batch_obj_pose, batch_status = batch_inputs

        # print(np.shape(batch_laser), 
        #       np.shape(batch_action), 
        #       np.shape(batch_cmd),
        #       np.shape(batch_cmd_next),
        #       np.shape(batch_obj_pose),
        #       np.shape(batch_status))
        # assert(False)

        feature_list = {'train/laser': _float_feature(np.reshape(batch_laser, (-1) )),
                        'train/action': _float_feature(np.reshape(batch_action, (-1))),
                        'train/cmd': _int64_feature(np.reshape(batch_cmd, (-1)).astype(int)),
                        'train/cmd_next': _int64_feature(np.reshape(batch_cmd_next, (-1)).astype(int)),
                        'train/obj_pose': _float_feature(np.reshape(batch_obj_pose, (-1))),
                        'train/status': _int64_feature(np.reshape(batch_status, (-1)).astype(int)),
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
                        'train/obj_pose': _float_feature(np.zeros((max_steps*2))),
                        'train/status': _int64_feature(np.zeros((max_steps)).astype(int)),
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
                   'train/obj_pose':  tf.FixedLenFeature([max_steps*2], tf.float32),
                   'train/status':  tf.FixedLenFeature([max_steps], tf.int64),
                   'train/lengths': tf.FixedLenFeature([], tf.int64),
                   }
        features = tf.parse_single_example(serialized_example, features=feature)

        return (features['train/laser'], 
                features['train/action'], 
                features['train/cmd'], 
                features['train/cmd_next'], 
                features['train/obj_pose'],  
                features['train/status'], 
                features['train/lengths']
                )


    file_list = os.listdir(tfrecord_file_path)
    print('reading tfrecord file lists:', file_list)
    tfrecord_file_list = []
    for file_name in file_list:
        tfrecord_file_list.append(os.path.join(tfrecord_file_path, file_name))
    files = tf.data.Dataset.list_files(tfrecord_file_list)
    # dataset = files.interleave(tf.data.TFRecordDataset)
    dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=8))
    # dataset = dataset.map(parse_fn)  # Parse the record into tensors.
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1000, 100))
    dataset = dataset.apply(tf.contrib.data.map_and_batch(parse_fn, batch_size, num_parallel_calls=8))
    dataset = dataset.prefetch(buffer_size=1000)
    return dataset


def test_read(tfrecord_path):
    dataset = tfrecord_dataset(tfrecord_path, 700, laser_dim, 64)

    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(5, 1))

    # dataset = dataset.repeat(1)  # Repeat the input indefinitely.
    # dataset = dataset.batch(64)

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
            start_time = time.time()
            next_inp_batch = sess.run(next_inp)

            print(n)
            print(time.time() - start_time)

            n+=1

        except tf.errors.OutOfRangeError:
            break

def write_multiple_tfrecords(source_list, tfrecord_path, max_file_num=1024):
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
    for x in xrange(len(file_path_number_list)/max_file_num+1):
        source_path_number_list = file_path_number_list[max_file_num*x:
                                                        min(max_file_num*(x+1), len(file_path_number_list))]                                          
        tfrecord_filename = os.path.join(tfrecord_path, str(x)+'.tfrecords')
        print('Creating '+tfrecord_filename)
        write_tfrecord(source_path_number_list, tfrecord_filename, 700, 20, 256)

        used_time = time.time() - start_time
        print('Created {}/{} tr files | time used (min): {:.1f}'.format(x, len(file_path_number_list)/max_file_num+1, used_time/60.))

if __name__ == '__main__':
    # source_filename = os.path.join(CWD[:-7], 'lan_nav_data/room/meta_file/validation_data.tar.gz')
    # tfrecord_filename = os.path.join(CWD[:-7], 'lan_nav_data/room/meta_file/train.tfrecords')  # address to save the TFRecords file
    # write_tfrecord(source_filename, tfrecord_filename, 700, 20, 64)

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
    #                os.path.join(CWD[:-7], 'lan_nav_data/room_new/ymiao-System-Product-Name_robot2'),
    #                ]
    # tfrecord_path = os.path.join(CWD[:-7], 'lan_nav_data/room_new/training_tr')

    source_list = [os.path.join(CWD[:-7], 'lan_nav_data/room_new/validation')]
    tfrecord_path = os.path.join(CWD[:-7], 'lan_nav_data/room_new/validation_tr')

    test_read(tfrecord_path)
    assert(False)

    if not os.path.exists(tfrecord_path): 
        os.makedirs(tfrecord_path)
    write_multiple_tfrecords(source_list, tfrecord_path, 10240)




