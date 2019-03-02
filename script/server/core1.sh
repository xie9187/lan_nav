#!/bin/bash

export ROS_IP=192.168.0.118
export ROS_MASTER_URI=http://192.168.0.118:11312

roscore -p 11312
