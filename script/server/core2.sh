#!/bin/bash

export ROS_IP=192.168.0.118
export ROS_MASTER_URI=http://192.168.0.118:11313

roscore -p 11313
