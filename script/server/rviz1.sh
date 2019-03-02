#!/bin/bash

export ROS_IP=192.168.0.100
export ROS_MASTER_URI=http://192.168.0.118:11312
export GAZEBO_MASTER_URI=http://192.168.0.118:11352

roslaunch launch/rviz.launch
