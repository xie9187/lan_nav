#!/bin/bash

export ROS_IP=192.168.0.100
export ROS_MASTER_URI=http://192.168.0.100:11312
export GAZEBO_MASTER_URI=http://192.168.0.100:11352

roslaunch launch/gazebo_room_simple.launch
