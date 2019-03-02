#!/bin/bash

export ROS_IP=192.168.0.117
export ROS_MASTER_URI=http://192.168.0.117:11312
export GAZEBO_MASTER_URI=http://192.168.0.117:11352

DISPLAY=:8 vglrun -d :7.0 roslaunch launch/gazebo_room_simple.launch
