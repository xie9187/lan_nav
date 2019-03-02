#!/bin/bash

export ROS_IP=192.168.0.100
export ROS_MASTER_URI=http://192.168.0.118:11313
export GAZEBO_MASTER_URI=http://192.168.0.118:11353

roslaunch launch/rviz.launch
