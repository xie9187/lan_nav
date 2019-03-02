#!/bin/bash

export ROS_IP=192.168.0.100
export ROS_MASTER_URI=http://192.168.0.100:11314

roscore -p 11314
