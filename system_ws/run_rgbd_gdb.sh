#!/bin/bash
# Script to run ORB-SLAM3 RGBD node under GDB

cd ~/Projects/teleoperation_spot/system_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

export VOCAB_PATH=~/Projects/teleoperation_spot/cpp/ORB_SLAM3/Vocabulary/ORBvoc.txt
export CONFIG_PATH=~/Projects/teleoperation_spot/system_ws/src/orbslam3_ros2/config/xtionA_rgbd.yaml

gdb --args install/orbslam3/lib/orbslam3/rgbd \
  $VOCAB_PATH \
  $CONFIG_PATH \
  --ros-args \
  -r /camera/color/image_raw:=/camA/rgb/image_raw \
  -r /camera/aligned_depth_to_color/image_raw:=/camA/depth/image \
  -r /camera/color/camera_info:=/camA/rgb/camera_info