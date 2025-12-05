#!/bin/bash
# Run ORB-SLAM3 RGBD under GDB with automatic debugging

cd ~/Projects/teleoperation_spot/system_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# Add ORB_SLAM3 library to path
export LD_LIBRARY_PATH=~/Projects/teleoperation_spot/cpp/ORB_SLAM3/lib:$LD_LIBRARY_PATH

export VOCAB_PATH=~/Projects/teleoperation_spot/cpp/ORB_SLAM3/Vocabulary/ORBvoc.txt
export CONFIG_PATH=~/Projects/teleoperation_spot/system_ws/src/orbslam3_ros2/config/xtionA_rgbd.yaml

echo "Starting ORB-SLAM3 RGBD node under GDB..."
echo "The node will run and crash. GDB will capture the backtrace."
echo ""
echo "Make sure a bag is playing in another terminal:"
echo "  ros2 bag play xtion_20251203_154457.bag/ --loop"
echo ""
read -p "Press Enter when bag is playing..."

gdb -batch -x gdb_commands.txt --args install/orbslam3/lib/orbslam3/rgbd \
  $VOCAB_PATH \
  $CONFIG_PATH \
  --ros-args \
  -r /camera/color/image_raw:=/camA/rgb/image_raw \
  -r /camera/aligned_depth_to_color/image_raw:=/camA/depth/image \
  -r /camera/color/camera_info:=/camA/rgb/camera_info

