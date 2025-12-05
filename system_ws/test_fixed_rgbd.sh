#!/bin/bash
# Test the fixed ORB-SLAM3 RGBD node

cd ~/Projects/teleoperation_spot/system_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# Add ORB_SLAM3 library to path
export LD_LIBRARY_PATH=~/Projects/teleoperation_spot/cpp/ORB_SLAM3/lib:$LD_LIBRARY_PATH

export VOCAB_PATH=~/Projects/teleoperation_spot/cpp/ORB_SLAM3/Vocabulary/ORBvoc.txt
export CONFIG_PATH=~/Projects/teleoperation_spot/system_ws/src/orbslam3_ros2/config/xtionA_rgbd.yaml

echo "==============================================="
echo "Testing FIXED ORB-SLAM3 RGBD Node"
echo "==============================================="
echo ""
echo "Make sure a bag is playing in another terminal:"
echo "  ros2 bag play xtion_20251203_154457.bag/ --loop"
echo ""
echo "Press Ctrl+C to stop the node when satisfied it's working."
echo ""
read -p "Press Enter when bag is playing..."

install/orbslam3/lib/orbslam3/rgbd \
  $VOCAB_PATH \
  $CONFIG_PATH \
  --ros-args \
  -r /camera/rgb:=/camA/rgb/image_raw \
  -r /camera/depth:=/camA/depth/image