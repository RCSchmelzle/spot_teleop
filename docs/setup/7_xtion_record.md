
first run tests on NiViewer 2
NiViewer2 -devices


cd ~/Projects/teleoperation_spot/system_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 -m xtion_bringup.xtion_mapper_gui

python3 -m xtion_bringup.gen_xtion_cameras
ros2 launch xtion_bringup multi_xtion.launch.py

rviz2


Run# ==== On HOST (your Ubuntu desktop) ====
cd ~/Projects/teleoperation_spot

# Allow root-in-docker to connect to your X server (safe to repeat)
xhost +local:root

# Start the existing container and attach
docker start -ai orbslam3_dev

# ==== Now INSIDE the container shell (root@... prompt) ====

# Sanity check that DISPLAY is set (usually ":1" or ":0")
echo $DISPLAY

# (Optional: test X)
xclock &

# Run ORB-SLAM3 on the TUM monocular sequence
cd /root/_multi_orbslam_ws/src/ORB_SLAM3

./Examples/Monocular/mono_tum \
  Vocabulary/ORBvoc.txt \
  Examples/Monocular/TUM1.yaml \
  /root/data/tum/rgbd_dataset_freiburg1_xyz



Record based on the generated names 
cd ~/Projects/teleoperation_spot/system_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 bag record -o xtion_test_bag \
  /camA/rgb/image_raw \
  /camA/rgb/camera_info \
  /camA/depth_raw/image \
  /camA/depth_raw/camera_info \
  /camB/rgb/image_raw \
  /camB/rgb/camera_info \
  /camB/depth_raw/image \
  /camB/depth_raw/camera_info \
  /tf \
  /tf_static





STAMP=$(date +%Y%m%d_%H%M%S)

ros2 bag record -o "xtion_${STAMP}.bag" \
  --regex \
  '/.*/rgb/image_raw|/.*/rgb/camera_info|/.*/depth_raw/image|/.*/depth_raw/camera_info|/tf|/tf_static'



#!/usr/bin/env bash
set -e

cd ~/Projects/teleoperation_spot/system_ws

source /opt/ros/jazzy/setup.bash
source install/setup.bash

STAMP=$(date +%Y%m%d_%H%M%S)

ros2 bag record -o "xtion_${STAMP}.bag" \
  --regex \
  '/.*/rgb/image_raw|/.*/rgb/camera_info|/.*/depth_raw/image|/.*/depth_raw/camera_info|/.*/depth/image|/.*/depth/camera_info|/tf|/tf_static'


ros2 bag play xtion_<timestamp>.bag


cd ~/Projects/teleoperation_spot/system_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 bag info xtion_


  Resolution Mode Parameters

  Add these parameters to the parameters=[{...}] section in single_asus_xtion.launch.py:

  parameters=[{
      'device_id': device_id,

      # Color/RGB resolution modes:
      'color_mode': '5',   # Options:
                           # '1' = 320x240 @ 30fps
                           # '5' = 640x480 @ 30fps (DEFAULT)
                           # '9' = 1280x1024 @ 30fps (SXGA - max for Xtion)

      # Depth resolution modes:
      'depth_mode': '5',   # Options:
                           # '1' = 320x240 @ 30fps (QVGA)
                           # '5' = 640x480 @ 30fps (VGA - DEFAULT, max for depth)
                           # Note: Depth is limited to 640x480 by hardware

      # IR resolution modes (if using IR instead of RGB):
      'ir_mode': '5',      # Same options as depth_mode
  }],

  Key Points

  - Default: Mode 5 (640x480 @ 30fps) for all streams
  - Max RGB: Mode 9 (1280x1024 @ 30fps)
  - Max Depth: Mode 5 (640x480 @ 30fps) - hardware limitation
  - Lower res options: Mode 1 (320x240) for faster processing/bandwidth

  Example Configurations

  High quality RGB with standard depth:
  'color_mode': '9',   # 1280x1024
  'depth_mode': '5',   # 640x480

  Balanced (current default):
  'color_mode': '5',   # 640x480
  'depth_mode': '5',   # 640x480

  Low bandwidth/fast processing:
  'color_mode': '1',   # 320x240
  'depth_mode': '1',   # 320x240



