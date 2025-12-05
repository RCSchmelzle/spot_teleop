#!/bin/bash
docker run --rm -it \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $HOME/Projects/teleoperation_spot/_multi_orbslam_ws:/root/_multi_orbslam_ws \
  multi_orbslam:noetic
