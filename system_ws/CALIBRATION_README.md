# Multi-Camera Extrinsic Calibration Tool

Complete end-to-end calibration tool for rigid camera extrinsics using ORB-SLAM3 trajectories and hand-eye calibration.

## Quick Start

```bash
cd ~/Projects/teleoperation_spot/system_ws
./extrinsic_camera_calibration.sh
```

## Features

- **Live Recording Mode**: Automatically bring up cameras, record, and calibrate
- **Existing Bag Processing**: Process previously recorded calibration bags
- **Full Pipeline**: ORB-SLAM3 → Trajectory Extraction → Time Sync → Hand-Eye → Extrinsics
- **Interactive GUI**: Name cameras visually from bag frames
- **Automatic Configuration**: Generate ORB-SLAM3 configs per camera
- **Robust Calibration**: RANSAC-based hand-eye with motion filtering

## How It Works

### Method: Hand-Eye Calibration from SLAM Trajectories

1. **Record Multi-Camera Data**: All cameras observe the same scene simultaneously
2. **SLAM Processing**: Run ORB-SLAM3 on each camera independently to get trajectories
3. **Time Synchronization**: Find overlapping timestamps between trajectories
4. **Relative Motion Extraction**: Compute camera motions between synchronized frames
5. **Hand-Eye Calibration**: Solve `AX = XB` where:
   - `A` = relative motions from camera A
   - `B` = relative motions from camera B
   - `X` = rigid transform from A to B (what we solve for)
6. **RANSAC Refinement**: Robustly handle outliers from tracking errors

### Why This Works

Since all cameras are rigidly attached and observing the same scene:
- Camera motions are related by a fixed rigid transform
- SLAM gives us egomotion in each camera's frame
- Hand-eye calibration recovers the fixed transform between frames

## Modes

### [0] Live Recording (Recommended)

Best for initial calibration or when you have physical cameras available.

**Workflow:**
1. Select option `[0]`
2. Cameras automatically launch
3. Follow on-screen recording instructions
4. Move rig smoothly for 30-60 seconds (rotations + translations)
5. Ctrl+C to stop recording
6. Automatic processing and calibration

**Tips for Good Calibration:**
- Point at textured surfaces (posters, patterns, objects)
- Avoid blank walls and featureless areas
- Include both rotations and translations
- Move slowly to avoid motion blur
- Record in well-lit environment

### [1-N] Existing Bags

Process previously recorded bags.

**Workflow:**
1. Select bag number from list
2. GUI shows camera frames - assign names
3. ORB-SLAM3 processes each camera
4. Trajectories extracted
5. Hand-eye calibration computes extrinsics

## Output

### Extrinsics Files

Saved to: `datasets/xtion_calibration_test/bags/<timestamp>/orbslam_config/extrinsics/`

Example: `camA_to_camB.yaml`
```yaml
transform:
  translation: [0.1234, -0.0567, 0.0890]  # meters [x, y, z]
  rotation: [0.9999, -0.0012, 0.0034,     # 3x3 rotation matrix (row-major)
             0.0012,  0.9999, 0.0056,
            -0.0034, -0.0056, 0.9999]
quaternion: [0.0012, 0.0028, -0.0006, 0.9999]  # [x, y, z, w]
calibrated: true
reference_camera: camA
target_camera: camB
```

### Trajectories

Saved to: `datasets/xtion_calibration_test/bags/<timestamp>/trajectories/`

- `<camera>_KeyFrameTrajectory.txt`: TUM format (timestamp tx ty tz qx qy qz qw)
- One trajectory per camera

## Algorithm Details

### Time Synchronization

- Samples trajectory at fixed rate (30 Hz default)
- Finds common time window across all cameras
- Interpolates poses using SLERP for rotations

### Motion Filtering

Rejects motion pairs that are:
- Too small (< 0.3° rotation, < 2mm translation)
- Too large (> 15° rotation, > 5cm translation per frame)
- Inconsistent between cameras (5x ratio threshold)

### Hand-Eye RANSAC

- Sample size: 8 motion pairs per hypothesis
- Iterations: 500
- Inlier thresholds: 0.5° rotation, 5mm translation
- Minimum inlier ratio: 30%

## Troubleshooting

### No trajectory overlap

**Symptom:** `ValueError: No overlapping time range`

**Causes:**
- ORB-SLAM3 tracking failed on one or both cameras
- Map resets created trajectories at different times
- Insufficient keyframes

**Solutions:**
1. Record new bag with better lighting and visual features
2. Move slower to maintain tracking
3. Avoid featureless areas (blank walls)
4. Check ORB-SLAM3 logs for tracking failures

### ORB-SLAM3 tracking fails

**Symptom:** Only 1-2 keyframes in trajectory, or "Fail to track local map!" in logs

**Solutions:**
1. **Improve environment**: Add visual texture (posters, patterns)
2. **Better lighting**: Ensure bright, even illumination
3. **Slower motion**: Give SLAM time to initialize and track
4. **Check calibration**: Verify camera parameters in YAML files
5. **Depth factor**: Ensure `DepthMapFactor: 1.0` for depth in meters

### Live recording not available

**Symptom:** Option `[0]` not shown

**Cause:** xtion_bringup not configured or xtion_mapping.yaml not found

**Solution:** Configure xtion_bringup first (see xtion setup docs)

## Dependencies

```bash
# Python packages
pip install PyQt5 pyyaml opencv-python scipy numpy

# ROS 2 packages
sudo apt install ros-jazzy-rosbag2-py

# ORB-SLAM3
# (Must be installed at ~/Projects/teleoperation_spot/cpp/ORB_SLAM3)
```

## Files

- `extrinsic_camera_calibration.sh` - Main tool
- `orbslam_bag_processor.sh` - Legacy tool (existing bags only)
- `scripts/bag_camera_mapper_gui.py` - Camera naming GUI
- `scripts/generate_orbslam_configs.py` - Config generator
- `scripts/calibrate_cameras_from_trajectories.py` - Calibration algorithm
- `scripts/trajectory_utils.py` - Core algorithms (hand-eye, sync, RANSAC)

## Theory References

Hand-eye calibration solves: `AX = XB`

Where:
- `A_i` = motion of camera A between frames i and i+1
- `B_i` = motion of camera B between frames i and i+1
- `X` = fixed rigid transform from A to B

Classic papers:
- Tsai & Lenz (1989): "A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/Eye Calibration"
- Strobl & Hirzinger (2006): "Optimal Hand-Eye Calibration"

Our implementation:
- Separates rotation and translation
- Uses SVD for rotation estimation
- Least squares for translation
- RANSAC for outlier rejection
- Motion filtering for data quality

## Support

For issues or questions, see:
- `docs/setup/10_orbslam3_w_xtion_rosbags_generalizable` - Full documentation
- GitHub issues: https://github.com/anthropics/claude-code/issues