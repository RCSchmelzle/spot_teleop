#!/usr/bin/env python3
"""
Multi-SLAM Stochastic Calibration

Runs ORB-SLAM3 N times to handle SLAM non-determinism, then calibrates each trajectory.
Saves all results to stochastic_extrinsics/ with scale normalization.
"""

import sys
import re
import time
import yaml
import json
import subprocess
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

# Configuration (will be set by user input)
NUM_SLAM_RUNS = 2
STRIDE_VALUES = [1, 2, 3, 5]  # Run calibration with each stride value

def get_user_configuration(headless=False):
    """Get SLAM run count and stride values from user."""
    if headless:
        print(f"\n{'='*70}")
        print("CONFIGURATION (headless mode - using defaults)")
        print(f"{'='*70}\n")
        return 2, [1, 2, 3, 5]

    print(f"\n{'='*70}")
    print("CONFIGURATION")
    print(f"{'='*70}\n")

    # Get number of SLAM runs
    while True:
        try:
            num_runs = input(f"Number of SLAM runs (default: 2): ").strip()
            if num_runs == "":
                num_runs = 2
                break
            num_runs = int(num_runs)
            if num_runs > 0:
                break
            print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")

    # Get stride values
    print("\nStride values (space-separated, e.g., '1 2 3 5')")
    while True:
        stride_input = input(f"Stride values (default: 1 2 3 5): ").strip()
        if stride_input == "":
            stride_values = [1, 2, 3, 5]
            break
        try:
            stride_values = [int(x) for x in stride_input.split()]
            if all(s > 0 for s in stride_values):
                break
            print("All stride values must be positive")
        except ValueError:
            print("Please enter valid numbers separated by spaces")

    return num_runs, stride_values

def normalize_transform_scale(T: np.ndarray, reference_scale: float = 1.0) -> np.ndarray:
    """
    Normalize translation to a reference scale.

    Args:
        T: 4x4 transformation matrix
        reference_scale: Target norm for translation vector (default 1.0)

    Returns:
        Normalized 4x4 transformation matrix
    """
    t = T[:3, 3]
    R = T[:3, :3]

    # Compute current scale
    current_scale = np.linalg.norm(t)

    if current_scale < 1e-6:
        # Translation is essentially zero, return as-is
        return T.copy()

    # Normalize translation
    scale_factor = reference_scale / current_scale
    t_normalized = t * scale_factor

    # Build normalized transform
    T_normalized = np.eye(4)
    T_normalized[:3, :3] = R
    T_normalized[:3, 3] = t_normalized

    return T_normalized


def extract_azimuth_elevation(R: np.ndarray) -> tuple:
    """
    Extract azimuth and elevation angles from rotation matrix.

    Args:
        R: 3x3 rotation matrix

    Returns:
        (azimuth_deg, elevation_deg): Azimuth and elevation in degrees
            - Azimuth: horizontal angle in XY plane (0° = +X, 90° = +Y)
            - Elevation: vertical angle from XY plane (-90° = -Z, +90° = +Z)
    """
    # Use the Z-axis (forward direction) of the rotation matrix
    forward = R[:, 2]  # Third column = Z-axis direction

    # Azimuth: angle in XY plane from X-axis
    azimuth_rad = np.arctan2(forward[1], forward[0])
    azimuth_deg = np.degrees(azimuth_rad)

    # Elevation: angle from XY plane
    xy_norm = np.sqrt(forward[0]**2 + forward[1]**2)
    elevation_rad = np.arctan2(forward[2], xy_norm)
    elevation_deg = np.degrees(elevation_rad)

    return azimuth_deg, elevation_deg


def create_config_with_atlas(config_file: str, atlas_file: Path, load_atlas: bool, work_dir: Path) -> Path:
    """
    Create a temporary config file with Atlas settings injected.

    Args:
        config_file: Path to original config file
        atlas_file: Path to Atlas file for save/load
        load_atlas: Whether to load existing atlas (True) or create new (False)
        work_dir: Directory to store temporary config

    Returns:
        Path to temporary config file with Atlas settings
    """
    # Read original config
    with open(config_file, 'r') as f:
        config_content = f.read()

    # Strip any existing Atlas settings to avoid duplicates
    config_content = re.sub(r'System\.SaveAtlasToFile:.*\n?', '', config_content)
    config_content = re.sub(r'System\.LoadAtlasFromFile:.*\n?', '', config_content)
    config_content = re.sub(r'#.*Atlas Settings.*\n?', '', config_content)
    config_content = re.sub(r'#.*atlas.*\n?', '', config_content, flags=re.IGNORECASE)
    config_content = re.sub(r'#-+\n#-+\n?', '', config_content)  # Clean up dividers

    # NOTE: Atlas saving is disabled due to Boost serialization crash
    # The KeyFrameTrajectory.txt will be saved instead
    # TODO: Debug ORB-SLAM3 Atlas serialization issue
    #
    # atlas_settings = f"""
    # # Atlas settings (shared multi-camera atlas)
    # System.SaveAtlasToFile: {atlas_file}
    # """
    # if load_atlas:
    #     atlas_settings += f'System.LoadAtlasFromFile: {atlas_file}\n'
    # config_content = config_content.rstrip() + '\n' + atlas_settings
    pass  # No atlas settings - just save trajectory

    # Write to temporary file
    temp_config = work_dir / "config_with_atlas.yaml"
    with open(temp_config, 'w') as f:
        f.write(config_content)

    return temp_config


def run_slam_for_camera(bag_path: str, camera_name: str, config_file: str,
                        rgb_topic: str, depth_topic: str,
                        vocab_path: str, output_dir: Path,
                        atlas_dir: Path, is_first_run: bool = True):
    """
    Run ORB-SLAM3 for a single camera with Atlas mode support.

    Args:
        bag_path: Path to the bag file
        camera_name: Name of the camera
        config_file: Path to ORB-SLAM3 config file
        rgb_topic: RGB image topic
        depth_topic: Depth image topic
        vocab_path: Path to ORB vocabulary
        output_dir: Directory for output trajectories
        atlas_dir: Directory for Atlas files (shared map)
        is_first_run: If True, creates new map; if False, loads existing map

    Returns:
        Path to trajectory file if successful, None otherwise
    """

    import os
    import time

    # Create temporary working directory
    work_dir = output_dir / f"{camera_name}_slam_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Create atlas directory if it doesn't exist
    atlas_dir.mkdir(parents=True, exist_ok=True)

    # Compute absolute path to system_ws
    system_ws = Path.home() / "Projects/teleoperation_spot/system_ws"

    # SHARED Atlas file - all cameras contribute to the same Atlas
    atlas_file = atlas_dir / "shared_atlas.osa"

    # Check atlas status and create config with atlas settings injected
    atlas_exists = atlas_file.exists()
    if atlas_exists:
        print(f"    Loading shared Atlas ({atlas_file.stat().st_size} bytes)")
        load_atlas = True
    else:
        print(f"    Creating shared Atlas")
        load_atlas = False

    # Create temp config with Atlas save/load settings in YAML
    temp_config = create_config_with_atlas(config_file, atlas_file, load_atlas, work_dir)

    # Source ROS and run SLAM with Atlas mode (settings now in YAML config)
    slam_cmd = f"""
    cd {work_dir} && \
    source /opt/ros/jazzy/setup.bash && \
    source {system_ws}/install/setup.bash && \
    export LD_LIBRARY_PATH="$HOME/Projects/teleoperation_spot/cpp/ORB_SLAM3/lib:$LD_LIBRARY_PATH" && \
    bash -c '
        # Cleanup function (only used for abnormal exits like Ctrl+C)
        cleanup() {{
            echo "Cleanup triggered..."
            kill -INT $SLAM_PID 2>/dev/null || true
            sleep 3
            kill -9 $SLAM_PID 2>/dev/null || true
            kill -9 $BAG_PID 2>/dev/null || true
            # Kill ROS daemon to avoid stale state
            ros2 daemon stop 2>/dev/null || true
        }}
        trap cleanup INT TERM

        # Start ORB-SLAM3 FIRST with simulated time (so viewer comes up before frames arrive)
        {system_ws}/install/orbslam3/lib/orbslam3/rgbd \
            {vocab_path} \
            {temp_config} \
            --ros-args \
            -p use_sim_time:=true \
            -r /camera/rgb:={rgb_topic} \
            -r /camera/depth:={depth_topic} \
            > slam.log 2>&1 &
        SLAM_PID=$!

        # Wait for SLAM to initialize (load vocabulary, start viewer)
        sleep 8

        # Now start bag playback with clock publishing
        ros2 bag play {bag_path} --clock > /dev/null 2>&1 &
        BAG_PID=$!

        # Wait for bag to finish
        wait $BAG_PID 2>/dev/null || true

        # Give SLAM time to process remaining frames
        echo "Bag finished, waiting for SLAM to process..."
        sleep 5

        # Gracefully stop SLAM (allows it to save trajectory and atlas)
        echo "Sending SIGINT to SLAM..."
        kill -INT $SLAM_PID 2>/dev/null || true

        # Wait for SLAM to finish saving (atlas can take a while)
        echo "Waiting for SLAM to save files..."
        for i in 1 2 3 4 5 6 7 8 9 10; do
            if ! kill -0 $SLAM_PID 2>/dev/null; then
                echo "SLAM exited gracefully"
                break
            fi
            sleep 1
        done

        # If still running after 10 seconds, force kill
        if kill -0 $SLAM_PID 2>/dev/null; then
            echo "SLAM still running, force killing..."
            kill -9 $SLAM_PID 2>/dev/null || true
        fi

        # Clean up ROS
        ros2 daemon stop 2>/dev/null || true

        # Clear trap and exit
        trap - INT TERM
        exit 0
    '
    """

    result = subprocess.run(slam_cmd, shell=True, executable='/bin/bash')

    # Check for trajectory file
    traj_file = work_dir / "KeyFrameTrajectory.txt"
    if traj_file.exists():
        # Copy to output location
        output_traj = output_dir / f"{camera_name}_KeyFrameTrajectory.txt"
        subprocess.run(f"cp {traj_file} {output_traj}", shell=True)

        # Clean up work directory on success
        subprocess.run(f"rm -rf {work_dir}", shell=True)

        return output_traj
    else:
        print(f"  WARNING: No trajectory generated for {camera_name}")
        # Keep work_dir for debugging - check slam.log for errors
        print(f"    Check {work_dir}/slam.log for details")
        return None


def cleanup_ros():
    """Kill any lingering ROS processes to ensure clean state."""
    print("Cleaning up ROS processes...")
    subprocess.run("ros2 daemon stop 2>/dev/null", shell=True, capture_output=True)
    subprocess.run("pkill -9 -f 'ros2 bag' 2>/dev/null", shell=True, capture_output=True)
    subprocess.run("pkill -9 -f 'orbslam3' 2>/dev/null", shell=True, capture_output=True)
    subprocess.run("pkill -9 -f '_ros2_daemon' 2>/dev/null", shell=True, capture_output=True)
    time.sleep(1)
    print("  Done.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Multi-SLAM Stochastic Calibration')
    parser.add_argument('session_dir', help='Path to session directory')
    parser.add_argument('--headless', '-H', action='store_true',
                        help='Run in headless mode (use default settings)')

    args = parser.parse_args()

    # Clean up any lingering ROS processes before starting
    cleanup_ros()

    session_dir = Path(args.session_dir)
    headless = args.headless

    config_dir = session_dir / "orbslam_config"
    trajectories_dir = session_dir / "trajectories"
    stochastic_dir = config_dir / "stochastic_extrinsics"
    atlas_dir = session_dir / "atlas"  # Directory for Atlas files (shared maps)

    # Load camera mapping
    mapping_file = config_dir / "camera_mapping.yaml"
    with open(mapping_file, 'r') as f:
        mapping_data = yaml.safe_load(f)

    # IMPORTANT: camera_names[0] is the reference camera for ALL calibrations
    # All transforms are from camera_names[0] to other cameras
    # This order is preserved across runs since we load from the same YAML file
    camera_names = [cam['name'] for cam in mapping_data['cameras']]

    # Find bag file
    bag_files = list(session_dir.glob("*.bag"))
    if not bag_files:
        print("Error: No bag file found in session directory")
        sys.exit(1)
    bag_path = str(bag_files[0])

    vocab_path = str(Path.home() / "Projects/teleoperation_spot/cpp/ORB_SLAM3/Vocabulary/ORBvoc.txt")

    # Get user configuration
    global NUM_SLAM_RUNS, STRIDE_VALUES
    NUM_SLAM_RUNS, STRIDE_VALUES = get_user_configuration(headless=headless)

    print(f"\n{'='*70}")
    print(f"STOCHASTIC MULTI-SLAM CALIBRATION WITH ATLAS MODE")
    print(f"{'='*70}")
    print(f"\nCameras: {', '.join(camera_names)}")
    print(f"SLAM runs: {NUM_SLAM_RUNS}")
    print(f"Stride values: {STRIDE_VALUES}")
    print(f"Total calibrations: {NUM_SLAM_RUNS} × {len(STRIDE_VALUES)} = {NUM_SLAM_RUNS * len(STRIDE_VALUES)}")
    print(f"Output: {stochastic_dir}")
    print(f"Atlas: {atlas_dir}")
    print(f"\nMode: Map merging (each run contributes to shared Atlas)")
    print()

    # Create/clear stochastic extrinsics directory (clear at start of full calibration session)
    import shutil
    if stochastic_dir.exists():
        shutil.rmtree(stochastic_dir)
    stochastic_dir.mkdir(parents=True, exist_ok=True)

    # Create atlas directory (don't clear - we want to keep building the map)
    atlas_dir.mkdir(parents=True, exist_ok=True)

    all_calibrations = []

    for slam_run in range(NUM_SLAM_RUNS):
        print(f"\n{'='*70}")
        print(f"SLAM RUN {slam_run + 1}/{NUM_SLAM_RUNS}")
        if slam_run == 0:
            print("(Creating initial Atlas)")
        else:
            print("(Loading and merging with existing Atlas)")
        print(f"{'='*70}\n")

        # Run SLAM for each camera with Atlas mode
        print("Running ORB-SLAM3 for all cameras...")
        traj_files = []
        for cam_data in mapping_data['cameras']:
            cam_name = cam_data['name']
            config_file = str(config_dir / f"{cam_name}_rgbd.yaml")
            rgb_topic = cam_data['topics']['rgb']
            depth_topic = cam_data['topics']['depth']

            print(f"  Processing {cam_name}...")

            # Create run-specific trajectory output path
            run_traj_path = trajectories_dir / f"run{slam_run+1:02d}_{cam_name}_KeyFrameTrajectory.txt"

            traj_file = run_slam_for_camera(
                bag_path, cam_name, config_file,
                rgb_topic, depth_topic,
                vocab_path, trajectories_dir,
                atlas_dir, is_first_run=(slam_run == 0)
            )

            # Rename trajectory to include run number
            if traj_file and traj_file.exists():
                traj_file.rename(run_traj_path)
                traj_files.append((cam_name, run_traj_path))
                print(f"    ✓ Trajectory saved: {run_traj_path.name}")
            else:
                print(f"    ✗ SLAM failed")

        if len(traj_files) < 2:
            print(f"  Skipping run {slam_run + 1}: insufficient trajectories")
            continue

        # Temporarily copy run-specific trajectories to expected locations for calibration
        temp_copies = []
        for cam_name, run_traj in traj_files:
            standard_path = trajectories_dir / f"{cam_name}_KeyFrameTrajectory.txt"
            subprocess.run(f"cp {run_traj} {standard_path}", shell=True)
            temp_copies.append(standard_path)

        # Run calibration with multiple stride values
        print(f"\n  Running calibrations with different strides on SLAM run {slam_run + 1}...")

        # Run calibration (use system_ws path, disable matplotlib display)
        system_ws = Path.home() / "Projects/teleoperation_spot/system_ws"
        calib_script = system_ws / "scripts/calibrate_cameras_from_trajectories.py"

        # Set MPLBACKEND to non-interactive to suppress plt.show()
        import os
        env = os.environ.copy()
        env['MPLBACKEND'] = 'Agg'  # Non-interactive backend

        # Run calibration for each stride value
        for stride in STRIDE_VALUES:
            print(f"\n    Stride {stride}...")

            # Clear extrinsics directory
            extrinsics_dir = config_dir / "extrinsics"
            if extrinsics_dir.exists():
                import shutil
                shutil.rmtree(extrinsics_dir)

            calib_result = subprocess.run(
                [sys.executable, str(calib_script), str(session_dir), str(stride)],
                capture_output=True,
                text=True,
                env=env
            )

            if calib_result.returncode != 0:
                print(f"      ✗ Calibration failed for stride {stride}")
                print(f"      STDOUT: {calib_result.stdout}")
                print(f"      STDERR: {calib_result.stderr}")
                continue

            # Load calibration results
            json_file = extrinsics_dir / "camera_poses.json"
            yaml_file = extrinsics_dir / f"{camera_names[0]}_to_{camera_names[1]}.yaml"

            if not json_file.exists():
                print(f"      ✗ Calibration failed: {json_file} not found")
                continue

            if not yaml_file.exists():
                print(f"      ✗ Calibration failed: {yaml_file} not found")
                continue

            with open(json_file, 'r') as f:
                json_data = json.load(f)
            with open(yaml_file, 'r') as f:
                yaml_data = yaml.safe_load(f)

            # Extract transform and normalize scale
            # NOTE: json_data[camera_names[1]] contains the transform from camera_names[0] to camera_names[1]
            # This is consistent across all SLAM runs since we always use camera_names[0] as reference
            if camera_names[1] in json_data:
                    T = np.eye(4)
                    T[:3, :3] = np.array(json_data[camera_names[1]]['rotation_matrix'])
                    T[:3, 3] = np.array(json_data[camera_names[1]]['translation_vector'])

                    # Normalize to unit translation norm
                    T_normalized = normalize_transform_scale(T, reference_scale=1.0)

                    # Extract azimuth and elevation from rotation
                    azimuth_deg, elevation_deg = extract_azimuth_elevation(T_normalized[:3, :3])

                    # Store both formats
                    calibration_data = {
                        'slam_run': slam_run + 1,
                        'stride': stride,
                        'json_format': {
                            camera_names[0]: {
                                "quaternion": [0.0, 0.0, 0.0, 1.0],
                                "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                "translation_vector": [0.0, 0.0, 0.0],
                                "azimuth_deg": 0.0,
                                "elevation_deg": 0.0
                            },
                            camera_names[1]: {
                                "quaternion": Rotation.from_matrix(T_normalized[:3, :3]).as_quat().tolist(),
                                "rotation_matrix": T_normalized[:3, :3].tolist(),
                                "translation_vector": T_normalized[:3, 3].tolist(),
                                "azimuth_deg": float(azimuth_deg),
                                "elevation_deg": float(elevation_deg)
                            }
                        },
                        'yaml_format': {
                            'calibrated': True,
                            'quaternion': Rotation.from_matrix(T_normalized[:3, :3]).as_quat().tolist(),
                            'reference_camera': camera_names[0],
                            'target_camera': camera_names[1],
                            'transform': {
                                'rotation': T_normalized[:3, :3].flatten().tolist(),
                                'translation': T_normalized[:3, 3].tolist()
                            },
                            'azimuth_deg': float(azimuth_deg),
                            'elevation_deg': float(elevation_deg)
                        },
                        'original_translation_norm': float(np.linalg.norm(T[:3, 3])),
                        'normalized_translation_norm': float(np.linalg.norm(T_normalized[:3, 3]))
                    }

                    all_calibrations.append(calibration_data)
                    print(f"      ✓ Calibration successful (original scale: {calibration_data['original_translation_norm']:.3f})")

                    # Save this run immediately to stochastic_extrinsics
                    run_num = len(all_calibrations)
                    json_file = stochastic_dir / f"run_{run_num:02d}_stride{stride}_poses.json"
                    yaml_file = stochastic_dir / f"run_{run_num:02d}_stride{stride}_extrinsic.yaml"

                    with open(json_file, 'w') as f:
                        json.dump(calibration_data['json_format'], f, indent=2)
                    with open(yaml_file, 'w') as f:
                        yaml.dump(calibration_data['yaml_format'], f, default_flow_style=False)

                    print(f"      Saved to stochastic_extrinsics/run_{run_num:02d}_stride{stride}_*")

    # Summary
    print(f"\n{'='*70}")
    print(f"STOCHASTIC CALIBRATION COMPLETE")
    print(f"{'='*70}\n")

    print(f"Saved {len(all_calibrations)} calibration runs to: {stochastic_dir}")
    print(f"Individual files: run_01 through run_{len(all_calibrations):02d}")
    print()

    if len(all_calibrations) > 0:
        print(f"\n{'='*70}")
        print("ALL CALIBRATED POSES")
        print(f"{'='*70}\n")

        for i, calib in enumerate(all_calibrations):
            cam1_data = calib['json_format'][camera_names[1]]
            t = cam1_data['translation_vector']
            q = cam1_data['quaternion']
            azimuth = cam1_data['azimuth_deg']
            elevation = cam1_data['elevation_deg']
            scale = calib['original_translation_norm']
            stride = calib['stride']

            # Calculate ratio x/max:y/max:z/max
            t_abs = [abs(t[0]), abs(t[1]), abs(t[2])]
            max_component = max(t_abs)
            if max_component > 1e-6:
                ratio = [t_abs[0]/max_component, t_abs[1]/max_component, t_abs[2]/max_component]
            else:
                ratio = [0.0, 0.0, 0.0]

            print(f"Run {i+1:02d} (Stride {stride}):")
            print(f"  Translation (norm): [{t[0]:8.4f}, {t[1]:8.4f}, {t[2]:8.4f}]")
            print(f"  Quaternion (xyzw):  [{q[0]:8.4f}, {q[1]:8.4f}, {q[2]:8.4f}, {q[3]:8.4f}]")
            print(f"  Azimuth/Elevation:  {azimuth:8.2f}° / {elevation:8.2f}°")
            print(f"  Original scale:     {scale:8.3f} m")
            print(f"  Ratio (x/max:y/max:z/max): {ratio[0]:.4f}:{ratio[1]:.4f}:{ratio[2]:.4f}")
            print()

        print(f"{'='*70}")
        print()

    print("Each run saved as:")
    print("  - run_XX_strideN_poses.json (JSON format)")
    print("  - run_XX_strideN_extrinsic.yaml (YAML format)")
    print(f"\nStride values used: {STRIDE_VALUES}")
    print(f"Total calibrations: {NUM_SLAM_RUNS} SLAM runs × {len(STRIDE_VALUES)} strides = {NUM_SLAM_RUNS * len(STRIDE_VALUES)} calibrations")


if __name__ == '__main__':
    main()