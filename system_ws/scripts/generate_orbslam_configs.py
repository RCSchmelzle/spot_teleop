#!/usr/bin/env python3
"""
Generate ORB-SLAM3 YAML configs and extrinsics placeholders from camera mapping.
"""

import sys
import yaml
from pathlib import Path


def load_template():
    """Load xtionA_rgbd.yaml as template"""
    template_path = Path(__file__).parent.parent / "src/orbslam3_ros2/config/xtionA_rgbd.yaml"
    with open(template_path, 'r') as f:
        return f.read()


def generate_camera_config(camera_name, template):
    """Generate config for one camera"""
    # For now, just use template as-is
    # Could customize per-camera if needed
    return template


def generate_extrinsics_placeholder(cam_from, cam_to):
    """Generate extrinsics placeholder YAML"""
    return f"""# Extrinsic calibration: {cam_from} to {cam_to}
# TODO: Calibrate using kalibr, manual measurement, or hand-eye calibration

transform:
  # Translation in meters [x, y, z]
  translation: [0.0, 0.0, 0.0]

  # Rotation matrix (row-major 3x3)
  rotation: [1.0, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, 1.0]

# Quaternion [x, y, z, w] (alternative to rotation matrix)
quaternion: [0.0, 0.0, 0.0, 1.0]

# Calibration status
calibrated: false

# Notes:
# - Use kalibr for accurate multi-camera calibration
# - Or manually measure distance/orientation between cameras
# - Update 'calibrated: true' when done
"""


def main():
    if len(sys.argv) != 2:
        print("Usage: generate_orbslam_configs.py <config_dir>")
        sys.exit(1)

    config_dir = Path(sys.argv[1])
    mapping_file = config_dir / "camera_mapping.yaml"

    if not mapping_file.exists():
        print(f"ERROR: Camera mapping not found: {mapping_file}")
        sys.exit(1)

    # Load camera mapping
    with open(mapping_file, 'r') as f:
        mapping = yaml.safe_load(f)

    cameras = mapping['cameras']
    print(f"Generating configs for {len(cameras)} cameras...")

    # Load template
    template = load_template()

    # Generate config for each camera
    for cam in cameras:
        name = cam['name']
        config_file = config_dir / f"{name}_rgbd.yaml"

        config_content = generate_camera_config(name, template)

        with open(config_file, 'w') as f:
            f.write(config_content)

        print(f"  ✓ {config_file.name}")

    # Generate extrinsics placeholders
    if len(cameras) > 1:
        extrinsics_dir = config_dir / "extrinsics"
        extrinsics_dir.mkdir(exist_ok=True)

        # Generate pairwise extrinsics
        for i in range(len(cameras)):
            for j in range(i + 1, len(cameras)):
                cam_from = cameras[i]['name']
                cam_to = cameras[j]['name']

                extr_file = extrinsics_dir / f"{cam_from}_to_{cam_to}.yaml"
                content = generate_extrinsics_placeholder(cam_from, cam_to)

                with open(extr_file, 'w') as f:
                    f.write(content)

                print(f"  ✓ extrinsics/{extr_file.name}")

    print(f"\nConfiguration generated in: {config_dir}")
    print("\nNext steps:")
    print("  1. Run ORB-SLAM3 on each camera to generate trajectories")
    print("  2. Calibrate extrinsics between cameras")
    print("  3. Update extrinsics/*.yaml files with calibration data")


if __name__ == '__main__':
    main()