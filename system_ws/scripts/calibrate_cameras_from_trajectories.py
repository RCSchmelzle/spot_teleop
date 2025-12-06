#!/usr/bin/env python3
"""
Interactive calibrator for computing rigid transforms between cameras from ORB-SLAM3 trajectories.
Uses trajectory_utils for time alignment, interpolation, and hand-eye calibration.

Multi-run mode: Runs calibration N times to account for SLAM and RANSAC non-determinism,
then computes mean and visualizes all variants.
"""

import sys
import yaml
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from scipy.spatial.transform import Rotation, Slerp

from trajectory_utils import (
    load_tum_trajectory,
    generate_synchronized_samples,
    get_synchronized_poses,
    calibrate_all_cameras_from_synchronized_poses,
    MotionFilterConfig,
    RansacConfig,
    PoseSE3,
)

# Number of calibration runs for multi-run mode
NUM_CALIBRATION_RUNS = 5


def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """Convert rotation matrix to Euler angles (roll, pitch, yaw) in degrees."""
    rot = Rotation.from_matrix(R)
    euler_rad = rot.as_euler('xyz', degrees=False)
    return tuple(np.rad2deg(euler_rad))


def average_quaternions(quats: List[np.ndarray]) -> np.ndarray:
    """
    Average multiple quaternions using the method from Markley et al. 2007.
    Handles quaternion double-cover (q and -q represent same rotation).

    Args:
        quats: List of quaternions as [x, y, z, w] numpy arrays

    Returns:
        Average quaternion [x, y, z, w]
    """
    if len(quats) == 1:
        return quats[0]

    # Convert to matrix form for averaging
    Q = np.array(quats)  # Shape: (N, 4)

    # Ensure all quaternions are in same hemisphere (handle double-cover)
    # Flip quaternions that are in opposite hemisphere from first
    for i in range(1, len(Q)):
        if np.dot(Q[0], Q[i]) < 0:
            Q[i] = -Q[i]

    # Compute average using eigenvector method
    M = Q.T @ Q  # 4x4 matrix
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    # Eigenvector with largest eigenvalue is the average
    avg_quat = eigenvectors[:, -1]

    # Normalize
    avg_quat = avg_quat / np.linalg.norm(avg_quat)

    return avg_quat


def compute_mean_transform(transforms: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean of multiple SE(3) transforms.

    Args:
        transforms: List of 4x4 transformation matrices

    Returns:
        mean_transform: Mean 4x4 transformation matrix
        std_stats: Dictionary with standard deviations
    """
    translations = []
    quaternions = []

    for T in transforms:
        t = T[:3, 3]
        R = T[:3, :3]
        rot = Rotation.from_matrix(R)
        q = rot.as_quat()  # [x, y, z, w]

        translations.append(t)
        quaternions.append(q)

    # Average translation
    mean_translation = np.mean(translations, axis=0)
    std_translation = np.std(translations, axis=0)

    # Average rotation (quaternion)
    mean_quat = average_quaternions(quaternions)
    mean_rotation = Rotation.from_quat(mean_quat).as_matrix()

    # Compute rotation standard deviation (in degrees)
    angles_from_mean = []
    mean_rot = Rotation.from_quat(mean_quat)
    for q in quaternions:
        rot = Rotation.from_quat(q)
        # Compute angle between this rotation and mean
        rel_rot = mean_rot.inv() * rot
        angle = np.linalg.norm(rel_rot.as_rotvec())
        angles_from_mean.append(np.rad2deg(angle))
    std_rotation = np.std(angles_from_mean)

    # Build mean transform
    mean_transform = np.eye(4)
    mean_transform[:3, :3] = mean_rotation
    mean_transform[:3, 3] = mean_translation

    std_stats = {
        'translation_std': std_translation,
        'rotation_std_deg': std_rotation
    }

    return mean_transform, std_stats


def display_extrinsics_text(X_dict: dict, camera_names: List[str], ref_idx: int = 0):
    """Display extrinsic calibration results as text."""
    print("\n" + "="*70)
    print("CAMERA EXTRINSIC CALIBRATION RESULTS")
    print("="*70)
    print(f"\nReference Camera: {camera_names[ref_idx]}")
    print("\nTransforms from reference camera to each camera:\n")

    for cam_idx in range(len(camera_names)):
        if cam_idx == ref_idx:
            print(f"{camera_names[cam_idx]} (reference):")
            print("  Translation (xyz): [0.000, 0.000, 0.000] meters")
            print("  Rotation (roll, pitch, yaw): [0.0, 0.0, 0.0] degrees")
            print()
            continue

        key = (ref_idx, cam_idx)
        if key not in X_dict:
            print(f"{camera_names[cam_idx]}: NO CALIBRATION DATA")
            print()
            continue

        X = X_dict[key]
        t = X[:3, 3]
        R = X[:3, :3]
        roll, pitch, yaw = rotation_matrix_to_euler(R)

        print(f"{camera_names[cam_idx]}:")
        print(f"  Translation (xyz): [{t[0]:7.4f}, {t[1]:7.4f}, {t[2]:7.4f}] meters")
        print(f"  Rotation (roll, pitch, yaw): [{roll:6.2f}, {pitch:6.2f}, {yaw:6.2f}] degrees")
        print()

    print("="*70 + "\n")


def save_extrinsics_yaml(X_dict: dict, camera_names: List[str], output_dir: Path, ref_idx: int = 0):
    """Save extrinsics to YAML files in output_dir/extrinsics/."""
    extrinsics_dir = output_dir / "extrinsics"
    extrinsics_dir.mkdir(parents=True, exist_ok=True)

    for cam_idx in range(len(camera_names)):
        if cam_idx == ref_idx:
            continue

        key = (ref_idx, cam_idx)
        if key not in X_dict:
            print(f"Warning: No calibration for {camera_names[cam_idx]}, skipping YAML output")
            continue

        X = X_dict[key]
        t = X[:3, 3].tolist()
        R = X[:3, :3]

        # Convert rotation to quaternion
        rot = Rotation.from_matrix(R)
        q = rot.as_quat()  # [x, y, z, w]

        # Create YAML content
        extrinsic_data = {
            'transform': {
                'translation': t,
                'rotation': R.flatten().tolist(),  # Row-major 3x3 matrix
            },
            'quaternion': q.tolist(),  # [x, y, z, w]
            'calibrated': True,
            'reference_camera': camera_names[ref_idx],
            'target_camera': camera_names[cam_idx],
        }

        # Save to file
        ref_name = camera_names[ref_idx]
        cam_name = camera_names[cam_idx]
        output_file = extrinsics_dir / f"{ref_name}_to_{cam_name}.yaml"

        with open(output_file, 'w') as f:
            yaml.dump(extrinsic_data, f, default_flow_style=False)

        print(f"Saved: {output_file}")


def save_extrinsics_json(X_dict: dict, camera_names: List[str], output_dir: Path, ref_idx: int = 0):
    """Save all camera poses in a single JSON file."""
    import json

    extrinsics_dir = output_dir / "extrinsics"
    extrinsics_dir.mkdir(parents=True, exist_ok=True)

    # Build the pose dictionary
    poses = {}

    # Reference camera (identity transform)
    ref_name = camera_names[ref_idx]
    poses[ref_name] = {
        "quaternion": [0.0, 0.0, 0.0, 1.0],  # [x, y, z, w] - identity rotation
        "rotation_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        "translation_vector": [0.0, 0.0, 0.0]
    }

    # Other cameras
    for cam_idx in range(len(camera_names)):
        if cam_idx == ref_idx:
            continue

        key = (ref_idx, cam_idx)
        if key not in X_dict:
            continue

        X = X_dict[key]
        t = X[:3, 3].tolist()
        R = X[:3, :3]

        # Convert rotation to quaternion
        rot = Rotation.from_matrix(R)
        q = rot.as_quat()  # [x, y, z, w]

        cam_name = camera_names[cam_idx]
        poses[cam_name] = {
            "quaternion": q.tolist(),  # [x, y, z, w]
            "rotation_matrix": R.tolist(),  # 3x3 matrix as list of lists
            "translation_vector": t  # [x, y, z]
        }

    # Save to JSON file
    output_file = extrinsics_dir / "camera_poses.json"
    with open(output_file, 'w') as f:
        json.dump(poses, f, indent=4)

    print(f"Saved: {output_file}")


def visualize_camera_rig_3d(X_dict: dict, camera_names: List[str], ref_idx: int = 0):
    """Optional 3D visualization of camera rig with frustums."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("Matplotlib not available, skipping 3D visualization")
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Camera frustum parameters
    frustum_depth = 0.1  # meters
    frustum_width = 0.08
    frustum_height = 0.06

    def draw_camera_frustum(ax, X: np.ndarray, label: str, color: str):
        """Draw a camera frustum at pose X."""
        # Define frustum corners in camera frame
        corners = np.array([
            [0, 0, 0],  # Camera center
            [-frustum_width/2, -frustum_height/2, frustum_depth],
            [frustum_width/2, -frustum_height/2, frustum_depth],
            [frustum_width/2, frustum_height/2, frustum_depth],
            [-frustum_width/2, frustum_height/2, frustum_depth],
        ])

        # Transform to world frame
        R = X[:3, :3]
        t = X[:3, 3]
        corners_world = (R @ corners.T).T + t

        # Draw edges
        center = corners_world[0]
        for i in range(1, 5):
            ax.plot([center[0], corners_world[i][0]],
                   [center[1], corners_world[i][1]],
                   [center[2], corners_world[i][2]],
                   color=color, linewidth=1.5)

        # Draw frustum rectangle
        for i in range(1, 5):
            j = i % 4 + 1
            ax.plot([corners_world[i][0], corners_world[j][0]],
                   [corners_world[i][1], corners_world[j][1]],
                   [corners_world[i][2], corners_world[j][2]],
                   color=color, linewidth=1.5)

        # Draw camera position
        ax.scatter([t[0]], [t[1]], [t[2]], color=color, s=100, marker='o')
        ax.text(t[0], t[1], t[2], f'  {label}', fontsize=10, color=color)

        # Fill frustum faces
        face_indices = [
            [1, 2, 3, 4],  # Far plane
        ]
        for face_idx in face_indices:
            face_corners = corners_world[face_idx]
            poly = Poly3DCollection([face_corners], alpha=0.2, facecolor=color, edgecolor=color)
            ax.add_collection3d(poly)

    # Draw reference camera at origin
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    X_identity = np.eye(4)
    draw_camera_frustum(ax, X_identity, camera_names[ref_idx], colors[ref_idx % len(colors)])

    # Draw other cameras
    for cam_idx in range(len(camera_names)):
        if cam_idx == ref_idx:
            continue

        key = (ref_idx, cam_idx)
        if key not in X_dict:
            continue

        X = X_dict[key]
        draw_camera_frustum(ax, X, camera_names[cam_idx], colors[cam_idx % len(colors)])

    # Set labels and limits
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('Multi-Camera Rig Extrinsic Calibration')

    # Equal aspect ratio
    all_positions = [np.array([0, 0, 0])]
    for cam_idx in range(len(camera_names)):
        if cam_idx == ref_idx:
            continue
        key = (ref_idx, cam_idx)
        if key in X_dict:
            all_positions.append(X_dict[key][:3, 3])

    all_positions = np.array(all_positions)
    max_range = np.max(np.ptp(all_positions, axis=0)) / 2.0
    mid_x = np.mean(all_positions[:, 0])
    mid_y = np.mean(all_positions[:, 1])
    mid_z = np.mean(all_positions[:, 2])
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()


def visualize_multi_run_results(all_X_dicts: List[dict], camera_names: List[str], ref_idx: int = 0):
    """Visualize all calibration runs plus mean."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Matplotlib not available, skipping multi-run visualization")
        return

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Draw reference camera
    ax.scatter([0], [0], [0], c='black', s=200, marker='o', label=f'{camera_names[ref_idx]} (ref)')
    ax.text(0, 0, 0, f'  {camera_names[ref_idx]}\n  (reference)', fontsize=12, fontweight='bold')

    # For each camera pair
    for cam_idx in range(len(camera_names)):
        if cam_idx == ref_idx:
            continue

        key = (ref_idx, cam_idx)

        # Collect all transforms for this camera across runs
        transforms = []
        for X_dict in all_X_dicts:
            if key in X_dict:
                transforms.append(X_dict[key])

        if len(transforms) == 0:
            continue

        # Compute mean
        mean_transform, std_stats = compute_mean_transform(transforms)

        # Extract positions
        positions = np.array([T[:3, 3] for T in transforms])
        mean_pos = mean_transform[:3, 3]

        # Draw individual runs (semi-transparent)
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  c='lightblue', s=100, alpha=0.4, marker='x')

        # Draw mean (solid)
        ax.scatter([mean_pos[0]], [mean_pos[1]], [mean_pos[2]],
                  c='red', s=300, marker='o', edgecolors='black', linewidths=2,
                  label=f'{camera_names[cam_idx]} (mean of {len(transforms)} runs)')

        ax.text(mean_pos[0], mean_pos[1], mean_pos[2],
               f'  {camera_names[cam_idx]}\n  (mean)', fontsize=11, fontweight='bold')

        # Draw std deviation ellipsoid (simplified as error bars)
        std_t = std_stats['translation_std']
        for i in range(3):
            line_start = mean_pos.copy()
            line_end = mean_pos.copy()
            line_start[i] -= std_t[i]
            line_end[i] += std_t[i]
            ax.plot([line_start[0], line_end[0]],
                   [line_start[1], line_end[1]],
                   [line_start[2], line_end[2]],
                   'r--', alpha=0.5, linewidth=2)

        # Print statistics
        print(f"\n{camera_names[cam_idx]} statistics ({len(transforms)} runs):")
        print(f"  Translation std: [{std_t[0]:.4f}, {std_t[1]:.4f}, {std_t[2]:.4f}] m")
        print(f"  Rotation std: {std_stats['rotation_std_deg']:.2f}°")

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title(f'Multi-Camera Calibration: {NUM_CALIBRATION_RUNS} Runs\n(Mean in Red, Individual Runs in Light Blue)')
    ax.legend()

    # Equal aspect ratio
    all_positions = [np.array([0, 0, 0])]
    for X_dict in all_X_dicts:
        for key, X in X_dict.items():
            all_positions.append(X[:3, 3])

    if len(all_positions) > 1:
        all_positions = np.array(all_positions)
        max_range = np.max(np.ptp(all_positions, axis=0)) / 2.0
        mid_x = np.mean(all_positions[:, 0])
        mid_y = np.mean(all_positions[:, 1])
        mid_z = np.mean(all_positions[:, 2])
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: calibrate_cameras_from_trajectories.py <session_dir>")
        print("\nExample:")
        print("  ./calibrate_cameras_from_trajectories.py ~/Projects/teleoperation_spot/datasets/xtion_calibration_test/bags/2025-12-05_14-30-22")
        sys.exit(1)

    session_dir = Path(sys.argv[1])
    trajectories_dir = session_dir / "trajectories"
    config_dir = session_dir / "orbslam_config"

    if not trajectories_dir.exists():
        print(f"Error: Trajectories directory not found: {trajectories_dir}")
        sys.exit(1)

    if not config_dir.exists():
        print(f"Error: Config directory not found: {config_dir}")
        sys.exit(1)

    # Load camera mapping to get camera names
    mapping_file = config_dir / "camera_mapping.yaml"
    if not mapping_file.exists():
        print(f"Error: Camera mapping not found: {mapping_file}")
        sys.exit(1)

    with open(mapping_file, 'r') as f:
        mapping_data = yaml.safe_load(f)

    camera_names = [cam['name'] for cam in mapping_data['cameras']]
    print(f"\nFound {len(camera_names)} cameras: {', '.join(camera_names)}")

    # Find trajectory files
    traj_files = []
    for cam_name in camera_names:
        traj_file = trajectories_dir / f"{cam_name}_KeyFrameTrajectory.txt"
        if not traj_file.exists():
            print(f"Error: Trajectory file not found: {traj_file}")
            sys.exit(1)
        traj_files.append(traj_file)

    print("\nLoading trajectories...")
    trajectories = [load_tum_trajectory(traj_file) for traj_file in traj_files]

    # Compute keyframe spacing statistics
    all_dts = []
    for i, (cam_name, traj) in enumerate(zip(camera_names, trajectories)):
        t_start = traj[0].t
        t_end = traj[-1].t
        duration = t_end - t_start

        # Compute time differences between consecutive keyframes
        times = np.array([pose.t for pose in traj])
        dts = np.diff(times)
        all_dts.extend(dts)

        mean_dt = np.mean(dts)
        median_dt = np.median(dts)
        std_dt = np.std(dts)

        print(f"  {cam_name}: {len(traj)} poses ({duration:.1f}s, {t_start:.2f} to {t_end:.2f})")
        print(f"    Keyframe spacing - mean: {mean_dt:.3f}s, median: {median_dt:.3f}s, std: {std_dt:.3f}s")

    # Compute adaptive epsilon based on overall keyframe spacing
    all_dts = np.array(all_dts)
    global_mean = np.mean(all_dts)
    global_median = np.median(all_dts)
    global_std = np.std(all_dts)

    # Set epsilon to mean + 1*std to accommodate typical variation
    # This gives tolerance on EACH side of the sample time
    epsilon_t = global_mean + 1.0 * global_std

    # Set sampling rate to be slower than typical keyframe rate
    dt_sample = max(0.2, global_median * 1.5)  # At least 0.2s, or 1.5x median spacing

    print(f"\nOverall keyframe spacing statistics:")
    print(f"  Mean: {global_mean:.3f}s, Median: {global_median:.3f}s, Std: {global_std:.3f}s")
    print(f"\nAdaptive synchronization parameters:")
    print(f"  dt_sample: {dt_sample:.3f}s ({1/dt_sample:.1f} Hz)")
    print(f"  epsilon_t: {epsilon_t:.3f}s (tolerance on each side)")
    print(f"  Total bracket window: {2*epsilon_t:.3f}s")

    print("\nGenerating synchronized samples...")
    valid_times = generate_synchronized_samples(
        trajectories,
        dt_sample=dt_sample,
        epsilon_t=epsilon_t
    )
    print(f"  Found {len(valid_times)} synchronized samples")

    if len(valid_times) < 10:
        print("Error: Insufficient synchronized samples for calibration")
        sys.exit(1)

    print("\nInterpolating poses at synchronized times...")
    all_synchronized_poses = []
    for tau_j in valid_times:
        synchronized_poses_at_tau = get_synchronized_poses(trajectories, tau_j)
        if synchronized_poses_at_tau is not None:
            all_synchronized_poses.append(synchronized_poses_at_tau)

    print(f"  Generated {len(all_synchronized_poses)} synchronized pose sets")

    print("\nComputing extrinsic calibrations (pairwise against reference camera)...")

    # Configure motion filtering - scale thresholds based on actual dt_sample
    # Default MotionFilterConfig assumes 30 FPS (dt=0.033s)
    # Scale max thresholds proportionally to our actual dt_sample
    dt_reference = 1/30.0  # Reference: 30 FPS
    scale_factor = dt_sample / dt_reference

    filter_cfg = MotionFilterConfig(
        min_rot_rad=np.deg2rad(0.3),          # Keep minimum unchanged (info threshold)
        min_trans=0.002,                       # Keep minimum unchanged
        max_rot_rad=np.deg2rad(15.0) * scale_factor,  # Scale max by time ratio
        max_trans=0.05 * scale_factor,         # Scale max by time ratio
        max_rot_ratio=5.0,                     # Keep ratio checks unchanged
        max_trans_ratio=5.0
    )

    print(f"\nMotion filter config (scaled for dt={dt_sample:.3f}s):")
    print(f"  Min rotation: {np.rad2deg(filter_cfg.min_rot_rad):.1f}°")
    print(f"  Max rotation: {np.rad2deg(filter_cfg.max_rot_rad):.1f}° (scaled by {scale_factor:.1f}x)")
    print(f"  Min translation: {filter_cfg.min_trans*1000:.1f}mm")
    print(f"  Max translation: {filter_cfg.max_trans*1000:.1f}mm (scaled by {scale_factor:.1f}x)")

    ransac_cfg = RansacConfig(
        max_iters=500,
        sample_size=8,
        rot_thresh_rad=np.deg2rad(0.5),
        trans_thresh=0.005,
        min_inlier_ratio=0.3
    )

    # Run calibration multiple times to account for RANSAC non-determinism
    print(f"\n{'='*70}")
    print(f"MULTI-RUN CALIBRATION ({NUM_CALIBRATION_RUNS} runs)")
    print(f"{'='*70}\n")

    all_X_dicts = []
    for run_idx in range(NUM_CALIBRATION_RUNS):
        print(f"\n--- Run {run_idx + 1}/{NUM_CALIBRATION_RUNS} ---")

        X_dict = calibrate_all_cameras_from_synchronized_poses(
            all_synchronized_poses,
            ref_cam_idx=0,
            stride=1,
            filter_cfg=filter_cfg,
            ransac_cfg=ransac_cfg,
        )

        all_X_dicts.append(X_dict)

        # Show quick result for this run
        for key, X in X_dict.items():
            t = X[:3, 3]
            print(f"  {camera_names[key[0]]} -> {camera_names[key[1]]}: t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")

    print(f"\n{'='*70}")
    print("COMPUTING MEAN ACROSS ALL RUNS")
    print(f"{'='*70}\n")

    # Compute mean transforms
    mean_X_dict = {}
    for cam_idx in range(len(camera_names)):
        if cam_idx == 0:  # Skip reference camera
            continue

        key = (0, cam_idx)

        # Collect all transforms for this camera pair
        transforms = []
        for X_dict in all_X_dicts:
            if key in X_dict:
                transforms.append(X_dict[key])

        if len(transforms) > 0:
            mean_transform, std_stats = compute_mean_transform(transforms)
            mean_X_dict[key] = mean_transform

            # Print statistics
            t_std = std_stats['translation_std']
            r_std = std_stats['rotation_std_deg']
            print(f"{camera_names[0]} -> {camera_names[cam_idx]}:")
            print(f"  Translation std: [{t_std[0]:.4f}, {t_std[1]:.4f}, {t_std[2]:.4f}] m")
            print(f"  Rotation std: {r_std:.2f}°")
            print(f"  Computed from {len(transforms)}/{NUM_CALIBRATION_RUNS} successful runs")
            print()

    # Display mean results as text
    display_extrinsics_text(mean_X_dict, camera_names, ref_idx=0)

    # Save mean to YAML (individual pairwise files)
    print("\nSaving MEAN extrinsics to YAML files...")
    save_extrinsics_yaml(mean_X_dict, camera_names, config_dir, ref_idx=0)

    # Save mean to JSON (single file with all camera poses)
    print("\nSaving MEAN camera poses to JSON...")
    save_extrinsics_json(mean_X_dict, camera_names, config_dir, ref_idx=0)

    # Visualize multi-run results
    print("\nGenerating multi-run visualization...")
    visualize_multi_run_results(all_X_dicts, camera_names, ref_idx=0)


if __name__ == '__main__':
    main()