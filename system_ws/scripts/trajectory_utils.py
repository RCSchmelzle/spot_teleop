#!/usr/bin/env python3
"""
Trajectory utilities for multi-camera extrinsic calibration.
Handles SE(3) pose representation, interpolation, and bracketing.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class PoseSE3:
    """SE(3) pose with timestamp"""
    t: float           # ROS time (seconds)
    T: np.ndarray      # 4x4 SE3 matrix (R|t)


def load_tum_trajectory(filepath: Path) -> List[PoseSE3]:
    """
    Load trajectory from TUM format file.
    Format: timestamp tx ty tz qx qy qz qw
    """
    trajectory = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 8:
                continue

            t = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

            # Normalize quaternion (TUM files should be normalized, but be defensive)
            q = np.array([qx, qy, qz, qw], dtype=np.float64)
            q /= np.linalg.norm(q)

            # Build SE3 matrix using scipy
            R = Rotation.from_quat(q).as_matrix()
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]

            trajectory.append(PoseSE3(t=t, T=T))

    if not trajectory:
        raise ValueError(f"No valid poses found in {filepath}")

    # Sort by time
    trajectory.sort(key=lambda p: p.t)
    return trajectory


def bracket_pose(traj: List[PoseSE3], t_query: float) -> Optional[Tuple[PoseSE3, PoseSE3]]:
    """
    Find poses bracketing t_query.

    Returns:
        (pose_before, pose_after) where pose_before.t <= t_query <= pose_after.t
        If t_query exactly equals a pose timestamp, returns (pose, pose)
        None if no valid bracket exists
    """
    if not traj:
        return None

    # Check bounds
    if t_query < traj[0].t or t_query > traj[-1].t:
        return None

    # Handle exact match at boundaries
    if t_query == traj[0].t:
        return (traj[0], traj[0])
    if t_query == traj[-1].t:
        return (traj[-1], traj[-1])

    # Binary search for bracketing poses
    left, right = 0, len(traj) - 1

    while left <= right:
        mid = (left + right) // 2

        # Exact match
        if traj[mid].t == t_query:
            return (traj[mid], traj[mid])

        if traj[mid].t < t_query:
            # Check if next pose brackets
            if mid + 1 < len(traj):
                if traj[mid + 1].t >= t_query:
                    return (traj[mid], traj[mid + 1])
            left = mid + 1
        else:
            right = mid - 1

    return None


def interpolate_pose(pose_before: PoseSE3, pose_after: PoseSE3, t_query: float) -> PoseSE3:
    """
    Interpolate SE(3) pose at t_query between two poses.

    Uses linear interpolation for translation and SLERP for rotation (via scipy).
    """
    dt = pose_after.t - pose_before.t
    if dt <= 0:
        # Degenerate or non-forward case: just return pose_before
        return pose_before

    alpha = (t_query - pose_before.t) / dt
    alpha = float(np.clip(alpha, 0.0, 1.0))

    # --- Translation: linear interpolation ---
    t_before = pose_before.T[:3, 3]
    t_after = pose_after.T[:3, 3]
    t_interp = (1.0 - alpha) * t_before + alpha * t_after

    # --- Rotation: SLERP ---
    R_before = pose_before.T[:3, :3]
    R_after  = pose_after.T[:3, :3]

    key_times = np.array([0.0, 1.0], dtype=np.float64)
    key_rots = Rotation.from_matrix(np.stack([R_before, R_after], axis=0))
    slerp = Slerp(key_times, key_rots)
    R_interp = slerp([alpha])[0]

    # --- Build SE(3) ---
    T_interp = np.eye(4, dtype=np.float64)
    T_interp[:3, :3] = R_interp.as_matrix()
    T_interp[:3, 3] = t_interp

    return PoseSE3(t=t_query, T=T_interp)


def compute_global_time_range(trajectories: List[List[PoseSE3]]) -> Tuple[float, float]:
    """
    Compute overlapping time range across all trajectories.

    Returns:
        (t_min, t_max) where all trajectories have data
    """
    if not trajectories or any(len(t) == 0 for t in trajectories):
        raise ValueError("All trajectories must be non-empty")

    # t_min = max of all trajectory starts
    t_min = max(traj[0].t for traj in trajectories)

    # t_max = min of all trajectory ends
    t_max = min(traj[-1].t for traj in trajectories)

    if t_min >= t_max:
        raise ValueError(f"No overlapping time range: t_min={t_min}, t_max={t_max}")

    return t_min, t_max


def generate_synchronized_samples(
    trajectories: List[List[PoseSE3]],
    dt_sample: float = 1/30.0,
    epsilon_t: float = 0.025
) -> List[float]:
    """
    Generate synchronized sample times where all cameras have valid poses.

    Args:
        trajectories: List of trajectories for each camera
        dt_sample: Sampling interval (default: 1/30 s)
        epsilon_t: Maximum time gap for bracketing (default: 0.025 s)
                  Note: Max spacing between samples is 2*epsilon_t

    Returns:
        List of valid sample times τ_j
    """
    # Get global time range
    t_min, t_max = compute_global_time_range(trajectories)

    # Generate candidate sample times (use linspace to avoid floating point accumulation)
    duration = t_max - t_min
    num_samples = int(np.ceil(duration / dt_sample)) + 1
    candidate_times = np.linspace(t_min, t_min + (num_samples - 1) * dt_sample, num_samples)

    # Filter to be within t_max
    candidate_times = candidate_times[candidate_times <= t_max]

    valid_times = []

    for tau_j in candidate_times:
        # Check if all cameras have valid brackets within epsilon_t
        all_valid = True

        for traj in trajectories:
            bracket = bracket_pose(traj, tau_j)

            if bracket is None:
                all_valid = False
                break

            pose_before, pose_after = bracket
            dt_before = tau_j - pose_before.t
            dt_after = pose_after.t - tau_j

            if dt_before > epsilon_t or dt_after > epsilon_t:
                all_valid = False
                break

        if all_valid:
            valid_times.append(float(tau_j))

    return valid_times


def get_synchronized_poses(
    trajectories: List[List[PoseSE3]],
    tau_j: float
) -> List[PoseSE3]:
    """
    Get interpolated poses for all cameras at time tau_j.

    Assumes tau_j is a valid synchronized time (from generate_synchronized_samples).
    """
    poses = []

    for traj in trajectories:
        bracket = bracket_pose(traj, tau_j)
        if bracket is None:
            raise ValueError(f"No valid bracket for time {tau_j}")

        pose_before, pose_after = bracket
        pose_interp = interpolate_pose(pose_before, pose_after, tau_j)
        poses.append(pose_interp)

    return poses


# ============================================================
# Hand-eye calibration utilities
# ============================================================

def invert_se3(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 SE(3) matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def rotation_angle(R: np.ndarray) -> float:
    """
    Returns the rotation angle (rad) of a 3x3 rotation matrix.
    """
    tr = np.trace(R)
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.arccos(cos_theta))


def motion_metrics(T: np.ndarray) -> Tuple[float, float]:
    """
    Given 4x4 SE(3) relative motion, returns (rot_angle_rad, trans_norm).
    """
    R = T[:3, :3]
    t = T[:3, 3]
    ang = rotation_angle(R)
    trans = float(np.linalg.norm(t))
    return ang, trans


def split_by_camera(all_synchronized_poses: List[List[PoseSE3]]
                    ) -> List[List[PoseSE3]]:
    """
    all_synchronized_poses[j][k] = pose of camera k at time tau_j.
    Returns per_cam[k] = list of poses for camera k over all tau_j.
    """
    num_samples = len(all_synchronized_poses)
    if num_samples == 0:
        return []

    num_cams = len(all_synchronized_poses[0])
    per_cam: List[List[PoseSE3]] = [[] for _ in range(num_cams)]

    for sample in all_synchronized_poses:
        assert len(sample) == num_cams
        for cam_idx, pose in enumerate(sample):
            per_cam[cam_idx].append(pose)

    return per_cam


def compute_relative_motions(poses: List[PoseSE3], stride: int = 1) -> List[np.ndarray]:
    """
    Compute relative motions A_i = T_i^{-1} T_{i+stride} for a single camera.

    stride=1 -> consecutive frames; larger stride can help increase per-step motion.
    """
    rel: List[np.ndarray] = []
    if stride <= 0:
        raise ValueError("stride must be >= 1")

    for i in range(len(poses) - stride):
        T_i = poses[i].T
        T_j = poses[i + stride].T
        A_ij = invert_se3(T_i) @ T_j
        rel.append(A_ij)
    return rel


@dataclass
class MotionFilterConfig:
    """
    Thresholds tuned for ~30 FPS handheld motion.
    Adjust as needed after inspecting your data.
    """
    # Drop tiny motions that carry almost no information
    min_rot_rad: float = np.deg2rad(0.3)   # < 0.3 deg -> tiny
    min_trans: float = 0.002               # < 2 mm -> tiny

    # Anything bigger than this in 1/30s is probably a glitch
    max_rot_rad: float = np.deg2rad(15.0)  # > 15 deg per sample -> suspicious
    max_trans: float = 0.05                # > 5 cm per sample -> suspicious

    # Consistency between cameras:
    max_rot_ratio: float = 5.0             # if one camera sees 5x more angle, weird
    max_trans_ratio: float = 5.0


def filter_motion_pairs(A_list: List[np.ndarray],
                        B_list: List[np.ndarray],
                        cfg: MotionFilterConfig
                        ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Filter motion pairs (Ai, Bi) using:
      - size thresholds (too small / too large)
      - inconsistency between A and B (ratios).
    Returns:
      A_filt, B_filt, kept_indices
    """
    assert len(A_list) == len(B_list)
    A_filt: List[np.ndarray] = []
    B_filt: List[np.ndarray] = []
    kept_indices: List[int] = []

    for i, (A, B) in enumerate(zip(A_list, B_list)):
        ang_A, trans_A = motion_metrics(A)
        ang_B, trans_B = motion_metrics(B)

        # 1) Size thresholds for each camera
        def bad_motion(ang, trans) -> bool:
            if ang < cfg.min_rot_rad and trans < cfg.min_trans:
                return True   # too tiny
            if ang > cfg.max_rot_rad or trans > cfg.max_trans:
                return True   # too big / implausible
            return False

        if bad_motion(ang_A, trans_A) or bad_motion(ang_B, trans_B):
            continue

        # 2) Inconsistency between cameras (order-of-magnitude check)
        eps = 1e-9
        if ang_A > eps and ang_B > eps:
            rot_ratio = max(ang_A, ang_B) / max(min(ang_A, ang_B), eps)
            if rot_ratio > cfg.max_rot_ratio:
                continue

        if trans_A > eps and trans_B > eps:
            trans_ratio = max(trans_A, trans_B) / max(min(trans_A, trans_B), eps)
            if trans_ratio > cfg.max_trans_ratio:
                continue

        # Passed all tests
        A_filt.append(A)
        B_filt.append(B)
        kept_indices.append(i)

    return A_filt, B_filt, kept_indices


def solve_hand_eye_ls(A_list: List[np.ndarray],
                      B_list: List[np.ndarray]) -> np.ndarray:
    """
    Solve A_i X = X B_i for X, where X maps points from camera B frame to camera A frame.

    Given relative motions Ai, Bi (4x4) in least squares.
    Returns X as a 4x4 SE(3) matrix.
    """
    assert len(A_list) == len(B_list)
    m = len(A_list)
    if m == 0:
        raise ValueError("No motion pairs provided to solve_hand_eye_ls")

    # ----- Solve for rotation R_X -----
    M_blocks = []
    for Ai, Bi in zip(A_list, B_list):
        R_A = Ai[:3, :3]
        R_B = Bi[:3, :3]
        # (R_A ⊗ I - I ⊗ R_B^T) vec(R_X) = 0
        M_blocks.append(np.kron(R_A, np.eye(3)) - np.kron(np.eye(3), R_B.T))
    M = np.vstack(M_blocks)  # (3*3*m, 9)

    _, _, Vt = np.linalg.svd(M)
    rvec = Vt[-1]  # nullspace
    R_X = rvec.reshape(3, 3)

    # Project to nearest proper rotation
    U, _, Vt_R = np.linalg.svd(R_X)
    R_X = U @ Vt_R
    if np.linalg.det(R_X) < 0:
        U[:, -1] *= -1
        R_X = U @ Vt_R

    # ----- Solve for translation t_X -----
    A_lin = []
    b_lin = []
    for Ai, Bi in zip(A_list, B_list):
        R_A = Ai[:3, :3]
        t_A = Ai[:3, 3]
        R_B = Bi[:3, :3]
        t_B = Bi[:3, 3]

        A_block = R_A - np.eye(3)
        b_block = R_X @ t_B - t_A

        A_lin.append(A_block)
        b_lin.append(b_block)

    A_lin = np.vstack(A_lin)           # (3m, 3)
    b_lin = np.concatenate(b_lin)      # (3m,)

    t_X, *_ = np.linalg.lstsq(A_lin, b_lin, rcond=None)

    X = np.eye(4, dtype=np.float64)
    X[:3, :3] = R_X
    X[:3, 3] = t_X
    return X


def hand_eye_residuals(X: np.ndarray,
                       A_list: List[np.ndarray],
                       B_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-pair rotation and translation residuals for A_i X vs X B_i.
    Returns (rot_errs_rad, trans_errs).
    """
    R_X = X[:3, :3]
    t_X = X[:3, 3]
    rot_errs = []
    trans_errs = []

    for Ai, Bi in zip(A_list, B_list):
        R_A, t_A = Ai[:3, :3], Ai[:3, 3]
        R_B, t_B = Bi[:3, :3], Bi[:3, 3]

        # Left and right sides of A_i X = X B_i
        R_L = R_A @ R_X
        t_L = R_A @ t_X + t_A

        R_R = R_X @ R_B
        t_R = R_X @ t_B + t_X

        R_err = R_L @ R_R.T
        ang_err = rotation_angle(R_err)
        trans_err = float(np.linalg.norm(t_L - t_R))

        rot_errs.append(ang_err)
        trans_errs.append(trans_err)

    return np.array(rot_errs), np.array(trans_errs)


@dataclass
class RansacConfig:
    max_iters: int = 500
    sample_size: int = 8                    # pairs per hypothesis
    rot_thresh_rad: float = np.deg2rad(0.5) # inlier if residual < 0.5 deg
    trans_thresh: float = 0.005             # inlier if residual < 5 mm
    min_inlier_ratio: float = 0.3


def solve_hand_eye_ransac(A_list: List[np.ndarray],
                          B_list: List[np.ndarray],
                          cfg: RansacConfig
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    RANSAC wrapper around solve_hand_eye_ls.
    Returns:
      X_best: 4x4 transform
      inlier_mask: boolean array over pairs
    """
    assert len(A_list) == len(B_list)
    n = len(A_list)
    if n == 0:
        raise ValueError("No motion pairs for RANSAC")
    if n < cfg.sample_size:
        raise ValueError("Not enough motion pairs for RANSAC")

    A_list = list(A_list)
    B_list = list(B_list)

    best_X = None
    best_inliers = None
    best_score = -1

    indices = np.arange(n)

    for _ in range(cfg.max_iters):
        sample_idx = np.random.choice(indices, size=cfg.sample_size, replace=False)
        A_sample = [A_list[i] for i in sample_idx]
        B_sample = [B_list[i] for i in sample_idx]

        try:
            X_candidate = solve_hand_eye_ls(A_sample, B_sample)
        except np.linalg.LinAlgError:
            continue

        rot_errs, trans_errs = hand_eye_residuals(X_candidate, A_list, B_list)
        inliers = (rot_errs < cfg.rot_thresh_rad) & (trans_errs < cfg.trans_thresh)

        num_inliers = int(inliers.sum())
        if num_inliers < cfg.min_inlier_ratio * n:
            continue

        score = num_inliers
        if score > best_score:
            best_score = score
            best_X = X_candidate
            best_inliers = inliers

    if best_X is None:
        # Fallback: plain LS on all pairs
        best_X = solve_hand_eye_ls(A_list, B_list)
        best_inliers = np.ones(n, dtype=bool)

    # Optional: re-fit LS on inliers
    A_in = [A_list[i] for i in range(n) if best_inliers[i]]
    B_in = [B_list[i] for i in range(n) if best_inliers[i]]
    X_refined = solve_hand_eye_ls(A_in, B_in)

    return X_refined, best_inliers


def get_motion_pairs_for_cameras(
    all_synchronized_poses: List[List[PoseSE3]],
    cam_idx_a: int,
    cam_idx_b: int,
    stride: int = 1,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    From all_synchronized_poses, extract relative motions for two cameras
    over all consecutive synchronized samples (with optional stride).
    Returns:
      A_list: relative motions for camera A
      B_list: relative motions for camera B
    """
    per_cam = split_by_camera(all_synchronized_poses)
    poses_a = per_cam[cam_idx_a]
    poses_b = per_cam[cam_idx_b]

    A_list = compute_relative_motions(poses_a, stride=stride)
    B_list = compute_relative_motions(poses_b, stride=stride)
    assert len(A_list) == len(B_list)
    return A_list, B_list


def calibrate_camera_pair_from_synchronized_poses(
    all_synchronized_poses: List[List[PoseSE3]],
    cam_idx_a: int,
    cam_idx_b: int,
    stride: int = 1,
    filter_cfg: Optional[MotionFilterConfig] = None,
    ransac_cfg: Optional[RansacConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Calibrate extrinsics between two cameras using hand-eye on their trajectories.

    Returns:
        X_A_to_B: 4x4 SE(3) from cam A to cam B
        inlier_mask: boolean mask over filtered motion pairs
        kept_indices: indices (in original A/B lists) of filtered pairs
    """
    if filter_cfg is None:
        filter_cfg = MotionFilterConfig()
    if ransac_cfg is None:
        ransac_cfg = RansacConfig()

    A_list, B_list = get_motion_pairs_for_cameras(
        all_synchronized_poses, cam_idx_a, cam_idx_b, stride=stride
    )

    # Filter
    A_filt, B_filt, kept_indices = filter_motion_pairs(A_list, B_list, filter_cfg)
    if len(A_filt) < 5:
        raise RuntimeError(f"Not enough valid motion pairs for cameras {cam_idx_a}->{cam_idx_b}")

    # Robust hand-eye
    X_A_to_B, inliers = solve_hand_eye_ransac(A_filt, B_filt, ransac_cfg)
    return X_A_to_B, inliers, kept_indices


def calibrate_all_cameras_from_synchronized_poses(
    all_synchronized_poses: List[List[PoseSE3]],
    ref_cam_idx: int = 0,
    stride: int = 1,
    filter_cfg: Optional[MotionFilterConfig] = None,
    ransac_cfg: Optional[RansacConfig] = None,
) -> dict:
    """
    Calibrate extrinsics from ref_cam_idx to every other camera.

    Returns:
        dict mapping (ref_cam_idx, cam_idx) -> 4x4 X_ref_to_cam
    """
    if filter_cfg is None:
        filter_cfg = MotionFilterConfig()
    if ransac_cfg is None:
        ransac_cfg = RansacConfig()

    per_cam = split_by_camera(all_synchronized_poses)
    num_cams = len(per_cam)
    results: dict = {}

    for cam_idx in range(num_cams):
        if cam_idx == ref_cam_idx:
            continue

        try:
            X_ref_to_cam, inliers, kept_idx = calibrate_camera_pair_from_synchronized_poses(
                all_synchronized_poses,
                cam_idx_a=ref_cam_idx,
                cam_idx_b=cam_idx,
                stride=stride,
                filter_cfg=filter_cfg,
                ransac_cfg=ransac_cfg,
            )
            results[(ref_cam_idx, cam_idx)] = X_ref_to_cam
        except RuntimeError as e:
            print(f"[WARN] Calibration failed for cam {ref_cam_idx}->{cam_idx}: {e}")

    return results