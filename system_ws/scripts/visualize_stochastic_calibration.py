#!/usr/bin/env python3
"""
Visualize Stochastic Calibration Results

Displays all calibration runs showing translation vectors, rotation angles,
and statistics across multiple SLAM runs.
"""

import sys
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_calibrations(stochastic_dir: Path):
    """Load all calibration results."""
    all_calibrations_file = stochastic_dir / "all_calibrations.json"

    if not all_calibrations_file.exists():
        print(f"Error: {all_calibrations_file} not found")
        sys.exit(1)

    with open(all_calibrations_file, 'r') as f:
        return json.load(f)


def visualize_calibrations(calibrations: list, camera_names: list):
    """Create comprehensive visualization of all calibration runs."""

    n_runs = len(calibrations)

    # Extract data
    translations = np.array([c['json_format'][camera_names[1]]['translation_vector']
                            for c in calibrations])
    quaternions = np.array([c['json_format'][camera_names[1]]['quaternion']
                           for c in calibrations])
    scales = np.array([c['original_translation_norm'] for c in calibrations])

    # Convert quaternions to euler angles (degrees)
    euler_angles = []
    for q in quaternions:
        rot = Rotation.from_quat(q)
        euler = rot.as_euler('xyz', degrees=True)
        euler_angles.append(euler)
    euler_angles = np.array(euler_angles)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. 3D plot of translation vectors
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    for i in range(n_runs):
        ax1.scatter([translations[i, 0]], [translations[i, 1]], [translations[i, 2]],
                    s=200, marker='o', label=f'Run {i+1}')
    ax1.set_xlabel('X (normalized)')
    ax1.set_ylabel('Y (normalized)')
    ax1.set_zlabel('Z (normalized)')
    ax1.set_title('Translation Vectors (Normalized to Unit Norm)')
    ax1.legend()
    ax1.grid(True)

    # 2. Translation components
    ax2 = fig.add_subplot(2, 3, 2)
    x = np.arange(n_runs) + 1
    width = 0.25
    ax2.bar(x - width, translations[:, 0], width, label='X', alpha=0.8)
    ax2.bar(x, translations[:, 1], width, label='Y', alpha=0.8)
    ax2.bar(x + width, translations[:, 2], width, label='Z', alpha=0.8)
    ax2.set_xlabel('Run Number')
    ax2.set_ylabel('Translation (normalized)')
    ax2.set_title('Translation Components by Run')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Euler angles
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(x, euler_angles[:, 0], 'o-', label='Roll (X)', linewidth=2, markersize=8)
    ax3.plot(x, euler_angles[:, 1], 's-', label='Pitch (Y)', linewidth=2, markersize=8)
    ax3.plot(x, euler_angles[:, 2], '^-', label='Yaw (Z)', linewidth=2, markersize=8)
    ax3.set_xlabel('Run Number')
    ax3.set_ylabel('Angle (degrees)')
    ax3.set_title('Rotation (Euler Angles XYZ)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Original scales
    ax4 = fig.add_subplot(2, 3, 4)
    for i in range(n_runs):
        ax4.bar([i+1], [scales[i]], alpha=0.7, label=f'Run {i+1}')
    ax4.set_xlabel('Run Number')
    ax4.set_ylabel('Translation Magnitude (meters)')
    ax4.set_title('Original SLAM Scale Before Normalization')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Quaternion components
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(x, quaternions[:, 0], 'o-', label='qx', linewidth=2, markersize=6)
    ax5.plot(x, quaternions[:, 1], 's-', label='qy', linewidth=2, markersize=6)
    ax5.plot(x, quaternions[:, 2], '^-', label='qz', linewidth=2, markersize=6)
    ax5.plot(x, quaternions[:, 3], 'd-', label='qw', linewidth=2, markersize=6)
    ax5.set_xlabel('Run Number')
    ax5.set_ylabel('Quaternion Component')
    ax5.set_title('Quaternion Components')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Individual run details
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    # Build per-run details
    run_details = f"""
INDIVIDUAL RUN DETAILS
{'='*40}

Cameras: {camera_names[0]} → {camera_names[1]}
Number of runs: {n_runs}

"""
    for i in range(n_runs):
        run_details += f"""RUN {i+1}:
  Translation (norm): [{translations[i, 0]:.4f}, {translations[i, 1]:.4f}, {translations[i, 2]:.4f}]
  Rotation (deg):     [{euler_angles[i, 0]:.2f}°, {euler_angles[i, 1]:.2f}°, {euler_angles[i, 2]:.2f}°]
  Quaternion:         [{quaternions[i, 0]:.4f}, {quaternions[i, 1]:.4f}, {quaternions[i, 2]:.4f}, {quaternions[i, 3]:.4f}]
  Original scale:     {scales[i]:.3f} m

"""

    ax6.text(0.1, 0.95, run_details, transform=ax6.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    return fig


def main():
    if len(sys.argv) < 2:
        print("Usage: visualize_stochastic_calibration.py <session_dir>")
        sys.exit(1)

    session_dir = Path(sys.argv[1])
    stochastic_dir = session_dir / "orbslam_config" / "stochastic_extrinsics"

    if not stochastic_dir.exists():
        print(f"Error: {stochastic_dir} does not exist")
        sys.exit(1)

    # Load calibrations
    calibrations = load_calibrations(stochastic_dir)

    if not calibrations:
        print("No calibrations found")
        sys.exit(1)

    # Get camera names from first calibration
    camera_names = list(calibrations[0]['json_format'].keys())

    print(f"\nLoaded {len(calibrations)} calibration runs")
    print(f"Cameras: {camera_names[0]} → {camera_names[1]}")
    print(f"\nGenerating visualization...")

    # Create visualization
    fig = visualize_calibrations(calibrations, camera_names)

    # Save figure
    output_file = stochastic_dir / "calibration_visualization.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_file}")

    # Show interactive plot
    plt.show()


if __name__ == '__main__':
    main()