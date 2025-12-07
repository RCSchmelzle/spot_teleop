#!/usr/bin/env python3
"""Visualize multiple TUM-format trajectories in 3D."""

import argparse
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_tum_trajectory(filepath):
    """Load TUM format trajectory: timestamp tx ty tz qx qy qz qw"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                data.append([float(x) for x in parts])
    return np.array(data) if data else None


def main():
    parser = argparse.ArgumentParser(description='Visualize TUM trajectories')
    parser.add_argument('files', nargs='+', help='Trajectory files to visualize')
    parser.add_argument('--labels', nargs='+', help='Labels for each trajectory')
    args = parser.parse_args()

    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.tab10.colors

    for i, filepath in enumerate(args.files):
        traj = load_tum_trajectory(filepath)
        if traj is None:
            print(f"Could not load: {filepath}")
            continue

        label = args.labels[i] if args.labels and i < len(args.labels) else Path(filepath).stem
        color = colors[i % len(colors)]

        # Plot trajectory
        ax.plot(traj[:, 1], traj[:, 2], traj[:, 3],
                label=f"{label} ({len(traj)} poses)", color=color, linewidth=1.5)

        # Mark start and end
        ax.scatter(*traj[0, 1:4], color=color, s=100, marker='o', edgecolors='black')
        ax.scatter(*traj[-1, 1:4], color=color, s=100, marker='s', edgecolors='black')

        # Print stats
        positions = traj[:, 1:4]
        print(f"\n{label}:")
        print(f"  Poses: {len(traj)}")
        print(f"  X range: {positions[:, 0].min():.3f} to {positions[:, 0].max():.3f}")
        print(f"  Y range: {positions[:, 1].min():.3f} to {positions[:, 1].max():.3f}")
        print(f"  Z range: {positions[:, 2].min():.3f} to {positions[:, 2].max():.3f}")
        print(f"  Center: [{positions.mean(axis=0)[0]:.3f}, {positions.mean(axis=0)[1]:.3f}, {positions.mean(axis=0)[2]:.3f}]")

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    ax.set_title('Camera Trajectories')

    # Equal aspect ratio
    max_range = 0
    for filepath in args.files:
        traj = load_tum_trajectory(filepath)
        if traj is not None:
            positions = traj[:, 1:4]
            max_range = max(max_range, np.ptp(positions, axis=0).max())

    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
