"""
Simple UAIBot demo for interpolation between two poses in SE(3).

Run from the project root:

    python3 examples/se3_interpolation_frames_demo.py

Choose how many intermediate frames are shown:

    python3 examples/se3_interpolation_frames_demo.py --points 8

The script saves:

    outputs/se3_interpolation_frames.html
"""

from pathlib import Path
import argparse
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEARCH_PATHS = [
    PROJECT_ROOT / "UAIbotPy" / "uaibot",
    PROJECT_ROOT / "UAIbotPy" / "uaibot" / "c_implementation" / "build",
    PROJECT_ROOT / "UAIbotPy",
]

for path in reversed(SEARCH_PATHS):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import uaibot as ub
import uaibot_cpp_bind as ub_cpp


def make_demo_poses():
    """Create two poses with different positions and orientations."""
    H_start = np.eye(4)
    H_start[:3, 3] = np.array([-0.45, -0.30, 0.20])

    R_goal = ub_cpp.exp_SO3(np.array([0.70, -0.35, 1.10]))
    p_goal = np.array([0.55, 0.35, 0.65])
    H_goal = ub_cpp.make_SE3(R_goal, p_goal)

    return np.asarray(H_start, dtype=float), np.asarray(H_goal, dtype=float)


def interpolate_poses(H_start, H_goal, n_intermediate):
    """Return start, intermediate poses, and goal."""
    if n_intermediate < 0:
        raise ValueError("n_intermediate must be nonnegative.")

    n_total = n_intermediate + 2
    poses = []

    for i in range(n_total):
        s = i / (n_total - 1)
        H = ub_cpp.interpolate_SE3_left(H_start, H_goal, s)
        poses.append(np.asarray(H, dtype=float))

    return poses


def create_demo_simulation(poses):
    """Create a UAIBot scene with one frame per interpolated pose."""
    objects = []

    for i, H in enumerate(poses):
        if i == 0:
            name = "H_start"
            size = 0.22
            marker_color = "green"
            marker_radius = 0.025
        elif i == len(poses) - 1:
            name = "H_goal"
            size = 0.22
            marker_color = "red"
            marker_radius = 0.025
        else:
            name = f"H_mid_{i:02d}"
            size = 0.13
            marker_color = "black"
            marker_radius = 0.018

        objects.append(ub.Frame(htm=H, name=name, size=size))

        marker = ub.Ball(
            htm=ub.Utils.trn(H[:3, 3]),
            name=f"p_{i:02d}",
            radius=marker_radius,
            color=marker_color,
            opacity=0.45,
        )
        objects.append(marker)

    return ub.Simulation(objects)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--points",
        type=int,
        default=15,
        help="Number of intermediate poses shown between start and goal.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs"),
        help="Directory where the UAIBot HTML file is saved.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="se3_interpolation_frames",
        help="Output HTML file name without extension.",
    )
    args = parser.parse_args()

    H_start = ub.Utils.trn([0, 0, 1])
    H_goal = H_start * ub.Utils.trn([1.3, 0.7, .3]) * ub.Utils.rotz(np.pi/3) * ub.Utils.rotx(np.pi/3)
    poses = interpolate_poses(H_start, H_goal, args.points)
    sim = create_demo_simulation(poses)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sim.save(str(output_dir), args.name)

    print(f"Intermediate points: {args.points}")
    print(f"Total frames shown: {len(poses)}")
    print(f"Saved UAIBot demo at: {output_dir / (args.name + '.html')}")


if __name__ == "__main__":
    main()
