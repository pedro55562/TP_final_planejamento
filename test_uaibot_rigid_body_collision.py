"""
Basic smoke tests for the rigid-body collision oracle embedded in UAIBot.

How to compile the UAIBot pybind module into the uaibot package folder:

    cmake -S UAIbotPy/uaibot/c_implementation \
          -B UAIbotPy/uaibot/c_implementation/build \
          -DPython3_EXECUTABLE="$(which python3)" \
          -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$PWD/UAIbotPy/uaibot"
    cmake --build UAIbotPy/uaibot/c_implementation/build --target uaibot_cpp_bind -j

Then run this file from the project root:

    PYTHONPATH="$PWD/UAIbotPy:$PWD/UAIbotPy/uaibot" python3 test_uaibot_rigid_body_collision.py

This script intentionally uses the Python high-level API:

    Utils.check_rigid_body_collision(...)

It does not call the C++ collision function directly. The small helper below
loads only uaibot.utils.utils so the script can run even in minimal environments
where optional visualization dependencies are not installed.
"""

import importlib.util
import os
from pathlib import Path
import sys
import types

import numpy as np


ROOT = Path(__file__).resolve().parent
UAIBOT_DIR = ROOT / "UAIbotPy" / "uaibot"


def load_utils_module():
    uaibot_path = str(UAIBOT_DIR)
    if uaibot_path not in sys.path:
        sys.path.insert(0, uaibot_path)

    os.environ["CPP_SO_FOUND"] = "1"

    # Minimal stubs for optional package-level dependencies not needed here.
    if "colour" not in sys.modules:
        colour_stub = types.ModuleType("colour")
        colour_stub.Color = object
        sys.modules["colour"] = colour_stub
    sys.modules.setdefault("quadprog", types.ModuleType("quadprog"))

    uaibot_pkg = sys.modules.setdefault("uaibot", types.ModuleType("uaibot"))
    uaibot_pkg.__path__ = [str(UAIBOT_DIR)]

    utils_pkg = sys.modules.setdefault("uaibot.utils", types.ModuleType("uaibot.utils"))
    utils_pkg.__path__ = [str(UAIBOT_DIR / "utils")]

    spec = importlib.util.spec_from_file_location(
        "uaibot.utils.utils",
        UAIBOT_DIR / "utils" / "utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


utils_module = load_utils_module()
Utils = utils_module.Utils
ub_cpp = utils_module.ub_cpp


def trn(x=0.0, y=0.0, z=0.0):
    H = np.eye(4, dtype=float)
    H[:3, 3] = [x, y, z]
    return H


def sphere(x=0.0, y=0.0, z=0.0, radius=0.2):
    return ub_cpp.CPP_GeometricPrimitives.create_sphere(trn(x, y, z), radius)


def check(robot_model, H_robot, obstacles, dist_tol=1e-3):
    return Utils.check_rigid_body_collision(
        robot_model=robot_model,
        htm=H_robot,
        obstacles=obstacles,
        tol=1e-4,
        dist_tol=dist_tol,
        no_iter_max=20,
    )


def assert_true(name, condition):
    if not condition:
        raise AssertionError(name)


def main():
    # 1. Free case: the robot body is far from the obstacle.
    robot_model = [sphere(radius=0.2)]
    far_obstacles = [sphere(x=2.0, radius=0.2)]
    free_result = check(robot_model, trn(), far_obstacles)

    assert_true("free case should be collision-free", free_result.is_free is True)
    assert_true("free case type should be 0", free_result.type == 0)
    assert_true("free case min_distance should be positive", free_result.min_distance > 0.0)

    # 2. Collision case: robot and obstacle overlap.
    overlapping_obstacles = [sphere(radius=0.2)]
    collision_result = check(robot_model, trn(), overlapping_obstacles)

    assert_true("overlap should collide", collision_result.is_free is False)
    assert_true("collision type should be 1", collision_result.type == 1)
    assert_true("collision min_distance should be below dist_tol", collision_result.min_distance < 1e-3)
    assert_true("collision info should identify robot/obstacle", collision_result.info == [1, 0, 0])

    # 3. H_robot transform changes the answer.
    obstacle_at_x1 = [sphere(x=1.0, radius=0.2)]
    before_motion = check(robot_model, trn(), obstacle_at_x1)
    after_motion = check(robot_model, trn(x=1.0), obstacle_at_x1)

    assert_true("robot before motion should be free", before_motion.is_free is True)
    assert_true("robot after motion should collide", after_motion.is_free is False)

    # 4. Inputs are not modified.
    local_robot = [sphere(x=0.1, radius=0.2)]
    fixed_obstacles = [sphere(x=1.0, radius=0.2)]
    robot_htm_before = np.array(local_robot[0].htm, dtype=float)
    obstacle_htm_before = np.array(fixed_obstacles[0].htm, dtype=float)

    check(local_robot, trn(x=0.5), fixed_obstacles)

    assert_true(
        "robot model primitive HTM should not be modified",
        np.allclose(np.array(local_robot[0].htm, dtype=float), robot_htm_before),
    )
    assert_true(
        "obstacle primitive HTM should not be modified",
        np.allclose(np.array(fixed_obstacles[0].htm, dtype=float), obstacle_htm_before),
    )

    print("All basic rigid-body collision oracle smoke tests passed.")


if __name__ == "__main__":
    main()
