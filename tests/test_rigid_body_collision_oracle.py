import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UAIBOT_DIR = ROOT / "UAIbotPy" / "uaibot"


def load_utils_module():
    path_str = str(UAIBOT_DIR)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

    os.environ["CPP_SO_FOUND"] = "1"

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


def call_oracle(robot_model, H_robot, obstacles, dist_tol=1e-3):
    return Utils.check_rigid_body_collision(
        robot_model=robot_model,
        htm=H_robot,
        obstacles=obstacles,
        tol=1e-4,
        dist_tol=dist_tol,
        no_iter_max=20,
    )


def test_rigid_body_collision_free_case():
    robot_model = [sphere(radius=0.2)]
    obstacles = [sphere(x=2.0, radius=0.2)]

    result = call_oracle(robot_model, trn(), obstacles)

    assert result.is_free is True
    assert result.type == 0
    assert result.robot_object_index == -1
    assert result.obstacle_index == -1
    assert result.info == []


def test_rigid_body_collision_detects_overlap():
    robot_model = [sphere(radius=0.2)]
    obstacles = [sphere(radius=0.2)]

    result = call_oracle(robot_model, trn(), obstacles)

    assert result.is_free is False
    assert result.type == 1
    assert result.robot_object_index == 0
    assert result.obstacle_index == 0
    assert result.info == [1, 0, 0]
    assert "Collision between robot object 0 and obstacle 0." == result.message


def test_rigid_body_collision_changes_with_robot_transform():
    robot_model = [sphere(radius=0.2)]
    obstacles = [sphere(x=1.0, radius=0.2)]

    free_result = call_oracle(robot_model, trn(), obstacles)
    collision_result = call_oracle(robot_model, trn(x=1.0), obstacles)

    assert free_result.is_free is True
    assert collision_result.is_free is False
    assert collision_result.type == 1


def test_rigid_body_collision_does_not_modify_original_objects():
    robot_model = [sphere(x=0.1, radius=0.2)]
    obstacles = [sphere(x=1.0, radius=0.2)]
    H_robot = trn(x=0.5)

    robot_htm_before = np.array(robot_model[0].htm, dtype=float)
    obstacle_htm_before = np.array(obstacles[0].htm, dtype=float)

    call_oracle(robot_model, H_robot, obstacles)

    assert np.allclose(np.array(robot_model[0].htm, dtype=float), robot_htm_before)
    assert np.allclose(np.array(obstacles[0].htm, dtype=float), obstacle_htm_before)


def test_rigid_body_collision_min_distance_values():
    robot_model = [sphere(radius=0.2)]

    free_result = call_oracle(robot_model, trn(), [sphere(x=2.0, radius=0.2)])
    collision_result = call_oracle(robot_model, trn(), [sphere(radius=0.2)])

    assert free_result.is_free is True
    assert free_result.min_distance > 0.0

    assert collision_result.is_free is False
    assert collision_result.min_distance < 1e-3
