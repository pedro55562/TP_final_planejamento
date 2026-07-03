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
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)

    loaded_cpp = sys.modules.get("uaibot_cpp_bind")
    loaded_cpp_path = Path(getattr(loaded_cpp, "__file__", "")).resolve() if loaded_cpp else None
    if loaded_cpp_path is None or loaded_cpp_path.parent != UAIBOT_DIR.resolve():
        sys.modules.pop("uaibot_cpp_bind", None)

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


def sphere(x=0.0, y=0.0, z=0.0, radius=0.08):
    return ub_cpp.CPP_GeometricPrimitives.create_sphere(trn(x, y, z), radius)


def plan(
    start,
    goal,
    obstacles,
    max_iterations=2500,
    shortcut_iterations=50,
    edge_resolution=0.04,
    output_resolution=0.02,
):
    return Utils.rrt_se3_bidirectional(
        h_start=start,
        h_goal=goal,
        position_bounds=np.array([[-1.1, 1.1], [-0.9, 0.9], [-0.15, 0.15]], dtype=float),
        ell=0.15,
        robot_model=[sphere(radius=0.08)],
        obstacles=obstacles,
        max_iterations=max_iterations,
        step_size=0.25,
        goal_tolerance=0.06,
        edge_resolution=edge_resolution,
        output_resolution=output_resolution,
        connect_resolution=0.04,
        goal_bias=0.08,
        other_tree_bias=0.30,
        shortcut_iterations=shortcut_iterations,
        collision_tol=1e-4,
        collision_dist_tol=1e-3,
        collision_no_iter_max=20,
    )


def assert_pose_free(H, robot_model, obstacles):
    result = Utils.check_rigid_body_collision(
        robot_model=robot_model,
        htm=H,
        obstacles=obstacles,
        tol=1e-4,
        dist_tol=1e-3,
        no_iter_max=20,
    )
    assert result.is_free is True


def assert_path_is_free(path, robot_model, obstacles):
    assert len(path) > 0
    for H in path:
        assert_pose_free(H, robot_model, obstacles)


def test_rrt_se3_without_obstacles_succeeds_and_hits_endpoints():
    H_start = trn(-0.4, 0.0, 0.0)
    H_goal = trn(0.4, 0.2, 0.0)

    result = plan(H_start, H_goal, obstacles=[], max_iterations=100)

    assert result.success is True
    assert len(result.path_discrete) >= 2
    assert np.allclose(result.path_discrete[0], H_start, atol=1e-9)
    assert np.allclose(result.path_discrete[-1], H_goal, atol=1e-9)
    assert result.total_iterations == result.iterations
    assert result.execution_time >= 0.0
    assert result.planning_time >= 0.0
    assert result.shortcut_time >= 0.0
    assert result.discretization_time >= 0.0
    assert result.raw_path_size >= result.shortcut_path_size
    assert result.shortcut_path_size == len(result.path)
    assert result.discrete_path_size == len(result.path_discrete)


def test_rrt_se3_with_simple_obstacle_returns_free_path():
    H_start = trn(-0.8, 0.0, 0.0)
    H_goal = trn(0.8, 0.0, 0.0)
    obstacles = [sphere(radius=0.25)]
    robot_model = [sphere(radius=0.08)]

    result = plan(H_start, H_goal, obstacles=obstacles, max_iterations=3500)

    assert result.success is True
    assert len(result.path_discrete) >= 2
    assert_path_is_free(result.path_discrete, robot_model, obstacles)


def test_rrt_se3_discrete_edges_are_pose_free():
    H_start = trn(-0.8, 0.0, 0.0)
    H_goal = trn(0.8, 0.0, 0.0)
    obstacles = [sphere(radius=0.25)]
    robot_model = [sphere(radius=0.08)]

    result = plan(H_start, H_goal, obstacles=obstacles, max_iterations=3500)

    assert result.success is True
    for H_a, H_b in zip(result.path_discrete[:-1], result.path_discrete[1:]):
        assert_pose_free(H_a, robot_model, obstacles)
        assert_pose_free(H_b, robot_model, obstacles)


def test_rrt_se3_shortcut_does_not_increase_waypoints_and_stays_free():
    H_start = trn(-0.4, 0.0, 0.0)
    H_goal = trn(0.4, 0.2, 0.0)
    robot_model = [sphere(radius=0.08)]

    no_shortcut = plan(H_start, H_goal, obstacles=[], max_iterations=100, shortcut_iterations=0)
    with_shortcut = plan(H_start, H_goal, obstacles=[], max_iterations=100, shortcut_iterations=50)

    assert no_shortcut.success is True
    assert with_shortcut.success is True
    assert len(with_shortcut.path) <= len(no_shortcut.path)
    assert_path_is_free(with_shortcut.path_discrete, robot_model, [])


def test_rrt_se3_output_resolution_controls_returned_discretization():
    H_start = trn(-0.4, 0.0, 0.0)
    H_goal = trn(0.4, 0.2, 0.0)

    coarse = plan(
        H_start,
        H_goal,
        obstacles=[],
        max_iterations=100,
        edge_resolution=0.04,
        output_resolution=0.20,
    )
    fine = plan(
        H_start,
        H_goal,
        obstacles=[],
        max_iterations=100,
        edge_resolution=0.04,
        output_resolution=0.05,
    )
    same_output_different_edge = plan(
        H_start,
        H_goal,
        obstacles=[],
        max_iterations=100,
        edge_resolution=0.20,
        output_resolution=0.05,
    )

    assert coarse.success is True
    assert fine.success is True
    assert same_output_different_edge.success is True
    assert len(fine.path_discrete) > len(coarse.path_discrete)
    assert len(same_output_different_edge.path_discrete) == len(fine.path_discrete)
    assert fine.discrete_path_size == len(fine.path_discrete)


def test_rrt_se3_failure_with_zero_iterations_and_blocked_direct_edge():
    H_start = trn(-0.8, 0.0, 0.0)
    H_goal = trn(0.8, 0.0, 0.0)
    obstacles = [sphere(radius=0.25)]

    result = plan(H_start, H_goal, obstacles=obstacles, max_iterations=0)

    assert result.success is False
    assert result.message == "Maximum number of iterations reached."
    assert result.path == []
    assert result.path_discrete == []
    assert result.total_iterations == 0
    assert result.raw_path_size == 0
    assert result.shortcut_path_size == 0
    assert result.discrete_path_size == 0
