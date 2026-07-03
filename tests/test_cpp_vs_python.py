import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
UAIBOT_PACKAGE = ROOT / "UAIbotPy" / "uaibot"

for path in (ROOT, SRC, UAIBOT_PACKAGE):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

sys.modules.setdefault("uaibot", types.ModuleType("uaibot"))

import uaibot_cpp_bind as se3_cpp
import SE3_utilities as py_se3
import rrt_SE3 as py_rrt


def as_array(x):
    return np.asarray(x, dtype=float)


def assert_close(cpp_value, py_value, tol=1e-9):
    cpp_arr = as_array(cpp_value)
    py_arr = as_array(py_value)

    if cpp_arr.shape != py_arr.shape and cpp_arr.size == py_arr.size:
        cpp_arr = cpp_arr.reshape(-1)
        py_arr = py_arr.reshape(-1)

    assert np.allclose(cpp_arr, py_arr, atol=tol, rtol=tol)


def assert_scalar_close(cpp_value, py_value, tol=1e-9):
    assert np.isclose(float(cpp_value), float(py_value), atol=tol, rtol=tol)


def make_test_poses():
    H1 = py_rrt.make_SE3(py_se3.exp_SO3([0.2, -0.1, 0.3]), [0.1, 0.2, -0.3])
    xi = py_se3.col([0.7, -0.25, 0.35, 0.25, -0.15, 0.45], 6)
    H2 = H1 * py_se3.exp_SE3(py_se3.hat_SE3(xi))
    return H1, H2


def test_skew_and_vee_so3_cpp_vs_python():
    w_cases = [
        [1.0, 2.0, 3.0],
        np.array([[0.2, -0.4, 0.8]]),
        np.array([[1e-9], [-2e-9], [3e-9]]),
    ]

    for w in w_cases:
        W_cpp = se3_cpp.skew(w)
        W_py = py_se3.skew(w)
        assert_close(W_cpp, W_py)
        assert_close(se3_cpp.vee_so3(W_cpp), py_se3.vee_so3(W_py))


def test_exp_and_log_so3_cpp_vs_python():
    axis = np.array([0.3, -0.4, 0.5], dtype=float)
    axis = axis / np.linalg.norm(axis)
    phi_cases = [
        np.array([1e-9, -2e-9, 3e-9]),
        np.array([0.3, -0.2, 0.5]),
        axis * (np.pi - 5e-7),
    ]

    for phi in phi_cases:
        R_cpp = se3_cpp.exp_SO3(phi)
        R_py = py_se3.exp_SO3(phi)
        assert_close(R_cpp, R_py, tol=1e-9)
        assert_close(se3_cpp.log_SO3(R_py), py_se3.log_SO3(R_py), tol=1e-8)


def test_jacobians_cpp_vs_python():
    phi_cases = [
        [1e-9, -2e-9, 3e-9],
        [0.2, -0.4, 0.1],
        [0.8, 0.1, -0.4],
    ]

    for phi in phi_cases:
        assert_close(se3_cpp.jac_left_SO3(phi), py_se3.jac_left_SO3(phi))
        assert_close(se3_cpp.inv_jac_left_SO3(phi), py_se3.inv_jac_left_SO3(phi))


def test_make_and_inverse_se3_cpp_vs_python():
    R = py_se3.exp_SO3([0.2, -0.3, 0.4])
    p = [1.0, 2.0, -1.0]

    H_cpp = se3_cpp.make_SE3(R, p)
    H_py = py_rrt.make_SE3(R, p)

    assert_close(H_cpp, H_py)
    assert_close(se3_cpp.inv_SE3(H_py), py_se3.inv_SE3(H_py))


def test_hat_vee_exp_log_se3_cpp_vs_python():
    xi_cases = [
        [1.0, 2.0, 3.0, 0.1, -0.2, 0.3],
        [0.5, -0.3, 0.2, 1e-9, -2e-9, 3e-9],
        [0.25, -0.15, 0.4, 0.3, -0.4, 0.5],
    ]

    for xi in xi_cases:
        A_cpp = se3_cpp.hat_SE3(xi)
        A_py = py_se3.hat_SE3(xi)
        H_py = py_se3.exp_SE3(A_py)

        assert_close(A_cpp, A_py)
        assert_close(se3_cpp.vee_SE3(A_cpp), py_se3.vee_SE3(A_py))
        assert_close(se3_cpp.exp_SE3(A_py), H_py)
        assert_close(se3_cpp.log_SE3(H_py), py_se3.log_SE3(H_py), tol=1e-8)


def test_metrics_cpp_vs_python():
    H1, H2 = make_test_poses()
    ell = 0.2
    G_py = py_rrt.make_metric_matrix(ell)
    G_cpp = se3_cpp.make_metric_matrix(ell)

    assert_close(G_cpp, G_py)
    assert_scalar_close(
        se3_cpp.metric_log_SE3_left(H1, H2, G_cpp),
        py_rrt.metric_log_SE3_left(H1, H2, G_py),
    )
    assert_scalar_close(
        se3_cpp.metric_log_SE3_right(H1, H2, G_cpp),
        py_rrt.metric_log_SE3_right(H1, H2, G_py),
    )
    assert_scalar_close(
        se3_cpp.metric_log_SE3_symmetric(H1, H2, G_cpp),
        py_rrt.metric_log_SE3_symmetric(H1, H2, G_py),
    )


def test_transform_and_object_point_metric_cpp_vs_python():
    H1, H2 = make_test_poses()
    q = [0.1, -0.2, 0.3]
    points = [
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1],
    ]

    assert_close(se3_cpp.transform_point(H1, q), py_rrt.transform_point(H1, q))
    assert_scalar_close(
        se3_cpp.metric_object_points(H1, H2, points),
        py_rrt.metric_object_points(H1, H2, points),
    )


@pytest.mark.parametrize("s", [0.0, 0.5, 1.0])
def test_interpolate_se3_left_cpp_vs_python(s):
    H1, H2 = make_test_poses()
    assert_close(
        se3_cpp.interpolate_SE3_left(H1, H2, s),
        py_rrt.interpolate_SE3_left(H1, H2, s),
        tol=1e-8,
    )


@pytest.mark.parametrize("s", [0.0, 0.5, 1.0])
def test_interpolate_se3_right_cpp_vs_python(s):
    H1, H2 = make_test_poses()
    assert_close(
        se3_cpp.interpolate_SE3_right(H1, H2, s),
        py_rrt.interpolate_SE3_right(H1, H2, s),
        tol=1e-8,
    )


def test_interpolate_se3_path_cpp_vs_python():
    H1, H2 = make_test_poses()
    cpp_path = se3_cpp.interpolate_SE3_path(H1, H2, 5)
    py_path = py_rrt.interpolate_SE3_path(H1, H2, 5)

    assert len(cpp_path) == len(py_path)
    for H_cpp, H_py in zip(cpp_path, py_path):
        assert_close(H_cpp, H_py, tol=1e-8)


def test_steer_se3_cpp_vs_python_and_step_size():
    H1, H2 = make_test_poses()
    G = se3_cpp.make_metric_matrix(0.2)
    step_size = 0.2

    H_new_cpp = se3_cpp.steer_SE3(H1, H2, step_size, G)
    H_new_py = py_rrt.steer_SE3(H1, H2, step_size, py_rrt.make_metric_matrix(0.2))

    assert_close(H_new_cpp, H_new_py, tol=1e-8)
    assert_scalar_close(
        se3_cpp.metric_log_SE3_left(H1, H_new_cpp, G),
        step_size,
        tol=1e-8,
    )


def test_quat_to_rot_cpp_vs_python():
    q_cases = [
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.2, -0.3, 0.1],
        np.array([[0.5], [-0.5], [0.5], [-0.5]]),
    ]

    for q in q_cases:
        assert_close(se3_cpp.quat_to_rot(q), py_rrt.quat_to_rot(q))


def test_sample_so3_uniform_properties():
    for _ in range(100):
        R = as_array(se3_cpp.sample_SO3_uniform())
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9, rtol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9, rtol=1e-9)


def test_sample_se3_uniform_box_properties():
    bounds = np.array([
        [-1.0, 2.0],
        [-3.0, 4.0],
        [0.5, 1.5],
    ])

    for _ in range(100):
        H = as_array(se3_cpp.sample_SE3_uniform_box(bounds))
        R = H[:3, :3]
        p = H[:3, 3]

        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9, rtol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9, rtol=1e-9)
        assert np.all(p >= bounds[:, 0])
        assert np.all(p <= bounds[:, 1])
