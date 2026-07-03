"""
Basic smoke tests for the SE(3) functions embedded in UAIBot's C++ binding.

How to compile the UAIBot pybind module into the uaibot package folder:

    cmake -S UAIbotPy/uaibot/c_implementation \
          -B UAIbotPy/uaibot/c_implementation/build \
          -DPython3_EXECUTABLE="$(which python3)" \
          -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$PWD/UAIbotPy/uaibot"
    cmake --build UAIbotPy/uaibot/c_implementation/build --target uaibot_cpp_bind -j

Then run this file from the project root:

    python3 examples/se3_cpp_smoke.py

Alternative direct CMake build without copying the module into uaibot:

    cmake -S UAIbotPy/uaibot/c_implementation \
          -B UAIbotPy/uaibot/c_implementation/build \
          -DPython3_EXECUTABLE="$(which python3)"
    cmake --build UAIbotPy/uaibot/c_implementation/build --target uaibot_cpp_bind -j

With the direct CMake build, this script also searches the CMake build folder
for uaibot_cpp_bind.
"""

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEARCH_PATHS = [
    PROJECT_ROOT / "UAIbotPy" / "uaibot",
    PROJECT_ROOT / "UAIbotPy" / "uaibot" / "c_implementation" / "build",
    PROJECT_ROOT / "UAIbotPy",
    PROJECT_ROOT / "src",
]

for path in reversed(SEARCH_PATHS):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import uaibot_cpp_bind as ub_cpp


def assert_close(name, A, B, tol=1e-9):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if not np.allclose(A, B, atol=tol, rtol=tol):
        err = float(np.max(np.abs(A - B)))
        raise AssertionError(f"{name} failed: err={err:.3e}, tol={tol:.3e}")


def assert_scalar_close(name, a, b, tol=1e-9):
    err = abs(float(a) - float(b))
    if err > tol:
        raise AssertionError(f"{name} failed: err={err:.3e}, tol={tol:.3e}")


def main():
    # SO(3): skew and vee_so3 preserve the angular vector convention.
    w = np.array([0.1, -0.2, 0.3])
    W = ub_cpp.skew(w)
    assert_close("vee_so3(skew(w))", ub_cpp.vee_so3(W), w)

    # SO(3): exp/log round trip.
    phi = np.array([0.25, -0.15, 0.4])
    R = ub_cpp.exp_SO3(phi)
    assert_close("R.T @ R", R.T @ R, np.eye(3))
    assert_scalar_close("det(R)", np.linalg.det(R), 1.0)
    assert_close("log_SO3(exp_SO3(phi))", ub_cpp.log_SO3(R), phi)

    # SE(3): xi = [v; w] is preserved by hat/vee and exp/log.
    xi = np.array([0.5, -0.2, 0.3, 0.1, -0.05, 0.2])
    A = ub_cpp.hat_SE3(xi)
    assert_close("vee_SE3(hat_SE3(xi))", ub_cpp.vee_SE3(A), xi)

    H1 = ub_cpp.make_SE3(np.eye(3), np.array([0.0, 0.0, 0.0]))
    H2 = H1 @ ub_cpp.exp_SE3(A)
    assert_close("log_SE3(exp_SE3(A))", ub_cpp.log_SE3(H2), A)
    assert_close("inv_SE3(H2) @ H2", ub_cpp.inv_SE3(H2) @ H2, np.eye(4))

    # Metric and steering: a limited step should have approximately step_size.
    G = ub_cpp.make_metric_matrix(0.2)
    full_distance = ub_cpp.metric_log_SE3_left(H1, H2, G)
    if full_distance <= 0.2:
        raise AssertionError("Test pose is too close for the steering check.")

    H_step = ub_cpp.steer_SE3(H1, H2, 0.2, G)
    step_distance = ub_cpp.metric_log_SE3_left(H1, H_step, G)
    assert_scalar_close("steer_SE3 step distance", step_distance, 0.2, tol=1e-8)

    # Interpolation endpoints.
    assert_close("interpolate s=0", ub_cpp.interpolate_SE3_left(H1, H2, 0.0), H1)
    assert_close("interpolate s=1", ub_cpp.interpolate_SE3_left(H1, H2, 1.0), H2)

    # Uniform sampling checks only SO(3) properties and position bounds.
    bounds = np.array([[-1.0, 2.0], [-3.0, 4.0], [0.5, 1.5]])
    for _ in range(20):
        H = ub_cpp.sample_SE3_uniform_box(bounds)
        R = H[:3, :3]
        p = H[:3, 3]
        assert_close("sample R.T @ R", R.T @ R, np.eye(3), tol=1e-9)
        assert_scalar_close("sample det(R)", np.linalg.det(R), 1.0, tol=1e-9)
        if not np.all((bounds[:, 0] <= p) & (p <= bounds[:, 1])):
            raise AssertionError(f"sample position out of bounds: {p}")

    print("All basic UAIBot SE(3) C++ smoke tests passed.")


if __name__ == "__main__":
    main()
