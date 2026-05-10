"""
SO(3) and SE(3) utilities.

Conventions used in this file
-----------------------------
1. All vectors are column np.matrix objects.
   Example: a 3D vector has shape (3, 1), not (3,).

2. All matrices are np.matrix objects.
   Example: a rotation matrix has shape (3, 3), and an HTM has shape (4, 4).

3. Matrix multiplication is done with *.
   Elementwise multiplication should be done explicitly with np.multiply when needed.

4. Twists are ordered as:

       xi = [v; w]

   where v is linear velocity and w is angular velocity.
"""

import numpy as np


_EPS = 1e-12
_SMALL_ANGLE = 1e-8
_NEAR_PI = 1e-6


# =============================================================================
# Basic shape/conversion utilities
# =============================================================================


def as_matrix(A, shape=None, name="matrix"):
    """
    Convert input to a float np.matrix and optionally validate its shape.

    Parameters
    ----------
    A : array-like
        Input data to be converted into np.matrix.

    shape : tuple or None
        Expected shape. If None, no shape validation is performed.

    name : str
        Name used in the error message if the shape is invalid.

    Returns
    -------
    M : np.matrix
        Converted matrix.
    """

    M = np.matrix(A, dtype=float)

    if shape is not None and M.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {M.shape}.")

    return M



def col(v, n=None, name="vector"):
    """
    Convert input to a float column np.matrix.

    Parameters
    ----------
    v : array-like
        Input vector. It can be passed as a list, 1D array, row matrix, or column
        matrix.

    n : int or None
        Expected vector dimension. If None, the dimension is not checked.

    name : str
        Name used in the error message if the shape is invalid.

    Returns
    -------
    v_col : np.matrix
        Column vector with shape (n, 1) if n is provided.
    """

    v_col = np.matrix(v, dtype=float)

    # If the user passes a row vector, transpose it.
    if v_col.shape[0] == 1 and v_col.shape[1] > 1:
        v_col = v_col.T

    # Flatten and reshape to a column vector. This also handles lists and arrays.
    v_col = np.matrix(np.asarray(v_col, dtype=float).reshape((-1, 1)))

    if n is not None and v_col.shape != (n, 1):
        raise ValueError(f"{name} must have shape ({n}, 1), got {v_col.shape}.")

    return v_col


# =============================================================================
# SO(3) utilities
# =============================================================================


def skew(w):
    """
    Return the skew-symmetric matrix associated with a 3D vector.

    For a vector w, this function returns S(w) such that:

        S(w) * v = w x v

    Parameters
    ----------
    w : array-like, shape (3,), (1, 3), or (3, 1)
        Input vector.

    Returns
    -------
    S : np.matrix, shape (3, 3)
        Skew-symmetric matrix.
    """

    w = col(w, 3, name="w")

    wx = float(w[0, 0])
    wy = float(w[1, 0])
    wz = float(w[2, 0])

    return np.matrix([
        [0.0, -wz, wy],
        [wz, 0.0, -wx],
        [-wy, wx, 0.0],
    ])



def vee_so3(W):
    """
    Convert a 3x3 skew-symmetric matrix into its associated 3D vector.

    This is the inverse operation of skew(w), assuming W is skew-symmetric:

        vee_so3(skew(w)) = w

    Parameters
    ----------
    W : array-like, shape (3, 3)
        Skew-symmetric matrix.

    Returns
    -------
    w : np.matrix, shape (3, 1)
        Vector associated with W.
    """

    W = as_matrix(W, shape=(3, 3), name="W")

    return np.matrix([
        [W[2, 1]],
        [W[0, 2]],
        [W[1, 0]],
    ])



def exp_SO3(phi):
    """
    Compute the exponential map from so(3) to SO(3).

    The input phi is a rotation vector. Its direction is the rotation axis and
    its norm is the rotation angle in radians.

    This function uses Rodrigues' formula:

        Exp(phi) = I + sin(theta)/theta * S(phi)
                     + (1 - cos(theta))/theta^2 * S(phi)^2

    with a Taylor approximation for small angles.

    Parameters
    ----------
    phi : array-like, shape (3,), (1, 3), or (3, 1)
        Rotation vector.

    Returns
    -------
    R : np.matrix, shape (3, 3)
        Rotation matrix.
    """

    phi = col(phi, 3, name="phi")
    theta = float(np.linalg.norm(phi))
    W = skew(phi)
    I = np.matrix(np.eye(3))

    if theta < _SMALL_ANGLE:
        # First terms of Rodrigues' formula near zero:
        # Exp(phi) ≈ I + S(phi) + 1/2 S(phi)^2.
        return I + W + 0.5 * W * W

    A = np.sin(theta) / theta
    B = (1.0 - np.cos(theta)) / (theta * theta)

    return I + A * W + B * W * W



def log_SO3(R):
    """
    Compute the logarithm map from SO(3) to so(3).

    The returned vector phi satisfies approximately:

        exp_SO3(phi) = R

    The angle is chosen in the interval [0, pi]. Near pi, a more robust axis
    extraction is used to reduce numerical problems.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix.

    Returns
    -------
    phi : np.matrix, shape (3, 1)
        Rotation vector.
    """

    R = as_matrix(R, shape=(3, 3), name="R")

    # Numerical safety for arccos. Due to floating-point errors, this value may
    # be slightly outside [-1, 1].
    c = 0.5 * (float(np.trace(R)) - 1.0)
    c = float(np.clip(c, -1.0, 1.0))
    theta = float(np.arccos(c))

    if theta < _SMALL_ANGLE:
        # For small angles:
        # log(R) ≈ vee(0.5 * (R - R.T)).
        W = 0.5 * (R - R.T)
        return vee_so3(W)

    if np.pi - theta < _NEAR_PI:
        # Near pi, sin(theta) is close to zero and the standard formula becomes
        # numerically fragile. We extract the rotation axis from the diagonal.
        A = 0.5 * (R + np.matrix(np.eye(3)))

        axis = np.zeros((3, 1), dtype=float)
        axis[0, 0] = np.sqrt(max(float(A[0, 0]), 0.0))
        axis[1, 0] = np.sqrt(max(float(A[1, 1]), 0.0))
        axis[2, 0] = np.sqrt(max(float(A[2, 2]), 0.0))

        # Recover signs using off-diagonal terms.
        if float(R[2, 1] - R[1, 2]) < 0.0:
            axis[0, 0] = -axis[0, 0]
        if float(R[0, 2] - R[2, 0]) < 0.0:
            axis[1, 0] = -axis[1, 0]
        if float(R[1, 0] - R[0, 1]) < 0.0:
            axis[2, 0] = -axis[2, 0]

        axis_norm = float(np.linalg.norm(axis))

        if axis_norm < _EPS:
            # Fallback. This case is rare, but can happen for noisy rotations.
            W = (1.0 / (2.0 * np.sin(theta))) * (R - R.T)
            axis = np.asarray(vee_so3(W), dtype=float)
            axis_norm = float(np.linalg.norm(axis))

            if axis_norm < _EPS:
                axis = np.array([[1.0], [0.0], [0.0]])
            else:
                axis = axis / axis_norm
        else:
            axis = axis / axis_norm

        return np.matrix(axis * theta)

    # General case.
    W = (1.0 / (2.0 * np.sin(theta))) * (R - R.T)
    axis = vee_so3(W)

    return axis * theta



def jac_left_SO3(phi):
    """
    Compute the left Jacobian of SO(3).

    The left Jacobian appears in the exponential map of SE(3). It maps the
    translational part of a twist when computing Exp_SE3.

    Formula:

        J(phi) = I + (1 - cos(theta))/theta^2 * S(phi)
                   + (theta - sin(theta))/theta^3 * S(phi)^2

    Parameters
    ----------
    phi : array-like, shape (3,), (1, 3), or (3, 1)
        Rotation vector.

    Returns
    -------
    J : np.matrix, shape (3, 3)
        Left Jacobian of SO(3).
    """

    phi = col(phi, 3, name="phi")
    theta = float(np.linalg.norm(phi))
    W = skew(phi)
    I = np.matrix(np.eye(3))

    if theta < _SMALL_ANGLE:
        # Series expansion:
        # J(phi) ≈ I + 1/2 S(phi) + 1/6 S(phi)^2.
        return I + 0.5 * W + (1.0 / 6.0) * W * W

    theta2 = theta * theta
    A = (1.0 - np.cos(theta)) / theta2
    B = (theta - np.sin(theta)) / (theta2 * theta)

    return I + A * W + B * W * W



def inv_jac_left_SO3(phi):
    """
    Compute the inverse of the left Jacobian of SO(3).

    This matrix is useful in the logarithm map of SE(3), where it recovers the
    translational twist coordinate from the translation vector.

    Parameters
    ----------
    phi : array-like, shape (3,), (1, 3), or (3, 1)
        Rotation vector.

    Returns
    -------
    J_inv : np.matrix, shape (3, 3)
        Inverse left Jacobian of SO(3).
    """

    phi = col(phi, 3, name="phi")
    theta = float(np.linalg.norm(phi))
    W = skew(phi)
    I = np.matrix(np.eye(3))

    if theta < _SMALL_ANGLE:
        # Series expansion:
        # J^{-1}(phi) ≈ I - 1/2 S(phi) + 1/12 S(phi)^2.
        return I - 0.5 * W + (1.0 / 12.0) * W * W

    half_theta = 0.5 * theta
    cot_half_theta = np.cos(half_theta) / np.sin(half_theta)
    theta2 = theta * theta

    C = (1.0 / theta2) * (1.0 - 0.5 * theta * cot_half_theta)

    return I - 0.5 * W + C * W * W


# =============================================================================
# SE(3) utilities
# =============================================================================


def inv_SE3(H):
    """
    Compute the inverse of a homogeneous transformation matrix in SE(3).

    If:

        H = [ R  p ]
            [ 0  1 ]

    then:

        H^{-1} = [ R.T  -R.T p ]
                 [  0      1   ]

    Parameters
    ----------
    H : array-like, shape (4, 4)
        Homogeneous transformation matrix.

    Returns
    -------
    H_inv : np.matrix, shape (4, 4)
        Inverse homogeneous transformation matrix.
    """

    H = as_matrix(H, shape=(4, 4), name="H")

    R = H[0:3, 0:3]
    p = H[0:3, 3]

    H_inv = np.matrix(np.eye(4))
    H_inv[0:3, 0:3] = R.T
    H_inv[0:3, 3] = -R.T * p

    return H_inv



def hat_SE3(xi):
    """
    Convert a twist vector into an se(3) matrix.

    The convention used here is:

        xi = [v; w]

    where v is linear velocity and w is angular velocity. The corresponding
    matrix is:

        hat(xi) = [ S(w)  v ]
                  [  0    0 ]

    Parameters
    ----------
    xi : array-like, shape (6,), (1, 6), or (6, 1)
        Twist vector ordered as [v; w].

    Returns
    -------
    A : np.matrix, shape (4, 4)
        Matrix representation of the twist in se(3).
    """

    xi = col(xi, 6, name="xi")

    v = xi[0:3, 0]
    w = xi[3:6, 0]

    A = np.matrix(np.zeros((4, 4)))
    A[0:3, 0:3] = skew(w)
    A[0:3, 3] = v

    return A



def vee_SE3(A):
    """
    Convert an se(3) matrix into its twist vector.

    This is the inverse operation of hat_SE3, assuming the convention:

        xi = [v; w]

    Parameters
    ----------
    A : array-like, shape (4, 4)
        Matrix in se(3).

    Returns
    -------
    xi : np.matrix, shape (6, 1)
        Twist vector ordered as [v; w].
    """

    A = as_matrix(A, shape=(4, 4), name="A")

    v = A[0:3, 3]
    w = vee_so3(A[0:3, 0:3])

    return np.vstack((v, w))



def exp_SE3(A):
    """
    Compute the exponential map from se(3) to SE(3).

    The input A is a 4x4 matrix in se(3):

        A = [ S(w)  v ]
            [  0    0 ]

    The output is:

        Exp(A) = [ Exp(S(w))  J(w) v ]
                 [     0        1   ]

    where J(w) is the left Jacobian of SO(3).

    Parameters
    ----------
    A : array-like, shape (4, 4)
        Matrix in se(3).

    Returns
    -------
    H : np.matrix, shape (4, 4)
        Homogeneous transformation matrix in SE(3).
    """

    A = as_matrix(A, shape=(4, 4), name="A")

    w_hat = A[0:3, 0:3]
    v = A[0:3, 3]

    phi = vee_so3(w_hat)
    R = exp_SO3(phi)
    J = jac_left_SO3(phi)
    p = J * v

    H = np.matrix(np.eye(4))
    H[0:3, 0:3] = R
    H[0:3, 3] = p

    return H



def log_SE3(H):
    """
    Compute the logarithm map from SE(3) to se(3).

    Given:

        H = [ R  p ]
            [ 0  1 ]

    this function computes A in se(3) such that:

        exp_SE3(A) ≈ H

    Parameters
    ----------
    H : array-like, shape (4, 4)
        Homogeneous transformation matrix.

    Returns
    -------
    A : np.matrix, shape (4, 4)
        Matrix in se(3).
    """

    H = as_matrix(H, shape=(4, 4), name="H")

    R = H[0:3, 0:3]
    p = H[0:3, 3]

    phi = log_SO3(R)
    J_inv = inv_jac_left_SO3(phi)
    v = J_inv * p

    A = np.matrix(np.zeros((4, 4)))
    A[0:3, 0:3] = skew(phi)
    A[0:3, 3] = v

    return A


# =============================================================================
# Integration utility
# =============================================================================


def propagate_htm(htm, xi, dt_step):
    """
    Propagate a homogeneous transformation matrix using a twist-like velocity.

    Convention
    ----------
    The twist-like vector is ordered as:

        xi = [v; w]

    where:

        v : linear velocity of the frame origin expressed in the world frame
        w : angular velocity expressed in the world frame

    The update rule is:

        R_next = Exp_SO3(w * dt) * R
        p_next = p + v * dt

    This is a simple first-order integrator for the translation and a Lie-group
    update for the rotation.

    Parameters
    ----------
    htm : array-like, shape (4, 4)
        Current homogeneous transformation matrix.

    xi : array-like, shape (6,), (1, 6), or (6, 1)
        Velocity vector ordered as [v; w].

    dt_step : float
        Integration time step.

    Returns
    -------
    H_next : np.matrix, shape (4, 4)
        Next homogeneous transformation matrix.
    """

    H = as_matrix(htm, shape=(4, 4), name="htm")
    xi = col(xi, 6, name="xi")

    p = H[0:3, 3]
    R = H[0:3, 0:3]

    v = xi[0:3, 0]
    w = xi[3:6, 0]

    R_next = exp_SO3(w * dt_step) * R
    p_next = p + v * dt_step

    H_next = np.matrix(np.eye(4))
    H_next[0:3, 0:3] = R_next
    H_next[0:3, 3] = p_next

    return H_next


# =============================================================================
# Tests
# =============================================================================


def _assert_close(name, A, B, tol=1e-9):
    """
    Assert that two matrices/vectors are numerically close.

    Parameters
    ----------
    name : str
        Name of the test being checked.

    A : array-like
        First object.

    B : array-like
        Second object.

    tol : float
        Maximum allowed infinity-norm error.

    Raises
    ------
    AssertionError
        If the maximum absolute error is larger than tol.
    """

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    err = float(np.max(np.abs(A - B)))

    if err > tol:
        raise AssertionError(f"{name} failed: error = {err:.3e}, tol = {tol:.3e}")



def test_col():
    """
    Test conversion of vectors to column np.matrix objects.
    """

    v1 = col([1, 2, 3], 3)
    v2 = col(np.array([1, 2, 3]), 3)
    v3 = col(np.matrix([[1, 2, 3]]), 3)

    _assert_close("test_col v1", v1, np.matrix([[1], [2], [3]]))
    _assert_close("test_col v2", v2, np.matrix([[1], [2], [3]]))
    _assert_close("test_col v3", v3, np.matrix([[1], [2], [3]]))



def test_skew_and_vee_so3():
    """
    Test skew(w), vee_so3(W), and the cross-product identity.
    """

    w = col([1.0, 2.0, 3.0], 3)
    v = col([-2.0, 5.0, 4.0], 3)

    S = skew(w)
    w_recovered = vee_so3(S)

    cross_expected = np.matrix(np.cross(np.asarray(w).reshape(3), np.asarray(v).reshape(3))).reshape((3, 1))
    cross_computed = S * v

    _assert_close("test_skew_and_vee_so3 vee", w_recovered, w)
    _assert_close("test_skew_and_vee_so3 cross", cross_computed, cross_expected)



def test_exp_log_SO3_general():
    """
    Test that log_SO3(exp_SO3(phi)) approximately recovers phi.
    """

    phi = col([0.3, -0.2, 0.5], 3)
    R = exp_SO3(phi)
    phi_recovered = log_SO3(R)

    _assert_close("test_exp_log_SO3_general", phi_recovered, phi, tol=1e-9)



def test_exp_log_SO3_small_angle():
    """
    Test SO(3) exponential and logarithm for a very small rotation.
    """

    phi = col([1e-9, -2e-9, 3e-9], 3)
    R = exp_SO3(phi)
    phi_recovered = log_SO3(R)

    _assert_close("test_exp_log_SO3_small_angle", phi_recovered, phi, tol=1e-9)



def test_rotation_matrix_properties():
    """
    Test that exp_SO3 returns a valid rotation matrix.
    """

    phi = col([0.8, 0.1, -0.4], 3)
    R = exp_SO3(phi)

    _assert_close("test_rotation_matrix_properties orthogonality", R.T * R, np.eye(3), tol=1e-9)
    _assert_close("test_rotation_matrix_properties determinant", [[np.linalg.det(R)]], [[1.0]], tol=1e-9)



def test_left_jacobian_inverse():
    """
    Test that inv_jac_left_SO3(phi) is approximately the inverse of jac_left_SO3(phi).
    """

    phi = col([0.2, -0.4, 0.1], 3)
    J = jac_left_SO3(phi)
    J_inv = inv_jac_left_SO3(phi)

    _assert_close("test_left_jacobian_inverse", J_inv * J, np.eye(3), tol=1e-9)



def test_inv_SE3():
    """
    Test that inv_SE3(H) produces the inverse homogeneous transformation.
    """

    R = exp_SO3([0.2, -0.3, 0.4])
    p = col([1.0, 2.0, -1.0], 3)

    H = np.matrix(np.eye(4))
    H[0:3, 0:3] = R
    H[0:3, 3] = p

    H_inv = inv_SE3(H)

    _assert_close("test_inv_SE3 left", H_inv * H, np.eye(4), tol=1e-9)
    _assert_close("test_inv_SE3 right", H * H_inv, np.eye(4), tol=1e-9)



def test_hat_and_vee_SE3():
    """
    Test hat_SE3(xi) and vee_SE3(A).
    """

    xi = col([1.0, 2.0, 3.0, 0.1, -0.2, 0.3], 6)
    A = hat_SE3(xi)
    xi_recovered = vee_SE3(A)

    _assert_close("test_hat_and_vee_SE3", xi_recovered, xi, tol=1e-12)



def test_exp_log_SE3():
    """
    Test that log_SE3(exp_SE3(A)) approximately recovers A.
    """

    xi = col([0.5, -0.3, 0.2, 0.1, 0.2, -0.1], 6)
    A = hat_SE3(xi)
    H = exp_SE3(A)
    A_recovered = log_SE3(H)

    _assert_close("test_exp_log_SE3", A_recovered, A, tol=1e-9)



def test_propagate_htm_zero_velocity():
    """
    Test that zero velocity keeps the homogeneous transformation unchanged.
    """

    H = np.matrix(np.eye(4))
    xi = col([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 6)

    H_next = propagate_htm(H, xi, 0.01)

    _assert_close("test_propagate_htm_zero_velocity", H_next, H, tol=1e-12)



def test_propagate_htm_translation():
    """
    Test translation integration in propagate_htm.
    """

    H = np.matrix(np.eye(4))
    xi = col([1.0, -2.0, 3.0, 0.0, 0.0, 0.0], 6)
    dt = 0.5

    H_next = propagate_htm(H, xi, dt)

    expected = np.matrix(np.eye(4))
    expected[0:3, 3] = col([0.5, -1.0, 1.5], 3)

    _assert_close("test_propagate_htm_translation", H_next, expected, tol=1e-12)



def test_propagate_htm_rotation():
    """
    Test rotation integration in propagate_htm.
    """

    H = np.matrix(np.eye(4))
    xi = col([0.0, 0.0, 0.0, 0.0, 0.0, np.pi], 6)
    dt = 0.5

    H_next = propagate_htm(H, xi, dt)
    R_expected = exp_SO3([0.0, 0.0, np.pi * dt])

    expected = np.matrix(np.eye(4))
    expected[0:3, 0:3] = R_expected

    _assert_close("test_propagate_htm_rotation", H_next, expected, tol=1e-12)



def run_tests():
    """
    Run all tests defined in this file.

    This is intentionally lightweight and does not require pytest. Later, these
    functions can be moved to a proper tests/ directory and executed with pytest.
    """

    tests = [
        test_col,
        test_skew_and_vee_so3,
        test_exp_log_SO3_general,
        test_exp_log_SO3_small_angle,
        test_rotation_matrix_properties,
        test_left_jacobian_inverse,
        test_inv_SE3,
        test_hat_and_vee_SE3,
        test_exp_log_SE3,
        test_propagate_htm_zero_velocity,
        test_propagate_htm_translation,
        test_propagate_htm_rotation,
    ]

    for test in tests:
        test()
        print(f"[OK] {test.__name__}")

    print("\nAll tests passed.")


if __name__ == "__main__":
    run_tests()
