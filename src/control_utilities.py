"""
UAIbot task-space controller.

This module provides an object-oriented task-space controller that depends on
UAIbot only for robot kinematics and for ub.Utils.dp_inv.

There is no animation, no simulation loop, and no integration logic in this
class. The controller only computes the control action for the current state.

This makes it usable in:

- UAIbot simulations
- custom simulations
- ROS loops
- real robot control loops

Main usage
----------
    controller = UAIbotTaskController(robot, htm_d=htm_d)

    u, r, Jr, Fr = controller.compute_control(q)

Conventions
-----------
1. This file follows the UAIbot convention and uses np.matrix.
2. Vectors are column matrices.
3. Matrix multiplication uses *.
4. The UAIbot damped pseudoinverse ub.Utils.dp_inv is used directly.
5. The geometric Jacobian returned by UAIbot is assumed to be ordered as:

       Jg = [ Jv ]
            [ Jw ]

   where Jv maps qdot to linear velocity and Jw maps qdot to angular velocity.

Default task
------------
If the user does not pass a task function and a task Jacobian, this class uses
this default end-effector pose task:

    r = [ s_e - s_d                 ]
        [ 1 - x_d.T * x_e           ]
        [ 1 - y_d.T * y_e           ]
        [ 1 - z_d.T * z_e           ]

where:
    s_e is the current end-effector position,
    s_d is the desired end-effector position,
    x_e, y_e, z_e are the current end-effector axes,
    x_d, y_d, z_d are the desired end-effector axes.

The controller enforces the first-order task dynamics:

    rdot = F(r)

through:

    Jr(q) * qdot = F(r)
    qdot = Jr(q)^+ * F(r)
"""

import numpy as np
import uaibot as ub

class UAIbotTaskController:
    """
    Generic task-space controller for UAIbot robots.

    The class stores the controller configuration and computes one control
    action at each call.

    It implements:

        r = task_function(robot, q, Jg, fk, **task_args)
        Jr = task_jacobian(robot, q, Jg, fk, **task_args)
        Fr = F(r, **F_args)
        u = ub.Utils.dp_inv(Jr, damping) * Fr

    By default, it uses:

        task_function = default_pose_task_function
        task_jacobian = default_pose_task_jacobian
        F             = F_linear

    Therefore, the default closed-loop task dynamics are:

        rdot = -gain * r

    with gain = 1.0.
    """

    def __init__(
        self,
        robot,
        htm_d=None,
        F=None,
        F_args=None,
        task_function=None,
        task_jacobian=None,
        task_args=None,
        damping=1e-3,
    ):
        """
        Create a task-space controller.

        Parameters
        ----------
        robot : uaibot.Robot
            UAIbot robot object. The controller uses it to compute forward
            kinematics and the geometric Jacobian.

        htm_d : None or array-like, shape (4, 4)
            Desired end-effector pose for the default pose task.

            If a custom task is used, this argument is optional.

        F : None or callable
            Task dynamics function. It must receive r and return F(r).

            Expected signature:

                F(r, **F_args) -> Fr

            If None, UAIbotTaskController.F_linear is used.

        F_args : None or dict
            Extra keyword arguments passed to F.

            Example:

                F_args={"gain": 2.0}

            This calls:

                F_linear(r, gain=2.0)

        task_function : None or callable
            Function that computes the task value r.

            Expected signature:

                task_function(robot, q, Jg, fk, **task_args) -> r

            If None, the default pose task is used.

        task_jacobian : None or callable
            Function that computes the task Jacobian Jr.

            Expected signature:

                task_jacobian(robot, q, Jg, fk, **task_args) -> Jr

            If None, the default pose task Jacobian is used.

        task_args : None or dict
            Extra keyword arguments passed to task_function and task_jacobian.

            Example:

                task_args={"htm_d": htm_d}

            For the default pose task, htm_d is inserted automatically.

        damping : float
            Damping parameter used by ub.Utils.dp_inv.
        """

        self.robot = robot

        self.htm_d = None
        if htm_d is not None:
            self.htm_d = self.as_htm(htm_d, name="htm_d")

        self.F = F if F is not None else self.F_linear
        self.F_args = {} if F_args is None else dict(F_args)

        self.task_function = (
            task_function
            if task_function is not None
            else self.default_pose_task_function
        )

        self.task_jacobian = (
            task_jacobian
            if task_jacobian is not None
            else self.default_pose_task_jacobian
        )

        self.task_args = {} if task_args is None else dict(task_args)

        # If htm_d was passed directly to the constructor, insert it into
        # task_args. If the user already passed task_args["htm_d"], keep the
        # user's value.
        if self.htm_d is not None:
            self.task_args.setdefault("htm_d", self.htm_d)

        self.damping = float(damping)

    # =========================================================================
    # Public control method
    # =========================================================================

    def compute_control(self, q=None):
        """
        Compute one control action for the current robot state.

        This method does not animate, integrate, or modify the robot. It only
        computes the control command.

        Parameters
        ----------
        q : None or array-like, shape (n, 1)
            Joint configuration.

            If None, self.robot.q is used.

            In a real robot or ROS loop, q should usually come from sensors.
            In a UAIbot simulation, q can be self.robot.q.

        Returns
        -------
        u : np.matrix, shape (n, 1)
            Joint velocity command.

        r : np.matrix, shape (m, 1)
            Task value.

        Jr : np.matrix, shape (m, n)
            Task Jacobian.

        Fr : np.matrix, shape (m, 1)
            Desired task derivative F(r).
        """

        if q is None:
            q = self.robot.q

        q = self.as_col(q, name="q")

        Jg, fk = self.robot.jac_geo(q)
        Jg = np.matrix(Jg, dtype=float)
        fk = self.as_htm(fk, name="fk")

        r = self.as_col(
            self.task_function(
                self.robot,
                q,
                Jg,
                fk,
                **self.task_args,
            ),
            name="r",
        )

        Jr = np.matrix(
            self.task_jacobian(
                self.robot,
                q,
                Jg,
                fk,
                **self.task_args,
            ),
            dtype=float,
        )

        if Jr.shape[0] != r.shape[0]:
            raise ValueError(
                "Task Jacobian row dimension must match task dimension. "
                f"Got Jr.shape={Jr.shape} and r.shape={r.shape}."
            )

        if Jr.shape[1] != q.shape[0]:
            raise ValueError(
                "Task Jacobian column dimension must match number of joints. "
                f"Got Jr.shape={Jr.shape} and q.shape={q.shape}."
            )

        Fr = self.evaluate_task_dynamics(self.F, r, self.F_args)

        u = ub.Utils.dp_inv(Jr, self.damping) * Fr

        return u, r, Jr, Fr

    # =========================================================================
    # Small helpers
    # =========================================================================

    @staticmethod
    def as_col(v, n=None, name="vector"):
        """
        Convert input to a column np.matrix.

        Parameters
        ----------
        v : array-like
            Input vector.

        n : int or None
            Expected vector dimension. If None, the dimension is not checked.

        name : str
            Name used in the error message.

        Returns
        -------
        v_col : np.matrix
            Column vector.
        """

        v_col = np.matrix(v, dtype=float)

        if v_col.shape[0] == 1 and v_col.shape[1] > 1:
            v_col = v_col.T

        v_col = np.matrix(np.asarray(v_col, dtype=float).reshape((-1, 1)))

        if n is not None and v_col.shape != (n, 1):
            raise ValueError(f"{name} must have shape ({n}, 1), got {v_col.shape}.")

        return v_col

    @staticmethod
    def as_htm(H, name="htm"):
        """
        Convert input to a 4x4 homogeneous transformation np.matrix.

        Parameters
        ----------
        H : array-like, shape (4, 4)
            Homogeneous transformation matrix.

        name : str
            Name used in the error message.

        Returns
        -------
        H : np.matrix, shape (4, 4)
            Homogeneous transformation matrix.
        """

        H = np.matrix(H, dtype=float)

        if H.shape != (4, 4):
            raise ValueError(f"{name} must have shape (4, 4), got {H.shape}.")

        return H

    @staticmethod
    def scalar(x):
        """
        Convert a 1x1 matrix-like value to float.

        Parameters
        ----------
        x : scalar-like
            Value to convert.

        Returns
        -------
        value : float
            Scalar value.
        """

        return float(np.asarray(x).reshape(-1)[0])

    @classmethod
    def evaluate_task_dynamics(cls, F, r, F_args=None):
        """
        Evaluate the task dynamics F(r).

        Parameters
        ----------
        F : callable
            Task dynamics function.

        r : np.matrix, shape (m, 1)
            Task value.

        F_args : None or dict
            Optional keyword arguments passed to F.

        Returns
        -------
        Fr : np.matrix, shape (m, 1)
            Desired task derivative.
        """

        if F_args is None:
            Fr = F(r)
        else:
            Fr = F(r, **F_args)

        return cls.as_col(Fr, r.shape[0], name="F(r)")

    # =========================================================================
    # Default pose task
    # =========================================================================

    @classmethod
    def compute_pose_task_error(cls, fk, htm_d):
        """
        Compute the default 6D pose task error.

        The task error is:

            r[0:3] = s_e - s_d
            r[3]   = 1 - x_d.T * x_e
            r[4]   = 1 - y_d.T * y_e
            r[5]   = 1 - z_d.T * z_e

        This orientation error is an axis-alignment error. It is not the
        logarithmic SO(3) error.

        Parameters
        ----------
        fk : array-like, shape (4, 4)
            Current end-effector homogeneous transformation.

        htm_d : array-like, shape (4, 4)
            Desired end-effector homogeneous transformation.

        Returns
        -------
        r : np.matrix, shape (6, 1)
            Pose task error.
        """

        fk = cls.as_htm(fk, name="fk")
        htm_d = cls.as_htm(htm_d, name="htm_d")

        x_e = fk[0:3, 0]
        y_e = fk[0:3, 1]
        z_e = fk[0:3, 2]
        s_e = fk[0:3, 3]

        x_d = htm_d[0:3, 0]
        y_d = htm_d[0:3, 1]
        z_d = htm_d[0:3, 2]
        s_d = htm_d[0:3, 3]

        r = np.matrix(np.zeros((6, 1)))
        r[0:3, 0] = s_e - s_d
        r[3, 0] = cls.scalar(1.0 - x_d.T * x_e)
        r[4, 0] = cls.scalar(1.0 - y_d.T * y_e)
        r[5, 0] = cls.scalar(1.0 - z_d.T * z_e)

        return r

    @classmethod
    def compute_pose_task_jacobian(cls, Jg, fk, htm_d):
        """
        Compute the Jacobian of the default pose task.

        The task Jacobian Jr satisfies:

            rdot = Jr(q) * qdot

        where r is the task error defined in compute_pose_task_error.

        Parameters
        ----------
        Jg : array-like, shape (6, n)
            Geometric Jacobian returned by UAIbot.

        fk : array-like, shape (4, 4)
            Current end-effector homogeneous transformation.

        htm_d : array-like, shape (4, 4)
            Desired end-effector homogeneous transformation.

        Returns
        -------
        Jr : np.matrix, shape (6, n)
            Task Jacobian.
        """

        Jg = np.matrix(Jg, dtype=float)
        fk = cls.as_htm(fk, name="fk")
        htm_d = cls.as_htm(htm_d, name="htm_d")

        if Jg.shape[0] != 6:
            raise ValueError(f"Jg must have shape (6, n), got {Jg.shape}.")

        n = Jg.shape[1]

        x_e = fk[0:3, 0]
        y_e = fk[0:3, 1]
        z_e = fk[0:3, 2]

        x_d = htm_d[0:3, 0]
        y_d = htm_d[0:3, 1]
        z_d = htm_d[0:3, 2]

        Jv = Jg[0:3, :]
        Jw = Jg[3:6, :]

        Jr = np.matrix(np.zeros((6, n)))
        Jr[0:3, :] = Jv
        Jr[3, :] = x_d.T * ub.Utils.S(x_e) * Jw
        Jr[4, :] = y_d.T * ub.Utils.S(y_e) * Jw
        Jr[5, :] = z_d.T * ub.Utils.S(z_e) * Jw

        return Jr

    @classmethod
    def default_pose_task_function(cls, robot, q, Jg, fk, htm_d, **kwargs):
        """
        Default task function wrapper used by the controller.

        Parameters
        ----------
        robot : uaibot.Robot
            UAIbot robot object. Included for a uniform custom-task interface.

        q : array-like, shape (n, 1)
            Current robot configuration. Included for a uniform custom-task
            interface.

        Jg : array-like, shape (6, n)
            Geometric Jacobian. Included for a uniform custom-task interface.

        fk : array-like, shape (4, 4)
            Current end-effector pose.

        htm_d : array-like, shape (4, 4)
            Desired end-effector pose.

        kwargs : dict
            Extra unused arguments.

        Returns
        -------
        r : np.matrix, shape (6, 1)
            Pose task error.
        """

        return cls.compute_pose_task_error(fk, htm_d)

    @classmethod
    def default_pose_task_jacobian(cls, robot, q, Jg, fk, htm_d, **kwargs):
        """
        Default task Jacobian wrapper used by the controller.

        Parameters
        ----------
        robot : uaibot.Robot
            UAIbot robot object. Included for a uniform custom-task interface.

        q : array-like, shape (n, 1)
            Current robot configuration. Included for a uniform custom-task
            interface.

        Jg : array-like, shape (6, n)
            Geometric Jacobian.

        fk : array-like, shape (4, 4)
            Current end-effector pose.

        htm_d : array-like, shape (4, 4)
            Desired end-effector pose.

        kwargs : dict
            Extra unused arguments.

        Returns
        -------
        Jr : np.matrix, shape (6, n)
            Pose task Jacobian.
        """

        return cls.compute_pose_task_jacobian(Jg, fk, htm_d)

    # =========================================================================
    # Task dynamics F(r)
    # =========================================================================

    @staticmethod
    def F_linear(r, gain=1.0):
        """
        Linear stable task dynamics.

        The dynamics are:

            F(r) = -gain * r

        Parameters
        ----------
        r : array-like, shape (m, 1)
            Task value.

        gain : float
            Positive convergence gain.

        Returns
        -------
        F : np.matrix, shape (m, 1)
            Desired task derivative.
        """

        r = UAIbotTaskController.as_col(r, name="r")
        return -float(gain) * r

    @staticmethod
    def F_componentwise_saturation(r, A=0.25, width=0.01):
        """
        Componentwise saturated task dynamics.

        This reproduces the logic:

            if abs(r_i) < width_i:
                F_i = -A_i * r_i / width_i
            elif r_i >= width_i:
                F_i = -A_i
            else:
                F_i = A_i

        Parameters
        ----------
        r : array-like, shape (m, 1)
            Task value.

        A : float or array-like, shape (m, 1)
            Maximum absolute value of each component of F.

        width : float or array-like, shape (m, 1)
            Width of the linear region around zero.

        Returns
        -------
        F : np.matrix, shape (m, 1)
            Desired task derivative.
        """

        r = UAIbotTaskController.as_col(r, name="r")
        m = r.shape[0]

        if np.isscalar(A):
            A = float(A) * np.ones((m, 1))
        else:
            A = np.asarray(UAIbotTaskController.as_col(A, m, name="A"), dtype=float)

        if np.isscalar(width):
            width = float(width) * np.ones((m, 1))
        else:
            width = np.asarray(
                UAIbotTaskController.as_col(width, m, name="width"),
                dtype=float,
            )

        F = np.matrix(np.zeros((m, 1)))

        for i in range(m):
            ri = UAIbotTaskController.scalar(r[i, 0])
            Ai = float(A[i, 0])
            wi = float(width[i, 0])

            if wi <= 0.0:
                raise ValueError("All width values must be positive.")

            if abs(ri) < wi:
                F[i, 0] = -Ai * (ri / wi)
            elif ri >= wi:
                F[i, 0] = -Ai
            else:
                F[i, 0] = Ai

        return F

    @staticmethod
    def F_tanh(r, A=0.25, width=0.01):
        """
        Smooth saturated task dynamics using tanh.

        The dynamics are:

            F_i = -A_i * tanh(r_i / width_i)

        Parameters
        ----------
        r : array-like, shape (m, 1)
            Task value.

        A : float or array-like, shape (m, 1)
            Saturation amplitude.

        width : float or array-like, shape (m, 1)
            Smoothness width. Smaller values make the function saturate faster.

        Returns
        -------
        F : np.matrix, shape (m, 1)
            Desired task derivative.
        """

        r = UAIbotTaskController.as_col(r, name="r")
        m = r.shape[0]

        if np.isscalar(A):
            A = float(A) * np.ones((m, 1))
        else:
            A = np.asarray(UAIbotTaskController.as_col(A, m, name="A"), dtype=float)

        if np.isscalar(width):
            width = float(width) * np.ones((m, 1))
        else:
            width = np.asarray(
                UAIbotTaskController.as_col(width, m, name="width"),
                dtype=float,
            )

        if np.any(width <= 0.0):
            raise ValueError("All width values must be positive.")

        F = -A * np.tanh(np.asarray(r, dtype=float) / width)

        return np.matrix(F)

    @staticmethod
    def F_normalized(r, max_norm=0.25, gain=1.0, eps=1e-12):
        """
        Norm-saturated vector field for task dynamics.

        This function first computes:

            F_raw = -gain * r

        and then saturates its Euclidean norm:

            ||F|| <= max_norm

        Parameters
        ----------
        r : array-like, shape (m, 1)
            Task value.

        max_norm : float
            Maximum norm of F.

        gain : float
            Linear gain before saturation.

        eps : float
            Small number used to avoid division by zero.

        Returns
        -------
        F : np.matrix, shape (m, 1)
            Desired task derivative.
        """

        r = UAIbotTaskController.as_col(r, name="r")

        F = -float(gain) * r
        norm_F = float(np.linalg.norm(F))

        if norm_F < eps or norm_F <= max_norm:
            return F

        return (float(max_norm) / norm_F) * F

    @staticmethod
    def F_sqrt(r, gain=1.0, eps=1e-6):
        """
        Nonlinear stabilizing task dynamics based on square-root scaling.

        The function is applied componentwise:

            F_i(r) = -gain * r_i / (sqrt(abs(r_i)) + eps)

        Parameters
        ----------
        r : array-like, shape (m, 1)
            Task value.

        gain : float
            Positive gain.

        eps : float
            Small value used to avoid division by zero.

        Returns
        -------
        F : np.matrix, shape (m, 1)
            Desired task derivative.
        """

        r = UAIbotTaskController.as_col(r, name="r")

        r_array = np.asarray(r, dtype=float)
        denominator = np.sqrt(np.abs(r_array)) + float(eps)

        F = -float(gain) * r_array / denominator

        return np.matrix(F)

    # =========================================================================
    # Catalogs
    # =========================================================================

    @classmethod
    def available_task_dynamics(cls):
        """
        Return a dictionary describing the predefined task dynamics functions.

        Returns
        -------
        dynamics : dict
            Dictionary where each key is a short name and each value contains
            the function object and a text description.
        """

        return {
            "linear": {
                "function": cls.F_linear,
                "description": "F(r) = -gain*r. Simple exponential convergence.",
            },
            "componentwise_saturation": {
                "function": cls.F_componentwise_saturation,
                "description": (
                    "Piecewise componentwise saturation. Far from zero, each "
                    "component has almost constant speed; near zero, it becomes linear."
                ),
            },
            "tanh": {
                "function": cls.F_tanh,
                "description": (
                    "Smooth saturation using -A*tanh(r/width). Similar to the "
                    "piecewise saturation, but differentiable."
                ),
            },
            "normalized": {
                "function": cls.F_normalized,
                "description": (
                    "Vector-norm saturation. Preserves the direction of -r while "
                    "limiting the norm of F(r)."
                ),
            },
            "sqrt": {
                "function": cls.F_sqrt,
                "description": (
                    "Nonlinear componentwise field: "
                    "F_i(r) = -gain*r_i/(sqrt(abs(r_i)) + eps)."
                ),
            },
        }

    @classmethod
    def available_task_functions(cls):
        """
        Return a dictionary describing the predefined task functions.

        Returns
        -------
        tasks : dict
            Dictionary where each key is a short name and each value contains
            the task function, task Jacobian, and a text description.
        """

        return {
            "pose_axis_alignment": {
                "task_function": cls.default_pose_task_function,
                "task_jacobian": cls.default_pose_task_jacobian,
                "description": (
                    "Default 6D end-effector pose task: position error plus "
                    "axis-alignment orientation error."
                ),
            },
        }


# =============================================================================
# Tests
# =============================================================================


def _assert_close(name, A, B, tol=1e-9):
    """
    Assert that two matrices/vectors are numerically close.
    """

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    err = float(np.max(np.abs(A - B)))

    if err > tol:
        raise AssertionError(f"{name} failed: error = {err:.3e}, tol = {tol:.3e}")



def _assert_finite(name, A):
    """
    Assert that all entries of an array-like object are finite.
    """

    A = np.asarray(A, dtype=float)

    if not np.all(np.isfinite(A)):
        raise AssertionError(f"{name} failed: array contains NaN or infinity.")



def test_F_linear():
    """
    Test the linear task dynamics.
    """

    r = np.matrix([[1.0], [-2.0], [3.0]])
    Fr = UAIbotTaskController.F_linear(r, gain=2.0)
    expected = np.matrix([[-2.0], [4.0], [-6.0]])

    _assert_close("test_F_linear", Fr, expected)



def test_F_componentwise_saturation():
    """
    Test the componentwise saturated task dynamics.
    """

    r = np.matrix([[0.0], [0.005], [0.02], [-0.02]])
    Fr = UAIbotTaskController.F_componentwise_saturation(r, A=0.25, width=0.01)
    expected = np.matrix([[0.0], [-0.125], [-0.25], [0.25]])

    _assert_close("test_F_componentwise_saturation", Fr, expected)



def test_F_sqrt():
    """
    Test the square-root task dynamics.
    """

    r = np.matrix([[0.0], [0.25], [-0.25]])
    Fr = UAIbotTaskController.F_sqrt(r, gain=2.0, eps=1e-6)

    expected = np.matrix([
        [0.0],
        [-2.0 * 0.25 / (np.sqrt(0.25) + 1e-6)],
        [2.0 * 0.25 / (np.sqrt(0.25) + 1e-6)],
    ])

    _assert_close("test_F_sqrt", Fr, expected)



def test_available_catalogs():
    """
    Test that the catalogs contain the expected entries.
    """

    dynamics = UAIbotTaskController.available_task_dynamics()
    tasks = UAIbotTaskController.available_task_functions()

    assert "linear" in dynamics
    assert "componentwise_saturation" in dynamics
    assert "tanh" in dynamics
    assert "normalized" in dynamics
    assert "sqrt" in dynamics
    assert "pose_axis_alignment" in tasks



def test_default_pose_task_controller():
    """
    Test the controller using the default pose task.

    This computes only one control action. It does not animate, integrate, or
    run a simulation.
    """

    robot = ub.Robot.create_kuka_kr5()
    htm_d = ub.Utils.trn([-0.3, 0.2, -0.3]) * robot.fkm()

    controller = UAIbotTaskController(
        robot=robot,
        htm_d=htm_d,
        F=UAIbotTaskController.F_componentwise_saturation,
        F_args={
            "A": [0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
            "width": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        },
        damping=1e-3,
    )

    u, r, Jr, Fr = controller.compute_control()

    n = robot.q.shape[0]

    assert u.shape == (n, 1)
    assert r.shape == (6, 1)
    assert Jr.shape == (6, n)
    assert Fr.shape == (6, 1)

    _assert_finite("test_default_pose_task_controller u", u)
    _assert_finite("test_default_pose_task_controller r", r)
    _assert_finite("test_default_pose_task_controller Jr", Jr)
    _assert_finite("test_default_pose_task_controller Fr", Fr)



def custom_pose_task_function(robot, q, Jg, fk, htm_d, **kwargs):
    """
    Custom task function used only for testing the custom-task interface.
    """

    return UAIbotTaskController.compute_pose_task_error(fk, htm_d)



def custom_pose_task_jacobian(robot, q, Jg, fk, htm_d, **kwargs):
    """
    Custom task Jacobian used only for testing the custom-task interface.
    """

    return UAIbotTaskController.compute_pose_task_jacobian(Jg, fk, htm_d)



def test_custom_task_controller():
    """
    Test the controller with explicitly passed task_function and task_jacobian.
    """

    robot = ub.Robot.create_kuka_kr5()
    htm_d = ub.Utils.trn([-0.3, 0.2, -0.3]) * robot.fkm()

    controller = UAIbotTaskController(
        robot=robot,
        task_function=custom_pose_task_function,
        task_jacobian=custom_pose_task_jacobian,
        task_args={"htm_d": htm_d},
        F=UAIbotTaskController.F_linear,
        F_args={"gain": 1.0},
        damping=1e-3,
    )

    u, r, Jr, Fr = controller.compute_control()

    n = robot.q.shape[0]

    assert u.shape == (n, 1)
    assert r.shape == (6, 1)
    assert Jr.shape == (6, n)
    assert Fr.shape == (6, 1)

    _assert_finite("test_custom_task_controller u", u)
    _assert_finite("test_custom_task_controller r", r)
    _assert_finite("test_custom_task_controller Jr", Jr)
    _assert_finite("test_custom_task_controller Fr", Fr)



def test_short_control_loop_without_animation():
    """
    Test a short control loop without animation or simulation.

    This manually integrates q outside the controller to show that the class is
    independent of UAIbot animation.
    """

    dt = 0.01
    steps = 5

    robot = ub.Robot.create_kuka_kr5()
    htm_d = ub.Utils.trn([-0.3, 0.2, -0.3]) * robot.fkm()

    controller = UAIbotTaskController(
        robot=robot,
        htm_d=htm_d,
        F=UAIbotTaskController.F_linear,
        F_args={"gain": 1.0},
        damping=1e-3,
    )

    q = robot.q

    hist_r = np.matrix(np.zeros((6, 0)))
    hist_u = np.matrix(np.zeros((robot.q.shape[0], 0)))

    for _ in range(steps):
        u, r, Jr, Fr = controller.compute_control(q=q)

        hist_r = np.block([hist_r, r])
        hist_u = np.block([hist_u, u])

        # Integration is external to the controller.
        q = q + u * dt

    assert hist_r.shape == (6, steps)
    assert hist_u.shape == (robot.q.shape[0], steps)

    _assert_finite("test_short_control_loop_without_animation hist_r", hist_r)
    _assert_finite("test_short_control_loop_without_animation hist_u", hist_u)



def run_tests():
    """
    Run all tests defined in this file.
    """

    tests = [
        test_F_linear,
        test_F_componentwise_saturation,
        test_F_sqrt,
        test_available_catalogs,
        test_default_pose_task_controller,
        test_custom_task_controller,
        test_short_control_loop_without_animation,
    ]

    for test in tests:
        test()
        print(f"[OK] {test.__name__}")

    print("\nAll tests passed.")


def demo_pose_regulation_visualization():
    """
    Run a visual demo of the default pose-regulation controller.

    This demo uses UAIbotTaskController only to compute the control action.
    The integration and animation are handled outside the controller.
    """
    import matplotlib.pyplot as plt

    dt = 0.01
    t = 0.0
    tmax = 6.0

    robot = ub.Robot.create_kuka_kr5()

    # Desired pose, same style as the original example.
    htm_d = ub.Utils.trn([-0.3, 0.2, -0.3]) * robot.fkm()

    # Desired frame visualization.
    frame_d = ub.Frame(htm=htm_d)

    sim = ub.Simulation([robot, frame_d])

    controller = UAIbotTaskController(
        robot=robot,
        htm_d=htm_d,
        F=UAIbotTaskController.F_tanh,
        F_args={
            "A": 0.25,
            "width": 0.01,
        },
        damping=1e-3,
    )

    n = robot.q.shape[0]

    hist_r = np.matrix(np.zeros((6, 0)))
    hist_u = np.matrix(np.zeros((n, 0)))
    hist_t = []

    # Current configuration.
    q = robot.q

    for _ in range(round(tmax / dt)):
        # Controller only computes the action.
        u, r, Jr, Fr = controller.compute_control(q=q)

        hist_r = np.block([hist_r, r])
        hist_u = np.block([hist_u, u])
        hist_t.append(t)

        # Integration is outside the controller.
        q_next = q + u * dt

        # Animation is also outside the controller.
        robot.add_ani_frame(time=t + dt, q=q_next)

        q = q_next
        t += dt

    sim.run()

    # Convert histories to arrays for plotting.
    hist_r_array = np.asarray(hist_r, dtype=float)
    hist_u_array = np.asarray(hist_u, dtype=float)

    # Plot task function history.
    plt.figure()
    for i in range(hist_r_array.shape[0]):
        plt.plot(hist_t, hist_r_array[i, :], label=f"r[{i}]")

    plt.xlabel("Tempo (s)")
    plt.ylabel("Função de tarefa")
    plt.title("Histórico da função de tarefa")
    plt.grid(True)

    # Plot control action history.
    plt.figure()
    for i in range(hist_u_array.shape[0]):
        plt.plot(hist_t, hist_u_array[i, :], label=f"u[{i}]")

    plt.xlabel("Tempo (s)")
    plt.ylabel("Ação de controle")
    plt.title("Histórico da ação de controle")
    plt.grid(True)

    plt.show()
    
    
    
if __name__ == "__main__":
    run_tests()
    # demo_pose_regulation_visualization()
    
