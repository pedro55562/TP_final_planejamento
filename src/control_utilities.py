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
For a constant desired pose:

    controller = UAIbotTaskController(robot, htm_d=htm_d)
    u, r, Jr, Fr, rt = controller.compute_control(q=q)

For a time-varying desired pose htm_d(t):

    controller = UAIbotTaskController(
        robot,
        htm_d=htm_d,
        time_derivative_mode="numeric",
    )
    u, r, Jr, Fr, rt = controller.compute_control(q=q, t=t)

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

The desired pose can be:

    htm_d           # constant pose
    htm_d(t)        # time-varying pose

For time-varying tasks, the controller uses:

    Jr(q,t) u = F(r) - partial r / partial t

Therefore:

    u = Jr(q,t)^+ * (F(r) - rt)

where:

    rt = partial r / partial t
"""

import numpy as np
import uaibot as ub


class UAIbotTaskController:
    """
    Generic task-space controller for UAIbot robots.

    The class stores the controller configuration and computes one control
    action at each call.

    It implements:

        r  = task_function(robot, q, Jg, fk, t, **task_args)
        Jr = task_jacobian(robot, q, Jg, fk, t, **task_args)
        Fr = F(r, **F_args)
        rt = partial r / partial t
        u  = ub.Utils.dp_inv(Jr, damping) * (Fr - rt)

    For static tasks, rt = 0.

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
        task_time_derivative=None,
        task_args=None,
        damping=1e-3,
        time_derivative_mode="none",
        time_derivative_eps=1e-4,
        numeric_time_derivative_method="five_point_forward",
    ):
        """
        Create a task-space controller.

        Parameters
        ----------
        robot : uaibot.Robot
            UAIbot robot object. The controller uses it to compute forward
            kinematics and the geometric Jacobian.

        htm_d : None, array-like, or callable
            Desired end-effector pose for the default pose task.

            It can be:

                htm_d        : constant 4x4 homogeneous transformation
                htm_d(t)     : callable returning a 4x4 homogeneous transformation

            If a custom task is used, this argument is optional.

        F : None or callable
            Task dynamics function. It must receive r and return F(r).

            Expected signature:

                F(r, **F_args) -> Fr

            If None, UAIbotTaskController.F_linear is used.

        F_args : None or dict
            Extra keyword arguments passed to F.

        task_function : None or callable
            Function that computes the task value r.

            Expected signature:

                task_function(robot, q, Jg, fk, t, **task_args) -> r

            If None, the default pose task is used.

        task_jacobian : None or callable
            Function that computes the task Jacobian Jr.

            Expected signature:

                task_jacobian(robot, q, Jg, fk, t, **task_args) -> Jr

            If None, the default pose task Jacobian is used.

        task_time_derivative : None or callable
            Function that computes the partial time derivative of the task.

            Expected signature:

                task_time_derivative(robot, q, Jg, fk, t, **task_args) -> rt

            It is required only when time_derivative_mode="analytic".

        task_args : None or dict
            Extra keyword arguments passed to task_function, task_jacobian and
            task_time_derivative.

            For the default pose task, htm_d is inserted automatically.

        damping : float
            Damping parameter used by ub.Utils.dp_inv.

        time_derivative_mode : str
            How to compute partial r / partial t. Options:

                "none"      : static task, rt = 0
                "analytic"  : user provides task_time_derivative
                "numeric"   : controller estimates rt numerically

        time_derivative_eps : float
            Step size used for numerical time differentiation.

        numeric_time_derivative_method : str
            Numerical differentiation method. Currently supported:

                "five_point_forward"

            This is a fourth-order one-sided formula:

                f'(t) ≈ (-25 f(t) + 48 f(t+h) - 36 f(t+2h)
                         + 16 f(t+3h) - 3 f(t+4h)) / (12 h)

            It is more accurate than the simple first-order forward difference
            and does not use the basic two-point central difference formula.
        """

        self.robot = robot

        self.htm_d = None
        if htm_d is not None:
            if callable(htm_d):
                self.htm_d = htm_d
            else:
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

        self.task_time_derivative = task_time_derivative
        self.task_args = {} if task_args is None else dict(task_args)

        # If htm_d was passed directly to the constructor, insert it into
        # task_args. If the user already passed task_args["htm_d"], keep the
        # user's value.
        if self.htm_d is not None:
            self.task_args.setdefault("htm_d", self.htm_d)

        self.damping = float(damping)

        valid_modes = {"none", "analytic", "numeric"}
        if time_derivative_mode not in valid_modes:
            raise ValueError(
                "time_derivative_mode must be 'none', 'analytic', or 'numeric'."
            )

        self.time_derivative_mode = time_derivative_mode
        self.time_derivative_eps = float(time_derivative_eps)
        self.numeric_time_derivative_method = numeric_time_derivative_method

        if self.time_derivative_eps <= 0.0:
            raise ValueError("time_derivative_eps must be positive.")

    # =========================================================================
    # Public control method
    # =========================================================================

    def compute_control(self, q=None, t=None):
        """
        Compute one control action for the current robot state.

        This method does not animate, integrate, or modify the robot. It only
        computes the control command.

        Parameters
        ----------
        q : None or array-like, shape (n, 1)
            Joint configuration.

            If None, self.robot.q is used.

        t : None or float
            Current time.

            It is required when the task depends explicitly on time, i.e. when:

                time_derivative_mode="analytic"
                time_derivative_mode="numeric"

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

        rt : np.matrix, shape (m, 1)
            Partial time derivative of the task, partial r / partial t.
        """

        if q is None:
            q = self.robot.q

        q = self.as_col(q, name="q")

        Jg, fk = self.robot.jac_geo(q)
        Jg = np.matrix(Jg, dtype=float)
        fk = self.as_htm(fk, name="fk")

        r = self.evaluate_task_function(q=q, Jg=Jg, fk=fk, t=t)
        Jr = self.evaluate_task_jacobian(q=q, Jg=Jg, fk=fk, t=t)

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
        rt = self.compute_task_time_derivative(q=q, Jg=Jg, fk=fk, t=t, r=r)

        u = ub.Utils.dp_inv(Jr, self.damping) * (Fr - rt)

        return u, r, Jr, Fr, rt

    # =========================================================================
    # Evaluation helpers
    # =========================================================================

    def evaluate_task_function(self, q, Jg, fk, t):
        """
        Evaluate the configured task function.

        Parameters
        ----------
        q : np.matrix, shape (n, 1)
            Current joint configuration.

        Jg : np.matrix, shape (6, n)
            Geometric Jacobian.

        fk : np.matrix, shape (4, 4)
            Current end-effector pose.

        t : None or float
            Current time.

        Returns
        -------
        r : np.matrix, shape (m, 1)
            Task value.
        """

        r = self.task_function(
            self.robot,
            q,
            Jg,
            fk,
            t,
            **self.task_args,
        )

        return self.as_col(r, name="r")

    def evaluate_task_jacobian(self, q, Jg, fk, t):
        """
        Evaluate the configured task Jacobian.

        Parameters
        ----------
        q : np.matrix, shape (n, 1)
            Current joint configuration.

        Jg : np.matrix, shape (6, n)
            Geometric Jacobian.

        fk : np.matrix, shape (4, 4)
            Current end-effector pose.

        t : None or float
            Current time.

        Returns
        -------
        Jr : np.matrix, shape (m, n)
            Task Jacobian.
        """

        Jr = self.task_jacobian(
            self.robot,
            q,
            Jg,
            fk,
            t,
            **self.task_args,
        )

        return np.matrix(Jr, dtype=float)

    def compute_task_time_derivative(self, q, Jg, fk, t, r):
        """
        Compute the partial time derivative of the task.

        The returned value is:

            rt = partial r / partial t

        Parameters
        ----------
        q : np.matrix, shape (n, 1)
            Current joint configuration.

        Jg : np.matrix, shape (6, n)
            Geometric Jacobian at q.

        fk : np.matrix, shape (4, 4)
            Forward kinematics at q.

        t : None or float
            Current time.

        r : np.matrix, shape (m, 1)
            Current task value.

        Returns
        -------
        rt : np.matrix, shape (m, 1)
            Partial time derivative of the task.
        """

        if self.time_derivative_mode == "none":
            return np.matrix(np.zeros(r.shape))

        if t is None:
            raise ValueError(
                "t must be provided when time_derivative_mode is not 'none'."
            )

        if self.time_derivative_mode == "analytic":
            if self.task_time_derivative is None:
                raise ValueError(
                    "task_time_derivative must be provided when "
                    "time_derivative_mode='analytic'."
                )

            rt = self.task_time_derivative(
                self.robot,
                q,
                Jg,
                fk,
                t,
                **self.task_args,
            )

            return self.as_col(rt, r.shape[0], name="rt")

        if self.time_derivative_mode == "numeric":
            return self.compute_task_time_derivative_numeric(
                q=q,
                Jg=Jg,
                fk=fk,
                t=t,
                r=r,
            )

        raise ValueError(
            "time_derivative_mode must be 'none', 'analytic', or 'numeric'."
        )

    def compute_task_time_derivative_numeric(self, q, Jg, fk, t, r):
        """
        Numerically estimate partial r / partial t.

        This method keeps q, Jg and fk fixed and only changes the time argument
        passed to the task function. Therefore it estimates a partial time
        derivative, not a total derivative.

        The default method is a fourth-order five-point forward formula:

            r_t ≈ (-25 r(t) + 48 r(t+h) - 36 r(t+2h)
                   + 16 r(t+3h) - 3 r(t+4h)) / (12 h)

        Parameters
        ----------
        q : np.matrix, shape (n, 1)
            Current joint configuration.

        Jg : np.matrix, shape (6, n)
            Geometric Jacobian at q.

        fk : np.matrix, shape (4, 4)
            Forward kinematics at q.

        t : float
            Current time.

        r : np.matrix, shape (m, 1)
            Current task value r(t).

        Returns
        -------
        rt : np.matrix, shape (m, 1)
            Numerical estimate of partial r / partial t.
        """

        if self.numeric_time_derivative_method != "five_point_forward":
            raise ValueError(
                "numeric_time_derivative_method must be 'five_point_forward'."
            )

        h = self.time_derivative_eps

        r0 = self.as_col(r, name="r0")
        r1 = self.evaluate_task_function(q=q, Jg=Jg, fk=fk, t=t + h)
        r2 = self.evaluate_task_function(q=q, Jg=Jg, fk=fk, t=t + 2.0 * h)
        r3 = self.evaluate_task_function(q=q, Jg=Jg, fk=fk, t=t + 3.0 * h)
        r4 = self.evaluate_task_function(q=q, Jg=Jg, fk=fk, t=t + 4.0 * h)

        return (-25.0 * r0 + 48.0 * r1 - 36.0 * r2 + 16.0 * r3 - 3.0 * r4) / (
            12.0 * h
        )

    # =========================================================================
    # Small helpers
    # =========================================================================

    @staticmethod
    def as_col(v, n=None, name="vector"):
        """
        Convert input to a column np.matrix.
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
        """

        H = np.matrix(H, dtype=float)

        if H.shape != (4, 4):
            raise ValueError(f"{name} must have shape (4, 4), got {H.shape}.")

        return H

    @staticmethod
    def scalar(x):
        """
        Convert a 1x1 matrix-like value to float.
        """

        return float(np.asarray(x).reshape(-1)[0])

    @classmethod
    def evaluate_task_dynamics(cls, F, r, F_args=None):
        """
        Evaluate the task dynamics F(r).
        """

        if F_args is None:
            Fr = F(r)
        else:
            Fr = F(r, **F_args)

        return cls.as_col(Fr, r.shape[0], name="F(r)")

    @classmethod
    def evaluate_htm_reference(cls, htm_d, t=None):
        """
        Evaluate a constant or time-varying desired pose.

        Parameters
        ----------
        htm_d : array-like or callable
            If array-like, it is interpreted as a constant desired pose.
            If callable, it must be htm_d(t).

        t : None or float
            Current time. Required when htm_d is callable.

        Returns
        -------
        H : np.matrix, shape (4, 4)
            Desired homogeneous transformation matrix.
        """

        if callable(htm_d):
            if t is None:
                raise ValueError("t must be provided when htm_d is callable.")
            return cls.as_htm(htm_d(t), name="htm_d(t)")

        return cls.as_htm(htm_d, name="htm_d")

    # =========================================================================
    # Default pose task
    # =========================================================================

    @classmethod
    def compute_pose_task_error(cls, fk, htm_d, t=None):
        """
        Compute the default 6D pose task error.

        The task error is:

            r[0:3] = s_e - s_d
            r[3]   = 1 - x_d.T * x_e
            r[4]   = 1 - y_d.T * y_e
            r[5]   = 1 - z_d.T * z_e

        The desired pose can be constant or time-varying.
        """

        fk = cls.as_htm(fk, name="fk")
        htm_ref = cls.evaluate_htm_reference(htm_d, t=t)

        x_e = fk[0:3, 0]
        y_e = fk[0:3, 1]
        z_e = fk[0:3, 2]
        s_e = fk[0:3, 3]

        x_d = htm_ref[0:3, 0]
        y_d = htm_ref[0:3, 1]
        z_d = htm_ref[0:3, 2]
        s_d = htm_ref[0:3, 3]

        r = np.matrix(np.zeros((6, 1)))
        r[0:3, 0] = s_e - s_d
        r[3, 0] = cls.scalar(1.0 - x_d.T * x_e)
        r[4, 0] = cls.scalar(1.0 - y_d.T * y_e)
        r[5, 0] = cls.scalar(1.0 - z_d.T * z_e)

        return r

    @classmethod
    def compute_pose_task_jacobian(cls, Jg, fk, htm_d, t=None):
        """
        Compute the Jacobian of the default pose task.

        The desired pose can be constant or time-varying.
        """

        Jg = np.matrix(Jg, dtype=float)
        fk = cls.as_htm(fk, name="fk")
        htm_ref = cls.evaluate_htm_reference(htm_d, t=t)

        if Jg.shape[0] != 6:
            raise ValueError(f"Jg must have shape (6, n), got {Jg.shape}.")

        n = Jg.shape[1]

        x_e = fk[0:3, 0]
        y_e = fk[0:3, 1]
        z_e = fk[0:3, 2]

        x_d = htm_ref[0:3, 0]
        y_d = htm_ref[0:3, 1]
        z_d = htm_ref[0:3, 2]

        Jv = Jg[0:3, :]
        Jw = Jg[3:6, :]

        Jr = np.matrix(np.zeros((6, n)))
        Jr[0:3, :] = Jv
        Jr[3, :] = x_d.T * ub.Utils.S(x_e) * Jw
        Jr[4, :] = y_d.T * ub.Utils.S(y_e) * Jw
        Jr[5, :] = z_d.T * ub.Utils.S(z_e) * Jw

        return Jr

    @classmethod
    def default_pose_task_function(cls, robot, q, Jg, fk, t, htm_d, **kwargs):
        """
        Default pose task function wrapper.
        """

        return cls.compute_pose_task_error(fk=fk, htm_d=htm_d, t=t)

    @classmethod
    def default_pose_task_jacobian(cls, robot, q, Jg, fk, t, htm_d, **kwargs):
        """
        Default pose task Jacobian wrapper.
        """

        return cls.compute_pose_task_jacobian(Jg=Jg, fk=fk, htm_d=htm_d, t=t)

    # =========================================================================
    # Pose tracking task using explicit axes
    # =========================================================================

    @classmethod
    def pose_tracking_task_function(
        cls,
        robot,
        q,
        Jg,
        fk,
        t,
        s_d,
        x_d,
        y_d,
        z_d,
        **kwargs,
    ):
        """
        Pose tracking task using explicit desired position and frame axes.

        The task is:

            r[0:3] = s_e(q) - s_d(t)
            r[3]   = 1 - x_d(t).T * x_e(q)
            r[4]   = 1 - y_d(t).T * y_e(q)
            r[5]   = 1 - z_d(t).T * z_e(q)
        """

        if t is None:
            raise ValueError("t must be provided for pose_tracking_task_function.")

        fk = cls.as_htm(fk, name="fk")

        x_e = fk[0:3, 0]
        y_e = fk[0:3, 1]
        z_e = fk[0:3, 2]
        s_e = fk[0:3, 3]

        s_ref = cls.as_col(s_d(t), 3, name="s_d(t)")
        x_ref = cls.as_col(x_d(t), 3, name="x_d(t)")
        y_ref = cls.as_col(y_d(t), 3, name="y_d(t)")
        z_ref = cls.as_col(z_d(t), 3, name="z_d(t)")

        r = np.matrix(np.zeros((6, 1)))
        r[0:3, 0] = s_e - s_ref
        r[3, 0] = cls.scalar(1.0 - x_ref.T * x_e)
        r[4, 0] = cls.scalar(1.0 - y_ref.T * y_e)
        r[5, 0] = cls.scalar(1.0 - z_ref.T * z_e)

        return r

    @classmethod
    def pose_tracking_task_jacobian(
        cls,
        robot,
        q,
        Jg,
        fk,
        t,
        x_d,
        y_d,
        z_d,
        **kwargs,
    ):
        """
        Pose tracking task Jacobian using explicit desired frame axes.

        The Jacobian is:

            Jr[0:3, :] = Jv
            Jr[3, :]   = x_d(t).T * S(x_e(q)) * Jw
            Jr[4, :]   = y_d(t).T * S(y_e(q)) * Jw
            Jr[5, :]   = z_d(t).T * S(z_e(q)) * Jw
        """

        if t is None:
            raise ValueError("t must be provided for pose_tracking_task_jacobian.")

        Jg = np.matrix(Jg, dtype=float)
        fk = cls.as_htm(fk, name="fk")

        if Jg.shape[0] != 6:
            raise ValueError(f"Jg must have shape (6, n), got {Jg.shape}.")

        n = Jg.shape[1]

        x_e = fk[0:3, 0]
        y_e = fk[0:3, 1]
        z_e = fk[0:3, 2]

        x_ref = cls.as_col(x_d(t), 3, name="x_d(t)")
        y_ref = cls.as_col(y_d(t), 3, name="y_d(t)")
        z_ref = cls.as_col(z_d(t), 3, name="z_d(t)")

        Jv = Jg[0:3, :]
        Jw = Jg[3:6, :]

        Jr = np.matrix(np.zeros((6, n)))
        Jr[0:3, :] = Jv
        Jr[3, :] = x_ref.T * ub.Utils.S(x_e) * Jw
        Jr[4, :] = y_ref.T * ub.Utils.S(y_e) * Jw
        Jr[5, :] = z_ref.T * ub.Utils.S(z_e) * Jw

        return Jr

    @classmethod
    def pose_tracking_task_time_derivative(
        cls,
        robot,
        q,
        Jg,
        fk,
        t,
        s_d_dot,
        x_d_dot,
        y_d_dot,
        z_d_dot,
        **kwargs,
    ):
        """
        Analytic partial time derivative for pose tracking.

        For:

            r[0:3] = s_e(q) - s_d(t)
            r[3]   = 1 - x_d(t).T * x_e(q)
            r[4]   = 1 - y_d(t).T * y_e(q)
            r[5]   = 1 - z_d(t).T * z_e(q)

        holding q fixed:

            partial r[0:3] / partial t = -s_d_dot(t)
            partial r[3]   / partial t = -x_d_dot(t).T * x_e(q)
            partial r[4]   / partial t = -y_d_dot(t).T * y_e(q)
            partial r[5]   / partial t = -z_d_dot(t).T * z_e(q)
        """

        if t is None:
            raise ValueError("t must be provided for pose_tracking_task_time_derivative.")

        fk = cls.as_htm(fk, name="fk")

        x_e = fk[0:3, 0]
        y_e = fk[0:3, 1]
        z_e = fk[0:3, 2]

        sd_dot = cls.as_col(s_d_dot(t), 3, name="s_d_dot(t)")
        xd_dot = cls.as_col(x_d_dot(t), 3, name="x_d_dot(t)")
        yd_dot = cls.as_col(y_d_dot(t), 3, name="y_d_dot(t)")
        zd_dot = cls.as_col(z_d_dot(t), 3, name="z_d_dot(t)")

        rt = np.matrix(np.zeros((6, 1)))
        rt[0:3, 0] = -sd_dot
        rt[3, 0] = cls.scalar(-xd_dot.T * x_e)
        rt[4, 0] = cls.scalar(-yd_dot.T * y_e)
        rt[5, 0] = cls.scalar(-zd_dot.T * z_e)

        return rt

    # =========================================================================
    # Task dynamics F(r)
    # =========================================================================

    @staticmethod
    def F_linear(r, gain=1.0):
        """
        Linear stable task dynamics: F(r) = -gain*r.
        """

        r = UAIbotTaskController.as_col(r, name="r")
        return -float(gain) * r

    @staticmethod
    def F_componentwise_saturation(r, A=0.25, width=0.01):
        """
        Componentwise saturated task dynamics.
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
        Smooth saturated task dynamics: F_i = -A_i*tanh(r_i/width_i).
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
        """

        return {
            "pose_axis_alignment": {
                "task_function": cls.default_pose_task_function,
                "task_jacobian": cls.default_pose_task_jacobian,
                "description": (
                    "Default 6D end-effector pose task. The desired pose can be "
                    "constant htm_d or time-varying htm_d(t)."
                ),
            },
            "pose_tracking_axes": {
                "task_function": cls.pose_tracking_task_function,
                "task_jacobian": cls.pose_tracking_task_jacobian,
                "task_time_derivative": cls.pose_tracking_task_time_derivative,
                "description": (
                    "6D pose tracking task using explicit s_d(t), x_d(t), y_d(t), "
                    "z_d(t), and optionally their derivatives."
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
    assert "pose_tracking_axes" in tasks



def test_default_pose_task_controller_constant_pose():
    """
    Test the controller using the default constant pose task.
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
        time_derivative_mode="none",
    )

    u, r, Jr, Fr, rt = controller.compute_control()

    n = robot.q.shape[0]

    assert u.shape == (n, 1)
    assert r.shape == (6, 1)
    assert Jr.shape == (6, n)
    assert Fr.shape == (6, 1)
    assert rt.shape == (6, 1)

    _assert_close("test_default_pose_task_controller_constant_pose rt", rt, np.zeros((6, 1)))
    _assert_finite("test_default_pose_task_controller_constant_pose u", u)
    _assert_finite("test_default_pose_task_controller_constant_pose r", r)
    _assert_finite("test_default_pose_task_controller_constant_pose Jr", Jr)
    _assert_finite("test_default_pose_task_controller_constant_pose Fr", Fr)



def custom_sine_task_function(robot, q, Jg, fk, t, **kwargs):
    """
    Custom 6D time-varying task used to test numerical time derivative.
    """

    return np.sin(t) * np.matrix(np.ones((6, 1)))



def custom_sine_task_jacobian(robot, q, Jg, fk, t, **kwargs):
    """
    Custom 6D task Jacobian used to test the interface.
    """

    return np.matrix(np.eye(6))



def test_numeric_time_derivative():
    """
    Test the fourth-order forward numerical time derivative.
    """

    robot = ub.Robot.create_kuka_kr5()

    controller = UAIbotTaskController(
        robot=robot,
        task_function=custom_sine_task_function,
        task_jacobian=custom_sine_task_jacobian,
        F=UAIbotTaskController.F_linear,
        F_args={"gain": 1.0},
        damping=1e-3,
        time_derivative_mode="numeric",
        time_derivative_eps=1e-4,
    )

    t = 0.7
    u, r, Jr, Fr, rt = controller.compute_control(t=t)

    expected_rt = np.cos(t) * np.matrix(np.ones((6, 1)))

    _assert_close("test_numeric_time_derivative", rt, expected_rt, tol=1e-7)
    _assert_finite("test_numeric_time_derivative u", u)



def test_analytic_pose_tracking_time_derivative():
    """
    Test the analytic time derivative for pose tracking with explicit axes.
    """

    robot = ub.Robot.create_kuka_kr5()
    q = robot.q
    Jg, fk = robot.jac_geo(q)

    s_d_dot = lambda t: np.matrix([[1.0], [2.0], [3.0]])
    x_d_dot = lambda t: np.matrix([[0.1], [0.0], [0.0]])
    y_d_dot = lambda t: np.matrix([[0.0], [0.2], [0.0]])
    z_d_dot = lambda t: np.matrix([[0.0], [0.0], [0.3]])

    rt = UAIbotTaskController.pose_tracking_task_time_derivative(
        robot=robot,
        q=q,
        Jg=Jg,
        fk=fk,
        t=0.0,
        s_d_dot=s_d_dot,
        x_d_dot=x_d_dot,
        y_d_dot=y_d_dot,
        z_d_dot=z_d_dot,
    )

    assert rt.shape == (6, 1)
    _assert_finite("test_analytic_pose_tracking_time_derivative rt", rt)



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
        time_derivative_mode="none",
    )

    q = robot.q

    hist_r = np.matrix(np.zeros((6, 0)))
    hist_u = np.matrix(np.zeros((robot.q.shape[0], 0)))

    for _ in range(steps):
        u, r, Jr, Fr, rt = controller.compute_control(q=q)

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
        test_default_pose_task_controller_constant_pose,
        test_numeric_time_derivative,
        test_analytic_pose_tracking_time_derivative,
        test_short_control_loop_without_animation,
    ]

    for test in tests:
        test()
        print(f"[OK] {test.__name__}")

    print("\nAll tests passed.")


if __name__ == "__main__":
    run_tests()
