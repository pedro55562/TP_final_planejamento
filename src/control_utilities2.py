"""
QP/CBF task-space controller for UAIbot.

This module extends UAIbotTaskController with CBF constraints solved through a
quadratic program.

It keeps the same philosophy as the base controller:

- no animation inside the controller
- no integration inside the controller
- works for constant pose regulation and time-varying trajectory tracking
- uses np.matrix and UAIbot conventions

QP solved
---------
The controller computes:

    y = F(r) - rt

and solves:

    minimize    ||Jr u - y||^2 + eps ||u||^2
    subject to  A u >= b

The inequality convention follows ub.Utils.solve_qp.

CBF convention
--------------
For a safety function h(q, t) >= 0, the CBF condition is:

    hdot + eta h >= 0

If hdot = dhdq * u, this gives:

    dhdq * u >= -eta h

which matches the form:

    A u >= b

Important
---------
This version does NOT consider objects carried by the robot.

That means there is no carried_object parameter and no CBF between a carried
object and the environment. The constraints considered here are only:

- robot-obstacle collision CBFs
- self-collision CBFs
- joint limit CBFs
- velocity limits
- extra user-defined CBF constraints
"""

import numpy as np
import uaibot as ub

from src.control_utilities import UAIbotTaskController


class UAIbotQPTaskController(UAIbotTaskController):
    """
    QP/CBF task-space controller for UAIbot robots.

    This class inherits the task-function machinery from UAIbotTaskController
    and replaces the final pseudoinverse control law by a constrained QP.

    It supports:

    - task tracking or pose regulation
    - time-varying references through htm_d(t)
    - numerical or analytic task time derivative
    - robot-obstacle collision CBFs
    - optional self-collision CBFs
    - joint limit CBFs
    - velocity limits
    - extra user-defined CBF constraints

    It does NOT include carried-object collision constraints.
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
        regularization=1e-3,
        eta=0.6,
        obstacles=None,
        obstacle_margin=0.025,
        care_obstacles=True,
        care_self_collision=False,
        self_collision_margin=0.025,
        distance_tol=0.0005,
        distance_no_iter_max=20,
        distance_max_dist=np.inf,
        distance_h=0,
        distance_eps=0,
        distance_mode="auto",
        care_joint_limits=True,
        joint_margin=2 * np.pi / 180,
        care_velocity_limits=True,
        max_qdot=1.5,
        extra_cbf_functions=None,
        extra_cbf_args=None,
    ):
        """
        Create a QP/CBF task-space controller.

        Parameters
        ----------
        robot : uaibot.Robot
            UAIbot robot object.

        htm_d : None, array-like, or callable
            Desired pose. Can be constant htm_d or time-varying htm_d(t).

        F, F_args, task_function, task_jacobian, task_time_derivative,
        task_args, damping, time_derivative_mode, time_derivative_eps,
        numeric_time_derivative_method
            Same meaning as in UAIbotTaskController.

        regularization : float
            Positive coefficient eps in the QP objective.

        eta : float
            CBF convergence/safety gain.

        obstacles : None or list
            Objects used for robot-obstacle CBFs.

        obstacle_margin : float
            Minimum desired distance from obstacles.

        care_obstacles : bool
            If True, add robot-obstacle collision CBF constraints.

        care_self_collision : bool
            If True, add self-collision CBF constraints using
            robot.compute_dist_auto.

        self_collision_margin : float
            Minimum desired self-collision distance.

        distance_tol : float
            Tolerance used by UAIbot distance computations.

        distance_no_iter_max : int
            Maximum number of iterations used by UAIbot distance computations.

        distance_max_dist : float
            Maximum distance used by UAIbot distance computations to skip
            expensive checks when possible.

        distance_h : float
            UAIbot generalized distance parameter h.

        distance_eps : float
            UAIbot generalized distance parameter eps.

        distance_mode : str
            UAIbot distance computation mode. Usually "auto", "c++" or "python".

        care_joint_limits : bool
            If True, add CBF constraints for joint limits.

        joint_margin : float
            Margin away from joint limits.

        care_velocity_limits : bool
            If True, add velocity box constraints.

        max_qdot : float or array-like
            Maximum absolute joint velocity. If scalar, the same limit is used
            for all joints.

        extra_cbf_functions : None or list of callables
            Additional CBF functions. Each function must have signature:

                cbf(robot, q, Jg, fk, t, **extra_cbf_args) -> A, b

            with inequality convention:

                A u >= b

        extra_cbf_args : None or dict
            Extra keyword arguments passed to all extra CBF functions.
        """

        super().__init__(
            robot=robot,
            htm_d=htm_d,
            F=F,
            F_args=F_args,
            task_function=task_function,
            task_jacobian=task_jacobian,
            task_time_derivative=task_time_derivative,
            task_args=task_args,
            damping=damping,
            time_derivative_mode=time_derivative_mode,
            time_derivative_eps=time_derivative_eps,
            numeric_time_derivative_method=numeric_time_derivative_method,
        )

        self.regularization = float(regularization)
        self.eta = float(eta)

        self.obstacles = [] if obstacles is None else list(obstacles)
        self.obstacle_margin = float(obstacle_margin)
        self.care_obstacles = bool(care_obstacles)

        self.care_self_collision = bool(care_self_collision)
        self.self_collision_margin = float(self_collision_margin)

        self.distance_tol = float(distance_tol)
        self.distance_no_iter_max = int(distance_no_iter_max)
        self.distance_max_dist = float(distance_max_dist)
        self.distance_h = float(distance_h)
        self.distance_eps = float(distance_eps)
        self.distance_mode = distance_mode

        # Warm-start structure for UAIbot auto-collision distance computation.
        self._old_auto_dist_struct = None

        self.care_joint_limits = bool(care_joint_limits)
        self.joint_margin = float(joint_margin)

        self.care_velocity_limits = bool(care_velocity_limits)
        self.max_qdot = max_qdot

        self.extra_cbf_functions = (
            [] if extra_cbf_functions is None else list(extra_cbf_functions)
        )
        self.extra_cbf_args = {} if extra_cbf_args is None else dict(extra_cbf_args)

        if self.regularization < 0.0:
            raise ValueError("regularization must be nonnegative.")

        if self.eta < 0.0:
            raise ValueError("eta must be nonnegative.")

    # =========================================================================
    # Public control method
    # =========================================================================

    def compute_control(self, q=None, t=None):
        """
        Compute one constrained control action.

        Parameters
        ----------
        q : None or array-like, shape (n, 1)
            Current joint configuration. If None, self.robot.q is used.

        t : None or float
            Current time. Required for time-varying tasks.

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
            Partial time derivative of the task.

        qp_data : dict
            Dictionary containing H, f, A, b and y.
        """

        if q is None:
            q = self.robot.q

        q = self.as_col(q, name="q")
        n = q.shape[0]

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

        if Jr.shape[1] != n:
            raise ValueError(
                "Task Jacobian column dimension must match number of joints. "
                f"Got Jr.shape={Jr.shape} and q.shape={q.shape}."
            )

        Fr = self.evaluate_task_dynamics(self.F, r, self.F_args)
        rt = self.compute_task_time_derivative(q=q, Jg=Jg, fk=fk, t=t, r=r)
        y = Fr - rt

        A, b = self.compute_cbf_constraints(q=q, Jg=Jg, fk=fk, t=t)

        H, f = self.build_qp_objective(
            Jr=Jr,
            y=y,
            n=n,
        )

        u = ub.Utils.solve_qp(H, f, A, b)
        u = self.as_col(u, n, name="u")

        qp_data = {
            "H": H,
            "f": f,
            "A": A,
            "b": b,
            "y": y,
        }

        return u, r, Jr, Fr, rt, qp_data

    # =========================================================================
    # QP objective
    # =========================================================================

    def build_qp_objective(self, Jr, y, n):
        """
        Build the QP objective matrices.

        The objective is compatible with ub.Utils.solve_qp:

            minimize 0.5 u.T H u + f.T u

        representing:

            ||Jr u - y||^2 + regularization ||u||^2
        """

        Jr = np.matrix(Jr, dtype=float)
        y = self.as_col(y, Jr.shape[0], name="y")

        H = 2.0 * (Jr.T * Jr + self.regularization * np.matrix(np.eye(n)))
        f = -2.0 * Jr.T * y

        return H, f

    # =========================================================================
    # CBF constraints
    # =========================================================================

    @staticmethod
    def empty_constraints(n):
        """
        Create empty inequality matrices A and b for A u >= b.
        """

        return np.matrix(np.zeros((0, n))), np.matrix(np.zeros((0, 1)))

    @staticmethod
    def stack_constraints(A, b, A_new, b_new):
        """
        Vertically stack inequality constraints.
        """

        if A_new is None or b_new is None:
            return A, b

        A_new = np.matrix(A_new, dtype=float)
        b_new = np.matrix(b_new, dtype=float)

        if b_new.shape[0] == 1 and b_new.shape[1] > 1:
            b_new = b_new.T

        b_new = np.matrix(np.asarray(b_new, dtype=float).reshape((-1, 1)))

        if A_new.shape[0] != b_new.shape[0]:
            raise ValueError(
                "A_new and b_new must have compatible row dimensions. "
                f"Got A_new.shape={A_new.shape}, b_new.shape={b_new.shape}."
            )

        return np.vstack((A, A_new)), np.vstack((b, b_new))

    def compute_cbf_constraints(self, q, Jg, fk, t=None):
        """
        Compute all active CBF and box constraints.

        All constraints follow the convention:

            A u >= b
        """

        q = self.as_col(q, name="q")
        n = q.shape[0]

        A, b = self.empty_constraints(n)

        if self.care_obstacles and len(self.obstacles) > 0:
            A_obs, b_obs = self.robot_obstacle_cbf_constraints(
                robot=self.robot,
                q=q,
                obstacles=self.obstacles,
                eta=self.eta,
                margin=self.obstacle_margin,
                distance_tol=self.distance_tol,
                distance_no_iter_max=self.distance_no_iter_max,
                distance_max_dist=self.distance_max_dist,
                distance_h=self.distance_h,
                distance_eps=self.distance_eps,
                distance_mode=self.distance_mode,
            )
            A, b = self.stack_constraints(A, b, A_obs, b_obs)

        if self.care_self_collision:
            A_self, b_self, self._old_auto_dist_struct = self.self_collision_cbf_constraints(
                robot=self.robot,
                q=q,
                eta=self.eta,
                margin=self.self_collision_margin,
                old_dist_struct=self._old_auto_dist_struct,
                distance_tol=self.distance_tol,
                distance_no_iter_max=self.distance_no_iter_max,
                distance_max_dist=self.distance_max_dist,
                distance_h=self.distance_h,
                distance_eps=self.distance_eps,
                distance_mode=self.distance_mode,
            )
            A, b = self.stack_constraints(A, b, A_self, b_self)

        if self.care_joint_limits:
            A_joint, b_joint = self.joint_limit_cbf_constraints(
                robot=self.robot,
                q=q,
                eta=self.eta,
                margin=self.joint_margin,
            )
            A, b = self.stack_constraints(A, b, A_joint, b_joint)

        if self.care_velocity_limits:
            A_vel, b_vel = self.velocity_limit_constraints(
                n=n,
                max_qdot=self.max_qdot,
            )
            A, b = self.stack_constraints(A, b, A_vel, b_vel)

        for cbf in self.extra_cbf_functions:
            A_extra, b_extra = cbf(
                self.robot,
                q,
                Jg,
                fk,
                t,
                **self.extra_cbf_args,
            )
            A, b = self.stack_constraints(A, b, A_extra, b_extra)

        return A, b

    @staticmethod
    def robot_obstacle_cbf_constraints(
        robot,
        q,
        obstacles,
        eta=0.6,
        margin=0.025,
        distance_tol=0.0005,
        distance_no_iter_max=20,
        distance_max_dist=np.inf,
        distance_h=0,
        distance_eps=0,
        distance_mode="auto",
    ):
        """
        CBF constraints between the robot and a list of obstacles.

        This follows the original code style:

            ds = robot.compute_dist(q=q, obj=obs)
            A  = ds.jac_dist_mat
            b  = -eta * (ds.dist_vect - margin)

        with inequality convention:

            A u >= b
        """

        q = UAIbotTaskController.as_col(q, name="q")
        n = q.shape[0]

        A, b = UAIbotQPTaskController.empty_constraints(n)

        for obs in obstacles:
            ds = robot.compute_dist(
                q=q,
                obj=obs,
                tol=distance_tol,
                no_iter_max=distance_no_iter_max,
                max_dist=distance_max_dist,
                h=distance_h,
                eps=distance_eps,
                mode=distance_mode,
            )
            A, b = UAIbotQPTaskController.stack_constraints(
                A,
                b,
                ds.jac_dist_mat,
                -float(eta) * (ds.dist_vect - float(margin)),
            )

        return A, b

    @staticmethod
    def self_collision_cbf_constraints(
        robot,
        q,
        eta=0.6,
        margin=0.025,
        old_dist_struct=None,
        distance_tol=0.0005,
        distance_no_iter_max=20,
        distance_max_dist=np.inf,
        distance_h=0,
        distance_eps=0,
        distance_mode="auto",
    ):
        """
        Self-collision CBF constraints using robot.compute_dist_auto.

        UAIbot's compute_dist_auto computes distances between non-sequential
        robot links. The expected returned structure has aggregate fields:

            dist_struct.jac_dist_mat
            dist_struct.dist_vect

        The CBF condition is:

            jac_dist_mat * u >= -eta * (dist_vect - margin)

        Parameters
        ----------
        robot : uaibot.Robot
            Robot object.

        q : array-like, shape (n, 1)
            Joint configuration.

        eta : float
            CBF gain.

        margin : float
            Minimum desired self-collision distance.

        old_dist_struct : None or DistStructRobotAuto
            Previous distance structure used to warm-start UAIbot's distance
            computation.

        distance_tol, distance_no_iter_max, distance_max_dist, distance_h,
        distance_eps, distance_mode
            Parameters forwarded to robot.compute_dist_auto.

        Returns
        -------
        A : np.matrix
            Constraint matrix.

        b : np.matrix
            Constraint vector.

        dist_struct : DistStructRobotAuto
            Distance structure returned by UAIbot. Store this and pass it back
            as old_dist_struct in the next control iteration for speed.
        """

        q = UAIbotTaskController.as_col(q, name="q")
        n = q.shape[0]

        dist_struct = robot.compute_dist_auto(
            q=q,
            old_dist_struct=old_dist_struct,
            tol=distance_tol,
            no_iter_max=distance_no_iter_max,
            max_dist=distance_max_dist,
            h=distance_h,
            eps=distance_eps,
            mode=distance_mode,
        )

        if not hasattr(dist_struct, "jac_dist_mat") or not hasattr(dist_struct, "dist_vect"):
            raise AttributeError(
                "robot.compute_dist_auto returned a structure without the expected "
                "aggregate attributes 'jac_dist_mat' and 'dist_vect'. Check the "
                "UAIbot version or adapt self_collision_cbf_constraints to the "
                "returned DistStructRobotAuto fields."
            )

        A, b = UAIbotQPTaskController.empty_constraints(n)
        A, b = UAIbotQPTaskController.stack_constraints(
            A,
            b,
            dist_struct.jac_dist_mat,
            -float(eta) * (dist_struct.dist_vect - float(margin)),
        )

        return A, b, dist_struct

    @staticmethod
    def joint_limit_cbf_constraints(robot, q, eta=0.6, margin=2 * np.pi / 180):
        """
        CBF constraints for joint limits.

        This follows the ub.Utils.solve_qp convention:

            A u >= b

        Lower limit:

            q_i >= q_min_i + margin
            u_i >= -eta * (q_i - q_min_i - margin)

        Upper limit:

            q_i <= q_max_i - margin
            -u_i >= -eta * (q_max_i - q_i - margin)
        """

        q = UAIbotTaskController.as_col(q, name="q")
        n = q.shape[0]

        q_min = UAIbotTaskController.as_col(robot.joint_limit[:, 0], n, name="q_min")
        q_max = UAIbotTaskController.as_col(robot.joint_limit[:, 1], n, name="q_max")

        I = np.matrix(np.eye(n))

        A = np.vstack((I, -I))
        b = np.vstack((
            -float(eta) * (q - q_min - float(margin)),
            -float(eta) * (q_max - q - float(margin)),
        ))

        return A, b

    @staticmethod
    def velocity_limit_constraints(n, max_qdot=1.5):
        """
        Box constraints for joint velocities.

        Convention:

            A u >= b

        Implements:

            -max_qdot <= u_i <= max_qdot
        """

        if np.isscalar(max_qdot):
            max_qdot = float(max_qdot) * np.matrix(np.ones((n, 1)))
        else:
            max_qdot = UAIbotTaskController.as_col(max_qdot, n, name="max_qdot")

        I = np.matrix(np.eye(n))

        A = np.vstack((I, -I))
        b = np.vstack((-max_qdot, -max_qdot))

        return A, b
