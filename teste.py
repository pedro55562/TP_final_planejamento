"""
Object transport demo using UAIbotTaskController.

This script adapts the original pick-and-transport example to use the
object-oriented task-space controller.

The controller itself does not animate or integrate. This script handles:

- scene creation
- reference trajectory creation
- time loop
- q integration
- UAIbot animation frames
- plotting

Controller usage in this script
-------------------------------
1. Approach phase:
   - task: position + z-axis alignment
   - desired pose: constant cube grasp pose
   - time_derivative_mode="none"

2. Transport phase:
   - task: full pose tracking
   - desired pose: htm_d(t), built from position spline + SQUAD orientation
   - time_derivative_mode="numeric"
"""

import numpy as np
import uaibot as ub
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation, Slerp

from src.control_utilities import UAIbotTaskController


# =============================================================================
# Quaternion/SQUAD utilities
# =============================================================================


def compute_control_quaternion(q_prev, q, q_next):
    """
    Compute a SQUAD control quaternion.

    Parameters
    ----------
    q_prev : scipy.spatial.transform.Rotation
        Previous orientation.

    q : scipy.spatial.transform.Rotation
        Current orientation.

    q_next : scipy.spatial.transform.Rotation
        Next orientation.

    Returns
    -------
    control : scipy.spatial.transform.Rotation
        Control quaternion used by SQUAD.
    """

    log_next = (q.inv() * q_next).as_rotvec()
    log_prev = (q.inv() * q_prev).as_rotvec()

    avg = -0.25 * (log_next + log_prev)
    control = q * Rotation.from_rotvec(avg)

    return control



def squad(q1, q2, a, b, u):
    """
    SQUAD interpolation between q1 and q2.

    Formula:

        SQUAD(q1,q2,a,b;u)
        = SLERP( SLERP(q1,q2;u), SLERP(a,b;u), 2u(1-u) )

    Parameters
    ----------
    q1, q2 : scipy.spatial.transform.Rotation
        Endpoint rotations.

    a, b : scipy.spatial.transform.Rotation
        Control rotations.

    u : float
        Interpolation parameter in [0, 1].

    Returns
    -------
    q_interp : scipy.spatial.transform.Rotation
        Interpolated rotation.
    """

    u = float(np.clip(u, 0.0, 1.0))

    key_times = [0.0, 1.0]

    rot_main = Rotation.from_quat([q1.as_quat(), q2.as_quat()])
    slerp_main = Slerp(key_times, rot_main)([u])[0]

    rot_control = Rotation.from_quat([a.as_quat(), b.as_quat()])
    slerp_control = Slerp(key_times, rot_control)([u])[0]

    final_param = 2.0 * u * (1.0 - u)

    rot_final = Rotation.from_quat([
        slerp_main.as_quat(),
        slerp_control.as_quat(),
    ])
    q_interp = Slerp(key_times, rot_final)([final_param])[0]

    return q_interp



def composite_squad(R_prev, R0, R1, R2, R_next, beta):
    """
    Composite SQUAD interpolation through three orientations.

    Parameters
    ----------
    R_prev : array-like, shape (3, 3)
        Previous neighbor orientation, used to compute the first control
        quaternion.

    R0 : array-like, shape (3, 3)
        Initial orientation.

    R1 : array-like, shape (3, 3)
        Intermediate orientation.

    R2 : array-like, shape (3, 3)
        Final orientation.

    R_next : array-like, shape (3, 3)
        Next neighbor orientation, used to compute the last control quaternion.

    beta : float
        Global interpolation parameter in [0, 1].

    Returns
    -------
    R_interp : np.matrix, shape (3, 3)
        Interpolated rotation matrix.
    """

    beta = float(np.clip(beta, 0.0, 1.0))

    q_prev = Rotation.from_matrix(np.asarray(R_prev, dtype=float))
    q0 = Rotation.from_matrix(np.asarray(R0, dtype=float))
    q1 = Rotation.from_matrix(np.asarray(R1, dtype=float))
    q2 = Rotation.from_matrix(np.asarray(R2, dtype=float))
    q_next = Rotation.from_matrix(np.asarray(R_next, dtype=float))

    if beta <= 0.5:
        u = 2.0 * beta
        a0 = compute_control_quaternion(q_prev, q0, q1)
        b0 = compute_control_quaternion(q0, q1, q2)
        q_interp = squad(q0, q1, a0, b0, u)
    else:
        u = 2.0 * (beta - 0.5)
        a1 = compute_control_quaternion(q0, q1, q2)
        b1 = compute_control_quaternion(q1, q2, q_next)
        q_interp = squad(q1, q2, a1, b1, u)

    return np.matrix(q_interp.as_matrix())


# =============================================================================
# Position spline and homogeneous transformation utilities
# =============================================================================


def as_col(v, n=None, name="vector"):
    """
    Convert input to column np.matrix.
    """

    return UAIbotTaskController.as_col(v, n=n, name=name)



def make_htm(R, p):
    """
    Build a homogeneous transformation matrix from rotation and position.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix.

    p : array-like, shape (3, 1)
        Position vector.

    Returns
    -------
    H : np.matrix, shape (4, 4)
        Homogeneous transformation matrix.
    """

    R = np.matrix(R, dtype=float).reshape((3, 3))
    p = as_col(p, 3, name="p")

    H = np.matrix(np.eye(4))
    H[0:3, 0:3] = R
    H[0:3, 3] = p

    return H



def quadratic_spline_3d(s0, s1, s2, beta):
    """
    Quadratic 3D spline passing through s0, s1 and s2.

    The parameter beta is in [0, 1]. The construction is equivalent to the
    original quadratic_spline_3D function.

    Parameters
    ----------
    s0, s1, s2 : array-like, shape (3, 1)
        Position control points.

    beta : float
        Interpolation parameter in [0, 1].

    Returns
    -------
    s : np.matrix, shape (3, 1)
        Interpolated position.
    """

    beta = float(np.clip(beta, 0.0, 1.0))

    s0 = as_col(s0, 3, name="s0")
    s1 = as_col(s1, 3, name="s1")
    s2 = as_col(s2, 3, name="s2")

    b = 4.0 * (s1 - s0) - (s2 - s0)
    a = (s2 - s0) - b
    c = s0

    return a * (beta ** 2) + b * beta + c



def create_transport_reference(
    s0,
    s1,
    s2,
    R_prev,
    R0,
    R1,
    R2,
    R_next,
    t_start,
    duration,
):
    """
    Create a time-varying desired pose htm_d(t).

    Position is generated by quadratic_spline_3d.
    Orientation is generated by composite_squad.

    Parameters
    ----------
    s0, s1, s2 : array-like, shape (3, 1)
        Position control points.

    R_prev, R0, R1, R2, R_next : array-like, shape (3, 3)
        Orientation control matrices for SQUAD.

    t_start : float
        Start time of the trajectory.

    duration : float
        Trajectory duration.

    Returns
    -------
    htm_d : callable
        Function htm_d(t) returning a 4x4 desired pose.
    """

    duration = float(duration)

    if duration <= 0.0:
        raise ValueError("duration must be positive.")

    def htm_d(t):
        beta = (float(t) - float(t_start)) / duration
        beta = float(np.clip(beta, 0.0, 1.0))

        s = quadratic_spline_3d(s0, s1, s2, beta)
        R = composite_squad(R_prev, R0, R1, R2, R_next, beta)

        return make_htm(R, s)

    return htm_d


# =============================================================================
# Custom task for the approach phase: position + z-axis alignment
# =============================================================================


def position_and_z_axis_task_function(robot, q, Jg, fk, t, htm_d, **kwargs):
    """
    Task used to approach the cube.

    This reproduces the original approach logic:

        r[0:3] = s_e - s_d
        r[3]   = 0
        r[4]   = 0
        r[5]   = 1 - z_d.T * z_e

    Parameters follow the UAIbotTaskController custom task interface.
    """

    fk = UAIbotTaskController.as_htm(fk, name="fk")
    htm_ref = UAIbotTaskController.evaluate_htm_reference(htm_d, t=t)

    s_e = fk[0:3, 3]
    z_e = fk[0:3, 2]

    s_d = htm_ref[0:3, 3]
    z_d = htm_ref[0:3, 2]

    r = np.matrix(np.zeros((6, 1)))
    r[0:3, 0] = s_e - s_d
    r[3, 0] = 0.0
    r[4, 0] = 0.0
    r[5, 0] = UAIbotTaskController.scalar(1.0 - z_d.T * z_e)

    return r



def position_and_z_axis_task_jacobian(robot, q, Jg, fk, t, htm_d, **kwargs):
    """
    Jacobian for position_and_z_axis_task_function.

    This reproduces the original approach Jacobian:

        Jr[0:3, :] = Jv
        Jr[3, :]   = 0
        Jr[4, :]   = 0
        Jr[5, :]   = z_d.T * S(z_e) * Jw
    """

    Jg = np.matrix(Jg, dtype=float)
    fk = UAIbotTaskController.as_htm(fk, name="fk")
    htm_ref = UAIbotTaskController.evaluate_htm_reference(htm_d, t=t)

    if Jg.shape[0] != 6:
        raise ValueError(f"Jg must have shape (6, n), got {Jg.shape}.")

    n = Jg.shape[1]

    z_e = fk[0:3, 2]
    z_d = htm_ref[0:3, 2]

    Jv = Jg[0:3, :]
    Jw = Jg[3:6, :]

    Jr = np.matrix(np.zeros((6, n)))
    Jr[0:3, :] = Jv
    Jr[3, :] = np.matrix(np.zeros((1, n)))
    Jr[4, :] = np.matrix(np.zeros((1, n)))
    Jr[5, :] = z_d.T * ub.Utils.S(z_e) * Jw

    return Jr



def cube_reached(robot, q, cube_grasp_htm, position_tol=0.01, z_axis_tol=0.06):
    """
    Check if the end-effector has reached the cube grasp pose.
    """

    _, fk = robot.jac_geo(q)
    fk = UAIbotTaskController.as_htm(fk, name="fk")
    cube_grasp_htm = UAIbotTaskController.as_htm(cube_grasp_htm, name="cube_grasp_htm")

    s_e = fk[0:3, 3]
    z_e = fk[0:3, 2]

    s_cube = cube_grasp_htm[0:3, 3]
    z_cube = cube_grasp_htm[0:3, 2]

    position_error = float(np.linalg.norm(s_e - s_cube))
    z_axis_error = float(np.linalg.norm(z_e - z_cube))

    return position_error < position_tol and z_axis_error < z_axis_tol


# =============================================================================
# Scene creation
# =============================================================================


def create_scene():
    """
    Create the robot and scene objects.

    Returns
    -------
    objects : dict
        Dictionary containing robot, cube, tables, obstacle, and visual objects.
    """

    robot = ub.Robot.create_kuka_kr5(ub.Utils.trn([0, 0, 0.2]))

    texture_table = ub.Texture(
        url="https://raw.githubusercontent.com/viniciusmgn/uaibot_content/master/contents/Textures/rough_metal.jpg",
        wrap_s="RepeatWrapping",
        wrap_t="RepeatWrapping",
        repeat=[1, 1],
    )

    material_table = ub.MeshMaterial(
        texture_map=texture_table,
        roughness=1,
        metalness=1,
        opacity=1,
    )

    table1 = ub.Box(
        name="table1",
        htm=ub.Utils.trn([0.8, 0, 0.15]),
        width=0.5,
        depth=0.5,
        height=0.4,
        mesh_material=material_table,
    )

    table2 = ub.Box(
        name="table2",
        htm=ub.Utils.trn([0, -0.8, 0.15]),
        width=0.5,
        depth=0.5,
        height=0.4,
        mesh_material=material_table,
    )

    table3 = ub.Box(
        name="table3",
        htm=ub.Utils.trn([0, 0, 0.1]),
        width=0.3,
        depth=0.3,
        height=0.2,
        mesh_material=material_table,
    )

    obstacle = ub.Box(
        name="obstacle",
        htm=ub.Utils.trn([0.8, -0.8, 0.5]),
        width=0.6,
        depth=0.6,
        height=1.0,
        mesh_material=material_table,
    )

    material_cube = ub.MeshMaterial(
        roughness=1,
        metalness=1,
        opacity=1,
        color="purple",
    )

    cube = ub.Box(
        name="cube",
        htm=ub.Utils.trn([0.8, 0, 0.4]),
        width=0.1,
        depth=0.1,
        height=0.1,
        mesh_material=material_cube,
    )

    ball_ref = ub.Ball(
        htm=np.identity(4),
        radius=0.02,
        color="cyan",
    )

    return {
        "robot": robot,
        "table1": table1,
        "table2": table2,
        "table3": table3,
        "obstacle": obstacle,
        "cube": cube,
        "ball_ref": ball_ref,
    }


# =============================================================================
# Plotting
# =============================================================================


def plot_histories(hist_t, hist_r, hist_u):
    """
    Plot task value and control action histories.
    """

    hist_r_array = np.asarray(hist_r, dtype=float)
    hist_u_array = np.asarray(hist_u, dtype=float)

    plt.figure()
    for i in range(hist_r_array.shape[0]):
        plt.plot(hist_t, hist_r_array[i, :], label=f"r[{i}]")

    plt.xlabel("Tempo (s)")
    plt.ylabel("Função de tarefa")
    plt.title("Histórico da função de tarefa")
    plt.grid(True)
    plt.legend()

    plt.figure()
    for i in range(hist_u_array.shape[0]):
        plt.plot(hist_t, hist_u_array[i, :], label=f"u[{i}]")

    plt.xlabel("Tempo (s)")
    plt.ylabel("Ação de controle")
    plt.title("Histórico da ação de controle")
    plt.grid(True)
    plt.legend()

    plt.show()


# =============================================================================
# Main demo
# =============================================================================


def run_transport_demo():
    """
    Run the adapted pick-and-transport demo.
    """

    dt = 0.01
    t = 0.0

    approach_tmax = 25.0
    transport_duration = 6.0
    transport_tmax = 9.0

    scene = create_scene()

    robot = scene["robot"]
    cube = scene["cube"]
    ball_ref = scene["ball_ref"]

    table1 = scene["table1"]
    table2 = scene["table2"]
    table3 = scene["table3"]
    obstacle = scene["obstacle"]

    # Desired grasp pose for the cube.
    cube_grasp_htm = ub.Utils.trn([0.8, 0, 0.45]) * ub.Utils.roty(np.pi)
    frame_get_cube = ub.Frame(htm=cube_grasp_htm)

    # -------------------------------------------------------------------------
    # Phase 1: approach cube and attach it
    # -------------------------------------------------------------------------

    approach_controller = UAIbotTaskController(
        robot=robot,
        htm_d=cube_grasp_htm,
        task_function=position_and_z_axis_task_function,
        task_jacobian=position_and_z_axis_task_jacobian,
        F=UAIbotTaskController.F_sqrt,
        damping=1e-3,
        time_derivative_mode="none",
    )

    q = robot.q

    for _ in range(round(approach_tmax / dt)):
        if cube_reached(robot, q, cube_grasp_htm):
            robot.attach_object(cube)
            break

        u, r, Jr, Fr, rt = approach_controller.compute_control(q=q)

        q_next = q + u * dt
        robot.add_ani_frame(time=t + dt, q=q_next)

        q = q_next
        t += dt

    # Current end-effector pose after reaching the cube.
    _, H0 = robot.jac_geo(q)
    H0 = UAIbotTaskController.as_htm(H0, name="H0")
    R0 = H0[0:3, 0:3]

    # -------------------------------------------------------------------------
    # Phase 2: define transport path
    # -------------------------------------------------------------------------

    H1 = (
        ub.Utils.trn([0, -0.3, 0.85])
        * ub.Utils.roty(np.pi / 2)
        * ub.Utils.rotx(np.pi / 2)
        * ub.Utils.rotz(np.pi / 2)
    )

    H2 = ub.Utils.trn([0, -0.8, 0.45]) * ub.Utils.rotx(np.pi)

    frame_h1 = ub.Frame(htm=H1)
    frame_h2 = ub.Frame(htm=H2)

    R1 = H1[0:3, 0:3]
    R2 = H2[0:3, 0:3]

    # Neighbors used by SQUAD. Here we reuse endpoints, as in the original code.
    R_prev = R0
    R_next = R2

    s0 = H0[0:3, 3]
    s1 = H1[0:3, 3]
    s2 = H2[0:3, 3]

    t_start_transport = t

    htm_d_transport = create_transport_reference(
        s0=s0,
        s1=s1,
        s2=s2,
        R_prev=R_prev,
        R0=R0,
        R1=R1,
        R2=R2,
        R_next=R_next,
        t_start=t_start_transport,
        duration=transport_duration,
    )

    transport_controller = UAIbotTaskController(
        robot=robot,
        htm_d=htm_d_transport,
        F=UAIbotTaskController.F_tanh,
        damping=1e-3,
        time_derivative_mode="numeric",
        time_derivative_eps=1e-4,
    )

    n = robot.q.shape[0]

    hist_r = np.matrix(np.zeros((6, 0)))
    hist_u = np.matrix(np.zeros((n, 0)))
    hist_t = []

    # -------------------------------------------------------------------------
    # Phase 2 loop: track transport trajectory
    # -------------------------------------------------------------------------

    for _ in range(round(transport_tmax / dt)):
        u, r, Jr, Fr, rt = transport_controller.compute_control(q=q, t=t)

        hist_r = np.block([hist_r, r])
        hist_u = np.block([hist_u, u])
        hist_t.append(t)

        q_next = q + u * dt
        robot.add_ani_frame(time=t + dt, q=q_next)

        H_ref = htm_d_transport(t)
        ball_ref.add_ani_frame(time=t, htm=ub.Utils.trn(H_ref[0:3, 3]))

        q = q_next
        t += dt

    robot.detach_object(cube)

    # -------------------------------------------------------------------------
    # Simulation and plots
    # -------------------------------------------------------------------------

    sim = ub.Simulation.create_sim_factory([
        robot,
        table1,
        table2,
        table3,
        obstacle,
        cube,
        frame_get_cube,
        frame_h1,
        frame_h2,
        ball_ref,
    ])

    sim.run()

    plot_histories(hist_t, hist_r, hist_u)


if __name__ == "__main__":
    run_transport_demo()
