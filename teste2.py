"""
Reactive QP/CBF pick-and-place demo with cube-environment CBF.

This script uses UAIbotQPTaskController in a single execution loop, following
what would typically be done in a ROS control node: at each cycle, the current
state is read, one control action is computed, the finite-state machine is
updated if necessary, and the robot configuration is integrated.

There is no time-parametrized trajectory tracking. Each mode defines one
constant target pose, and the controller reacts to the current error:

    time_derivative_mode = "none"

Important conventions
---------------------
1. The cube is not included in the robot-obstacle list.
   This means robot-cube collision is intentionally ignored, allowing the
   end-effector to approach and grasp the object.

2. Cube-environment collision is handled by an extra CBF defined in this script.
   This restriction is activated only while the cube is attached to the
   end-effector.

3. During placement, cube-table collision with the destination table is not
   considered, because the cube must be allowed to touch that table.

4. Immediately after grasping, cube-table collision with the pickup table is not
   considered either, because the cube starts close to or in contact with that
   table.

Requirement
-----------
This file assumes that src/control_utilities2.py contains the version of
UAIbotQPTaskController without built-in carried_object / care_carried_object
logic, but with support for extra_cbf_functions.
"""

import numpy as np
import uaibot as ub
import matplotlib.pyplot as plt

from src.control_utilities2 import UAIbotQPTaskController


# =============================================================================
# General parameters
# =============================================================================

# Integration and control step.
dt = 0.01

# Maximum total simulation time.
tmax = 25.0

# Task convergence tolerances.
pose_target_tol  = 0.005
place_target_tol = 0.005
final_target_tol = 0.0005

# Grasp-specific tolerances.
grasp_position_tol = 0.008
grasp_z_axis_tol = 0.04

# Task dynamics parameters for F(r) = -A*tanh(r/width).
task_A = 0.4
task_width = 0.07

# QP/controller parameters.
damping = 3e-3
regularization = 3e-3
eta = 0.6
max_qdot = 1.2

# Barrier-function safety margins.
default_obstacle_margin = 0.02
grasp_obstacle_margin = 0.012
place_obstacle_margin = 0.012
cube_obstacle_margin = 0.025

default_self_collision_margin = 0.008
grasp_self_collision_margin = 0.005
joint_margin = 2 * np.pi / 180

# Switches for constraint classes.
care_self_collision = True
care_joint_limits = True
care_velocity_limits = True

# Parameters used by UAIbot distance routines.
distance_tol = 0.0005
distance_no_iter_max = 20
distance_max_dist = np.inf
distance_h = 0
distance_eps = 0
distance_mode = "auto"

# Vertical clearances above the cube and above the destination table.
approach_clearance = 0.12
place_clearance = 0.12

# Finite-state machine modes.
mode_pregrasp = 0
mode_grasp = 1
mode_lift = 2
mode_move_above_place = 3
mode_place = 4
mode_finish_above_place = 5
mode_done = 6


# =============================================================================
# Task dynamics
# =============================================================================


def task_dynamics(r):
    """
    Compute the desired task dynamics.

    This function is used in all modes. The wrapper avoids passing F_args as a
    dictionary to the controller, keeping the parameters explicitly defined at
    the top of the file.
    """

    return UAIbotQPTaskController.F_tanh(r, A=task_A, width=task_width)


# =============================================================================
# Custom task for the grasping phase
# =============================================================================


def position_and_z_axis_task_function(robot, q, Jg, fk, t, htm_d, **kwargs):
    """
    Compute the task used to approach the cube.

    The task uses end-effector position and z-axis alignment:

        r[0:3] = s_e - s_d
        r[3]   = 0
        r[4]   = 0
        r[5]   = 1 - z_d.T * z_e

    This is useful during grasping because, at that moment, position and the
    approach axis are more important than controlling the full orientation.
    """

    fk = UAIbotQPTaskController.as_htm(fk, name="fk")
    htm_ref = UAIbotQPTaskController.evaluate_htm_reference(htm_d, t=t)

    s_e = fk[0:3, 3]
    z_e = fk[0:3, 2]

    s_d = htm_ref[0:3, 3]
    z_d = htm_ref[0:3, 2]

    r = np.matrix(np.zeros((6, 1)))
    r[0:3, 0] = s_e - s_d
    r[3, 0] = 0.0
    r[4, 0] = 0.0
    r[5, 0] = UAIbotQPTaskController.scalar(1.0 - z_d.T * z_e)

    return r



def position_and_z_axis_task_jacobian(robot, q, Jg, fk, t, htm_d, **kwargs):
    """
    Compute the Jacobian for the position plus z-axis alignment task.

    The Jacobian is assembled as:

        Jr[0:3, :] = Jv
        Jr[3, :]   = 0
        Jr[4, :]   = 0
        Jr[5, :]   = z_d.T * S(z_e) * Jw

    where Jv and Jw are the linear and angular parts of the end-effector
    geometric Jacobian.
    """

    Jg = np.matrix(Jg, dtype=float)
    fk = UAIbotQPTaskController.as_htm(fk, name="fk")
    htm_ref = UAIbotQPTaskController.evaluate_htm_reference(htm_d, t=t)

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


# =============================================================================
# Extra CBF: cube vs. scene
# =============================================================================


def cube_environment_cbf(
    robot,
    q,
    Jg,
    fk,
    t,
    cube,
    cube_obstacles,
    margin,
):
    """
    Build CBF constraints between the cube and selected scene objects.

    This function is the place where the cube-environment restriction is built.

    The function assumes that the cube is rigidly attached to the end-effector
    whenever this CBF is active. Therefore, the velocity of the relevant point on
    the cube is computed from the linear and angular velocity of the
    end-effector.

    The objects checked against the cube are passed through cube_obstacles. This
    selection changes depending on the mode:

    - after grasping, the pickup table is removed from the list;
    - during transport, the relevant scene obstacles are included;
    - during placement, the destination table is removed from the list.

    The inequality convention is:

        A u >= b

    For h = dist - margin >= 0, the CBF condition used here is:

        d(dist)/dq * u >= -eta * (dist - margin)
    """

    q = UAIbotQPTaskController.as_col(q, name="q")
    Jg = np.matrix(Jg, dtype=float)
    fk = UAIbotQPTaskController.as_htm(fk, name="fk")

    n = q.shape[0]
    A = np.matrix(np.zeros((0, n)))
    b = np.matrix(np.zeros((0, 1)))

    if cube_obstacles is None or len(cube_obstacles) == 0:
        return A, b

    s_e = fk[0:3, 3]
    Jv = Jg[0:3, :]
    Jw = Jg[3:6, :]

    for obs in cube_obstacles:
        try:
            point_cube, point_obs, dist, _ = cube.compute_dist(
                obs,
                h=distance_h,
                eps=distance_eps,
                mode=distance_mode,
            )
        except TypeError:
            point_cube, point_obs, dist, _ = cube.compute_dist(obs)

        dist = max(float(dist), 1e-9)

        point_cube = UAIbotQPTaskController.as_col(point_cube, 3, name="point_cube")
        point_obs = UAIbotQPTaskController.as_col(point_obs, 3, name="point_obs")

        direction = point_cube - point_obs
        lever = point_cube - s_e

        cross_term = np.cross(
            np.asarray(lever.T).reshape(3),
            np.asarray(direction.T).reshape(3),
        )
        cross_term = np.matrix(cross_term).reshape((1, 3))

        jac_dist = (direction.T * Jv + cross_term * Jw) / dist

        A = np.vstack((A, jac_dist))
        b = np.vstack((b, np.matrix([[-eta * (dist - margin)]])))

    return A, b



def make_cube_environment_cbf(cube, cube_obstacles, margin):
    """
    Create a cube-specific CBF function for a selected obstacle list.

    The returned function closes over cube, cube_obstacles, and margin. This
    lets the controller receive a simple CBF function without needing an
    extra_cbf_args dictionary.
    """

    def cbf(robot, q, Jg, fk, t):
        return cube_environment_cbf(
            robot=robot,
            q=q,
            Jg=Jg,
            fk=fk,
            t=t,
            cube=cube,
            cube_obstacles=cube_obstacles,
            margin=margin,
        )

    return cbf


# =============================================================================
# Error and mode-switch helpers
# =============================================================================


def compute_grasp_errors(robot, q, grasp_htm):
    """
    Compute the errors used to validate the grasp.

    Returns the end-effector position error and the error between the current
    z-axis and the desired grasp z-axis.
    """

    _, fk = robot.jac_geo(q)

    fk = UAIbotQPTaskController.as_htm(fk, name="fk")
    grasp_htm = UAIbotQPTaskController.as_htm(grasp_htm, name="grasp_htm")

    s_e = fk[0:3, 3]
    z_e = fk[0:3, 2]

    s_d = grasp_htm[0:3, 3]
    z_d = grasp_htm[0:3, 2]

    position_error = float(np.linalg.norm(s_e - s_d))
    z_axis_error = float(np.linalg.norm(z_e - z_d))

    return position_error, z_axis_error



def check_target_reached(robot, q, r, target_htm, use_grasp_check, target_tol):
    """
    Check whether the active target has been reached.

    For the grasp mode, a dedicated position and z-axis check is used. For all
    other modes, the norm of the task function is used.
    """

    if use_grasp_check:
        position_error, z_axis_error = compute_grasp_errors(robot, q, target_htm)
        return position_error <= grasp_position_tol and z_axis_error <= grasp_z_axis_tol

    return float(np.linalg.norm(r)) <= target_tol



def mode_name(mode):
    """
    Return a readable name for the current mode.
    """

    if mode == mode_pregrasp:
        return "pregrasp_above_cube"
    if mode == mode_grasp:
        return "grasp_cube"
    if mode == mode_lift:
        return "lift_after_grasp"
    if mode == mode_move_above_place:
        return "move_above_final_table"
    if mode == mode_place:
        return "place_cube_on_table"
    if mode == mode_finish_above_place:
        return "finish_above_final_table"
    if mode == mode_done:
        return "done"
    return "unknown"



def print_loop_status(robot, q, r, mode, target_htm, use_grasp_check):
    """
    Print a compact status line for the current loop iteration.
    """

    r_norm = float(np.linalg.norm(r))

    if use_grasp_check:
        position_error, z_axis_error = compute_grasp_errors(robot, q, target_htm)

        print(
            f"mode={mode} ({mode_name(mode)}), "
            f"||r||={r_norm:.4f}, "
            f"pos={position_error:.4f}, z={z_axis_error:.4f}"
        )
    else:
        print(f"mode={mode} ({mode_name(mode)}), ||r||={r_norm:.4f}")


# =============================================================================
# Scene creation
# =============================================================================


def create_scene():
    """
    Create the robot, cube, tables, and obstacle used in the scene.
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

    table_pick = ub.Box(
        name="table_pick",
        htm=ub.Utils.trn([0.8, 0, 0.15]),
        width=0.5,
        depth=0.5,
        height=0.4,
        mesh_material=material_table,
    )

    table_place = ub.Box(
        name="table_place",
        htm=ub.Utils.trn([0, -0.8, 0.15]),
        width=0.5,
        depth=0.5,
        height=0.4,
        mesh_material=material_table,
    )

    table_base = ub.Box(
        name="table_base",
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

    # List used only for robot-environment collision constraints.
    # The cube is kept out of this list to allow approach and grasping.
    environment_obstacles = [table_pick, table_place, obstacle]

    return robot, cube, table_pick, table_place, table_base, obstacle, environment_obstacles


# =============================================================================
# Mode configuration
# =============================================================================


def get_mode_configuration(
    mode,
    cube_pregrasp_htm,
    cube_grasp_htm,
    final_above_htm,
    final_place_htm,
    cube_cbf_after_pick,
    cube_cbf_while_carrying,
    cube_cbf_during_place,
):
    """
    Return the control parameters associated with the current mode.

    The function returns the tuple:

        target_htm,
        use_grasp_task,
        use_grasp_check,
        target_tol,
        obstacle_margin,
        self_collision_margin,
        cube_cbf_function

    This avoids dictionaries while keeping explicit what changes from one mode
    to the next.
    """

    if mode == mode_pregrasp:
        return (
            cube_pregrasp_htm,
            False,
            False,
            pose_target_tol,
            default_obstacle_margin,
            default_self_collision_margin,
            None,
        )

    if mode == mode_grasp:
        return (
            cube_grasp_htm,
            True,
            True,
            pose_target_tol,
            grasp_obstacle_margin,
            grasp_self_collision_margin,
            None,
        )

    if mode == mode_lift:
        return (
            cube_pregrasp_htm,
            False,
            False,
            pose_target_tol,
            default_obstacle_margin,
            default_self_collision_margin,
            cube_cbf_after_pick,
        )

    if mode == mode_move_above_place:
        return (
            final_above_htm,
            False,
            False,
            pose_target_tol,
            default_obstacle_margin,
            default_self_collision_margin,
            cube_cbf_while_carrying,
        )

    if mode == mode_place:
        return (
            final_place_htm,
            False,
            False,
            place_target_tol,
            place_obstacle_margin,
            default_self_collision_margin,
            cube_cbf_during_place,
        )

    if mode == mode_finish_above_place:
        return (
            final_above_htm,
            False,
            False,
            final_target_tol,
            default_obstacle_margin,
            default_self_collision_margin,
            None,
        )

    raise ValueError(f"Invalid mode: {mode}")


# =============================================================================
# Controller creation
# =============================================================================


def create_qp_controller(
    robot,
    target_htm,
    environment_obstacles,
    use_grasp_task=False,
    obstacle_margin=default_obstacle_margin,
    self_collision_margin=default_self_collision_margin,
    cube_environment_cbf_function=None,
):
    """
    Create the QP/CBF controller for the active target.

    When cube_environment_cbf_function is not None, the corresponding
    cube-scene CBF is added as an extra constraint.
    """

    if use_grasp_task:
        task_function = position_and_z_axis_task_function
        task_jacobian = position_and_z_axis_task_jacobian
    else:
        task_function = None
        task_jacobian = None

    if cube_environment_cbf_function is None:
        extra_cbf_functions = None
    else:
        extra_cbf_functions = [cube_environment_cbf_function]

    return UAIbotQPTaskController(
        robot=robot,
        htm_d=target_htm,
        F=task_dynamics,
        F_args=None,
        task_function=task_function,
        task_jacobian=task_jacobian,
        damping=damping,
        time_derivative_mode="none",
        regularization=regularization,
        eta=eta,
        obstacles=environment_obstacles,
        obstacle_margin=obstacle_margin,
        care_obstacles=True,
        care_self_collision=care_self_collision,
        self_collision_margin=self_collision_margin,
        care_joint_limits=care_joint_limits,
        joint_margin=joint_margin,
        care_velocity_limits=care_velocity_limits,
        max_qdot=max_qdot,
        distance_tol=distance_tol,
        distance_no_iter_max=distance_no_iter_max,
        distance_max_dist=distance_max_dist,
        distance_h=distance_h,
        distance_eps=distance_eps,
        distance_mode=distance_mode,
        extra_cbf_functions=extra_cbf_functions,
        extra_cbf_args=None,
    )


# =============================================================================
# Plots
# =============================================================================


def plot_histories(hist_t, hist_r_norm, hist_u, hist_mode):
    """
    Plot the task norm, active mode, and control actions.
    """

    if len(hist_u) > 0:
        hist_u_array = np.asarray(np.hstack(hist_u), dtype=float)
    else:
        hist_u_array = np.zeros((1, 0))

    plt.figure()
    plt.plot(hist_t, hist_r_norm)
    plt.xlabel("Tempo (s)")
    plt.ylabel("||r||")
    plt.title("Norma da função de tarefa")
    plt.grid(True)

    plt.figure()
    plt.step(hist_t, hist_mode, where="post")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Modo")
    plt.title("Modo ativo")
    plt.grid(True)

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
# Main execution
# =============================================================================


def run_pick_place_demo():
    """
    Run the reactive pick-and-place demo in a single control loop.
    """

    (
        robot,
        cube,
        table_pick,
        table_place,
        table_base,
        obstacle,
        environment_obstacles,
    ) = create_scene()

    cube_grasp_htm = ub.Utils.trn([0.8, 0, 0.45]) * ub.Utils.roty(np.pi)
    cube_pregrasp_htm = (
        ub.Utils.trn([0.8, 0, 0.45 + approach_clearance])
        * ub.Utils.roty(np.pi)
    )

    final_place_htm = ub.Utils.trn([0, -0.8, 0.45]) * ub.Utils.rotx(np.pi)
    final_above_htm = (
        ub.Utils.trn([0, -0.8, 0.45 + place_clearance])
        * ub.Utils.rotx(np.pi)
    )

    # -------------------------------------------------------------------------
    # Cube-scene constraint construction
    # -------------------------------------------------------------------------
    # These CBFs protect the cube while it is attached to the end-effector. Each
    # one uses a different list of scene objects.
    #
    # After grasping: the pickup table is ignored.
    # During transport: pickup table, destination table, and obstacle are used.
    # During placement: the destination table is ignored.
    # -------------------------------------------------------------------------

    cube_cbf_after_pick = make_cube_environment_cbf(
        cube=cube,
        cube_obstacles=[table_place, obstacle],
        margin=cube_obstacle_margin,
    )

    cube_cbf_while_carrying = make_cube_environment_cbf(
        cube=cube,
        cube_obstacles=[table_pick, table_place, obstacle],
        margin=cube_obstacle_margin,
    )

    cube_cbf_during_place = make_cube_environment_cbf(
        cube=cube,
        cube_obstacles=[table_pick, obstacle],
        margin=cube_obstacle_margin,
    )

    frames = [
        ub.Frame(cube_pregrasp_htm, size=0.12),
        ub.Frame(cube_grasp_htm, size=0.12),
        ub.Frame(final_above_htm, size=0.12),
        ub.Frame(final_place_htm, size=0.12),
    ]

    sim = ub.Simulation.create_sim_factory([
        robot,
        table_pick,
        table_place,
        table_base,
        obstacle,
        cube,
        *frames,
    ])

    q = np.matrix(robot.q)
    t = 0.0
    mode = mode_pregrasp
    previous_mode = None
    controller = None
    cube_attached = False

    hist_t = []
    hist_r_norm = []
    hist_u = []
    hist_mode = []

    # -------------------------------------------------------------------------
    # Single execution loop
    # -------------------------------------------------------------------------

    while t < tmax and mode != mode_done:
        if mode != previous_mode:
            (
                target_htm,
                use_grasp_task,
                use_grasp_check,
                target_tol,
                obstacle_margin,
                self_collision_margin,
                cube_environment_cbf_function,
            ) = get_mode_configuration(
                mode=mode,
                cube_pregrasp_htm=cube_pregrasp_htm,
                cube_grasp_htm=cube_grasp_htm,
                final_above_htm=final_above_htm,
                final_place_htm=final_place_htm,
                cube_cbf_after_pick=cube_cbf_after_pick,
                cube_cbf_while_carrying=cube_cbf_while_carrying,
                cube_cbf_during_place=cube_cbf_during_place,
            )

            controller = create_qp_controller(
                robot=robot,
                target_htm=target_htm,
                environment_obstacles=environment_obstacles,
                use_grasp_task=use_grasp_task,
                obstacle_margin=obstacle_margin,
                self_collision_margin=self_collision_margin,
                cube_environment_cbf_function=cube_environment_cbf_function,
            )

            previous_mode = mode
            print(f"\nEntering mode {mode}: {mode_name(mode)}")

        u, r, Jr, Fr, rt, qp_data = controller.compute_control(q=q, t=t)
        r_norm = float(np.linalg.norm(r))

        print_loop_status(
            robot=robot,
            q=q,
            r=r,
            mode=mode,
            target_htm=target_htm,
            use_grasp_check=use_grasp_check,
        )

        reached = check_target_reached(
            robot=robot,
            q=q,
            r=r,
            target_htm=target_htm,
            use_grasp_check=use_grasp_check,
            target_tol=target_tol,
        )

        if reached:
            if mode == mode_grasp and not cube_attached:
                robot.attach_object(cube)
                cube_attached = True
                print("Cube attached.")

            elif mode == mode_place and cube_attached:
                robot.detach_object(cube)
                cube_attached = False
                print("Cube detached.")

            mode += 1
            continue

        q_next = q + u * dt
        t_next = t + dt

        robot.add_ani_frame(time=t_next, q=q_next)

        hist_t.append(t)
        hist_r_norm.append(r_norm)
        hist_u.append(np.matrix(u))
        hist_mode.append(mode)

        q = q_next
        t = t_next

    if cube_attached:
        robot.detach_object(cube)
        print("Cube detached at the end because it was still attached.")

    sim.run()
    # plot_histories(hist_t, hist_r_norm, hist_u, hist_mode)


if __name__ == "__main__":
    run_pick_place_demo()
