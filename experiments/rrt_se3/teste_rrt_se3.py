import numpy as np
import os
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
LOCAL_UAIBOTPY = PROJECT_ROOT / "UAIbotPy"

experiment_dir_str = str(EXPERIMENT_DIR)
local_uaibotpy_str = str(LOCAL_UAIBOTPY)

if local_uaibotpy_str not in sys.path:
    sys.path.insert(0, local_uaibotpy_str)

import uaibot as ub

# Depois de carregar o UAIBot local, devolvemos o experimento ao topo.
# Assim, "from setup import *" usa experiments/rrt_se3/setup.py.
if local_uaibotpy_str in sys.path:
    sys.path.remove(local_uaibotpy_str)
if experiment_dir_str in sys.path:
    sys.path.remove(experiment_dir_str)
sys.path.insert(0, experiment_dir_str)

from scipy.linalg import expm

from aux_functions import *


def carregar_htm(nome_arquivo):
    caminho_arquivo = EXPERIMENT_DIR / nome_arquivo

    htms = []
    matriz_atual = []

    with open(caminho_arquivo, "r") as f:
        for linha in f:
            linha = linha.strip()

            if linha == "":
                if matriz_atual:
                    htms.append(np.matrix(matriz_atual))
                    matriz_atual = []
            else:
                valores = [float(v) for v in linha.split()]
                matriz_atual.append(valores)

        if matriz_atual:
            htms.append(np.matrix(matriz_atual))

    return np.array(htms)


def draw_pc(path, sim, color="white", radius=0.02):
    sl = []
    for htm in path:
        sl.append(htm[0:3, 3])
    pc = ub.PointCloud(size=radius, color=color, points=sl)
    sim.add(pc)


def propagate_htm(htm, xi, dt_step):
    p = htm[0:3, 3].reshape(3, 1)
    R = htm[0:3, 0:3]

    v = np.asarray(xi[0:3]).reshape(3, 1)
    w = np.asarray(xi[3:6]).reshape(3, 1)

    R_next = expm(ub.Utils.S(w) * dt_step) @ R
    p_next = p + v * dt_step

    htm_next = np.eye(4)
    htm_next[0:3, 0:3] = R_next
    htm_next[0:3, 3] = p_next.flatten()

    return np.matrix(htm_next)


def modulation(H, H_target, lam):
    d = np.linalg.norm(log_SE3(ub.Utils.inv_htm(H) @ H_target))
    return 1.0 - np.exp(-lam * d) * (1.0 + lam * d)


def eval_xid_from_state(state_htm, htm_path, kt1, kt2, kt3, kn1, kn2, ds, lambdaa):
    xid, dist, idx = ub.Robot.vector_field_SE3(
        state=state_htm,
        curve=htm_path,
        kt1=kt1,
        kt2=kt2,
        kt3=kt3,
        kn1=kn1,
        kn2=kn2,
        ds=ds,
        delta=1e-3,
    )

    xid = np.asarray(xid, dtype=float).reshape(6, 1)

    # Ajuste igual ao seu codigo antigo
    xid[0:3, :] = xid[0:3, :] + ub.Utils.S(xid[3:6, :]) @ state_htm[0:3, -1].reshape(3, 1)

    alpha = modulation(state_htm, htm_path[-1], lam=lambdaa)

    return xid * alpha, dist, idx


def saturate_twist(xi, xi_max):
    xi = np.asarray(xi, dtype=float).reshape(6, 1)
    xi_max = np.asarray(xi_max, dtype=float).reshape(6, 1)

    return np.minimum(np.maximum(xi, -xi_max), xi_max)


def get_position_bounds():
    return np.array([
        [-1.2, 2.2],
        [-0.2, 3.3],
        [0.1, 1.45],
    ], dtype=float)


def get_rrt_options():
    return {
        "ell": 0.15,
        "max_iterations": 6000,
        "step_size": 0.25,
        "goal_tolerance": 0.08,
        "edge_resolution": 0.015,
        "output_resolution": 0.005,
        "connect_resolution": 0.06,
        "goal_bias": 0.08,
        "other_tree_bias": 0.35,
        "shortcut_iterations": 80,
        "collision_tol": 1e-4,
        "collision_dist_tol": 1e-3,
        "collision_no_iter_max": 20,
    }


def create_robot_body():
    return ub.Cylinder(
        htm=ub.Utils.trn([0, 0, 0]) * ub.Utils.roty(np.pi),
        name="robot_body",
        radius=0.3,
        height=0.17,
        color="cyan",
        opacity=0.55,
    )


def _box(htm, width, depth, height, mesh_material=None):
    kwargs = {"mesh_material": mesh_material} if mesh_material is not None else {}
    return ub.Box(htm=htm, width=width, depth=depth, height=height, **kwargs)


def _cylinder(htm, height, radius, mesh_material=None):
    kwargs = {"mesh_material": mesh_material} if mesh_material is not None else {}
    return ub.Cylinder(htm=htm, height=height, radius=radius, **kwargs)


def create_obstacles(material_wood=None, material_steel=None):
    piso = _box(
        htm=ub.Utils.trn([0, 0, -0.2]),
        width=7,
        depth=7,
        height=0.05,
        mesh_material=material_wood,
    )

    teto = _box(
        htm=ub.Utils.trn([0, 0, 1.74]),
        width=7,
        depth=7,
        height=0.05,
        mesh_material=material_wood,
    )

    parede_frente = _box(
        htm=ub.Utils.trn([0, 2, 0.8]),
        width=3,
        depth=0.1,
        height=1.9,
        mesh_material=material_wood,
    )

    parede_fundo = _box(
        htm=ub.Utils.trn([0, 3.5, 0.8]),
        width=7,
        depth=0.1,
        height=1.9,
        mesh_material=material_wood,
    )

    parede_lateral = _box(
        htm=ub.Utils.trn([-1.5, 2.75, 0.8]) * ub.Utils.rotz(np.pi / 2),
        width=1.5,
        depth=0.1,
        height=1.9,
        mesh_material=material_wood,
    )

    parede_sup = _box(
        htm=ub.Utils.trn([1.3, 2.42, 1.37]) * ub.Utils.rotz(np.pi / 2),
        width=0.75,
        depth=0.1,
        height=0.95,
        mesh_material=material_steel,
    )

    parede_inf = _box(
        htm=ub.Utils.trn([1.3, 2.42, -0.5]) * ub.Utils.rotz(np.pi / 2),
        width=0.75,
        depth=0.1,
        height=0.95,
        mesh_material=material_steel,
    )

    parede_sup_lat = _box(
        htm=ub.Utils.trn([1.3, 3.16, 0.8]) * ub.Utils.rotz(np.pi / 2),
        width=0.74,
        depth=0.1,
        height=1.9,
        mesh_material=material_steel,
    )

    pilar = _cylinder(
        htm=ub.Utils.trn([1.35, 1, 1]),
        height=2,
        radius=0.05,
        mesh_material=material_steel,
    )

    unknown_obs = [parede_sup, parede_sup_lat, pilar]
    known_obs = [parede_frente, piso, teto, parede_fundo, parede_lateral, parede_inf]
    return known_obs + unknown_obs


def create_rrt_test_scenario():
    robot_body = create_robot_body()
    robot_collision_model = [robot_body]
    H_start = np.matrix(ub.Utils.trn([0, 0, 0.1]) * ub.Utils.roty(np.pi))
    htm_ref = carregar_htm("data/caminho.txt")
    H_goal = np.matrix(htm_ref[-1])
    obstacles = create_obstacles()
    return H_start, H_goal, get_position_bounds(), robot_collision_model, obstacles


def plan_rrt_path(H_start, H_goal, robot_collision_model, obstacles):
    options = get_rrt_options()
    result = ub.Utils.rrt_se3_bidirectional(
        h_start=H_start,
        h_goal=H_goal,
        position_bounds=get_position_bounds(),
        robot_model=robot_collision_model,
        obstacles=obstacles,
        **options,
    )

    print("RRT success:", result.success)
    print("RRT message:", result.message)
    print("RRT iterations:", result.iterations)
    print("RRT nodes start:", result.number_of_nodes_start)
    print("RRT nodes goal:", result.number_of_nodes_goal)
    print("RRT waypoints:", len(result.path))
    print("RRT discrete poses:", len(result.path_discrete))
    print("RRT time: ",result.execution_time," s")
    if not result.success:
        raise RuntimeError(result.message)

    return np.array([np.matrix(H) for H in result.path_discrete])


def main():
    # ============================================================
    # Simulacao
    # ============================================================

    sim = ub.Simulation.create_sim_hill()

    # ============================================================
    # Drone
    # ============================================================

    robot_body = create_robot_body()

    robot_3d_model = ub.Model3D(
        url="https://cdn.jsdelivr.net/gh/pedro55562/SE3_CBF_ASSETS@main/TEMA12_DRONA6.obj",
        scale=0.0009,
        mesh_material=ub.MeshMaterial.create_rough_metal(),
    )

    robot_frame = ub.Frame(size=0.10)

    robot_rigid_3d = ub.RigidObject(
        list_model_3d=[robot_3d_model],
        htm=ub.Utils.trn([0, 0, -0.05]) * ub.Utils.roty(np.pi),
    )

    robot_UAV = ub.Group(
        list_of_objects=[robot_body, robot_rigid_3d, robot_frame],
        htm=ub.Utils.trn([0, 0, 0.1]) * ub.Utils.roty(np.pi),
    )

    sim.add([robot_UAV])

    # ============================================================
    # Ambiente e obstaculos
    # ============================================================

    texture_steel = ub.Texture(
        url="https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/Textures/rough_metal.jpg",
        wrap_s="RepeatWrapping",
        wrap_t="RepeatWrapping",
        repeat=[4, 4],
    )

    material_steel = ub.MeshMaterial(
        metalness=0.7,
        clearcoat=1,
        roughness=0.5,
        normal_scale=[0.5, 0.5],
        texture_map=texture_steel,
    )

    material_wood = ub.MeshMaterial.create_wood()
    all_obs = create_obstacles(material_wood=material_wood, material_steel=material_steel)
    sim.add(all_obs)

    # ============================================================
    # Caminho planejado com RRT e alvo
    # ============================================================

    htm_ref = carregar_htm("data/caminho.txt")
    htm_target = np.matrix(htm_ref[-1])

    robot_collision_model = [robot_body]
    H_start = np.matrix(robot_UAV.htm)

    htm_path = plan_rrt_path(
        H_start=H_start,
        H_goal=htm_target,
        robot_collision_model=robot_collision_model,
        obstacles=all_obs,
    )

    frame_target = ub.Frame(htm=htm_target, size=0.18)
    sim.add([frame_target])

    # Caminho planejado pelo RRT em branco
    draw_pc(path=htm_path, sim=sim, color="white", radius=0.018)

    # ============================================================
    # Parametros do vector field
    # ============================================================

    dt = 0.01
    dt_num = 0.085
    t_max = 80.0

    kt1 = 9
    kt2 = 1
    kt3 = 1

    kn1 = 1
    kn2 = 1

    lambdaa = 15.0

    xi_max = np.array([
        [2.5],
        [2.5],
        [2.5],
        [2.5],
        [2.5],
        [2.5],
    ])

    # ============================================================
    # Loop de simulacao
    # ============================================================

    H = np.matrix(robot_UAV.htm)

    path_followed = []
    error_list = []

    ball_tr = ub.Ball(htm=np.identity(4), radius=0.025, color="cyan")
    sim.add([ball_tr])

    for k in range(int(t_max / dt)):
        t = k * dt

        err = np.linalg.norm(log_SE3(ub.Utils.inv_htm(H) @ htm_target))
        error_list.append(err)

        if err < 0.025:
            print("Chegou no alvo.")
            print("Erro final:", err)
            break

        xid, dist, idx = eval_xid_from_state(
            state_htm=H,
            htm_path=htm_path,
            kt1=kt1,
            kt2=kt2,
            kt3=kt3,
            kn1=kn1,
            kn2=kn2,
            ds=dt_num,
            lambdaa=lambdaa,
        )

        xid = saturate_twist(xid, xi_max)

        H = propagate_htm(H, xid, dt)

        robot_UAV.add_ani_frame(time=t, htm=H)
        ball_tr.add_ani_frame(time=t, htm=htm_path[idx])

        path_followed.append(H.copy())

    # ============================================================
    # Desenhar caminho seguido e salvar animacao
    # ============================================================

    if len(path_followed) > 0:
        draw_pc(path=path_followed, sim=sim, color="magenta", radius=0.012)
        print("Erro final:", error_list[-1])

    SAVE_ANIMATION = True

    if SAVE_ANIMATION:
        sim.save(
            address=str(PROJECT_ROOT / "outputs"),
            file_name="se3_vectorfield_only",
        )


if __name__ == "__main__":
    main()
