import uaibot as ub
import numpy as np
from src.SE3_utilities import *
import os

_EPS = 1e-12
_SMALL_ANGLE = 1e-8
_NEAR_PI = 1e-6


# =============================================================================
# Funcoes auxiliares para SE(3)
# =============================================================================


def make_SE3(R, p):
    """
    Monta uma matriz homogenea H em SE(3).

        H = [ R  p ]
            [ 0  1 ]
    """

    R = as_matrix(R, shape=(3, 3), name="R")
    p = col(p, 3, name="p")

    H = np.matrix(np.eye(4))
    H[0:3, 0:3] = R
    H[0:3, 3] = p

    return H


def get_position(H):
    """
    Retorna a posicao p de uma HTM.
    """

    H = as_matrix(H, shape=(4, 4), name="H")
    return H[0:3, 3]


def get_rotation(H):
    """
    Retorna a rotacao R de uma HTM.
    """

    H = as_matrix(H, shape=(4, 4), name="H")
    return H[0:3, 0:3]


def to_np(H):
    """
    Converte np.matrix para np.array.
    O UAIBot costuma aceitar melhor np.array.
    """

    return np.asarray(H, dtype=float)


# =============================================================================
# Metricas / distance-like functions em SE(3)
# =============================================================================


def make_metric_matrix(ell):
    """
    Cria a matriz G usada na metrica logaritmica.

    A convencao do twist eh:

        xi = [v; w]

    A metrica fica:

        d(H1,H2)^2 = ||v||^2 + ell^2 ||w||^2

    Logo, 1 radiano vale ell metros.
    """

    ell = float(ell)

    if ell <= 0.0:
        raise ValueError("ell must be positive.")

    G = np.matrix(np.diag([
        1.0, 1.0, 1.0,
        ell**2, ell**2, ell**2,
    ]))

    return G


def metric_log_SE3_left(H1, H2, G):
    """
    Metrica logaritmica left-invariant em SE(3).

        H_rel = inv(H1) * H2
        A = log(H_rel)
        xi = vee(A)
        d = sqrt(xi.T * G * xi)

    Essa eh a mais importante para comecar.
    """

    H1 = as_matrix(H1, shape=(4, 4), name="H1")
    H2 = as_matrix(H2, shape=(4, 4), name="H2")
    G = as_matrix(G, shape=(6, 6), name="G")

    H_rel = inv_SE3(H1) * H2
    A = log_SE3(H_rel)
    xi = vee_SE3(A)

    d2 = (xi.T * G * xi).item()

    return np.sqrt(max(d2, 0.0))


def metric_log_SE3_right(H1, H2, G):
    """
    Metrica logaritmica right-invariant em SE(3).

        H_rel = H2 * inv(H1)
        A = log(H_rel)
        xi = vee(A)
        d = sqrt(xi.T * G * xi)
    """

    H1 = as_matrix(H1, shape=(4, 4), name="H1")
    H2 = as_matrix(H2, shape=(4, 4), name="H2")
    G = as_matrix(G, shape=(6, 6), name="G")

    H_rel = H2 * inv_SE3(H1)
    A = log_SE3(H_rel)
    xi = vee_SE3(A)

    d2 = (xi.T * G * xi).item()

    return np.sqrt(max(d2, 0.0))


def metric_log_SE3_symmetric(H1, H2, G):
    """
    Combinacao pratica das metricas left e right.

    Nao eh necessaria para comecar, mas eh util para comparar.
    """

    d_left = metric_log_SE3_left(H1, H2, G)
    d_right = metric_log_SE3_right(H1, H2, G)

    return np.sqrt(0.5 * (d_left**2 + d_right**2))


def transform_point(H, q):
    """
    Transforma um ponto q fixo no corpo pela pose H.
    """

    H = as_matrix(H, shape=(4, 4), name="H")
    q = col(q, 3, name="q")

    q_hom = np.vstack((q, np.matrix([[1.0]])))
    p_hom = H * q_hom

    return p_hom[0:3, 0]


def metric_object_points(H1, H2, body_points):
    """
    Metrica induzida por pontos do objeto.

    Ela mede o RMS do deslocamento dos pontos do corpo:

        d = sqrt( 1/N sum ||H1 qi - H2 qi||^2 )

    Boa para um copo, uma caixa, uma ferramenta, etc.
    """

    if len(body_points) == 0:
        raise ValueError("body_points must not be empty.")

    acc = 0.0

    for q in body_points:
        p1 = transform_point(H1, q)
        p2 = transform_point(H2, q)

        e = p2 - p1
        acc += float(e.T * e)

    return np.sqrt(acc / len(body_points))


# =============================================================================
# Interpolacao em SE(3)
# =============================================================================


def interpolate_SE3_left(H1, H2, s):
    """
    Interpolacao em SE(3) usando exponencial e logaritmo.

        H(s) = H1 * exp( s * log( inv(H1) * H2 ) )

    com s em [0,1].
    """

    H1 = as_matrix(H1, shape=(4, 4), name="H1")
    H2 = as_matrix(H2, shape=(4, 4), name="H2")

    s = float(np.clip(s, 0.0, 1.0))

    A = log_SE3(inv_SE3(H1) * H2)

    return H1 * exp_SE3(s * A)


def interpolate_SE3_right(H1, H2, s):
    """
    Interpolacao right-invariant.

        H(s) = exp( s * log( H2 * inv(H1) ) ) * H1

    Para o projeto, eu comecaria usando a left.
    """

    H1 = as_matrix(H1, shape=(4, 4), name="H1")
    H2 = as_matrix(H2, shape=(4, 4), name="H2")

    s = float(np.clip(s, 0.0, 1.0))

    A = log_SE3(H2 * inv_SE3(H1))

    return exp_SE3(s * A) * H1


def interpolate_SE3_path(H1, H2, n_points):
    """
    Gera uma lista de poses interpoladas entre H1 e H2.
    Inclui H1 e H2.
    """

    n_points = int(n_points)

    if n_points < 2:
        raise ValueError("n_points must be at least 2.")

    path = []

    for i in range(n_points):
        s = i / (n_points - 1)
        path.append(interpolate_SE3_left(H1, H2, s))

    return path


def steer_SE3(H1, H2, step_size, G):
    """
    Um passo limitado em SE(3).

    Nao eh o RRT. Isso eh so o bloco de steering:

        H_new = H1 * exp(alpha * log(inv(H1)*H2))

    com alpha escolhido para andar no maximo step_size.
    """

    step_size = float(step_size)

    if step_size <= 0.0:
        raise ValueError("step_size must be positive.")

    d = metric_log_SE3_left(H1, H2, G)

    if d < _EPS:
        return as_matrix(H1, shape=(4, 4), name="H1").copy()

    alpha = min(1.0, step_size / d)

    return interpolate_SE3_left(H1, H2, alpha)


# =============================================================================
# Amostragem uniforme em B x SO(3)
# =============================================================================


def quat_to_rot(q):
    """
    Converte quaternion unitario para matriz de rotacao.

    Convencao:

        q = [qw, qx, qy, qz]
    """

    q = col(q, 4, name="q")

    n = float(np.linalg.norm(q))

    if n < _EPS:
        raise ValueError("Quaternion norm is too close to zero.")

    q = q / n

    qw = float(q[0, 0])
    qx = float(q[1, 0])
    qy = float(q[2, 0])
    qz = float(q[3, 0])

    R = np.matrix([
        [
            1.0 - 2.0 * (qy*qy + qz*qz),
            2.0 * (qx*qy - qw*qz),
            2.0 * (qx*qz + qw*qy),
        ],
        [
            2.0 * (qx*qy + qw*qz),
            1.0 - 2.0 * (qx*qx + qz*qz),
            2.0 * (qy*qz - qw*qx),
        ],
        [
            2.0 * (qx*qz - qw*qy),
            2.0 * (qy*qz + qw*qx),
            1.0 - 2.0 * (qx*qx + qy*qy),
        ],
    ])

    return R


def sample_SO3_uniform():
    """
    Amostra R em SO(3) de forma uniforme.

    Metodo:
        1. amostra q em R4 com normal padrao
        2. normaliza q para a esfera S3
        3. converte q para matriz de rotacao
    """

    q = np.random.normal(size=(4, 1))
    q = q / np.linalg.norm(q)

    return quat_to_rot(q)


def sample_SE3_uniform_box(position_bounds):
    """
    Amostra uniformemente em B x SO(3).

    position_bounds:
        [
            [x_min, x_max],
            [y_min, y_max],
            [z_min, z_max],
        ]

    Observacao:
        Nao existe distribuicao uniforme normalizada em todo SE(3),
        pois a parte translacional eh infinita.
    """

    if len(position_bounds) != 3:
        raise ValueError("position_bounds must have three intervals.")

    x_min, x_max = position_bounds[0]
    y_min, y_max = position_bounds[1]
    z_min, z_max = position_bounds[2]

    p = np.matrix([
        [np.random.uniform(x_min, x_max)],
        [np.random.uniform(y_min, y_max)],
        [np.random.uniform(z_min, z_max)],
    ])

    R = sample_SO3_uniform()

    return make_SE3(R, p)


# =============================================================================
# Testes numericos
# =============================================================================


def assert_close(name, A, B, tol=1e-9):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    err = float(np.max(np.abs(A - B)))

    if err > tol:
        raise AssertionError(f"{name} failed: err = {err:.3e}, tol = {tol:.3e}")


def assert_scalar_close(name, a, b, tol=1e-9):
    err = abs(float(a) - float(b))

    if err > tol:
        raise AssertionError(f"{name} failed: err = {err:.3e}, tol = {tol:.3e}")


def test_make_SE3():
    R = exp_SO3([0.2, -0.1, 0.4])
    p = [1.0, 2.0, 3.0]

    H = make_SE3(R, p)

    assert_close("test_make_SE3 R", H[0:3, 0:3], R)
    assert_close("test_make_SE3 p", H[0:3, 3], col(p, 3))
    assert_close("test_make_SE3 last row", H[3, :], np.matrix([[0.0, 0.0, 0.0, 1.0]]))


def test_metric_zero():
    H = make_SE3(exp_SO3([0.2, 0.1, -0.3]), [0.1, 0.2, 0.3])
    G = make_metric_matrix(0.15)

    d = metric_log_SE3_left(H, H, G)

    assert_scalar_close("test_metric_zero", d, 0.0, tol=1e-10)


def test_metric_translation():
    H1 = make_SE3(np.eye(3), [0.0, 0.0, 0.0])
    H2 = make_SE3(np.eye(3), [1.0, 0.0, 0.0])
    G = make_metric_matrix(0.15)

    d = metric_log_SE3_left(H1, H2, G)

    assert_scalar_close("test_metric_translation", d, 1.0, tol=1e-9)


def test_metric_rotation_value():
    H1 = make_SE3(np.eye(3), [0.0, 0.0, 0.0])
    H2 = make_SE3(exp_SO3([0.0, 0.0, 1.0]), [0.0, 0.0, 0.0])

    ell = 0.15
    G = make_metric_matrix(ell)

    d = metric_log_SE3_left(H1, H2, G)

    assert_scalar_close("test_metric_rotation_value", d, ell, tol=1e-9)


def test_interpolation_endpoints():
    H1 = make_SE3(np.eye(3), [0.0, 0.0, 0.0])

    xi = col([0.5, -0.2, 0.3, 0.2, -0.1, 0.4], 6)
    H2 = H1 * exp_SE3(hat_SE3(xi))

    Hs0 = interpolate_SE3_left(H1, H2, 0.0)
    Hs1 = interpolate_SE3_left(H1, H2, 1.0)

    assert_close("test_interpolation_endpoints s0", Hs0, H1, tol=1e-9)
    assert_close("test_interpolation_endpoints s1", Hs1, H2, tol=1e-9)


def test_interpolation_half_log():
    H1 = make_SE3(np.eye(3), [0.0, 0.0, 0.0])

    xi = col([0.5, 0.2, -0.1, 0.0, 0.0, np.pi / 2.0], 6)
    H2 = H1 * exp_SE3(hat_SE3(xi))

    Hhalf = interpolate_SE3_left(H1, H2, 0.5)

    A_full = log_SE3(inv_SE3(H1) * H2)
    A_half = log_SE3(inv_SE3(H1) * Hhalf)

    assert_close("test_interpolation_half_log", A_half, 0.5 * A_full, tol=1e-8)


def test_steer_step_size():
    H1 = make_SE3(np.eye(3), [0.0, 0.0, 0.0])

    xi = col([1.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2.0], 6)
    H2 = H1 * exp_SE3(hat_SE3(xi))

    G = make_metric_matrix(0.2)
    step_size = 0.2

    Hnew = steer_SE3(H1, H2, step_size, G)
    d = metric_log_SE3_left(H1, Hnew, G)

    assert_scalar_close("test_steer_step_size", d, step_size, tol=1e-8)


def test_sample_SO3_uniform_properties():
    for _ in range(100):
        R = sample_SO3_uniform()

        assert_close("sample_SO3_uniform orthogonality", R.T * R, np.eye(3), tol=1e-9)
        assert_scalar_close("sample_SO3_uniform det", np.linalg.det(R), 1.0, tol=1e-9)


def test_sample_SE3_uniform_box_bounds():
    bounds = [
        [-1.0, 2.0],
        [-3.0, 4.0],
        [0.5, 1.5],
    ]

    for _ in range(100):
        H = sample_SE3_uniform_box(bounds)
        p = get_position(H)

        x = float(p[0, 0])
        y = float(p[1, 0])
        z = float(p[2, 0])

        if not (bounds[0][0] <= x <= bounds[0][1]):
            raise AssertionError("x out of bounds.")

        if not (bounds[1][0] <= y <= bounds[1][1]):
            raise AssertionError("y out of bounds.")

        if not (bounds[2][0] <= z <= bounds[2][1]):
            raise AssertionError("z out of bounds.")


def test_object_points_metric_zero():
    H = make_SE3(exp_SO3([0.1, 0.2, 0.3]), [0.2, -0.1, 0.4])

    points = [
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1],
    ]

    d = metric_object_points(H, H, points)

    assert_scalar_close("test_object_points_metric_zero", d, 0.0, tol=1e-12)


def run_tests():
    tests = [
        test_make_SE3,
        test_metric_zero,
        test_metric_translation,
        test_metric_rotation_value,
        test_interpolation_endpoints,
        test_interpolation_half_log,
        test_steer_step_size,
        test_sample_SO3_uniform_properties,
        test_sample_SE3_uniform_box_bounds,
        test_object_points_metric_zero,
    ]

    for test in tests:
        test()
        print(f"[OK] {test.__name__}")

    print("\nAll primitive tests passed.")


# =============================================================================
# Exemplo UAIBot: interpolacao em SE(3)
# =============================================================================


def demo_uaibot_interpolation():
    """
    Gera um HTML com uma caixa e um frame se movendo por interpolacao em SE(3).

    Saida:
        ./uaibot_out/se3_interpolation.html
    """

    import uaibot as ub

    out_dir = os.path.join(os.getcwd(), "uaibot_out")
    os.makedirs(out_dir, exist_ok=True)

    # Pose inicial
    H1 = make_SE3(
        exp_SO3([0.0, 0.0, 0.0]),
        [0.0, 0.0, 0.15],
    )

    # Pose final criada por um deslocamento em SE(3)
    xi = col([
        0.75, 0.35, 0.20,   # v
        0.70, 0.15, 1.10,   # w
    ], 6)

    H2 = H1 * exp_SE3(hat_SE3(xi))

    # Objeto animado
    moving_box = ub.Box(
        htm=to_np(H1),
        name="moving_box",
        width=0.12,
        depth=0.08,
        height=0.08,
        color="red",
        opacity=0.85,
    )

    moving_frame = ub.Frame(
        htm=to_np(H1),
        name="moving_frame",
        size=0.18,
    )

    # Frames fixos para marcar inicio e fim
    start_frame = ub.Frame(
        htm=to_np(H1),
        name="start_frame",
        size=0.22,
    )

    goal_frame = ub.Frame(
        htm=to_np(H2),
        name="goal_frame",
        size=0.22,
    )

    n_frames = 160
    dt = 0.03

    for i in range(n_frames):
        s = i / (n_frames - 1)
        t = i * dt

        Hs = interpolate_SE3_left(H1, H2, s)

        moving_box.add_ani_frame(time=t, htm=to_np(Hs))
        moving_frame.add_ani_frame(time=t, htm=to_np(Hs))

    sim = ub.Simulation([
        start_frame,
        goal_frame,
        moving_box,
        moving_frame,
    ])

    sim.save(out_dir, "se3_interpolation")

    print("\nUAIBot demo saved at:")
    print(os.path.join(out_dir, "se3_interpolation.html"))


if __name__ == "__main__":
    run_tests()

    # Descomente para gerar a animacao no UAIBot:
    demo_uaibot_interpolation()