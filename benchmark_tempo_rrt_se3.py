import argparse
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "rrt_se3"
LOCAL_UAIBOTPY = PROJECT_ROOT / "UAIbotPy"

for path in (LOCAL_UAIBOTPY, EXPERIMENT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import uaibot as ub

from teste_rrt_se3 import create_rrt_test_scenario, get_rrt_options


def run_once(scenario, options):
    H_start, H_goal, position_bounds, robot_collision_model, obstacles = scenario

    t0 = time.perf_counter()

    try:
        result = ub.Utils.rrt_se3_bidirectional(
            h_start=H_start,
            h_goal=H_goal,
            position_bounds=position_bounds,
            robot_model=robot_collision_model,
            obstacles=obstacles,
            **options,
        )

        external_time = time.perf_counter() - t0

        if result.success:
            if hasattr(result, "execution_time"):
                execution_time = result.execution_time
            else:
                execution_time = external_time

            return True, float(execution_time)

        return False, np.nan

    except Exception as e:
        print("Execution failed:", e)
        return False, np.nan


def compute_stats(times, n_runs, n_success):
    success_times = np.asarray([t for t in times if np.isfinite(t)], dtype=float)
    n_failure = n_runs - n_success
    success_rate = 100.0 * n_success / n_runs if n_runs > 0 else 0.0

    if success_times.size == 0:
        return {
            "n_runs": n_runs,
            "n_success": n_success,
            "n_failure": n_failure,
            "success_rate": success_rate,
            "mean_execution_time": np.nan,
            "std_execution_time": np.nan,
            "min_execution_time": np.nan,
            "max_execution_time": np.nan,
            "median_execution_time": np.nan,
            "success_times": success_times,
        }

    return {
        "n_runs": n_runs,
        "n_success": n_success,
        "n_failure": n_failure,
        "success_rate": success_rate,
        "mean_execution_time": float(np.mean(success_times)),
        "std_execution_time": float(np.std(success_times, ddof=0)),
        "min_execution_time": float(np.min(success_times)),
        "max_execution_time": float(np.max(success_times)),
        "median_execution_time": float(np.median(success_times)),
        "success_times": success_times,
    }


def _format_seconds(value):
    if not np.isfinite(value):
        return "nan"
    return f"{value:.6f}"


def print_summary(stats):
    print()
    print("===== Benchmark RRT SE(3) =====")
    print(f"Runs: {stats['n_runs']}")
    print(f"Successes: {stats['n_success']}")
    print(f"Failures: {stats['n_failure']}")
    print(f"Success rate: {stats['success_rate']:.2f} %")
    print()
    print("Execution time:")
    print(f"  mean   = {_format_seconds(stats['mean_execution_time'])} s")
    print(f"  std    = {_format_seconds(stats['std_execution_time'])} s")
    print(f"  min    = {_format_seconds(stats['min_execution_time'])} s")
    print(f"  max    = {_format_seconds(stats['max_execution_time'])} s")
    print(f"  median = {_format_seconds(stats['median_execution_time'])} s")
    print()
    print("Successful execution times:")
    if stats["success_times"].size == 0:
        print("[]")
    else:
        print("[")
        print("  " + ", ".join(f"{t:.6f}" for t in stats["success_times"]))
        print("]")


def print_latex_results(stats):
    n_runs = stats["n_runs"]
    n_success = stats["n_success"]
    success_rate = stats["success_rate"]
    mean_time = _format_seconds(stats["mean_execution_time"])
    std_time = _format_seconds(stats["std_execution_time"])
    min_time = _format_seconds(stats["min_execution_time"])
    max_time = _format_seconds(stats["max_execution_time"])

    print()
    print("===== Bloco LaTeX temporario =====")
    print(r"\section{Resultados preliminares}")
    print()
    print(
        "O planejador RRT bidirecional em \\(SE(3)\\) foi avaliado no "
        f"cenário de teste por meio de {n_runs} execuções independentes. "
        "Em cada execução, o mesmo problema de planejamento foi resolvido, "
        "sem fixação de semente aleatória, de modo a observar a variação "
        "natural do método baseado em amostragem."
    )
    print()
    print(
        f"Das {n_runs} execuções realizadas, o planejador obteve sucesso em "
        f"{n_success} casos, correspondendo a uma taxa de sucesso de "
        f"{success_rate:.2f}\\%. O tempo médio de execução entre as execuções "
        f"bem-sucedidas foi de \\( {mean_time} \\,\\mathrm{{s}}\\), com "
        f"desvio padrão de \\( {std_time} \\,\\mathrm{{s}}\\). O menor tempo "
        f"observado foi de \\( {min_time} \\,\\mathrm{{s}}\\), enquanto o "
        f"maior tempo foi de \\( {max_time} \\,\\mathrm{{s}}\\)."
    )
    print()
    print(
        "Esses resultados indicam que o planejador foi capaz de encontrar "
        "caminhos livres de colisão de forma consistente no cenário "
        "considerado. A variação observada no tempo de execução é esperada, "
        "pois o RRT é um método baseado em amostragem aleatória e, portanto, "
        "diferentes execuções podem gerar árvores e conexões distintas."
    )
    print()
    print(
        "Como esta análise é preliminar, os valores reportados serão "
        "utilizados apenas para caracterizar o comportamento geral do "
        "planejador no cenário testado. Uma avaliação mais detalhada pode "
        "considerar outros cenários, diferentes resoluções de discretização e "
        "diferentes parâmetros de amostragem."
    )


def run_benchmark(n_runs=100):
    scenario = create_rrt_test_scenario()
    options = get_rrt_options()

    times = []
    records = []

    for run_id in range(1, n_runs + 1):
        success, execution_time = run_once(scenario, options)
        times.append(execution_time)
        records.append({
            "run_id": run_id,
            "success": success,
            "execution_time": execution_time,
        })

        time_text = "nan" if not np.isfinite(execution_time) else f"{execution_time:.6f}"
        print(f"[{run_id}/{n_runs}] success={success} time={time_text}")

    n_success = sum(1 for record in records if record["success"])
    stats = compute_stats(times, n_runs, n_success)
    print_summary(stats)
    print_latex_results(stats)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    run_benchmark(args.runs)


if __name__ == "__main__":
    main()
