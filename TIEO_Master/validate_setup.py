"""Fast formula, budget and reproducibility checks for the master code."""

from __future__ import annotations

from tempfile import TemporaryDirectory

import numpy as np

from benchmarks import get_benchmarks, validate_known_optima
from experiment_utils import create_numbered_output_root
from standard_optimizers import OPTIMIZERS, OptimizerConfig, run_optimizer
from tieo_core import TIEO, TIEOConfig


def main() -> None:
    with TemporaryDirectory() as temporary_directory:
        base = f"{temporary_directory}/results"
        first_output = create_numbered_output_root(base)
        second_output = create_numbered_output_root(base)
        if first_output.name != "results_1" or second_output.name != "results_2":
            raise AssertionError("automatic result-folder numbering check failed")

    optimum_values = validate_known_optima()
    tolerances = {
        33: 1e-8,
        34: 1e-10,
        35: 1e-10,
        39: 1e-8,
    }
    for benchmark_id, value in optimum_values.items():
        benchmark = get_benchmarks(30)[benchmark_id - 1]
        tolerance = tolerances.get(benchmark_id, 1e-10)
        error = abs(value - benchmark.optimum)
        if error > tolerance:
            raise AssertionError(
                f"F{benchmark_id} optimum check failed: value={value}, expected={benchmark.optimum}, error={error}"
            )

    sphere = get_benchmarks(5)[0]
    for optimizer_name in OPTIMIZERS:
        standard_result = run_optimizer(
            optimizer_name,
            sphere.function,
            sphere.bounds,
            OptimizerConfig(population=10, evaluation_budget=60, seed=1234),
        )
        if standard_result["eval_count"] != 60:
            raise AssertionError(f"{optimizer_name} budget check failed")
        if np.any(np.diff(standard_result["convergence_fitness"]) > 0):
            raise AssertionError(f"{optimizer_name} convergence is not monotone")

    for state_mode in ["baseline", "quantile", "adaptive_zscore", "relative_deviation"]:
        optimizer = TIEO(
            sphere.function,
            sphere.bounds,
            state_mode=state_mode,
            config=TIEOConfig(population=10, max_tribes=3, evaluation_budget=120, seed=1234),
        )
        method_result = optimizer.run()
        if method_result["eval_count"] != 120:
            raise AssertionError(f"{state_mode} budget check failed: {method_result['eval_count']} != 120")

    reward_optimizer = TIEO(
        sphere.function,
        sphere.bounds,
        state_mode="baseline",
        config=TIEOConfig(
            population=10,
            max_tribes=3,
            evaluation_budget=120,
            reward_mode="evaluation_aware",
            seed=1234,
        ),
    )
    reward_result = reward_optimizer.run()
    if reward_result["eval_count"] != 120:
        raise AssertionError("evaluation-aware reward budget check failed")
    if not np.all(np.isfinite(reward_result["Q"])):
        raise AssertionError("evaluation-aware reward produced a non-finite Q-table")
    if not np.all(np.isfinite(reward_result["reward_sums"])):
        raise AssertionError("evaluation-aware reward produced non-finite reward totals")

    restart_optimizer = TIEO(
        sphere.function,
        sphere.bounds,
        state_mode="baseline",
        config=TIEOConfig(
            population=10,
            max_tribes=3,
            evaluation_budget=120,
            restart_fraction=0.20,
            stagnation_patience=20,
            max_restarts=1,
            stagnation_relative_tolerance=1e9,
            seed=1234,
        ),
    )
    restart_result = restart_optimizer.run()
    if restart_result["eval_count"] != 120 or restart_result["restart_count"] != 1:
        raise AssertionError("stagnation-restart budget or trigger check failed")
    if np.any(np.diff(restart_result["convergence_fitness"]) > 0):
        raise AssertionError("stagnation restart broke best-so-far convergence")

    weight_optimizer = TIEO(
        sphere.function,
        sphere.bounds,
        state_mode="baseline",
        config=TIEOConfig(
            population=10,
            max_tribes=3,
            evaluation_budget=120,
            weight_schedule="linear",
            w_start=0.9,
            w_end=0.15,
            seed=1234,
        ),
    )
    weight_result = weight_optimizer.run()
    if weight_result["eval_count"] != 120:
        raise AssertionError("decreasing-autonomy-weight budget check failed")
    if not np.isclose(weight_result["final_autonomy_weight"], 0.15):
        raise AssertionError("decreasing-autonomy-weight endpoint check failed")

    refinement_optimizer = TIEO(
        sphere.function,
        sphere.bounds,
        state_mode="quantile",
        config=TIEOConfig(
            population=10,
            max_tribes=3,
            evaluation_budget=120,
            local_refinement_fraction=0.05,
            seed=1234,
        ),
    )
    refinement_result = refinement_optimizer.run()
    if refinement_result["eval_count"] != 120:
        raise AssertionError("quantile local-refinement budget check failed")
    if refinement_result["local_refinement_evaluations"] != 6:
        raise AssertionError("quantile local-refinement reservation check failed")

    results = []
    for _ in range(2):
        optimizer = TIEO(
            sphere.function,
            sphere.bounds,
            state_mode="baseline",
            config=TIEOConfig(population=10, max_tribes=3, evaluation_budget=200, seed=1234),
        )
        results.append(optimizer.run())

    if results[0]["eval_count"] != 200:
        raise AssertionError(f"budget check failed: {results[0]['eval_count']} != 200")
    if not np.array_equal(results[0]["convergence_fitness"], results[1]["convergence_fitness"]):
        raise AssertionError("reproducibility check failed")
    if np.any(np.diff(results[0]["convergence_fitness"]) > 0):
        raise AssertionError("best-so-far convergence is not monotone")

    print("All benchmark, budget, variant and reproducibility checks passed.")


if __name__ == "__main__":
    main()
