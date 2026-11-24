"""Shared experiment runner, statistics and plots for TIEO methods."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks import Benchmark, select_benchmarks
from tieo_core import TIEO, TIEOConfig
from standard_optimizers import OptimizerConfig, run_optimizer


METHODS = {
    "baseline": {"state_mode": "baseline", "reward_mode": "binary"},
    "quantile": {"state_mode": "quantile", "reward_mode": "binary"},
    "adaptive_zscore": {"state_mode": "adaptive_zscore", "reward_mode": "binary"},
    "relative_deviation": {"state_mode": "relative_deviation", "reward_mode": "binary"},
    "evaluation_aware_reward": {"state_mode": "baseline", "reward_mode": "evaluation_aware"},
    "stagnation_restart": {
        "state_mode": "baseline",
        "reward_mode": "binary",
        "config_overrides": {"restart_fraction": 0.05, "stagnation_patience": 1_500, "max_restarts": 2},
    },
    "decreasing_autonomy_weight": {
        "state_mode": "baseline",
        "reward_mode": "binary",
        "config_overrides": {"weight_schedule": "linear", "w_start": 0.9, "w_end": 0.15},
    },
}

QUANTILE_METHODS = {
    "quantile": {"state_mode": "quantile", "reward_mode": "binary"},
    "quantile_local_refinement": {
        "state_mode": "quantile",
        "reward_mode": "binary",
        "config_overrides": {"local_refinement_fraction": 0.05, "local_refinement_method": "Powell"},
    },
    "quantile_decreasing_autonomy": {
        "state_mode": "quantile",
        "reward_mode": "binary",
        "config_overrides": {"weight_schedule": "linear", "w_start": 0.5, "w_end": 0.15},
    },
    "quantile_adaptive_thresholds": {
        "state_mode": "quantile",
        "reward_mode": "binary",
        "config_overrides": {
            "quantile_schedule": "adaptive",
            "quantile_lower_start": 0.20,
            "quantile_upper_start": 0.80,
        },
    },
    "quantile_minimum_occupancy": {
        "state_mode": "quantile",
        "reward_mode": "binary",
        "config_overrides": {"quantile_min_tribe_size": 5},
    },
    "quantile_diversity_restart": {
        "state_mode": "quantile",
        "reward_mode": "binary",
        "config_overrides": {
            "restart_fraction": 0.10,
            "stagnation_patience": 1_500,
            "max_restarts": 2,
            "restart_requires_low_diversity": True,
            "restart_diversity_threshold": 0.05,
        },
    },
}


def create_numbered_output_root(base: str | Path) -> Path:
    """Atomically create and return base_1, base_2, ... without overwriting a run."""
    base_path = Path(base)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = base_path.parent / f"{base_path.name}_{index}"
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            index += 1


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _json_array(values: np.ndarray) -> str:
    return json.dumps(np.asarray(values).tolist(), separators=(",", ":"))


def _interpolate_error(
    result: dict,
    optimum: float,
    checkpoints: np.ndarray,
) -> np.ndarray:
    evaluations = np.asarray(result["convergence_evals"], dtype=float)
    errors = np.maximum(np.abs(np.asarray(result["convergence_fitness"]) - optimum), 1e-16)
    return np.interp(checkpoints, evaluations, errors, left=errors[0], right=errors[-1])


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (benchmark_id, benchmark_name, dimension, optimum), group in runs.groupby(
        ["benchmark_id", "benchmark", "dimension", "optimum"], sort=True
    ):
        values = group["final_fitness"].to_numpy(float)
        errors = group["final_error"].to_numpy(float)
        rows.append(
            {
                "benchmark_id": benchmark_id,
                "benchmark": benchmark_name,
                "dimension": dimension,
                "optimum": optimum,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "median": float(np.median(values)),
                "best": float(np.min(values)),
                "worst": float(np.max(values)),
                "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
                "mean_error": float(np.mean(errors)),
                "median_error": float(np.median(errors)),
                "std_error": float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0,
                "best_error": float(np.min(errors)),
                "worst_error": float(np.max(errors)),
                "success_rate_1e-8": float(np.mean(errors <= 1e-8)),
                "avg_evaluations": float(group["evaluations"].mean()),
                "avg_runtime_seconds": float(group["runtime_seconds"].mean()),
                "runs": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values("benchmark_id").reset_index(drop=True)


def summarize_convergence(convergence: pd.DataFrame) -> pd.DataFrame:
    summary = (
        convergence.groupby(["benchmark_id", "benchmark", "evaluation"], sort=True)["error"]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def plot_method_summary(method_name: str, summary: pd.DataFrame, output_dir: Path) -> None:
    values = np.log10(np.maximum(summary["mean_error"].to_numpy(float), 1e-16))
    labels = [f"F{value}" for value in summary["benchmark_id"]]
    fig, axis = plt.subplots(figsize=(14, 5))
    axis.bar(labels, values, color="#3567a8")
    axis.set_ylabel("log10(mean final error)")
    axis.set_xlabel("Benchmark")
    axis.set_title(f"{method_name}: mean final error")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "mean_error_summary.png", dpi=180)
    plt.close(fig)


def plot_method_convergence(method_name: str, convergence: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "convergence_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for benchmark_id, group in convergence.groupby("benchmark_id", sort=True):
        group = group.sort_values("evaluation")
        x = group["evaluation"].to_numpy(float)
        mean = np.maximum(group["mean"].to_numpy(float), 1e-16)
        std = group["std"].to_numpy(float)
        lower = np.maximum(mean - std, 1e-16)
        upper = np.maximum(mean + std, 1e-16)
        name = str(group["benchmark"].iloc[0])
        fig, axis = plt.subplots(figsize=(7, 4.5))
        axis.semilogy(x, mean, label="mean error", color="#3567a8")
        axis.fill_between(x, lower, upper, color="#3567a8", alpha=0.2, label="mean ± std")
        axis.set_xlabel("Objective evaluations")
        axis.set_ylabel("Absolute error")
        axis.set_title(f"{method_name} — F{benchmark_id} {name}")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"F{int(benchmark_id):02d}_{_slug(name)}.png", dpi=160)
        plt.close(fig)


def run_suite(
    method_name: str,
    state_mode: str,
    reward_mode: str = "binary",
    config_overrides: dict | None = None,
    runs: int = 1,
    evaluation_budget: int = 15_000,
    population: int = 50,
    max_tribes: int = 5,
    dimension: int = 30,
    base_seed: int = 1000,
    benchmark_ids: Iterable[int] | None = None,
    output_root: str | Path = "results",
    make_plots: bool = True,
    verbose: bool = True,
) -> dict[str, pd.DataFrame | Path]:
    if state_mode not in TIEO.VALID_STATE_MODES:
        raise ValueError(f"Unsupported state mode: {state_mode}")
    if reward_mode not in TIEO.VALID_REWARD_MODES:
        raise ValueError(f"Unsupported reward mode: {reward_mode}")
    benchmarks = select_benchmarks(benchmark_ids, default_dimension=dimension)
    output_dir = Path(output_root) / method_name
    output_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict] = []
    convergence_rows: list[dict] = []
    checkpoint_count = min(101, max(2, evaluation_budget - population + 1))
    checkpoints = np.unique(np.linspace(population, evaluation_budget, checkpoint_count, dtype=int))

    for benchmark in benchmarks:
        if verbose:
            print(f"\n[{method_name}] F{benchmark.id:02d} {benchmark.name}")
        for run_index in range(runs):
            seed = int(base_seed + benchmark.id * 10_000 + run_index)
            config_values = dict(
                population=population,
                max_tribes=max_tribes,
                evaluation_budget=evaluation_budget,
                reward_mode=reward_mode,
                seed=seed,
            )
            config_values.update(config_overrides or {})
            config = TIEOConfig(**config_values)
            optimizer = TIEO(
                objective=benchmark.function,
                bounds=benchmark.bounds,
                state_mode=state_mode,
                config=config,
            )
            start = time.perf_counter()
            result = optimizer.run()
            elapsed = time.perf_counter() - start
            final_error = abs(float(result["best_f"]) - benchmark.optimum)
            run_rows.append(
                {
                    "method": method_name,
                    "state_mode": state_mode,
                    "reward_mode": reward_mode,
                    "weight_schedule": config.weight_schedule,
                    "quantile_schedule": config.quantile_schedule,
                    "quantile_min_tribe_size": config.quantile_min_tribe_size,
                    "restart_fraction": config.restart_fraction,
                    "stagnation_patience": config.stagnation_patience,
                    "restart_requires_low_diversity": config.restart_requires_low_diversity,
                    "restart_diversity_threshold": config.restart_diversity_threshold,
                    "local_refinement_fraction": config.local_refinement_fraction,
                    "benchmark_id": benchmark.id,
                    "benchmark": benchmark.name,
                    "category": benchmark.category,
                    "dimension": benchmark.dimension,
                    "optimum": benchmark.optimum,
                    "run": run_index + 1,
                    "seed": seed,
                    "final_fitness": float(result["best_f"]),
                    "final_error": final_error,
                    "evaluations": int(result["eval_count"]),
                    "iterations": int(result["iterations"]),
                    "runtime_seconds": elapsed,
                    "q_table": _json_array(result["Q"]),
                    "state_visits": _json_array(result["state_visits"]),
                    "action_counts": _json_array(result["action_counts"]),
                    "action_successes": _json_array(result["action_successes"]),
                    "reward_sums": _json_array(result["reward_sums"]),
                    "action_history_evals": _json_array(result["action_history_evals"]),
                    "action_history_actions": _json_array(result["action_history_actions"]),
                    "action_history_successes": _json_array(result["action_history_successes"]),
                    "restart_count": int(result["restart_count"]),
                    "restart_evaluations": _json_array(result["restart_evaluations"]),
                    "local_refinement_evaluations": int(result["local_refinement_evaluations"]),
                    "final_autonomy_weight": float(result["final_autonomy_weight"]),
                    "best_x": _json_array(result["best_x"]),
                }
            )
            interpolated = _interpolate_error(result, benchmark.optimum, checkpoints)
            for evaluation, error in zip(checkpoints, interpolated):
                convergence_rows.append(
                    {
                        "method": method_name,
                        "benchmark_id": benchmark.id,
                        "benchmark": benchmark.name,
                        "run": run_index + 1,
                        "seed": seed,
                        "evaluation": int(evaluation),
                        "error": float(error),
                    }
                )
            if verbose:
                print(
                    f"  Run {run_index + 1:02d}/{runs} | best={result['best_f']:.6e} "
                    f"| error={final_error:.3e} | evals={result['eval_count']} | {elapsed:.3f}s"
                )

    run_frame = pd.DataFrame(run_rows)
    convergence_frame = pd.DataFrame(convergence_rows)
    summary_frame = summarize_runs(run_frame)
    convergence_summary = summarize_convergence(convergence_frame)

    run_frame.to_csv(output_dir / "all_runs.csv", index=False)
    summary_frame.to_csv(output_dir / "summary.csv", index=False)
    convergence_frame.to_csv(output_dir / "convergence_runs.csv", index=False)
    convergence_summary.to_csv(output_dir / "convergence_summary.csv", index=False)
    try:
        summary_frame.to_excel(output_dir / "summary.xlsx", index=False)
    except ImportError:
        pass

    if make_plots:
        plot_method_summary(method_name, summary_frame, output_dir)
        plot_method_convergence(method_name, convergence_summary, output_dir)

    return {
        "runs": run_frame,
        "summary": summary_frame,
        "convergence": convergence_frame,
        "convergence_summary": convergence_summary,
        "output_dir": output_dir,
    }


def run_standard_suite(
    method_name: str,
    runs: int = 1,
    evaluation_budget: int = 15_000,
    population: int = 50,
    dimension: int = 30,
    base_seed: int = 1000,
    benchmark_ids: Iterable[int] | None = None,
    output_root: str | Path = "results",
    make_plots: bool = True,
    verbose: bool = True,
) -> dict[str, pd.DataFrame | Path]:
    """Run a non-TIEO reference optimizer with the same budgets, seeds and outputs."""

    benchmarks = select_benchmarks(benchmark_ids, default_dimension=dimension)
    output_dir = Path(output_root) / method_name
    output_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict] = []
    convergence_rows: list[dict] = []
    checkpoint_count = min(101, max(2, evaluation_budget - population + 1))
    checkpoints = np.unique(np.linspace(population, evaluation_budget, checkpoint_count, dtype=int))

    for benchmark in benchmarks:
        if verbose:
            print(f"\n[{method_name}] F{benchmark.id:02d} {benchmark.name}")
        for run_index in range(runs):
            seed = int(base_seed + benchmark.id * 10_000 + run_index)
            config = OptimizerConfig(population=population, evaluation_budget=evaluation_budget, seed=seed)
            start = time.perf_counter()
            result = run_optimizer(method_name, benchmark.function, benchmark.bounds, config)
            elapsed = time.perf_counter() - start
            final_error = abs(float(result["best_f"]) - benchmark.optimum)
            run_rows.append(
                {
                    "method": method_name,
                    "state_mode": "",
                    "reward_mode": "",
                    "benchmark_id": benchmark.id,
                    "benchmark": benchmark.name,
                    "category": benchmark.category,
                    "dimension": benchmark.dimension,
                    "optimum": benchmark.optimum,
                    "run": run_index + 1,
                    "seed": seed,
                    "final_fitness": float(result["best_f"]),
                    "final_error": final_error,
                    "evaluations": int(result["eval_count"]),
                    "iterations": int(result["iterations"]),
                    "runtime_seconds": elapsed,
                    "best_x": _json_array(result["best_x"]),
                }
            )
            interpolated = _interpolate_error(result, benchmark.optimum, checkpoints)
            for evaluation, error in zip(checkpoints, interpolated):
                convergence_rows.append(
                    {
                        "method": method_name,
                        "benchmark_id": benchmark.id,
                        "benchmark": benchmark.name,
                        "run": run_index + 1,
                        "seed": seed,
                        "evaluation": int(evaluation),
                        "error": float(error),
                    }
                )
            if verbose:
                print(
                    f"  Run {run_index + 1:02d}/{runs} | best={result['best_f']:.6e} "
                    f"| error={final_error:.3e} | evals={result['eval_count']} | {elapsed:.3f}s"
                )

    run_frame = pd.DataFrame(run_rows)
    convergence_frame = pd.DataFrame(convergence_rows)
    summary_frame = summarize_runs(run_frame)
    convergence_summary = summarize_convergence(convergence_frame)
    run_frame.to_csv(output_dir / "all_runs.csv", index=False)
    summary_frame.to_csv(output_dir / "summary.csv", index=False)
    convergence_frame.to_csv(output_dir / "convergence_runs.csv", index=False)
    convergence_summary.to_csv(output_dir / "convergence_summary.csv", index=False)
    try:
        summary_frame.to_excel(output_dir / "summary.xlsx", index=False)
    except ImportError:
        pass
    if make_plots:
        plot_method_summary(method_name, summary_frame, output_dir)
        plot_method_convergence(method_name, convergence_summary, output_dir)
    return {
        "runs": run_frame,
        "summary": summary_frame,
        "convergence": convergence_frame,
        "convergence_summary": convergence_summary,
        "output_dir": output_dir,
    }


def parse_benchmark_ids(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def build_method_parser(method_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run {method_name} on the TIEO 40-function suite")
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="independent runs per benchmark (default: 1; increase explicitly when needed)",
    )
    parser.add_argument("--budget", type=int, default=15_000)
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--max-tribes", type=int, default=5)
    parser.add_argument("--dimension", type=int, default=30, help="dimension for scalable functions")
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--benchmarks", type=str, default=None, help="comma-separated IDs; default is all 40")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="explicit output folder; default automatically creates results_1, results_2, ...",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def method_main(
    method_name: str,
    state_mode: str,
    reward_mode: str = "binary",
    config_overrides: dict | None = None,
) -> None:
    parser = build_method_parser(method_name)
    args = parser.parse_args()
    output_root = args.output_root or create_numbered_output_root(Path(__file__).parent / "results")
    run_suite(
        method_name=method_name,
        state_mode=state_mode,
        reward_mode=reward_mode,
        config_overrides=config_overrides,
        runs=args.runs,
        evaluation_budget=args.budget,
        population=args.population,
        max_tribes=args.max_tribes,
        dimension=args.dimension,
        base_seed=args.base_seed,
        benchmark_ids=parse_benchmark_ids(args.benchmarks),
        output_root=output_root,
        make_plots=not args.no_plots,
        verbose=not args.quiet,
    )
    print(f"\nResults saved to: {(output_root / method_name).resolve()}")

