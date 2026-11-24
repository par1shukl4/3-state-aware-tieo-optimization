"""Compare TIEO variants with ten standard metaheuristic optimizers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_all import (
    _paired_statistics,
    _plot_average_ranks,
    _plot_comparative_convergence,
    _plot_mean_errors,
    _plot_outcomes,
    _rank_tables,
)
from experiment_utils import (
    create_numbered_output_root,
    parse_benchmark_ids,
    run_standard_suite,
    run_suite,
)


TIEO_METHODS = {
    "baseline": {"state_mode": "baseline", "reward_mode": "binary"},
    "quantile_local_refinement": {
        "state_mode": "quantile",
        "reward_mode": "binary",
        "config_overrides": {"local_refinement_fraction": 0.05, "local_refinement_method": "Powell"},
    },
}

REFERENCE_METHODS = [
    "simulated_annealing",
    "pso",
    "differential_evolution",
    "abc",
    "wolf_pack_search",
    "salp_swarm",
    "satin_bowerbird",
    "marine_predators",
    "sine_cosine",
    "gravitational_search",
]

DISPLAY_NAMES = {
    "baseline": "Baseline TIEO",
    "quantile_local_refinement": "Quantile + Local",
    "simulated_annealing": "SA",
    "pso": "PSO",
    "differential_evolution": "DE",
    "abc": "ABC",
    "wolf_pack_search": "WPS",
    "salp_swarm": "SSA",
    "satin_bowerbird": "SBO",
    "marine_predators": "MPA",
    "sine_cosine": "SCA",
    "gravitational_search": "GSA",
}


def _score_tables(summary: pd.DataFrame, runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    method_count = summary["method"].nunique()
    detailed = summary.copy()
    detailed["rank"] = detailed.groupby("benchmark_id")["mean_error"].rank(method="average")
    detailed["rank_score_0_100"] = 100 * (method_count - detailed["rank"]) / max(method_count - 1, 1)

    rows = []
    for method, group in runs.groupby("method", sort=False):
        errors = np.maximum(group["final_error"].to_numpy(float), 1e-16)
        method_detail = detailed[detailed["method"] == method]
        rows.append(
            {
                "method": method,
                "geometric_mean_error": float(10 ** np.mean(np.log10(errors))),
                "mean_log10_error": float(np.mean(np.log10(errors))),
                "std_log10_error": float(np.std(np.log10(errors), ddof=1)) if len(errors) > 1 else 0.0,
                "mean_runtime_seconds": float(group["runtime_seconds"].mean()),
                "total_runtime_seconds": float(group["runtime_seconds"].sum()),
                "average_rank": float(method_detail["rank"].mean()),
                "rank_score_0_100": float(method_detail["rank_score_0_100"].mean()),
            }
        )
    overall = pd.DataFrame(rows).sort_values(["average_rank", "mean_log10_error"]).reset_index(drop=True)
    return detailed.sort_values(["benchmark_id", "rank"]), overall


def _plot_algorithm_scores(overall: pd.DataFrame, output_dir: Path) -> None:
    ordered = overall.sort_values("rank_score_0_100", ascending=False)
    fig, axis = plt.subplots(figsize=(12, 5.5))
    labels = ordered["method"].map(DISPLAY_NAMES).fillna(ordered["method"])
    axis.bar(labels, ordered["rank_score_0_100"], color="#4f7f62")
    axis.set_ylabel("Algorithm rank score (0–100; higher is better)")
    axis.set_title("Algorithm score across selected benchmark functions")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "algorithm_rank_score.png", dpi=190)
    plt.close(fig)


def _plot_improvement_probability(baseline_runs: pd.DataFrame, output_dir: Path, budget: int) -> None:
    action_names = ["Autonomy", "Diplomacy", "War"]
    bins = np.linspace(0.0, 1.0, 11)
    successes = np.zeros((3, 10), dtype=float)
    counts = np.zeros((3, 10), dtype=float)
    for row in baseline_runs.itertuples():
        evaluations = np.asarray(json.loads(row.action_history_evals), dtype=float)
        actions = np.asarray(json.loads(row.action_history_actions), dtype=int)
        outcomes = np.asarray(json.loads(row.action_history_successes), dtype=bool)
        progress_bins = np.minimum(np.digitize(evaluations / budget, bins) - 1, 9)
        for action in range(3):
            for bin_index in range(10):
                mask = (actions == action) & (progress_bins == bin_index)
                counts[action, bin_index] += np.sum(mask)
                successes[action, bin_index] += np.sum(outcomes[mask])
    probabilities = np.divide(successes, counts, out=np.full_like(successes, np.nan), where=counts > 0)
    centers = (bins[:-1] + bins[1:]) * 50
    rows = []
    fig, axis = plt.subplots(figsize=(8, 5))
    for action, name in enumerate(action_names):
        axis.plot(centers, probabilities[action], marker="o", label=name)
        for index, probability in enumerate(probabilities[action]):
            rows.append(
                {
                    "action": name,
                    "progress_bin_start": bins[index],
                    "progress_bin_end": bins[index + 1],
                    "attempts": int(counts[action, index]),
                    "improvement_probability": probability,
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "tieo_fitness_improvement_probability.csv", index=False)
    axis.set_xlabel("Evaluation budget used (%)")
    axis.set_ylabel("Probability of fitness improvement")
    axis.set_ylim(0, 1)
    axis.set_title("Fitness-improvement probability of TIEO iterative equations")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "tieo_fitness_improvement_probability.png", dpi=190)
    plt.close(fig)


def _configuration_scores(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot(index="benchmark_id", columns="method", values="mean_error")
    ranks = pivot.rank(axis=1, method="average")
    count = len(ranks.columns)
    scores = 100 * (count - ranks.mean()) / max(count - 1, 1)
    return scores.rename("rank_score_0_100").reset_index().rename(columns={"index": "method"})


def _run_sensitivity(args: argparse.Namespace, output_root: Path) -> None:
    sensitivity_root = output_root / "sensitivity"
    benchmark_ids = parse_benchmark_ids(args.benchmarks)

    tribe_summaries = []
    for maximum_tribes in [2, 3, 4, 5, 6]:
        name = f"tieo_max_tribes_{maximum_tribes}"
        result = run_suite(
            method_name=name,
            state_mode="baseline",
            config_overrides={"w": 0.5},
            runs=args.sensitivity_runs,
            evaluation_budget=args.budget,
            population=args.population,
            max_tribes=maximum_tribes,
            dimension=args.dimension,
            base_seed=args.base_seed,
            benchmark_ids=benchmark_ids,
            output_root=sensitivity_root / "maximum_tribes",
            make_plots=False,
            verbose=not args.quiet,
        )
        frame = result["summary"].copy()
        frame["method"] = name
        frame["maximum_tribes"] = maximum_tribes
        tribe_summaries.append(frame)
    tribe_summary = pd.concat(tribe_summaries, ignore_index=True)
    tribe_scores = _configuration_scores(tribe_summary)
    tribe_scores["maximum_tribes"] = tribe_scores["method"].str.extract(r"(\d+)$").astype(int)
    tribe_scores.to_csv(sensitivity_root / "maximum_tribes_scores.csv", index=False)

    weight_summaries = []
    weights = [0.1, 0.3, 0.5, 0.7, 0.9]
    for weight in weights:
        name = f"tieo_weight_{str(weight).replace('.', '_')}"
        result = run_suite(
            method_name=name,
            state_mode="baseline",
            config_overrides={"w": weight},
            runs=args.sensitivity_runs,
            evaluation_budget=args.budget,
            population=args.population,
            max_tribes=5,
            dimension=args.dimension,
            base_seed=args.base_seed,
            benchmark_ids=benchmark_ids,
            output_root=sensitivity_root / "learning_weight",
            make_plots=False,
            verbose=not args.quiet,
        )
        frame = result["summary"].copy()
        frame["method"] = name
        frame["weight"] = weight
        weight_summaries.append(frame)
    weight_summary = pd.concat(weight_summaries, ignore_index=True)
    weight_scores = _configuration_scores(weight_summary)
    weight_map = {f"tieo_weight_{str(value).replace('.', '_')}": value for value in weights}
    weight_scores["weight"] = weight_scores["method"].map(weight_map)
    weight_scores.to_csv(sensitivity_root / "learning_weight_scores.csv", index=False)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    ordered = tribe_scores.sort_values("maximum_tribes")
    axis.plot(ordered["maximum_tribes"], ordered["rank_score_0_100"], marker="o")
    axis.set_xlabel("Maximum tribal number N")
    axis.set_ylabel("Algorithm rank score (0–100)")
    axis.set_title("Algorithm score under maximum tribal numbers (w = 0.5)")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(sensitivity_root / "score_by_maximum_tribes.png", dpi=190)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    ordered = weight_scores.sort_values("weight")
    axis.plot(ordered["weight"], ordered["rank_score_0_100"], marker="o", color="#8a5a44")
    axis.set_xlabel("Learning weight factor w")
    axis.set_ylabel("Algorithm rank score (0–100)")
    axis.set_title("Algorithm score under learning weight values (N = 5)")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(sensitivity_root / "score_by_learning_weight.png", dpi=190)
    plt.close(fig)


def run_comparison(args: argparse.Namespace) -> Path:
    if args.output_root is None:
        output_root = create_numbered_output_root(Path(__file__).parent / "results")
    else:
        output_root = Path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
    comparison_dir = output_root / "twelve_algorithm_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    benchmark_ids = parse_benchmark_ids(args.benchmarks)
    run_frames, summary_frames, convergence_frames = [], [], []

    for method, config in TIEO_METHODS.items():
        result = run_suite(
            method_name=method,
            state_mode=config["state_mode"],
            reward_mode=config["reward_mode"],
            config_overrides=config.get("config_overrides"),
            runs=args.runs,
            evaluation_budget=args.budget,
            population=args.population,
            max_tribes=args.max_tribes,
            dimension=args.dimension,
            base_seed=args.base_seed,
            benchmark_ids=benchmark_ids,
            output_root=output_root,
            make_plots=not args.no_individual_plots,
            verbose=not args.quiet,
        )
        run_frames.append(result["runs"])
        summary = result["summary"].copy(); summary["method"] = method; summary_frames.append(summary)
        convergence = result["convergence_summary"].copy(); convergence["method"] = method; convergence_frames.append(convergence)

    for method in REFERENCE_METHODS:
        result = run_standard_suite(
            method_name=method,
            runs=args.runs,
            evaluation_budget=args.budget,
            population=args.population,
            dimension=args.dimension,
            base_seed=args.base_seed,
            benchmark_ids=benchmark_ids,
            output_root=output_root,
            make_plots=not args.no_individual_plots,
            verbose=not args.quiet,
        )
        run_frames.append(result["runs"])
        summary = result["summary"].copy(); summary["method"] = method; summary_frames.append(summary)
        convergence = result["convergence_summary"].copy(); convergence["method"] = method; convergence_frames.append(convergence)

    all_runs = pd.concat(run_frames, ignore_index=True)
    all_summary = pd.concat(summary_frames, ignore_index=True)
    all_convergence = pd.concat(convergence_frames, ignore_index=True)
    method_names = list(TIEO_METHODS) + REFERENCE_METHODS
    paired = _paired_statistics(all_runs, method_names, "baseline")
    rank_table, average_ranks = _rank_tables(all_summary)
    detailed, overall = _score_tables(all_summary, all_runs)

    all_runs.to_csv(comparison_dir / "all_algorithms_runs.csv", index=False)
    all_summary.to_csv(comparison_dir / "all_algorithms_summary.csv", index=False)
    paired.to_csv(comparison_dir / "paired_statistics_vs_baseline.csv", index=False)
    rank_table.to_csv(comparison_dir / "rank_table.csv", index=False)
    average_ranks.to_csv(comparison_dir / "average_ranks.csv", index=False)
    detailed.to_csv(comparison_dir / "benchmark_mean_std_time_rank_score.csv", index=False)
    overall.to_csv(comparison_dir / "algorithm_mean_std_time_rank_score.csv", index=False)
    try:
        detailed.to_excel(comparison_dir / "benchmark_mean_std_time_rank_score.xlsx", index=False)
        overall.to_excel(comparison_dir / "algorithm_mean_std_time_rank_score.xlsx", index=False)
    except ImportError:
        pass

    plot_summary = all_summary.copy()
    plot_summary["method"] = plot_summary["method"].map(DISPLAY_NAMES).fillna(plot_summary["method"])
    plot_ranks = average_ranks.copy()
    plot_ranks["method"] = plot_ranks["method"].map(DISPLAY_NAMES).fillna(plot_ranks["method"])
    plot_paired = paired.copy()
    plot_paired["method"] = plot_paired["method"].map(DISPLAY_NAMES).fillna(plot_paired["method"])
    plot_convergence = all_convergence.copy()
    plot_convergence["method"] = plot_convergence["method"].map(DISPLAY_NAMES).fillna(plot_convergence["method"])
    _plot_mean_errors(plot_summary, comparison_dir)
    _plot_average_ranks(plot_ranks, comparison_dir)
    _plot_outcomes(plot_paired, comparison_dir, "Baseline TIEO")
    _plot_comparative_convergence(plot_convergence, comparison_dir)
    _plot_algorithm_scores(overall, comparison_dir)
    _plot_improvement_probability(run_frames[0], comparison_dir, args.budget)
    if not args.skip_sensitivity:
        _run_sensitivity(args, output_root)
    return comparison_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare TIEO with ten standard metaheuristics")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--sensitivity-runs", type=int, default=1)
    parser.add_argument("--budget", type=int, default=15_000)
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--max-tribes", type=int, default=5)
    parser.add_argument("--dimension", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--benchmarks", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--no-individual-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


if __name__ == "__main__":
    destination = run_comparison(build_parser().parse_args())
    print(f"\nTwelve-algorithm comparison saved to: {destination.resolve()}")
