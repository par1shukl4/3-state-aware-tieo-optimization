"""Run all corrected TIEO methods and create comparative tables and plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_utils import METHODS, create_numbered_output_root, parse_benchmark_ids, run_suite

try:
    from scipy.stats import wilcoxon
except ImportError:  # Statistical output remains optional; optimization does not.
    wilcoxon = None


def _holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for position, original_index in enumerate(order):
        value = min(1.0, (count - position) * p_values[original_index])
        running = max(running, value)
        adjusted[original_index] = running
    return adjusted.tolist()


def _paired_statistics(
    all_runs: pd.DataFrame,
    method_names: list[str] | None = None,
    reference_method: str = "baseline",
) -> pd.DataFrame:
    baseline = all_runs[all_runs["method"] == reference_method]
    rows = []
    candidates = method_names or list(METHODS)
    for method in [name for name in candidates if name != reference_method]:
        method_runs = all_runs[all_runs["method"] == method]
        for benchmark_id in sorted(all_runs["benchmark_id"].unique()):
            baseline_group = baseline[baseline["benchmark_id"] == benchmark_id]
            left = baseline_group[["seed", "final_error"]]
            right = method_runs[method_runs["benchmark_id"] == benchmark_id][["seed", "final_error"]]
            paired = left.merge(right, on="seed", suffixes=("_baseline", "_method"))
            base_values = paired["final_error_baseline"].to_numpy(float)
            method_values = paired["final_error_method"].to_numpy(float)
            if len(paired) == 0:
                continue
            log_base = np.log10(np.maximum(base_values, 1e-300))
            log_method = np.log10(np.maximum(method_values, 1e-300))
            log_difference = log_method - log_base
            exact_ties = method_values == base_values
            if wilcoxon is None or np.all(exact_ties):
                p_value = 1.0
            else:
                try:
                    p_value = float(wilcoxon(log_difference, zero_method="pratt").pvalue)
                except ValueError:
                    p_value = 1.0
            rows.append(
                {
                    "method": method,
                    "reference_method": reference_method,
                    "benchmark_id": int(benchmark_id),
                    "benchmark": str(baseline_group["benchmark"].iloc[0]),
                    "mean_error_baseline": float(np.mean(base_values)),
                    "mean_error_method": float(np.mean(method_values)),
                    "median_paired_difference": float(np.median(method_values - base_values)),
                    "run_wins": int(np.sum((method_values < base_values) & ~exact_ties)),
                    "run_ties": int(np.sum(exact_ties)),
                    "run_losses": int(np.sum((method_values > base_values) & ~exact_ties)),
                    "wilcoxon_p": p_value,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    adjusted = []
    for _, group in frame.groupby("method", sort=False):
        adjusted.extend(zip(group.index, _holm_adjust(group["wilcoxon_p"].tolist())))
    adjusted_map = dict(adjusted)
    frame["holm_p"] = [adjusted_map[index] for index in frame.index]
    frame["outcome_vs_baseline"] = np.where(
        (frame["holm_p"] < 0.05) & (frame["mean_error_method"] < frame["mean_error_baseline"]),
        "win",
        np.where(
            (frame["holm_p"] < 0.05) & (frame["mean_error_method"] > frame["mean_error_baseline"]),
            "loss",
            "tie",
        ),
    )
    return frame.sort_values(["method", "benchmark_id"]).reset_index(drop=True)


def _plot_mean_errors(summary: pd.DataFrame, output_dir: Path) -> None:
    pivot = summary.pivot(index="benchmark_id", columns="method", values="mean_error").sort_index()
    values = np.log10(np.maximum(pivot.to_numpy(float), 1e-16))
    x = np.arange(len(pivot))
    width = 0.8 / len(pivot.columns)
    fig, axis = plt.subplots(figsize=(18, 6))
    for index, method in enumerate(pivot.columns):
        axis.bar(x + (index - (len(pivot.columns) - 1) / 2) * width, values[:, index], width, label=method)
    axis.set_xticks(x, [f"F{value}" for value in pivot.index], rotation=0)
    axis.set_ylabel("log10(mean final error)")
    axis.set_xlabel("Benchmark")
    axis.set_title("Corrected TIEO methods: mean final error")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=min(len(pivot.columns), 5))
    fig.tight_layout()
    fig.savefig(output_dir / "all_methods_mean_error.png", dpi=180)
    plt.close(fig)


def _rank_tables(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = summary.pivot(index="benchmark_id", columns="method", values="mean_error").sort_index()
    rank_table = pivot.rank(axis=1, method="average", ascending=True)
    average = rank_table.mean(axis=0).sort_values().rename("average_rank").reset_index()
    return rank_table.reset_index(), average


def _plot_average_ranks(average_ranks: pd.DataFrame, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(average_ranks["method"], average_ranks["average_rank"], color="#4f7f62")
    axis.set_ylabel("Average rank (lower is better)")
    axis.set_title("Average rank across benchmarks")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "average_ranks.png", dpi=180)
    plt.close(fig)


def _plot_outcomes(stats: pd.DataFrame, output_dir: Path, reference_method: str = "baseline") -> None:
    if stats.empty:
        return
    counts = stats.groupby(["method", "outcome_vs_baseline"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=["win", "tie", "loss"], fill_value=0)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(counts))
    colors = {"win": "#3f8f5f", "tie": "#999999", "loss": "#b84a4a"}
    for outcome in ["win", "tie", "loss"]:
        axis.bar(counts.index, counts[outcome], bottom=bottom, label=outcome, color=colors[outcome])
        bottom += counts[outcome].to_numpy()
    axis.set_ylabel("Number of benchmarks")
    axis.set_title(f"Statistical outcomes versus {reference_method}")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "wins_ties_losses.png", dpi=180)
    plt.close(fig)


def _plot_comparative_convergence(convergence: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "convergence_comparison"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for benchmark_id, group in convergence.groupby("benchmark_id", sort=True):
        fig, axis = plt.subplots(figsize=(7, 4.5))
        for method, method_group in group.groupby("method", sort=False):
            method_group = method_group.sort_values("evaluation")
            axis.semilogy(
                method_group["evaluation"],
                np.maximum(method_group["mean"], 1e-16),
                label=method,
            )
        benchmark = str(group["benchmark"].iloc[0])
        axis.set_xlabel("Objective evaluations")
        axis.set_ylabel("Mean absolute error")
        axis.set_title(f"F{benchmark_id} {benchmark}")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"F{int(benchmark_id):02d}.png", dpi=160)
        plt.close(fig)


def compare_methods(
    args: argparse.Namespace,
    method_configs: dict,
    reference_method: str,
    comparison_name: str = "comparison",
) -> Path:
    if args.output_root is None:
        if args.reuse:
            raise ValueError("--reuse requires an explicit --output-root folder")
        output_root = create_numbered_output_root(Path(__file__).parent / "results")
    else:
        output_root = Path(args.output_root)
    comparison_dir = output_root / comparison_name
    comparison_dir.mkdir(parents=True, exist_ok=True)
    run_frames = []
    summary_frames = []
    convergence_frames = []

    for method_name, method_config in method_configs.items():
        state_mode = method_config["state_mode"]
        reward_mode = method_config["reward_mode"]
        config_overrides = method_config.get("config_overrides")
        method_dir = output_root / method_name
        if args.reuse and (method_dir / "all_runs.csv").exists():
            runs = pd.read_csv(method_dir / "all_runs.csv")
            summary = pd.read_csv(method_dir / "summary.csv")
            convergence = pd.read_csv(method_dir / "convergence_summary.csv")
        else:
            result = run_suite(
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
                make_plots=not args.no_individual_plots,
                verbose=not args.quiet,
            )
            runs = result["runs"]
            summary = result["summary"]
            convergence = result["convergence_summary"]
        summary = summary.copy()
        summary["method"] = method_name
        convergence = convergence.copy()
        convergence["method"] = method_name
        run_frames.append(runs)
        summary_frames.append(summary)
        convergence_frames.append(convergence)

    all_runs = pd.concat(run_frames, ignore_index=True)
    all_summary = pd.concat(summary_frames, ignore_index=True)
    all_convergence = pd.concat(convergence_frames, ignore_index=True)
    paired_stats = _paired_statistics(all_runs, list(method_configs), reference_method)
    rank_table, average_ranks = _rank_tables(all_summary)

    all_runs.to_csv(comparison_dir / "all_methods_runs.csv", index=False)
    all_summary.to_csv(comparison_dir / "all_methods_summary.csv", index=False)
    paired_stats.to_csv(comparison_dir / f"paired_statistics_vs_{reference_method}.csv", index=False)
    rank_table.to_csv(comparison_dir / "rank_table.csv", index=False)
    average_ranks.to_csv(comparison_dir / "average_ranks.csv", index=False)

    _plot_mean_errors(all_summary, comparison_dir)
    _plot_average_ranks(average_ranks, comparison_dir)
    _plot_outcomes(paired_stats, comparison_dir, reference_method)
    _plot_comparative_convergence(all_convergence, comparison_dir)
    return comparison_dir


def compare_all(args: argparse.Namespace) -> Path:
    return compare_methods(args, METHODS, reference_method="baseline")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and compare all corrected TIEO methods")
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="independent runs per benchmark (default: 1; increase explicitly when needed)",
    )
    parser.add_argument("--budget", type=int, default=15_000)
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--max-tribes", type=int, default=5)
    parser.add_argument("--dimension", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--benchmarks", type=str, default=None, help="comma-separated IDs; default is all 40")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="explicit output folder; default automatically creates results_1, results_2, ...",
    )
    parser.add_argument("--reuse", action="store_true", help="reuse existing method CSV files")
    parser.add_argument("--no-individual-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


if __name__ == "__main__":
    destination = compare_all(build_parser().parse_args())
    print(f"\nComparative results saved to: {destination.resolve()}")

