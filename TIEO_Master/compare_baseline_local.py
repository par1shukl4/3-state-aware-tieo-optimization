"""Paired comparison of baseline TIEO and quantile TIEO with local refinement."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_all import build_parser, compare_methods


METHODS = {
    "baseline": {"state_mode": "baseline", "reward_mode": "binary"},
    "quantile_local_refinement": {
        "state_mode": "quantile",
        "reward_mode": "binary",
        "config_overrides": {"local_refinement_fraction": 0.05, "local_refinement_method": "Powell"},
    },
}


def _paired_frame(output_root: Path) -> pd.DataFrame:
    baseline = pd.read_csv(output_root / "baseline" / "all_runs.csv")
    local = pd.read_csv(output_root / "quantile_local_refinement" / "all_runs.csv")
    columns = ["benchmark_id", "benchmark", "category", "run", "seed", "final_error"]
    paired = baseline[columns].merge(
        local[columns],
        on=["benchmark_id", "benchmark", "category", "run", "seed"],
        suffixes=("_baseline", "_local"),
    )
    paired["log10_improvement"] = np.log10(np.maximum(paired["final_error_baseline"], 1e-16)) - np.log10(
        np.maximum(paired["final_error_local"], 1e-16)
    )
    paired["outcome"] = np.where(
        paired["final_error_local"] < paired["final_error_baseline"],
        "win",
        np.where(paired["final_error_local"] > paired["final_error_baseline"], "loss", "tie"),
    )
    return paired


def _plot_paired_scatter(paired: pd.DataFrame, output_dir: Path) -> None:
    grouped = paired.groupby(["benchmark_id", "benchmark"], as_index=False)[
        ["final_error_baseline", "final_error_local"]
    ].mean()
    baseline = np.maximum(grouped["final_error_baseline"].to_numpy(float), 1e-16)
    local = np.maximum(grouped["final_error_local"].to_numpy(float), 1e-16)
    lower = min(baseline.min(), local.min())
    upper = max(baseline.max(), local.max())
    fig, axis = plt.subplots(figsize=(8, 7))
    axis.loglog(baseline, local, "o", color="#3567a8", alpha=0.8)
    axis.plot([lower, upper], [lower, upper], "--", color="#555555", label="equal performance")
    for row, x, y in zip(grouped.itertuples(), baseline, local):
        axis.annotate(f"F{row.benchmark_id}", (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axis.set_xlabel("Baseline mean final error")
    axis.set_ylabel("Quantile + local refinement mean final error")
    axis.set_title("Paired final errors (points below line favor local refinement)")
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "paired_final_error_scatter.png", dpi=190)
    plt.close(fig)


def _plot_function_improvement(paired: pd.DataFrame, output_dir: Path) -> None:
    grouped = paired.groupby("benchmark_id", as_index=False)["log10_improvement"].mean()
    colors = np.where(grouped["log10_improvement"] >= 0, "#3f8f5f", "#b84a4a")
    fig, axis = plt.subplots(figsize=(15, 5.5))
    axis.bar([f"F{value}" for value in grouped["benchmark_id"]], grouped["log10_improvement"], color=colors)
    axis.axhline(0.0, color="#333333", linewidth=1)
    axis.set_xlabel("Benchmark")
    axis.set_ylabel("Mean log10 improvement (positive favors local refinement)")
    axis.set_title("Per-function improvement over baseline")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "per_function_improvement.png", dpi=190)
    plt.close(fig)


def _plot_category_improvement(paired: pd.DataFrame, output_dir: Path) -> None:
    categories = sorted(paired["category"].unique())
    values = [paired.loc[paired["category"] == category, "log10_improvement"].to_numpy(float) for category in categories]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(values, tick_labels=categories, showmeans=True)
    axis.axhline(0.0, color="#555555", linestyle="--")
    axis.set_ylabel("Paired log10 improvement")
    axis.set_title("Improvement distribution by benchmark category")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "category_improvement_boxplot.png", dpi=190)
    plt.close(fig)


def _plot_convergence_improvement(output_root: Path, output_dir: Path) -> None:
    baseline = pd.read_csv(output_root / "baseline" / "convergence_summary.csv")
    local = pd.read_csv(output_root / "quantile_local_refinement" / "convergence_summary.csv")
    merged = baseline.merge(
        local,
        on=["benchmark_id", "benchmark", "evaluation"],
        suffixes=("_baseline", "_local"),
    )
    merged["progress"] = merged.groupby("benchmark_id")["evaluation"].transform(lambda values: values / values.max())
    merged["log10_improvement"] = np.log10(np.maximum(merged["mean_baseline"], 1e-16)) - np.log10(
        np.maximum(merged["mean_local"], 1e-16)
    )
    summary = merged.groupby("progress")["log10_improvement"].agg(
        median="median",
        lower=lambda values: values.quantile(0.25),
        upper=lambda values: values.quantile(0.75),
    )
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(summary.index * 100, summary["median"], color="#3567a8", label="median improvement")
    axis.fill_between(summary.index * 100, summary["lower"], summary["upper"], color="#3567a8", alpha=0.2, label="IQR")
    axis.axhline(0.0, color="#555555", linestyle="--")
    axis.set_xlabel("Evaluation budget used (%)")
    axis.set_ylabel("log10 improvement over baseline")
    axis.set_title("Paired convergence improvement across benchmarks")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "convergence_improvement_overview.png", dpi=190)
    plt.close(fig)


def create_detailed_outputs(output_root: Path, comparison_dir: Path) -> None:
    paired = _paired_frame(output_root)
    paired.to_csv(comparison_dir / "paired_function_comparison.csv", index=False)
    _plot_paired_scatter(paired, comparison_dir)
    _plot_function_improvement(paired, comparison_dir)
    _plot_category_improvement(paired, comparison_dir)
    _plot_convergence_improvement(output_root, comparison_dir)


if __name__ == "__main__":
    parser = build_parser()
    parser.description = "Compare baseline TIEO with quantile TIEO plus local refinement"
    destination = compare_methods(
        parser.parse_args(),
        METHODS,
        reference_method="baseline",
        comparison_name="baseline_vs_quantile_local",
    )
    create_detailed_outputs(destination.parent, destination)
    print(f"\nDetailed paired comparison saved to: {destination.resolve()}")
