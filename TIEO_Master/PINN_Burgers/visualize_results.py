"""Publication-oriented figures for Burgers PINN optimization results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .burgers_reference import BurgersReference
from .hyperparameter_space import NAMES


def plot_optimization_convergence(evaluations: np.ndarray, fitness: np.ndarray, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.semilogy(evaluations, np.maximum(fitness, 1e-16), color="#3567a8")
    axis.set_xlabel("PINN training trials")
    axis.set_ylabel("Best validation fitness")
    axis.set_title("Quantile + Local TIEO hyperparameter convergence")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "optimization_convergence.png", dpi=190)
    plt.close(fig)


def plot_hyperparameter_trials(trials: pd.DataFrame, output_dir: Path) -> None:
    if trials.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    fitness = np.maximum(trials["fitness"].to_numpy(float), 1e-16)
    for axis, name in zip(axes.ravel(), NAMES):
        scatter = axis.scatter(trials["trial"], trials[name], c=np.log10(fitness), cmap="viridis", s=25)
        axis.set_title(name.replace("_", " "))
        axis.set_xlabel("Trial")
        axis.grid(alpha=0.2)
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label="log10(fitness)", shrink=0.8)
    fig.suptitle("Hyperparameter search trace", y=1.02)
    fig.savefig(output_dir / "hyperparameter_search_trace.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def _heatmap(values: np.ndarray, reference: BurgersReference, title: str, label: str, path: Path, cmap: str) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(
        values,
        extent=[reference.x.min(), reference.x.max(), reference.t.max(), reference.t.min()],
        aspect="auto",
        cmap=cmap,
    )
    axis.set_xlabel("x")
    axis.set_ylabel("t")
    axis.set_title(title)
    fig.colorbar(image, ax=axis, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_solution_fields(reference: BurgersReference, prediction: np.ndarray, output_dir: Path) -> None:
    error = np.abs(prediction - reference.u)
    _heatmap(reference.u, reference, "Numerical reference solution", "u(x,t)", output_dir / "reference_solution.png", "coolwarm")
    _heatmap(prediction, reference, "PINN predicted solution", "û(x,t)", output_dir / "pinn_prediction.png", "coolwarm")
    _heatmap(error, reference, "Absolute PINN error", "|û-u|", output_dir / "absolute_error.png", "magma")


def plot_cross_sections(reference: BurgersReference, prediction: np.ndarray, output_dir: Path) -> None:
    requested_times = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for axis, requested in zip(axes.ravel(), requested_times):
        index = int(np.argmin(np.abs(reference.t - requested)))
        axis.plot(reference.x, reference.u[index], label="Reference", color="#222222")
        axis.plot(reference.x, prediction[index], "--", label="PINN", color="#3567a8")
        axis.set_title(f"t = {reference.t[index]:.2f}")
        axis.set_xlabel("x")
        axis.set_ylabel("u")
        axis.grid(alpha=0.25)
    axes.ravel()[-1].axis("off")
    axes.ravel()[0].legend()
    fig.suptitle("Burgers solution cross-sections")
    fig.tight_layout()
    fig.savefig(output_dir / "solution_cross_sections.png", dpi=190)
    plt.close(fig)


def plot_loss_history(history: list[dict[str, float]], output_dir: Path) -> None:
    frame = pd.DataFrame(history)
    frame.to_csv(output_dir / "final_training_loss_history.csv", index=False)
    fig, axis = plt.subplots(figsize=(8, 5))
    for column, label in [
        ("total_loss", "Total"),
        ("pde_loss", "PDE"),
        ("initial_loss", "Initial"),
        ("boundary_loss", "Boundary"),
    ]:
        axis.semilogy(frame["step"], np.maximum(frame[column], 1e-16), label=label)
    axis.set_xlabel("Training step")
    axis.set_ylabel("Loss")
    axis.set_title("Final PINN training losses")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "final_training_loss.png", dpi=190)
    plt.close(fig)


def create_all_figures(
    reference: BurgersReference,
    prediction: np.ndarray,
    trials: pd.DataFrame,
    convergence_evals: np.ndarray,
    convergence_fitness: np.ndarray,
    loss_history: list[dict[str, float]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_optimization_convergence(convergence_evals, convergence_fitness, output_dir)
    plot_hyperparameter_trials(trials, output_dir)
    plot_solution_fields(reference, prediction, output_dir)
    plot_cross_sections(reference, prediction, output_dir)
    plot_loss_history(loss_history, output_dir)
