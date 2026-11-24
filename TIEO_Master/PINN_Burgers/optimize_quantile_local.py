"""Run Quantile + Local TIEO hyperparameter optimization for a Burgers PINN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiment_utils import create_numbered_output_root
from tieo_core import TIEO, TIEOConfig

from .burgers_pinn import TrainingConfig, create_training_data, predict_grid, train_pinn
from .burgers_reference import solve_reference
from .hyperparameter_space import BOUNDS, NAMES, decode
from .pinn_objective import PINNObjective
from .visualize_results import create_all_figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize Burgers PINN hyperparameters with Quantile + Local TIEO")
    parser.add_argument("--budget", type=int, default=60, help="number of PINN training trials")
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--max-tribes", type=int, default=4)
    parser.add_argument("--screen-steps", type=int, default=1_000)
    parser.add_argument("--final-steps", type=int, default=10_000)
    parser.add_argument("--lbfgs-steps", type=int, default=500)
    parser.add_argument("--collocation-points", type=int, default=2_000)
    parser.add_argument("--initial-points", type=int, default=256)
    parser.add_argument("--boundary-points", type=int, default=256)
    parser.add_argument("--validation-points", type=int, default=1_024)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--hidden-width", type=int, default=50)
    parser.add_argument("--reference-nx", type=int, default=257)
    parser.add_argument("--reference-nt", type=int, default=101)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="small smoke run for installation verification")
    return parser


def _apply_quick_settings(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.budget = 20
    args.population = 4
    args.max_tribes = 3
    args.screen_steps = 5
    args.final_steps = 20
    args.lbfgs_steps = 0
    args.collocation_points = 64
    args.initial_points = 32
    args.boundary_points = 32
    args.validation_points = 64
    args.hidden_layers = 2
    args.hidden_width = 12
    args.reference_nx = 65
    args.reference_nt = 21


def main() -> Path:
    args = build_parser().parse_args()
    _apply_quick_settings(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    package_dir = Path(__file__).resolve().parent
    output_root = args.output_root or create_numbered_output_root(package_dir / "results")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("Solving the numerical Burgers reference...")
    reference = solve_reference(nx=args.reference_nx, nt=args.reference_nt)
    np.savez_compressed(
        output_root / "burgers_reference.npz",
        x=reference.x,
        t=reference.t,
        u=reference.u,
        viscosity=reference.viscosity,
    )

    screen_config = TrainingConfig(
        hidden_layers=args.hidden_layers,
        hidden_width=args.hidden_width,
        adam_steps=args.screen_steps,
        lbfgs_steps=0,
        collocation_points=args.collocation_points,
        initial_points=args.initial_points,
        boundary_points=args.boundary_points,
        validation_points=args.validation_points,
        seed=args.seed,
        device=args.device,
    )
    objective = PINNObjective(reference, screen_config, output_root)
    optimizer = TIEO(
        objective=objective,
        bounds=BOUNDS,
        state_mode="quantile",
        config=TIEOConfig(
            population=args.population,
            max_tribes=args.max_tribes,
            evaluation_budget=args.budget,
            local_refinement_fraction=0.05,
            local_refinement_method="Powell",
            seed=args.seed,
        ),
    )
    print("Starting Quantile + Local TIEO hyperparameter search...")
    optimization = optimizer.run()
    best_vector = np.asarray(optimization["best_x"], dtype=float)
    best_hyperparameters = decode(best_vector)

    final_config = TrainingConfig(
        hidden_layers=args.hidden_layers,
        hidden_width=args.hidden_width,
        adam_steps=args.final_steps,
        lbfgs_steps=args.lbfgs_steps,
        collocation_points=args.collocation_points,
        initial_points=args.initial_points,
        boundary_points=args.boundary_points,
        validation_points=max(args.validation_points, 2_048 if not args.quick else args.validation_points),
        seed=args.seed + 10_000,
        device=args.device,
    )
    final_data = create_training_data(reference, final_config)
    print("Retraining the best configuration with an independent initialization...")
    final_result = train_pinn(best_hyperparameters, reference, final_data, final_config)
    prediction = predict_grid(final_result.model, reference, args.device)
    absolute_error = np.abs(prediction - reference.u)
    full_relative_l2 = float(np.linalg.norm(prediction - reference.u) / np.linalg.norm(reference.u))
    metrics = {
        "screen_best_fitness": float(optimization["best_f"]),
        "final_validation_fitness": final_result.fitness,
        "final_validation_relative_l2": final_result.relative_l2,
        "full_grid_relative_l2": full_relative_l2,
        "full_grid_rmse": float(np.sqrt(np.mean(absolute_error**2))),
        "maximum_absolute_error": float(np.max(absolute_error)),
        "pde_rmse": final_result.pde_rmse,
        "outer_evaluations": int(optimization["eval_count"]),
        "unique_training_trials": len(objective.records),
        "local_refinement_evaluations": int(optimization["local_refinement_evaluations"]),
    }
    with (output_root / "best_hyperparameters.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "search_vector": {name: float(value) for name, value in zip(NAMES, best_vector)},
                "decoded": best_hyperparameters.to_dict(),
            },
            handle,
            indent=2,
        )
    with (output_root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    pd.DataFrame(
        {
            "evaluation": optimization["convergence_evals"],
            "best_fitness": optimization["convergence_fitness"],
        }
    ).to_csv(output_root / "optimization_convergence.csv", index=False)
    np.savez_compressed(output_root / "final_solution.npz", prediction=prediction, error=absolute_error)
    torch.save(final_result.model.state_dict(), output_root / "best_pinn_state.pt")
    create_all_figures(
        reference,
        prediction,
        objective.frame(),
        optimization["convergence_evals"],
        optimization["convergence_fitness"],
        final_result.loss_history,
        output_root,
    )
    print(json.dumps(metrics, indent=2))
    print(f"\nPINN results saved to: {output_root.resolve()}")
    return output_root


if __name__ == "__main__":
    main()

