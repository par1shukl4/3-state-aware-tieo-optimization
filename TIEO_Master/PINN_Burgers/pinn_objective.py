"""Deterministic outer-loop fitness function for PINN hyperparameter search."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .burgers_pinn import PINNData, TrainingConfig, create_training_data, train_pinn
from .burgers_reference import BurgersReference
from .hyperparameter_space import NAMES, decode, vector_record


class PINNObjective:
    def __init__(
        self,
        reference: BurgersReference,
        training_config: TrainingConfig,
        output_dir: Path,
    ) -> None:
        self.reference = reference
        self.training_config = training_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data: PINNData = create_training_data(reference, training_config)
        self.records: list[dict] = []
        self.cache: dict[tuple[float, ...], float] = {}
        self.trial_file = self.output_dir / "hyperparameter_trials.csv"
        self._initialize_csv()

    def _initialize_csv(self) -> None:
        fieldnames = [
            "trial",
            *NAMES,
            "learning_rate",
            "weight_decay",
            "pde_weight",
            "initial_weight",
            "boundary_weight",
            "fitness",
            "relative_l2",
            "pde_rmse",
            "runtime_seconds",
        ]
        with self.trial_file.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    def __call__(self, vector: np.ndarray) -> float:
        values = np.asarray(vector, dtype=float)
        cache_key = tuple(np.round(values, 12))
        if cache_key in self.cache:
            return self.cache[cache_key]
        hyperparameters = decode(values)
        start = time.perf_counter()
        result = train_pinn(hyperparameters, self.reference, self.data, self.training_config)
        elapsed = time.perf_counter() - start
        record = {
            "trial": len(self.records) + 1,
            **vector_record(values),
            **hyperparameters.to_dict(),
            "fitness": result.fitness,
            "relative_l2": result.relative_l2,
            "pde_rmse": result.pde_rmse,
            "runtime_seconds": elapsed,
        }
        self.records.append(record)
        with self.trial_file.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=record.keys())
            writer.writerow(record)
        self.cache[cache_key] = result.fitness
        print(
            f"Trial {record['trial']:03d} | fitness={result.fitness:.4e} "
            f"| relative_L2={result.relative_l2:.4e} | PDE_RMSE={result.pde_rmse:.4e} "
            f"| {elapsed:.2f}s"
        )
        return result.fitness

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

