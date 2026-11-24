"""Continuous PINN hyperparameter space used by Quantile + Local TIEO."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


NAMES = [
    "log10_learning_rate",
    "log10_weight_decay",
    "log10_pde_weight",
    "log10_initial_weight",
    "log10_boundary_weight",
    "activation_scale",
]

BOUNDS = np.array(
    [
        [-5.0, -2.0],
        [-8.0, -3.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [0.5, 5.0],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class PINNHyperparameters:
    learning_rate: float
    weight_decay: float
    pde_weight: float
    initial_weight: float
    boundary_weight: float
    activation_scale: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def decode(vector: np.ndarray) -> PINNHyperparameters:
    values = np.clip(np.asarray(vector, dtype=float), BOUNDS[:, 0], BOUNDS[:, 1])
    if values.shape != (6,):
        raise ValueError("PINN hyperparameter vector must contain six values")
    return PINNHyperparameters(
        learning_rate=float(10.0 ** values[0]),
        weight_decay=float(10.0 ** values[1]),
        pde_weight=float(10.0 ** values[2]),
        initial_weight=float(10.0 ** values[3]),
        boundary_weight=float(10.0 ** values[4]),
        activation_scale=float(values[5]),
    )


def vector_record(vector: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(NAMES, np.asarray(vector, dtype=float))}

