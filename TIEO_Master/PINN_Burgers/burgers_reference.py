"""Finite-difference reference solution for the viscous Burgers equation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import diags


@dataclass
class BurgersReference:
    x: np.ndarray
    t: np.ndarray
    u: np.ndarray
    viscosity: float

    def evaluate(self, x_values: np.ndarray, t_values: np.ndarray) -> np.ndarray:
        interpolator = RegularGridInterpolator(
            (self.t, self.x),
            self.u,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        points = np.column_stack([np.asarray(t_values).ravel(), np.asarray(x_values).ravel()])
        return interpolator(points).reshape(np.asarray(x_values).shape)


def solve_reference(
    nx: int = 257,
    nt: int = 101,
    viscosity: float = 0.01 / np.pi,
) -> BurgersReference:
    """Solve u_t + u*u_x - viscosity*u_xx = 0 with zero Dirichlet boundaries."""

    if nx < 17 or nt < 3:
        raise ValueError("reference grid is too small")
    x = np.linspace(-1.0, 1.0, nx)
    t = np.linspace(0.0, 1.0, nt)
    dx = x[1] - x[0]
    initial = -np.sin(np.pi * x[1:-1])

    def right_hand_side(_: float, interior: np.ndarray) -> np.ndarray:
        full = np.zeros(nx, dtype=float)
        full[1:-1] = interior
        left, center, right = full[:-2], full[1:-1], full[2:]
        left_speed = np.maximum(np.abs(left), np.abs(center))
        right_speed = np.maximum(np.abs(center), np.abs(right))
        left_flux = 0.25 * (left**2 + center**2) - 0.5 * left_speed * (center - left)
        right_flux = 0.25 * (center**2 + right**2) - 0.5 * right_speed * (right - center)
        convection = (right_flux - left_flux) / dx
        second = (full[2:] - 2.0 * full[1:-1] + full[:-2]) / (dx * dx)
        return -convection + viscosity * second

    solution = solve_ivp(
        right_hand_side,
        (0.0, 1.0),
        initial,
        t_eval=t,
        method="BDF",
        rtol=1e-8,
        atol=1e-10,
        jac_sparsity=diags(
            [np.ones(nx - 3), np.ones(nx - 2), np.ones(nx - 3)],
            offsets=[-1, 0, 1],
            shape=(nx - 2, nx - 2),
            format="csr",
        ),
    )
    if not solution.success:
        raise RuntimeError(f"reference solver failed: {solution.message}")
    u = np.zeros((nt, nx), dtype=float)
    u[:, 1:-1] = solution.y.T
    return BurgersReference(x=x, t=t, u=u, viscosity=viscosity)

