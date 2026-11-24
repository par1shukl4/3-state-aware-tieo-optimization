"""PyTorch PINN model and deterministic training for Burgers' equation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import qmc
from torch import nn

from .burgers_reference import BurgersReference
from .hyperparameter_space import PINNHyperparameters


@dataclass
class TrainingConfig:
    hidden_layers: int = 4
    hidden_width: int = 50
    adam_steps: int = 1_000
    lbfgs_steps: int = 0
    collocation_points: int = 2_000
    initial_points: int = 256
    boundary_points: int = 256
    validation_points: int = 1_024
    seed: int = 1234
    device: str = "cpu"


@dataclass
class PINNData:
    collocation: np.ndarray
    initial_x: np.ndarray
    boundary_t: np.ndarray
    validation_xt: np.ndarray
    validation_u: np.ndarray


@dataclass
class TrainingResult:
    model: nn.Module
    fitness: float
    relative_l2: float
    pde_rmse: float
    loss_history: list[dict[str, float]]


class BurgersPINN(nn.Module):
    def __init__(self, hidden_layers: int, hidden_width: int, activation_scale: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_width = 2
        for _ in range(hidden_layers):
            linear = nn.Linear(input_width, hidden_width)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend([linear, nn.Tanh()])
            input_width = hidden_width
        output = nn.Linear(input_width, 1)
        nn.init.xavier_normal_(output.weight)
        nn.init.zeros_(output.bias)
        self.network = nn.Sequential(*layers, output)
        self.activation_scale = float(activation_scale)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        normalized = torch.column_stack([xt[:, 0], 2.0 * xt[:, 1] - 1.0])
        value = normalized
        for layer in self.network:
            if isinstance(layer, nn.Tanh):
                value = torch.tanh(self.activation_scale * value)
            else:
                value = layer(value)
        return value


def create_training_data(reference: BurgersReference, config: TrainingConfig) -> PINNData:
    rng = np.random.default_rng(config.seed)
    sampler = qmc.LatinHypercube(d=2, seed=config.seed)
    unit = sampler.random(config.collocation_points)
    collocation = np.column_stack([2.0 * unit[:, 0] - 1.0, unit[:, 1]])
    initial_x = rng.uniform(-1.0, 1.0, size=(config.initial_points, 1))
    boundary_t = rng.uniform(0.0, 1.0, size=(config.boundary_points, 1))
    validation_unit = qmc.LatinHypercube(d=2, seed=config.seed + 1).random(config.validation_points)
    validation_xt = np.column_stack([2.0 * validation_unit[:, 0] - 1.0, validation_unit[:, 1]])
    validation_u = reference.evaluate(validation_xt[:, 0], validation_xt[:, 1]).reshape(-1, 1)
    return PINNData(
        collocation=collocation.astype(np.float32),
        initial_x=initial_x.astype(np.float32),
        boundary_t=boundary_t.astype(np.float32),
        validation_xt=validation_xt.astype(np.float32),
        validation_u=validation_u.astype(np.float32),
    )


def _tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _pde_residual(model: nn.Module, xt_values: torch.Tensor, viscosity: float) -> torch.Tensor:
    xt = xt_values.detach().clone().requires_grad_(True)
    u = model(xt)
    first = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True)[0]
    u_x = first[:, 0:1]
    u_t = first[:, 1:2]
    second = torch.autograd.grad(u_x, xt, torch.ones_like(u_x), create_graph=True)[0]
    u_xx = second[:, 0:1]
    return u_t + u * u_x - viscosity * u_xx


def train_pinn(
    hyperparameters: PINNHyperparameters,
    reference: BurgersReference,
    data: PINNData,
    config: TrainingConfig,
) -> TrainingResult:
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(config.device)
    model = BurgersPINN(config.hidden_layers, config.hidden_width, hyperparameters.activation_scale).to(device)

    collocation = _tensor(data.collocation, device)
    initial_x = _tensor(data.initial_x, device)
    initial_xt = torch.column_stack([initial_x[:, 0], torch.zeros(len(initial_x), device=device)])
    initial_target = -torch.sin(torch.pi * initial_x)
    boundary_t = _tensor(data.boundary_t, device)
    left_xt = torch.column_stack([-torch.ones(len(boundary_t), device=device), boundary_t[:, 0]])
    right_xt = torch.column_stack([torch.ones(len(boundary_t), device=device), boundary_t[:, 0]])
    history: list[dict[str, float]] = []

    def losses() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = _pde_residual(model, collocation, reference.viscosity)
        pde_loss = torch.mean(residual**2)
        initial_loss = torch.mean((model(initial_xt) - initial_target) ** 2)
        boundary_loss = torch.mean(model(left_xt) ** 2) + torch.mean(model(right_xt) ** 2)
        total = (
            hyperparameters.pde_weight * pde_loss
            + hyperparameters.initial_weight * initial_loss
            + hyperparameters.boundary_weight * boundary_loss
        )
        return total, pde_loss, initial_loss, boundary_loss

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )
    record_every = max(1, config.adam_steps // 200)
    for step in range(config.adam_steps):
        optimizer.zero_grad(set_to_none=True)
        total, pde_loss, initial_loss, boundary_loss = losses()
        total.backward()
        optimizer.step()
        if step % record_every == 0 or step == config.adam_steps - 1:
            history.append(
                {
                    "step": float(step + 1),
                    "total_loss": float(total.detach().cpu()),
                    "pde_loss": float(pde_loss.detach().cpu()),
                    "initial_loss": float(initial_loss.detach().cpu()),
                    "boundary_loss": float(boundary_loss.detach().cpu()),
                }
            )

    if config.lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            max_iter=config.lbfgs_steps,
            tolerance_grad=1e-9,
            tolerance_change=1e-11,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            total, _, _, _ = losses()
            total.backward()
            return total

        lbfgs.step(closure)
        total, pde_loss, initial_loss, boundary_loss = losses()
        history.append(
            {
                "step": float(config.adam_steps + config.lbfgs_steps),
                "total_loss": float(total.detach().cpu()),
                "pde_loss": float(pde_loss.detach().cpu()),
                "initial_loss": float(initial_loss.detach().cpu()),
                "boundary_loss": float(boundary_loss.detach().cpu()),
            }
        )

    validation_xt = _tensor(data.validation_xt, device)
    validation_target = _tensor(data.validation_u, device)
    with torch.no_grad():
        prediction = model(validation_xt)
        relative_l2 = torch.linalg.vector_norm(prediction - validation_target) / torch.linalg.vector_norm(validation_target)
    residual = _pde_residual(model, validation_xt, reference.viscosity)
    pde_rmse = torch.sqrt(torch.mean(residual**2))
    fitness = float(relative_l2.detach().cpu()) + 0.05 * float(pde_rmse.detach().cpu())
    return TrainingResult(
        model=model,
        fitness=fitness,
        relative_l2=float(relative_l2.detach().cpu()),
        pde_rmse=float(pde_rmse.detach().cpu()),
        loss_history=history,
    )


def predict_grid(model: nn.Module, reference: BurgersReference, device: str = "cpu") -> np.ndarray:
    xx, tt = np.meshgrid(reference.x, reference.t)
    points = np.column_stack([xx.ravel(), tt.ravel()]).astype(np.float32)
    with torch.no_grad():
        prediction = model(_tensor(points, torch.device(device))).detach().cpu().numpy()
    return prediction.reshape(reference.u.shape)
