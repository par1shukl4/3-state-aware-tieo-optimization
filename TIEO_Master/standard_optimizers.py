"""Budget-controlled reference implementations of ten population optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray


@dataclass
class OptimizerConfig:
    population: int = 50
    evaluation_budget: int = 15_000
    seed: int = 1000


class ObjectiveTracker:
    def __init__(self, objective: Callable[[Array], float], bounds: Array, budget: int, seed: int) -> None:
        self.objective = objective
        self.bounds = np.asarray(bounds, dtype=float)
        self.lower = self.bounds[:, 0]
        self.upper = self.bounds[:, 1]
        self.range = self.upper - self.lower
        self.dimension = len(self.lower)
        self.budget = int(budget)
        self.rng = np.random.default_rng(seed)
        self.evaluations = 0
        self.best_f = np.inf
        self.best_x = np.zeros(self.dimension)
        self.convergence_evals: list[int] = []
        self.convergence_fitness: list[float] = []

    @property
    def remaining(self) -> int:
        return self.budget - self.evaluations

    def evaluate(self, points: Array) -> Array:
        matrix = np.atleast_2d(np.asarray(points, dtype=float))
        if len(matrix) > self.remaining:
            raise RuntimeError("evaluation budget would be exceeded")
        values = []
        for point in matrix:
            value = float(self.objective(point))
            if not np.isfinite(value):
                value = np.inf
            self.evaluations += 1
            if value < self.best_f:
                self.best_f = value
                self.best_x = point.copy()
            self.convergence_evals.append(self.evaluations)
            self.convergence_fitness.append(self.best_f)
            values.append(value)
        return np.asarray(values, dtype=float)

    def uniform(self, count: int) -> Array:
        return self.rng.uniform(self.lower, self.upper, size=(count, self.dimension))

    def finish_budget(self) -> None:
        while self.remaining:
            self.evaluate(self.uniform(min(self.remaining, 64)))

    def result(self, iterations: int) -> dict:
        return {
            "best_x": self.best_x.copy(),
            "best_f": float(self.best_f),
            "eval_count": self.evaluations,
            "iterations": iterations,
            "convergence_evals": np.asarray(self.convergence_evals, dtype=int),
            "convergence_fitness": np.asarray(self.convergence_fitness, dtype=float),
        }


def _initialize(tracker: ObjectiveTracker, population: int) -> tuple[Array, Array]:
    count = min(population, tracker.remaining)
    positions = tracker.uniform(count)
    return positions, tracker.evaluate(positions)


def _available(tracker: ObjectiveTracker, candidates: Array) -> tuple[Array, Array]:
    selected = np.asarray(candidates, dtype=float)[: tracker.remaining]
    if len(selected) == 0:
        return selected, np.empty(0)
    selected = np.clip(selected, tracker.lower, tracker.upper)
    return selected, tracker.evaluate(selected)


def _simulated_annealing(tracker: ObjectiveTracker, config: OptimizerConfig) -> int:
    from scipy.optimize import dual_annealing

    def objective(point: Array) -> float:
        if tracker.remaining <= 0:
            return tracker.best_f
        return float(tracker.evaluate(np.asarray(point)[None, :])[0])

    dual_annealing(
        objective,
        bounds=list(map(tuple, tracker.bounds)),
        maxfun=tracker.budget,
        seed=config.seed,
        no_local_search=True,
    )
    tracker.finish_budget()
    return tracker.evaluations


def _differential_evolution(tracker: ObjectiveTracker, config: OptimizerConfig) -> int:
    from scipy.optimize import differential_evolution

    count = min(config.population, config.evaluation_budget)
    initial = tracker.uniform(count)
    generations = max(0, (config.evaluation_budget - count) // count)

    def objective(point: Array) -> float:
        if tracker.remaining <= 0:
            return tracker.best_f
        return float(tracker.evaluate(np.asarray(point)[None, :])[0])

    differential_evolution(
        objective,
        bounds=list(map(tuple, tracker.bounds)),
        init=initial,
        maxiter=generations,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=False,
        updating="immediate",
        workers=1,
        seed=config.seed,
    )
    tracker.finish_budget()
    return generations


def _pso(tracker: ObjectiveTracker, config: OptimizerConfig) -> int:
    positions, fitness = _initialize(tracker, config.population)
    velocities = tracker.rng.uniform(-0.1, 0.1, positions.shape) * tracker.range
    personal = positions.copy()
    personal_fitness = fitness.copy()
    iterations = 0
    while tracker.remaining:
        progress = tracker.evaluations / tracker.budget
        inertia = 0.9 - 0.5 * progress
        r1 = tracker.rng.random(positions.shape)
        r2 = tracker.rng.random(positions.shape)
        velocities = inertia * velocities + 1.7 * r1 * (personal - positions) + 1.7 * r2 * (tracker.best_x - positions)
        candidates, values = _available(tracker, positions + velocities)
        count = len(values)
        improved = values < personal_fitness[:count]
        positions[:count] = candidates
        improved_indices = np.flatnonzero(improved)
        personal[improved_indices] = candidates[improved_indices]
        personal_fitness[improved_indices] = values[improved_indices]
        iterations += 1
    return iterations


def _abc(tracker: ObjectiveTracker, config: OptimizerConfig) -> int:
    positions, fitness = _initialize(tracker, config.population)
    trials = np.zeros(len(positions), dtype=int)
    iterations = 0
    while tracker.remaining:
        for phase in range(2):
            if not tracker.remaining:
                break
            if phase == 0:
                selected = np.arange(len(positions))
            else:
                ranks = np.argsort(np.argsort(fitness))
                probabilities = (len(fitness) - ranks) / np.sum(len(fitness) - ranks)
                selected = tracker.rng.choice(len(positions), size=len(positions), p=probabilities)
            partners = tracker.rng.integers(0, len(positions), size=len(selected))
            partners = np.where(partners == selected, (partners + 1) % len(positions), partners)
            phi = tracker.rng.uniform(-1, 1, size=(len(selected), tracker.dimension))
            candidates, values = _available(tracker, positions[selected] + phi * (positions[selected] - positions[partners]))
            count = len(values)
            chosen = selected[:count]
            improved = values < fitness[chosen]
            positions[chosen[improved]] = candidates[improved]
            fitness[chosen[improved]] = values[improved]
            trials[chosen] += 1
            trials[chosen[improved]] = 0
        scouts = np.flatnonzero(trials > max(10, tracker.dimension))
        if scouts.size and tracker.remaining:
            scouts = scouts[: tracker.remaining]
            candidates, values = _available(tracker, tracker.uniform(len(scouts)))
            positions[scouts[: len(values)]] = candidates
            fitness[scouts[: len(values)]] = values
            trials[scouts[: len(values)]] = 0
        iterations += 1
    return iterations


def _wps(tracker: ObjectiveTracker, config: OptimizerConfig) -> int:
    positions, fitness = _initialize(tracker, config.population)
    iterations = 0
    while tracker.remaining:
        progress = tracker.evaluations / tracker.budget
        leader = positions[np.argmin(fitness)]
        step = (0.3 * (1 - progress) + 0.01) * tracker.range
        scout = positions + tracker.rng.uniform(-1, 1, positions.shape) * step
        besiege = leader + tracker.rng.random(positions.shape) * (leader - positions)
        candidates = np.where(tracker.rng.random((len(positions), 1)) < 0.5, scout, besiege)
        candidates, values = _available(tracker, candidates)
        count = len(values)
        improved = values < fitness[:count]
        improved_indices = np.flatnonzero(improved)
        positions[improved_indices] = candidates[improved_indices]
        fitness[improved_indices] = values[improved_indices]
        iterations += 1
    return iterations


def _ssa(tracker: ObjectiveTracker, config: OptimizerConfig) -> int:
    positions, fitness = _initialize(tracker, config.population)
    iterations = 0
    while tracker.remaining:
        order = np.argsort(fitness)
        positions, fitness = positions[order], fitness[order]
        progress = tracker.evaluations / tracker.budget
        c1 = 2 * np.exp(-((4 * progress) ** 2))
        candidates = positions.copy()
        c2 = tracker.rng.random(tracker.dimension)
        c3 = tracker.rng.random(tracker.dimension)
        candidates[0] = tracker.best_x + np.where(c3 < 0.5, 1, -1) * c1 * ((tracker.upper - tracker.lower) * c2 + tracker.lower)
        for index in range(1, len(candidates)):
            candidates[index] = 0.5 * (positions[index] + candidates[index - 1])
        candidates, values = _available(tracker, candidates)
        positions[: len(values)] = candidates
        fitness[: len(values)] = values
        iterations += 1
    return iterations


def _sbo(tracker: ObjectiveTracker, config: OptimizerConfig) -> int:
    positions, fitness = _initialize(tracker, config.population)
    iterations = 0
    while tracker.remaining:
        ranks = np.argsort(np.argsort(fitness))
        probabilities = (len(fitness) - ranks) / np.sum(len(fitness) - ranks)
        targets = tracker.rng.choice(len(positions), size=len(positions), p=probabilities)
        progress = tracker.evaluations / tracker.budget
        mutation = 0.05 * (1 - progress) * tracker.rng.normal(size=positions.shape) * tracker.range
        candidates = positions + tracker.rng.random(positions.shape) * (positions[targets] - positions) + mutation
        candidates, values = _available(tracker, candidates)
        count = len(values)
        improved = values < fitness[:count]
        improved_indices = np.flatnonzero(improved)
        positions[improved_indices] = candidates[improved_indices]
        fitness[improved_indices] = values[improved_indices]
        iterations += 1
    return iterations


def _mpa(tracker: ObjectiveTracker, config: OptimizerConfig) -> int:
    positions, fitness = _initialize(tracker, config.population)
    iterations = 0
    while tracker.remaining:
        progress = tracker.evaluations / tracker.budget
        brownian = tracker.rng.normal(size=positions.shape)
        levy = tracker.rng.standard_t(df=1.5, size=positions.shape) * 0.05
        if progress < 1 / 3:
            steps = brownian * (tracker.best_x - brownian * positions)
        elif progress < 2 / 3:
            mask = np.arange(len(positions))[:, None] < len(positions) / 2
            steps = np.where(mask, levy * (tracker.best_x - positions), brownian * (tracker.best_x - positions))
        else:
            steps = levy * (tracker.best_x - positions)
        candidates, values = _available(tracker, positions + (1 - progress) * steps)
        count = len(values)
        improved = values < fitness[:count]
        improved_indices = np.flatnonzero(improved)
        positions[improved_indices] = candidates[improved_indices]
        fitness[improved_indices] = values[improved_indices]
        iterations += 1
    return iterations


def _sca(tracker: ObjectiveTracker, config: OptimizerConfig) -> int:
    positions, fitness = _initialize(tracker, config.population)
    iterations = 0
    while tracker.remaining:
        progress = tracker.evaluations / tracker.budget
        r1 = 2 * (1 - progress)
        r2 = tracker.rng.uniform(0, 2 * np.pi, positions.shape)
        r3 = tracker.rng.uniform(0, 2, positions.shape)
        trig = np.where(tracker.rng.random(positions.shape) < 0.5, np.sin(r2), np.cos(r2))
        candidates, values = _available(tracker, positions + r1 * trig * np.abs(r3 * tracker.best_x - positions))
        count = len(values)
        improved = values < fitness[:count]
        improved_indices = np.flatnonzero(improved)
        positions[improved_indices] = candidates[improved_indices]
        fitness[improved_indices] = values[improved_indices]
        iterations += 1
    return iterations


def _gsa(tracker: ObjectiveTracker, config: OptimizerConfig) -> int:
    positions, fitness = _initialize(tracker, config.population)
    velocities = np.zeros_like(positions)
    iterations = 0
    while tracker.remaining:
        progress = tracker.evaluations / tracker.budget
        shifted = fitness.max() - fitness + 1e-12
        masses = shifted / shifted.sum()
        elite_count = max(2, int(len(positions) * (1 - 0.8 * progress)))
        elite = np.argsort(fitness)[:elite_count]
        acceleration = np.zeros_like(positions)
        gravitational = 100 * np.exp(-20 * progress)
        for source in elite:
            delta = positions[source] - positions
            distance = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-12
            acceleration += tracker.rng.random((len(positions), 1)) * gravitational * masses[source] * delta / distance
        velocities = tracker.rng.random(velocities.shape) * velocities + acceleration
        candidates, values = _available(tracker, positions + velocities)
        positions[: len(values)] = candidates
        fitness[: len(values)] = values
        iterations += 1
    return iterations


OPTIMIZERS = {
    "simulated_annealing": _simulated_annealing,
    "pso": _pso,
    "differential_evolution": _differential_evolution,
    "abc": _abc,
    "wolf_pack_search": _wps,
    "salp_swarm": _ssa,
    "satin_bowerbird": _sbo,
    "marine_predators": _mpa,
    "sine_cosine": _sca,
    "gravitational_search": _gsa,
}


def run_optimizer(
    name: str,
    objective: Callable[[Array], float],
    bounds: Array,
    config: OptimizerConfig,
) -> dict:
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {name}")
    tracker = ObjectiveTracker(objective, bounds, config.evaluation_budget, config.seed)
    iterations = OPTIMIZERS[name](tracker, config)
    tracker.finish_budget()
    return tracker.result(iterations)
