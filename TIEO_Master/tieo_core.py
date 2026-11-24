"""Shared corrected TIEO implementation used by every state-classification method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.cluster import KMeans


Array = np.ndarray


@dataclass
class TIEOConfig:
    population: int = 50
    max_tribes: int = 5
    evaluation_budget: int = 15_000
    w: float = 0.5
    weight_schedule: str = "constant"
    w_start: float = 0.9
    w_end: float = 0.15
    gamma: float = 0.9
    relative_delta: float = 0.10
    quantile_schedule: str = "fixed"
    quantile_lower_start: float = 0.20
    quantile_upper_start: float = 0.80
    quantile_min_tribe_size: int = 0
    reward_mode: str = "binary"
    restart_fraction: float = 0.0
    stagnation_patience: int = 1_500
    max_restarts: int = 2
    stagnation_relative_tolerance: float = 1e-8
    restart_requires_low_diversity: bool = False
    restart_diversity_threshold: float = 0.05
    local_refinement_fraction: float = 0.0
    local_refinement_method: str = "Powell"
    seed: int = 1000


class TIEO:
    """Tribe-based optimizer with a small shared tabular Q learner.

    Public methods select a state classifier and/or reward model. All operators,
    acceptance rules, budgets, random-number generation and logging stay equal.
    """

    VALID_STATE_MODES = {"baseline", "quantile", "adaptive_zscore", "relative_deviation"}
    VALID_REWARD_MODES = {"binary", "evaluation_aware"}
    VALID_WEIGHT_SCHEDULES = {"constant", "linear"}
    VALID_QUANTILE_SCHEDULES = {"fixed", "adaptive"}

    def __init__(
        self,
        objective: Callable[[Array], float],
        bounds: Array,
        state_mode: str = "baseline",
        config: TIEOConfig | None = None,
    ) -> None:
        if state_mode not in self.VALID_STATE_MODES:
            raise ValueError(f"Unknown state mode: {state_mode}")

        self.objective = objective
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (dimension, 2)")
        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError("every lower bound must be smaller than its upper bound")

        self.state_mode = state_mode
        self.config = config or TIEOConfig()
        if self.config.reward_mode not in self.VALID_REWARD_MODES:
            raise ValueError(f"Unknown reward mode: {self.config.reward_mode}")
        if self.config.weight_schedule not in self.VALID_WEIGHT_SCHEDULES:
            raise ValueError(f"Unknown weight schedule: {self.config.weight_schedule}")
        if self.config.quantile_schedule not in self.VALID_QUANTILE_SCHEDULES:
            raise ValueError(f"Unknown quantile schedule: {self.config.quantile_schedule}")
        if not 0.0 < self.config.quantile_lower_start < self.config.quantile_upper_start < 1.0:
            raise ValueError("quantile start thresholds must satisfy 0 < lower < upper < 1")
        if self.config.quantile_min_tribe_size < 0:
            raise ValueError("quantile_min_tribe_size cannot be negative")
        if not 0.0 <= self.config.restart_fraction <= 1.0:
            raise ValueError("restart_fraction must be between 0 and 1")
        if self.config.stagnation_patience < 1:
            raise ValueError("stagnation_patience must be positive")
        if self.config.max_restarts < 0:
            raise ValueError("max_restarts cannot be negative")
        if not 0.0 <= self.config.restart_diversity_threshold <= 1.0:
            raise ValueError("restart_diversity_threshold must be between 0 and 1")
        if not 0.0 <= self.config.local_refinement_fraction < 1.0:
            raise ValueError("local_refinement_fraction must be in [0, 1)")
        if self.config.population < 2:
            raise ValueError("population must be at least 2")
        if self.config.evaluation_budget < self.config.population:
            raise ValueError("evaluation budget must cover population initialization")

        self.dimension = self.bounds.shape[0]
        self.lower = self.bounds[:, 0]
        self.upper = self.bounds[:, 1]
        self.range = self.upper - self.lower
        self.rng = np.random.default_rng(self.config.seed)
        self.eval_count = 0
        self._evaluation_limit = self.config.evaluation_budget
        self.state_count = 2 if state_mode == "baseline" else 3
        self.Q = np.zeros((self.state_count, 3), dtype=float)
        self.action_counts = np.zeros((self.state_count, 3), dtype=int)
        self.action_successes = np.zeros((self.state_count, 3), dtype=int)
        self.reward_sums = np.zeros((self.state_count, 3), dtype=float)
        self.state_visits = np.zeros(self.state_count, dtype=int)
        self.restart_evaluations: list[int] = []
        self.action_history_evals: list[int] = []
        self.action_history_actions: list[int] = []
        self.action_history_successes: list[bool] = []

    @property
    def remaining_evaluations(self) -> int:
        return self._evaluation_limit - self.eval_count

    def evaluate(self, points: Array) -> Array | float:
        values = np.asarray(points, dtype=float)
        one_point = values.ndim == 1
        matrix = values[None, :] if one_point else values
        if matrix.ndim != 2 or matrix.shape[1] != self.dimension:
            raise ValueError("candidate shape does not match benchmark dimension")
        if matrix.shape[0] > self.remaining_evaluations:
            raise RuntimeError("objective evaluation budget would be exceeded")

        results = np.array([float(self.objective(row)) for row in matrix], dtype=float)
        if not np.all(np.isfinite(results)):
            results = np.where(np.isfinite(results), results, np.inf)
        self.eval_count += matrix.shape[0]
        return float(results[0]) if one_point else results

    def _initialize(self) -> tuple[Array, Array]:
        population = self.rng.uniform(
            self.lower,
            self.upper,
            size=(self.config.population, self.dimension),
        )
        fitness = np.asarray(self.evaluate(population), dtype=float)
        return population, fitness

    def _create_tribes(self, population: Array, fitness: Array) -> list[dict]:
        maximum = min(self.config.max_tribes, self.config.population)
        tribe_count = int(self.rng.integers(2, maximum + 1))
        random_state = int(self.rng.integers(0, np.iinfo(np.int32).max))
        labels = KMeans(n_clusters=tribe_count, n_init=10, random_state=random_state).fit_predict(population)
        tribes = []
        for label in range(tribe_count):
            members = np.flatnonzero(labels == label).tolist()
            if members:
                tribes.append({"members": members})
        self._refresh_tribes(tribes, population, fitness)
        return tribes

    @staticmethod
    def _refresh_tribes(tribes: list[dict], population: Array, fitness: Array) -> None:
        for tribe in tribes:
            members = tribe["members"]
            if not members:
                tribe.update(leader=None, average=np.inf, leader_fitness=np.inf)
                continue
            member_fitness = fitness[members]
            local_best = int(np.argmin(member_fitness))
            leader = int(members[local_best])
            tribe.update(
                leader=leader,
                average=float(np.mean(member_fitness)),
                leader_fitness=float(fitness[leader]),
            )

    def _classify_states(self, tribes: list[dict]) -> list[int]:
        averages = np.array([tribe["average"] for tribe in tribes], dtype=float)
        if self.state_mode == "baseline":
            center = float(np.mean(averages))
            return [0 if value < center else 1 for value in averages]

        if averages.size < 3:
            center = float(np.mean(averages))
            return [0 if value < center else 2 for value in averages]

        if self.state_mode == "quantile":
            eligible = np.ones(averages.size, dtype=bool)
            if self.config.quantile_min_tribe_size:
                eligible = np.array(
                    [len(tribe["members"]) >= self.config.quantile_min_tribe_size for tribe in tribes],
                    dtype=bool,
                )
                if np.sum(eligible) < 3:
                    center = float(np.mean(averages))
                    return [0 if value < center else 2 for value in averages]
            if self.config.quantile_schedule == "adaptive":
                progress = min(1.0, self.eval_count / self.config.evaluation_budget)
                lower_quantile = self.config.quantile_lower_start + (1.0 / 3.0 - self.config.quantile_lower_start) * progress
                upper_quantile = self.config.quantile_upper_start + (2.0 / 3.0 - self.config.quantile_upper_start) * progress
            else:
                lower_quantile, upper_quantile = 1.0 / 3.0, 2.0 / 3.0
            lower, upper = np.quantile(averages[eligible], [lower_quantile, upper_quantile])
            center = float(np.mean(averages[eligible]))
            return [
                (0 if value < center else 2)
                if not eligible[index]
                else (0 if value <= lower else 2 if value > upper else 1)
                for index, value in enumerate(averages)
            ]

        if self.state_mode == "adaptive_zscore":
            standard_deviation = float(np.std(averages))
            if standard_deviation < 1e-12:
                return [1] * averages.size
            z_scores = (averages - np.mean(averages)) / standard_deviation
            threshold = max(0.5, float(np.median(np.abs(z_scores))))
            return [0 if z <= -threshold else 2 if z >= threshold else 1 for z in z_scores]

        center = float(np.mean(averages))
        robust_scale = max(
            abs(center),
            float(np.median(np.abs(averages - center))),
            1e-12,
        )
        deviations = (averages - center) / robust_scale
        delta = self.config.relative_delta
        return [0 if value < -delta else 2 if value > delta else 1 for value in deviations]

    def _select_action(self, state: int, progress: float, tribe_count: int) -> int:
        if tribe_count <= 1:
            return 0
        q_values = self.Q[state]
        if np.allclose(q_values, q_values[0]):
            return int(self.rng.integers(0, 3))
        exploitation_probability = 0.6 + 0.3 * progress
        if self.rng.random() > exploitation_probability:
            return int(self.rng.integers(0, 3))
        maximum = np.max(q_values)
        tied = np.flatnonzero(np.isclose(q_values, maximum))
        return int(self.rng.choice(tied))

    def _limited_members(self, members: list[int], leader: int | None = None) -> Array:
        candidates = np.asarray(members, dtype=int)
        if leader is not None and candidates.size > 1:
            candidates = candidates[candidates != leader]
        if candidates.size > self.remaining_evaluations:
            candidates = self.rng.choice(candidates, size=self.remaining_evaluations, replace=False)
        return np.asarray(candidates, dtype=int)

    def _calculate_reward(
        self,
        old_stat: float,
        new_stat: float,
        population_fitness: Array,
        evaluations_used: int,
        binary_reward: float,
    ) -> float:
        """Return the configured Q-learning reward.

        ``evaluation_aware`` measures signed improvement relative to a robust
        fitness scale, clips extreme values, and divides by objective calls.
        The original binary reward remains untouched for all existing methods.
        """

        if self.config.reward_mode == "binary":
            return float(binary_reward)

        values = np.asarray(population_fitness, dtype=float)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = max(abs(float(old_stat)), mad, 1e-12)
        normalized_improvement = (float(old_stat) - float(new_stat)) / scale
        clipped = float(np.clip(normalized_improvement, -1.0, 1.0))
        return clipped / max(int(evaluations_used), 1)

    def _autonomy_weight(self) -> float:
        if self.config.weight_schedule == "constant":
            return float(self.config.w)
        progress = min(1.0, self.eval_count / self.config.evaluation_budget)
        return float(self.config.w_start + (self.config.w_end - self.config.w_start) * progress)

    def _restart_worst(self, population: Array, fitness: Array) -> int:
        """Reinitialize the worst configured fraction while preserving the best."""

        if self.config.restart_fraction <= 0.0 or self.remaining_evaluations <= 0:
            return 0
        restart_count = max(1, int(np.ceil(self.config.restart_fraction * self.config.population)))
        restart_count = min(restart_count, self.remaining_evaluations, self.config.population - 1)
        if restart_count <= 0:
            return 0

        protected = int(np.argmin(fitness))
        worst_first = np.argsort(fitness)[::-1]
        selected = np.array([index for index in worst_first if index != protected][:restart_count], dtype=int)
        if selected.size == 0:
            return 0
        new_positions = self.rng.uniform(self.lower, self.upper, size=(selected.size, self.dimension))
        new_fitness = np.asarray(self.evaluate(new_positions), dtype=float)
        population[selected] = new_positions
        fitness[selected] = new_fitness
        self.restart_evaluations.append(self.eval_count)
        return int(selected.size)

    def _population_diversity(self, population: Array) -> float:
        """Return mean coordinate spread normalized by each search-range width."""

        return float(np.mean(np.std(population, axis=0) / np.maximum(self.range, 1e-12)))

    def _local_refine(
        self,
        best_x: Array,
        best_f: float,
        convergence_evals: list[int],
        convergence_fitness: list[float],
    ) -> tuple[Array, float, int]:
        """Spend the reserved budget refining the best point with bounded Powell search."""

        from scipy.optimize import minimize

        start_evaluations = self.eval_count
        local_best_x = best_x.copy()
        local_best_f = float(best_f)

        def local_objective(point: Array) -> float:
            nonlocal local_best_x, local_best_f
            if self.remaining_evaluations <= 0:
                return local_best_f
            candidate = np.clip(np.asarray(point, dtype=float), self.lower, self.upper)
            value = float(self.evaluate(candidate))
            if value < local_best_f:
                local_best_x = candidate.copy()
                local_best_f = value
            convergence_evals.append(self.eval_count)
            convergence_fitness.append(local_best_f)
            return value

        available = self.remaining_evaluations
        if available > 0:
            minimize(
                local_objective,
                local_best_x,
                method=self.config.local_refinement_method,
                bounds=list(map(tuple, self.bounds)),
                options={"maxfev": available, "maxiter": available, "disp": False},
            )

        # A local solver may terminate early. Use any remainder for shrinking
        # Gaussian probes so every method retains the same exact budget.
        total_reserved = max(1, self.config.evaluation_budget - start_evaluations)
        while self.remaining_evaluations > 0:
            used_fraction = (self.eval_count - start_evaluations) / total_reserved
            scale = max(1e-4, 0.05 * (1.0 - used_fraction))
            candidate = np.clip(
                local_best_x + self.rng.normal(size=self.dimension) * scale * self.range,
                self.lower,
                self.upper,
            )
            value = float(self.evaluate(candidate))
            if value < local_best_f:
                local_best_x = candidate.copy()
                local_best_f = value
            convergence_evals.append(self.eval_count)
            convergence_fitness.append(local_best_f)

        return local_best_x, local_best_f, self.eval_count - start_evaluations

    def _autonomy(self, tribe: dict, population: Array, fitness: Array) -> tuple[float, bool]:
        indices = self._limited_members(tribe["members"], tribe["leader"])
        if indices.size == 0:
            return 0.0, False

        old_positions = population[indices].copy()
        old_fitness = fitness[indices].copy()
        reward_scale_fitness = fitness.copy()
        leader = population[tribe["leader"]].copy()
        branch = self.rng.random(old_positions.shape) < 0.5
        sigma = np.abs(leader - old_positions) + 1e-12
        normal_candidates = self.rng.normal(loc=leader, scale=sigma)
        guided_candidates = old_positions + self._autonomy_weight() * self.rng.normal(size=old_positions.shape) * (leader - old_positions)
        candidates = np.where(branch, normal_candidates, guided_candidates)

        if len(tribe["members"]) == 1:
            candidates += self.rng.normal(0.0, 0.01 * self.range, size=candidates.shape)

        candidates = np.clip(candidates, self.lower, self.upper)
        candidate_fitness = np.asarray(self.evaluate(candidates), dtype=float)
        improved = candidate_fitness < old_fitness
        accepted_fitness = np.where(improved, candidate_fitness, old_fitness)
        population[indices[improved]] = candidates[improved]
        fitness[indices[improved]] = candidate_fitness[improved]
        reward = self._calculate_reward(
            old_stat=float(np.mean(old_fitness)),
            new_stat=float(np.mean(accepted_fitness)),
            population_fitness=reward_scale_fitness,
            evaluations_used=indices.size,
            binary_reward=float(0.5 * np.mean(improved)),
        )
        return reward, bool(np.any(improved))

    def _diplomacy(
        self,
        tribe: dict,
        target: dict,
        population: Array,
        fitness: Array,
    ) -> tuple[float, bool]:
        indices = self._limited_members(tribe["members"])
        if indices.size == 0:
            return 0.0, False

        old_positions = population[indices].copy()
        old_fitness = fitness[indices].copy()
        reward_scale_fitness = fitness.copy()
        target_leader = population[target["leader"]].copy()
        tribe_average = float(np.mean(old_fitness))
        distances = np.linalg.norm(target_leader - old_positions, axis=1) + 1e-12
        worse = old_fitness > tribe_average

        candidates = old_positions.copy()
        if np.any(worse):
            random_normal = self.rng.normal(size=int(np.sum(worse)))
            candidates[worse] = random_normal[:, None] * (old_positions[worse] + np.sqrt(distances[worse])[:, None])
        if np.any(~worse):
            mask = self.rng.integers(0, 2, size=(int(np.sum(~worse)), self.dimension))
            candidates[~worse] = mask * old_positions[~worse] + (1 - mask) * target_leader

        candidates = np.clip(candidates, self.lower, self.upper)
        candidate_fitness = np.asarray(self.evaluate(candidates), dtype=float)
        improved = candidate_fitness < old_fitness
        accepted_fitness = np.where(improved, candidate_fitness, old_fitness)
        population[indices[improved]] = candidates[improved]
        fitness[indices[improved]] = candidate_fitness[improved]
        reward = self._calculate_reward(
            old_stat=float(np.mean(old_fitness)),
            new_stat=float(np.mean(accepted_fitness)),
            population_fitness=reward_scale_fitness,
            evaluations_used=indices.size,
            binary_reward=float(0.5 * np.mean(improved)),
        )
        return reward, bool(np.any(improved))

    def _war(
        self,
        tribe: dict,
        target: dict,
        population: Array,
        fitness: Array,
    ) -> tuple[float, bool]:
        if self.remaining_evaluations <= 0:
            return 0.0, False
        winner, loser = (tribe, target) if tribe["average"] <= target["average"] else (target, tribe)
        if not loser["members"]:
            return 0.0, False

        captured = max(loser["members"], key=lambda index: fitness[index])
        old_fitness = float(fitness[captured])
        reward_scale_fitness = fitness.copy()
        winner_leader = population[winner["leader"]]
        loser_leader = population[loser["leader"]]
        sigma = np.abs(winner_leader - loser_leader) + 1e-12
        candidate = self.rng.normal(loc=winner_leader, scale=sigma)
        candidate = np.clip(candidate, self.lower, self.upper)
        candidate_fitness = float(self.evaluate(candidate))

        population[captured] = candidate
        fitness[captured] = candidate_fitness
        loser["members"].remove(captured)
        winner["members"].append(captured)
        improved = candidate_fitness < old_fitness
        reward = self._calculate_reward(
            old_stat=old_fitness,
            new_stat=candidate_fitness,
            population_fitness=reward_scale_fitness,
            evaluations_used=1,
            binary_reward=0.5 if improved else 0.0,
        )
        return reward, improved

    def _update_q(self, state: int, action: int, reward: float, next_state: int, progress: float) -> None:
        learning_rate = max(0.1, 1.0 - 0.9 * progress)
        target = reward + self.config.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += learning_rate * (target - self.Q[state, action])

    def run(self) -> dict:
        refinement_reserve = min(
            int(round(self.config.local_refinement_fraction * self.config.evaluation_budget)),
            self.config.evaluation_budget - self.config.population,
        )
        self._evaluation_limit = self.config.evaluation_budget - refinement_reserve
        population, fitness = self._initialize()
        tribes = self._create_tribes(population, fitness)
        best_index = int(np.argmin(fitness))
        best_x = population[best_index].copy()
        best_f = float(fitness[best_index])
        convergence_evals = [self.eval_count]
        convergence_fitness = [best_f]
        last_meaningful_improvement_eval = self.eval_count
        iterations = 0

        while self.remaining_evaluations > 0:
            tribes = [tribe for tribe in tribes if tribe["members"]]
            if len(tribes) <= 1:
                tribes = self._create_tribes(population, fitness)
            self._refresh_tribes(tribes, population, fitness)
            made_evaluation = False

            for tribe in list(tribes):
                if self.remaining_evaluations <= 0 or not tribe["members"]:
                    break
                self._refresh_tribes(tribes, population, fitness)
                states = self._classify_states(tribes)
                tribe_index = tribes.index(tribe)
                state = states[tribe_index]
                progress = self.eval_count / self.config.evaluation_budget
                action = self._select_action(state, progress, len(tribes))
                before = self.eval_count

                if action == 0:
                    reward, success = self._autonomy(tribe, population, fitness)
                else:
                    targets = [candidate for candidate in tribes if candidate is not tribe and candidate["members"]]
                    if not targets:
                        action = 0
                        reward, success = self._autonomy(tribe, population, fitness)
                    else:
                        target = targets[int(self.rng.integers(0, len(targets)))]
                        if action == 1:
                            reward, success = self._diplomacy(tribe, target, population, fitness)
                        else:
                            reward, success = self._war(tribe, target, population, fitness)

                if self.eval_count == before:
                    continue
                made_evaluation = True
                self.state_visits[state] += 1
                self.action_counts[state, action] += 1
                self.reward_sums[state, action] += reward
                if success:
                    self.action_successes[state, action] += 1
                self.action_history_evals.append(self.eval_count)
                self.action_history_actions.append(action)
                self.action_history_successes.append(success)

                tribes = [candidate for candidate in tribes if candidate["members"]]
                self._refresh_tribes(tribes, population, fitness)
                if tribe in tribes:
                    next_states = self._classify_states(tribes)
                    next_state = next_states[tribes.index(tribe)]
                else:
                    next_state = state
                self._update_q(state, action, reward, next_state, progress)

                current_best = int(np.argmin(fitness))
                if fitness[current_best] < best_f:
                    previous_best = best_f
                    best_f = float(fitness[current_best])
                    best_x = population[current_best].copy()
                    meaningful_threshold = max(
                        1e-12,
                        self.config.stagnation_relative_tolerance * max(abs(previous_best), 1.0),
                    )
                    if previous_best - best_f > meaningful_threshold:
                        last_meaningful_improvement_eval = self.eval_count
                convergence_evals.append(self.eval_count)
                convergence_fitness.append(best_f)

                restart_due = (
                    self.config.restart_fraction > 0.0
                    and len(self.restart_evaluations) < self.config.max_restarts
                    and self.eval_count - last_meaningful_improvement_eval >= self.config.stagnation_patience
                    and (
                        not self.config.restart_requires_low_diversity
                        or self._population_diversity(population) <= self.config.restart_diversity_threshold
                    )
                )
                if restart_due and self._restart_worst(population, fitness) > 0:
                    current_best = int(np.argmin(fitness))
                    if fitness[current_best] < best_f:
                        best_f = float(fitness[current_best])
                        best_x = population[current_best].copy()
                    convergence_evals.append(self.eval_count)
                    convergence_fitness.append(best_f)
                    last_meaningful_improvement_eval = self.eval_count
                    tribes = self._create_tribes(population, fitness)
                    break

            iterations += 1
            if not made_evaluation:
                break

        self._evaluation_limit = self.config.evaluation_budget
        local_refinement_evaluations = 0
        if refinement_reserve > 0:
            best_x, best_f, local_refinement_evaluations = self._local_refine(
                best_x,
                best_f,
                convergence_evals,
                convergence_fitness,
            )

        return {
            "best_x": best_x,
            "best_f": best_f,
            "eval_count": self.eval_count,
            "iterations": iterations,
            "Q": self.Q.copy(),
            "state_visits": self.state_visits.copy(),
            "action_counts": self.action_counts.copy(),
            "action_successes": self.action_successes.copy(),
            "reward_sums": self.reward_sums.copy(),
            "action_history_evals": np.asarray(self.action_history_evals, dtype=int),
            "action_history_actions": np.asarray(self.action_history_actions, dtype=int),
            "action_history_successes": np.asarray(self.action_history_successes, dtype=bool),
            "restart_count": len(self.restart_evaluations),
            "restart_evaluations": np.asarray(self.restart_evaluations, dtype=int),
            "local_refinement_evaluations": local_refinement_evaluations,
            "final_autonomy_weight": self._autonomy_weight(),
            "convergence_evals": np.asarray(convergence_evals, dtype=int),
            "convergence_fitness": np.asarray(convergence_fitness, dtype=float),
        }

