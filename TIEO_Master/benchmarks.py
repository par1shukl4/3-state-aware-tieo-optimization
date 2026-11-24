"""Canonical definitions for the 40 TIEO benchmark functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class Benchmark:
    id: int
    name: str
    function: Callable[[Array], float]
    bounds: Array
    optimum: float
    category: str

    @property
    def dimension(self) -> int:
        return int(self.bounds.shape[0])


def _bounds(dimension: int, lower: float, upper: float) -> Array:
    return np.tile(np.array([[lower, upper]], dtype=float), (dimension, 1))


def sphere(x: Array) -> float:
    return float(np.sum(x**2))


def sum_squares(x: Array) -> float:
    weights = np.arange(1, x.size + 1, dtype=float)
    return float(np.sum(weights * x**2))


def sum_different_powers(x: Array) -> float:
    powers = np.arange(2, x.size + 2, dtype=float)
    return float(np.sum(np.abs(x) ** powers))


def rotated_hyper_ellipsoid(x: Array) -> float:
    return float(np.sum(np.cumsum(x) ** 2))


def axis_parallel_hyper_ellipsoid(x: Array) -> float:
    weights = np.arange(1, x.size + 1, dtype=float)
    return float(np.sum(weights * x**2))


def zakharov(x: Array) -> float:
    weights = 0.5 * np.arange(1, x.size + 1, dtype=float)
    linear = float(np.dot(weights, x))
    return float(np.sum(x**2) + linear**2 + linear**4)


def schwefel_12(x: Array) -> float:
    return float(np.sum(np.cumsum(x) ** 2))


def schwefel_222(x: Array) -> float:
    absolute = np.abs(x)
    return float(np.sum(absolute) + np.prod(absolute))


def booth(x: Array) -> float:
    a, b = x
    return float((a + 2.0 * b - 7.0) ** 2 + (2.0 * a + b - 5.0) ** 2)


def matyas(x: Array) -> float:
    a, b = x
    return float(0.26 * (a**2 + b**2) - 0.48 * a * b)


def rosenbrock(x: Array) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2))


def beale(x: Array) -> float:
    a, b = x
    return float(
        (1.5 - a + a * b) ** 2
        + (2.25 - a + a * b**2) ** 2
        + (2.625 - a + a * b**3) ** 2
    )


def three_hump_camel(x: Array) -> float:
    a, b = x
    return float(2.0 * a**2 - 1.05 * a**4 + a**6 / 6.0 + a * b + b**2)


def six_hump_camel(x: Array) -> float:
    a, b = x
    return float((4.0 - 2.1 * a**2 + a**4 / 3.0) * a**2 + a * b + (-4.0 + 4.0 * b**2) * b**2)


def bukin_6(x: Array) -> float:
    a, b = x
    return float(100.0 * np.sqrt(abs(b - 0.01 * a**2)) + 0.01 * abs(a + 10.0))


def dixon_price(x: Array) -> float:
    first = (x[0] - 1.0) ** 2
    indices = np.arange(2, x.size + 1, dtype=float)
    rest = np.sum(indices * (2.0 * x[1:] ** 2 - x[:-1]) ** 2)
    return float(first + rest)


def colville(x: Array) -> float:
    x1, x2, x3, x4 = x
    return float(
        100.0 * (x1**2 - x2) ** 2
        + (x1 - 1.0) ** 2
        + (x3 - 1.0) ** 2
        + 90.0 * (x3**2 - x4) ** 2
        + 10.1 * ((x2 - 1.0) ** 2 + (x4 - 1.0) ** 2)
        + 19.8 * (x2 - 1.0) * (x4 - 1.0)
    )


def powell(x: Array) -> float:
    blocks = x.reshape(-1, 4)
    a, b, c, d = blocks.T
    return float(
        np.sum(
            (a + 10.0 * b) ** 2
            + 5.0 * (c - d) ** 2
            + (b - 2.0 * c) ** 4
            + 10.0 * (a - d) ** 4
        )
    )


def rastrigin(x: Array) -> float:
    return float(10.0 * x.size + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


def ackley(x: Array) -> float:
    square_mean = np.mean(x**2)
    cosine_mean = np.mean(np.cos(2.0 * np.pi * x))
    return float(-20.0 * np.exp(-0.2 * np.sqrt(square_mean)) - np.exp(cosine_mean) + 20.0 + np.e)


def griewank(x: Array) -> float:
    indices = np.sqrt(np.arange(1, x.size + 1, dtype=float))
    return float(np.sum(x**2) / 4000.0 - np.prod(np.cos(x / indices)) + 1.0)


def schwefel(x: Array) -> float:
    return float(418.9828872724338 * x.size - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def levy(x: Array) -> float:
    w = 1.0 + (x - 1.0) / 4.0
    first = np.sin(np.pi * w[0]) ** 2
    middle = np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0) ** 2))
    last = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)
    return float(first + middle + last)


def michalewicz(x: Array, m: float = 10.0) -> float:
    indices = np.arange(1, x.size + 1, dtype=float)
    return float(-np.sum(np.sin(x) * np.sin(indices * x**2 / np.pi) ** (2.0 * m)))


def styblinski_tang(x: Array) -> float:
    return float(0.5 * np.sum(x**4 - 16.0 * x**2 + 5.0 * x))


def eggholder(x: Array) -> float:
    a, b = x
    return float(-(b + 47.0) * np.sin(np.sqrt(abs(a / 2.0 + b + 47.0))) - a * np.sin(np.sqrt(abs(a - b - 47.0))))


def drop_wave(x: Array) -> float:
    radius2 = float(np.sum(x**2))
    return float(-(1.0 + np.cos(12.0 * np.sqrt(radius2))) / (0.5 * radius2 + 2.0))


def schaffer_2(x: Array) -> float:
    a, b = x
    radius2 = a**2 + b**2
    return float(0.5 + (np.sin(a**2 - b**2) ** 2 - 0.5) / (1.0 + 0.001 * radius2) ** 2)


def schaffer_4(x: Array) -> float:
    a, b = x
    radius2 = a**2 + b**2
    numerator = np.cos(np.sin(abs(a**2 - b**2))) ** 2 - 0.5
    return float(0.5 + numerator / (1.0 + 0.001 * radius2) ** 2)


_FOXHOLE_AXIS = np.array([-32.0, -16.0, 0.0, 16.0, 32.0], dtype=float)
_FOXHOLE_POINTS = np.array([(a, b) for b in _FOXHOLE_AXIS for a in _FOXHOLE_AXIS], dtype=float).T


def de_jong_5(x: Array) -> float:
    denominators = np.arange(1, 26, dtype=float) + np.sum((x[:, None] - _FOXHOLE_POINTS) ** 6, axis=0)
    return float(1.0 / (0.002 + np.sum(1.0 / denominators)))


_SHEKEL_A = np.array(
    [
        [4.0, 4.0, 4.0, 4.0],
        [1.0, 1.0, 1.0, 1.0],
        [8.0, 8.0, 8.0, 8.0],
        [6.0, 6.0, 6.0, 6.0],
        [3.0, 7.0, 3.0, 7.0],
        [2.0, 9.0, 2.0, 9.0],
        [5.0, 5.0, 3.0, 3.0],
        [8.0, 1.0, 8.0, 1.0],
        [6.0, 2.0, 6.0, 2.0],
        [7.0, 3.6, 7.0, 3.6],
    ]
)
_SHEKEL_C = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])


def shekel_10(x: Array) -> float:
    return float(-np.sum(1.0 / (np.sum((x - _SHEKEL_A) ** 2, axis=1) + _SHEKEL_C)))


def perm_function(x: Array, beta: float = 0.5) -> float:
    indices = np.arange(1, x.size + 1, dtype=float)
    total = 0.0
    for power in range(1, x.size + 1):
        inner = np.sum((indices**power + beta) * ((x / indices) ** power - 1.0))
        total += inner**2
    return float(total)


def branin(x: Array) -> float:
    a, b = x
    term = b - 5.1 * a**2 / (4.0 * np.pi**2) + 5.0 * a / np.pi - 6.0
    return float(term**2 + 10.0 * (1.0 - 1.0 / (8.0 * np.pi)) * np.cos(a) + 10.0)


def goldstein_price(x: Array) -> float:
    a, b = x
    first = 1.0 + (a + b + 1.0) ** 2 * (
        19.0 - 14.0 * a + 3.0 * a**2 - 14.0 * b + 6.0 * a * b + 3.0 * b**2
    )
    second = 30.0 + (2.0 * a - 3.0 * b) ** 2 * (
        18.0 - 32.0 * a + 12.0 * a**2 + 48.0 * b - 36.0 * a * b + 27.0 * b**2
    )
    return float(first * second)


def easom(x: Array) -> float:
    a, b = x
    return float(-np.cos(a) * np.cos(b) * np.exp(-((a - np.pi) ** 2 + (b - np.pi) ** 2)))


_HARTMANN_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_HARTMANN3_A = np.array([[3.0, 10.0, 30.0], [0.1, 10.0, 35.0], [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]])
_HARTMANN3_P = 1e-4 * np.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]])
_HARTMANN4_A = np.array([[10.0, 3.0, 17.0, 3.5], [0.05, 10.0, 17.0, 0.1], [3.0, 3.5, 1.7, 10.0], [17.0, 8.0, 0.05, 10.0]])
_HARTMANN4_P = 1e-4 * np.array([[1312, 1696, 5569, 124], [2329, 4135, 8307, 3736], [2348, 1451, 3522, 2883], [4047, 8828, 8732, 5743]])
_HARTMANN6_A = np.array([[10.0, 3.0, 17.0, 3.5, 1.7, 8.0], [0.05, 10.0, 17.0, 0.1, 8.0, 14.0], [3.0, 3.5, 1.7, 10.0, 17.0, 8.0], [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]])
_HARTMANN6_P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991], [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])


def _hartmann(x: Array, a: Array, p: Array) -> float:
    return float(-np.sum(_HARTMANN_ALPHA * np.exp(-np.sum(a * (x - p) ** 2, axis=1))))


def hartmann_3(x: Array) -> float:
    return _hartmann(x, _HARTMANN3_A, _HARTMANN3_P)


def hartmann_4(x: Array) -> float:
    outer = -_hartmann(x, _HARTMANN4_A, _HARTMANN4_P)
    return float((1.1 - outer) / 0.839)


def hartmann_6(x: Array) -> float:
    return _hartmann(x, _HARTMANN6_A, _HARTMANN6_P)


def forrester(x: Array) -> float:
    value = float(x[0])
    return float((6.0 * value - 2.0) ** 2 * np.sin(12.0 * value - 4.0))


def katsuura(x: Array) -> float:
    dimension = x.size
    powers = 2.0 ** np.arange(1, 33, dtype=float)
    terms = []
    for index, value in enumerate(x, start=1):
        fractional = np.abs(powers * value - np.rint(powers * value)) / powers
        terms.append((1.0 + index * np.sum(fractional)) ** (10.0 / dimension**1.2))
    scale = 10.0 / dimension**2
    return float(scale * np.prod(terms) - scale)


def get_benchmarks(default_dimension: int = 30) -> list[Benchmark]:
    """Return the fixed 40-function suite in the user-specified order."""

    d = int(default_dimension)
    suite = [
        Benchmark(1, "Sphere", sphere, _bounds(d, -100, 100), 0.0, "unimodal"),
        Benchmark(2, "Sum Squares", sum_squares, _bounds(d, -10, 10), 0.0, "unimodal"),
        Benchmark(3, "Sum Different Powers", sum_different_powers, _bounds(d, -1, 1), 0.0, "unimodal"),
        Benchmark(4, "Rotated Hyper-Ellipsoid", rotated_hyper_ellipsoid, _bounds(d, -65.536, 65.536), 0.0, "unimodal"),
        Benchmark(5, "Axis Parallel Hyper-Ellipsoid", axis_parallel_hyper_ellipsoid, _bounds(d, -5.12, 5.12), 0.0, "unimodal"),
        Benchmark(6, "Zakharov", zakharov, _bounds(d, -5, 10), 0.0, "unimodal"),
        Benchmark(7, "Schwefel 1.2", schwefel_12, _bounds(d, -100, 100), 0.0, "unimodal"),
        Benchmark(8, "Schwefel 2.22", schwefel_222, _bounds(d, -10, 10), 0.0, "unimodal"),
        Benchmark(9, "Booth", booth, _bounds(2, -10, 10), 0.0, "unimodal"),
        Benchmark(10, "Matyas", matyas, _bounds(2, -10, 10), 0.0, "unimodal"),
        Benchmark(11, "Rosenbrock", rosenbrock, _bounds(d, -30, 30), 0.0, "valley"),
        Benchmark(12, "Beale", beale, _bounds(2, -4.5, 4.5), 0.0, "valley"),
        Benchmark(13, "Three-Hump Camel", three_hump_camel, _bounds(2, -5, 5), 0.0, "valley"),
        Benchmark(14, "Six-Hump Camel", six_hump_camel, np.array([[-3, 3], [-2, 2]], dtype=float), -1.031628453489877, "valley"),
        Benchmark(15, "Bukin N.6", bukin_6, np.array([[-15, -5], [-3, 3]], dtype=float), 0.0, "valley"),
        Benchmark(16, "Dixon-Price", dixon_price, _bounds(d, -10, 10), 0.0, "valley"),
        Benchmark(17, "Colville", colville, _bounds(4, -10, 10), 0.0, "valley"),
        Benchmark(18, "Powell", powell, _bounds(4, -4, 5), 0.0, "valley"),
        Benchmark(19, "Rastrigin", rastrigin, _bounds(d, -5.12, 5.12), 0.0, "multimodal"),
        Benchmark(20, "Ackley", ackley, _bounds(d, -32.768, 32.768), 0.0, "multimodal"),
        Benchmark(21, "Griewank", griewank, _bounds(d, -600, 600), 0.0, "multimodal"),
        Benchmark(22, "Schwefel", schwefel, _bounds(d, -500, 500), 0.0, "multimodal"),
        Benchmark(23, "Levy", levy, _bounds(d, -10, 10), 0.0, "multimodal"),
        Benchmark(24, "Michalewicz", michalewicz, _bounds(2, 0, np.pi), -1.801303410098553, "multimodal"),
        Benchmark(25, "Styblinski-Tang", styblinski_tang, _bounds(d, -5, 5), -39.16599 * d, "multimodal"),
        Benchmark(26, "Eggholder", eggholder, _bounds(2, -512, 512), -959.6406627106155, "multimodal"),
        Benchmark(27, "Drop-Wave", drop_wave, _bounds(2, -5.12, 5.12), -1.0, "multimodal"),
        Benchmark(28, "Schaffer N.2", schaffer_2, _bounds(2, -100, 100), 0.0, "multimodal"),
        Benchmark(29, "Schaffer N.4", schaffer_4, _bounds(2, -100, 100), 0.292578632035980, "multimodal"),
        Benchmark(30, "De Jong N.5", de_jong_5, _bounds(2, -65.536, 65.536), 0.998003837794449, "multimodal"),
        Benchmark(31, "Shekel m=10", shekel_10, _bounds(4, 0, 10), -10.536409816692, "multimodal"),
        Benchmark(32, "Perm d,beta", perm_function, _bounds(10, -10, 10), 0.0, "multimodal"),
        Benchmark(33, "Branin", branin, np.array([[-5, 10], [0, 15]], dtype=float), 0.397887357729738, "classic"),
        Benchmark(34, "Goldstein-Price", goldstein_price, _bounds(2, -2, 2), 3.0, "classic"),
        Benchmark(35, "Easom", easom, _bounds(2, -100, 100), -1.0, "classic"),
        Benchmark(36, "Hartmann 3D", hartmann_3, _bounds(3, 0, 1), -3.86278214782076, "classic"),
        Benchmark(37, "Hartmann 4D", hartmann_4, _bounds(4, 0, 1), -3.134494141, "classic"),
        Benchmark(38, "Hartmann 6D", hartmann_6, _bounds(6, 0, 1), -3.32236801141551, "classic"),
        Benchmark(39, "Forrester", forrester, _bounds(1, 0, 1), -6.02074005576708, "classic"),
        Benchmark(40, "Katsuura", katsuura, _bounds(d, -100, 100), 0.0, "classic"),
    ]
    return suite


def select_benchmarks(ids: Iterable[int] | None = None, default_dimension: int = 30) -> list[Benchmark]:
    suite = get_benchmarks(default_dimension)
    if ids is None:
        return suite
    selected = {int(value) for value in ids}
    unknown = selected.difference({benchmark.id for benchmark in suite})
    if unknown:
        raise ValueError(f"Unknown benchmark IDs: {sorted(unknown)}")
    return [benchmark for benchmark in suite if benchmark.id in selected]


def validate_known_optima() -> dict[int, float]:
    """Evaluate known minimizers used for regression tests where compactly available."""

    points = {
        1: np.zeros(30),
        9: np.array([1.0, 3.0]),
        10: np.zeros(2),
        12: np.array([3.0, 0.5]),
        13: np.zeros(2),
        15: np.array([-10.0, 1.0]),
        17: np.ones(4),
        27: np.zeros(2),
        33: np.array([-np.pi, 12.275]),
        34: np.array([0.0, -1.0]),
        35: np.array([np.pi, np.pi]),
        39: np.array([0.757248757841856]),
    }
    registry = {benchmark.id: benchmark for benchmark in get_benchmarks(30)}
    return {identifier: registry[identifier].function(point) for identifier, point in points.items()}

