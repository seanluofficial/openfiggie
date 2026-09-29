"""Small statistics helpers: means with normal-approximation CIs, and bootstrap CIs."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Estimate:
    mean: float
    stderr: float
    n: int

    @property
    def ci95(self) -> tuple[float, float]:
        half = 1.96 * self.stderr
        return self.mean - half, self.mean + half

    def __str__(self) -> str:
        return f"{self.mean:+.2f} ± {1.96 * self.stderr:.2f}"


def estimate(xs: Sequence[float]) -> Estimate:
    n = len(xs)
    if n == 0:
        return Estimate(math.nan, math.nan, 0)
    mean = sum(xs) / n
    if n == 1:
        return Estimate(mean, math.nan, 1)
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return Estimate(mean, math.sqrt(var / n), n)


def bootstrap_ci(
    xs: Sequence[float], n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(rng.choices(xs, k=n)) / n for _ in range(n_resamples))
    lo = means[int(alpha / 2 * n_resamples)]
    hi = means[int((1 - alpha / 2) * n_resamples) - 1]
    return lo, hi
