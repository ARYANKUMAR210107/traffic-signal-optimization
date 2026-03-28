from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

FitnessFunc = Callable[[np.ndarray], float]

np.random.seed(42)


def grasshopper_optimization(
    dim: int,
    fitness_func: FitnessFunc,
    bounds: tuple[float, float] = (5.0, 60.0),
    population_size: int = 30,
    iterations: int = 80,
    c_max: float = 1.0,
    c_min: float = 0.00004,
    f: float = 0.5,
    l: float = 1.5,
    seed: int = 42,
) -> Tuple[np.ndarray, float, List[float]]:
    rng = np.random.default_rng(seed)
    lb, ub = bounds
    pos = rng.uniform(lb, ub, size=(population_size, dim))
    pos += rng.normal(0, 2.0, size=pos.shape)
    pos = np.clip(pos, lb, ub)
    for i in range(population_size):
        if pos[i].sum() > 120:
            pos[i] *= 120.0 / pos[i].sum()

    fitness_cache: dict[tuple, float] = {}
    _CACHE_LIMIT = 1000

    def evaluate(ind: np.ndarray) -> float:
        key = tuple(np.round(ind, 3))
        if key not in fitness_cache:
            if len(fitness_cache) >= _CACHE_LIMIT:
                fitness_cache.pop(next(iter(fitness_cache)))
            fitness_cache[key] = float(fitness_func(ind))
        return fitness_cache[key]

    fitness = np.array([evaluate(p) for p in pos])
    gbest_idx = np.argmin(fitness)
    gbest = pos[gbest_idx].copy()
    gbest_fit = float(fitness[gbest_idx])

    history: List[float] = [gbest_fit]

    for t in range(iterations):
        c = c_max - (c_max - c_min) * (t / iterations)

        for i in range(population_size):
            s_i = np.zeros(dim)
            for j in range(population_size):
                if i == j:
                    continue
                dist = np.linalg.norm(pos[j] - pos[i]) + 1e-9
                r = 2 + rng.random()  # social interaction weight
                s_ij = f * np.exp(-l * dist) - np.exp(-dist)
                direction = (pos[j] - pos[i]) / dist
                s_i += r * s_ij * direction

            pos[i] = c * ((ub - lb) / 2 * s_i) + gbest
            pos[i] = np.clip(pos[i], lb, ub)
            if pos[i].sum() > 120:
                pos[i] *= 120.0 / pos[i].sum()

        pos += rng.normal(0, 2.0, size=pos.shape)
        pos = np.clip(pos, lb, ub)
        for i in range(population_size):
            if pos[i].sum() > 120:
                pos[i] *= 120.0 / pos[i].sum()

        fitness = np.array([evaluate(p) for p in pos])
        gbest_idx = np.argmin(fitness)
        if fitness[gbest_idx] < gbest_fit:
            gbest_fit = float(fitness[gbest_idx])
            gbest = pos[gbest_idx].copy()
        history.append(gbest_fit)
        if t % 10 == 0 or t == iterations - 1:
            print(f"[GOA] Iter {t:03d} Best {gbest_fit:.4f} Cache {len(fitness_cache)}")

    return gbest, gbest_fit, history


__all__ = ["grasshopper_optimization"]
