from __future__ import annotations

import random
from typing import Callable, List, Tuple

import numpy as np

FitnessFunc = Callable[[np.ndarray], float]

random.seed(42)
np.random.seed(42)


def _init_population(pop_size: int, dim: int, bounds: tuple[float, float], rng: np.random.Generator):
    low, high = bounds
    return rng.uniform(low, high, size=(pop_size, dim))


def _tournament(pop: np.ndarray, fits: np.ndarray, k: int, rng: np.random.Generator):
    idx = rng.integers(0, len(pop), size=k)
    best_idx = idx[np.argmin(fits[idx])]
    return pop[best_idx].copy()


def genetic_algorithm(
    dim: int,
    fitness_func: FitnessFunc,
    bounds: tuple[float, float] = (5.0, 60.0),
    pop_size: int = 60,
    generations: int = 80,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.1,
    mutation_scale: float = 5.0,
    elite_size: int = 0,
    seed: int = 42,
) -> Tuple[np.ndarray, float, List[float]]:
    rng = np.random.default_rng(seed)
    population = _init_population(pop_size, dim, bounds, rng)
    fitness_cache: dict[tuple, float] = {}
    _CACHE_LIMIT = 1000

    def evaluate(ind: np.ndarray) -> float:
        key = tuple(np.round(ind, 3))
        if key not in fitness_cache:
            if len(fitness_cache) >= _CACHE_LIMIT:
                fitness_cache.pop(next(iter(fitness_cache)))
            fitness_cache[key] = float(fitness_func(ind))
        return fitness_cache[key]

    best_hist = []
    for gen in range(generations):
        fitness_values = np.array([evaluate(ind) for ind in population])
        gen_best_idx = np.argmin(fitness_values)
        gen_best_fit = fitness_values[gen_best_idx]
        best_hist.append(float(gen_best_fit))
        if gen % 10 == 0 or gen == generations - 1:
            print(f"[GA] Gen {gen:03d} Best {gen_best_fit:.4f} Cache {len(fitness_cache)}")

        # Elitism
        elite_indices = fitness_values.argsort()[:max(1, elite_size)] if elite_size > 0 else []
        new_pop = [population[i].copy() for i in elite_indices]
        while len(new_pop) < pop_size:
            parent1 = _tournament(population, fitness_values, k=3, rng=rng)
            parent2 = _tournament(population, fitness_values, k=3, rng=rng)

            # Crossover (blend)
            if rng.random() < crossover_rate:
                alpha = rng.uniform(0, 1, size=dim)
                child = alpha * parent1 + (1 - alpha) * parent2
            else:
                child = parent1.copy()

            # Mutation
            if rng.random() < mutation_rate:
                child += rng.normal(0, mutation_scale, size=dim)

            # Exploration noise
            child += rng.normal(0, 2.0, size=dim)

            # Clip to bounds
            child = np.clip(child, bounds[0], bounds[1])
            if child.sum() > 120:
                child *= 120.0 / child.sum()
            new_pop.append(child)

        population = np.vstack(new_pop)

    # Final evaluation
    fitness_values = np.array([evaluate(ind) for ind in population])
    best_idx = np.argmin(fitness_values)
    return population[best_idx], float(fitness_values[best_idx]), best_hist


__all__ = ["genetic_algorithm"]
