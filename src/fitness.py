from __future__ import annotations

from typing import Hashable, Sequence, Tuple, Dict, List
import random
import numpy as np
import networkx as nx

random.seed(42)
np.random.seed(42)

MIN_GREEN = 5.0
YELLOW_TIME = 3.0
MAX_CYCLE = 120.0
VEH_CAPACITY_PER_SEC = 1.2

_PATH_CACHE: dict = {}
_CACHE_LIMIT = 1000


def _get_path_cache(G: nx.Graph, path: Sequence[Hashable]) -> dict:
    key = (id(G), tuple(path))
    if key not in _PATH_CACHE:
        if len(_PATH_CACHE) >= _CACHE_LIMIT:
            _PATH_CACHE.pop(next(iter(_PATH_CACHE)))

        travel = nx.path_weight(G, path, weight="priority_weight")
        edges = list(zip(path[:-1], path[1:]))

        weights = []
        for u, v in edges:
            data = G[u][v]
            weights.append(data.get("weight", data.get("priority_weight", 1.0)))

        _PATH_CACHE[key] = {
            "travel": float(travel),
            "weights": np.asarray(weights, dtype=float),
        }

    return _PATH_CACHE[key]


def signal_delay_cost(path, signal_nodes, timings):
    timings = np.clip(np.asarray(timings), MIN_GREEN, 60.0)
    node_to_time = dict(zip(signal_nodes, timings))

    delay = 0.0
    base_wait = 20.0

    for node in path:
        if node in node_to_time:
            green = node_to_time[node]
            cycle = MAX_CYCLE + YELLOW_TIME
            red_fraction = (cycle - green) / cycle

            delay += red_fraction * base_wait * 1.2

    return delay


def _simulate_cycles(G, path, signal_nodes, timings, cycles=40, seed=42):
    cache = _get_path_cache(G, path)
    rng = np.random.default_rng(seed)

    timings = np.clip(np.asarray(timings), MIN_GREEN, 60)

    queue = np.zeros(len(signal_nodes))

    total_travel = 0
    total_signal = 0
    total_queue = 0
    total_stops = 0
    per_cycle_delay: List[float] = []

    edge_weights = cache["weights"]

    for t in range(cycles):

        # realistic traffic spikes
        if rng.random() < 0.3:
            arrivals = rng.uniform(20, 40, size=len(signal_nodes))
        else:
            arrivals = rng.uniform(5, 15, size=len(signal_nodes))

        service = timings * VEH_CAPACITY_PER_SEC

        # single controlled queue growth
        queue = queue * 1.15 + arrivals - service
        queue = np.maximum(queue, 0)

        # no hard clipping (keep sensitivity)
        queue = np.minimum(queue, 300)

        red_fraction = np.clip((MAX_CYCLE - timings) / MAX_CYCLE, 0, 1)

        # nonlinear signal delay (no clip)
        signal_delay = red_fraction * (queue + arrivals) ** 1.25 * 2.5

        stops = red_fraction * (queue + arrivals)

        # nonlinear congestion
        congestion = 1 + (np.mean(queue) ** 1.2) * 0.03
        travel_time = edge_weights.sum() * congestion

        # balanced contributions
        total_travel += travel_time * 0.4
        total_signal += signal_delay.sum() * 2.5
        total_queue += queue.sum() * 1.8
        total_stops += stops.sum()

        # consistent delay tracking
        per_cycle_delay.append(
            travel_time * 0.4 + signal_delay.sum() * 2.5 + queue.sum() * 0.5
        )

    metrics = {
        "travel_time": total_travel,
        "signal_delay": total_signal,
        "queue_length": total_queue,
        "stops": total_stops,
    }

    # stronger imbalance penalty
    imbalance_penalty = np.std(timings) * 20.0

    real_delay = total_travel + total_signal + total_queue * 0.5 + imbalance_penalty

    return metrics, per_cycle_delay, real_delay


def evaluate_solution(G, path, signal_nodes, timings):
    timings = np.clip(np.asarray(timings, dtype=float), MIN_GREEN, 60.0)
    if timings.sum() > MAX_CYCLE:
        timings = timings * (MAX_CYCLE / timings.sum())

    metrics, per_cycle, real_delay = _simulate_cycles(G, path, signal_nodes, timings)

    return real_delay, metrics, per_cycle, real_delay


def fitness(G, path, signal_nodes, timings):
    real_delay, _, _, _ = evaluate_solution(G, path, signal_nodes, timings)
    return real_delay


__all__ = ["signal_delay_cost", "evaluate_solution", "fitness"]
