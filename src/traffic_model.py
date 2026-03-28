from __future__ import annotations

import numpy as np
import networkx as nx


def apply_traffic(
    G: nx.Graph,
    delay_lookup: dict[int, float],
    hour: int,
    peak_multiplier: float = 1.5,
    noise_std: float = 0.05,
    seed: int = 42,
) -> nx.Graph:
    """
    Update edge weights to reflect traffic-induced delay.

    weight = distance * (1 + peak_multiplier * delay_factor + noise)
    """
    rng = np.random.default_rng(seed)
    delay_factor = delay_lookup.get(hour % 24, np.mean(list(delay_lookup.values())))

    for u, v, data in G.edges(data=True):
        distance = data.get("distance", 1.0)
        noise = abs(rng.normal(0, noise_std))
        weight = distance * (1 + peak_multiplier * delay_factor + noise)
        data["weight"] = weight
        data["priority_weight"] = weight * 0.7  # ambulance priority lane / signal pre-emption

    return G


__all__ = ["apply_traffic"]
