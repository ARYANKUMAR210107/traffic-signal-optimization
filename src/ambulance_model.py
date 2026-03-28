from __future__ import annotations

import math
import random
from typing import Hashable, Tuple

import networkx as nx

random.seed(42)

# Cache shortest routes keyed by graph identity and endpoints
_ROUTE_CACHE: dict[tuple[int, Hashable, Hashable, bool], tuple[tuple[Hashable, ...], float]] = {}
_CACHE_LIMIT = 1000


def _euclid(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest_node(G: nx.Graph, lat: float, lon: float) -> Hashable:
    """
    Find the nearest graph node to a (lat, lon) point.
    """
    target = (lon, lat)  # graph uses (lon, lat)
    best_node = None
    best_dist = float("inf")
    for node in G.nodes:
        d = _euclid(node, target)
        if d < best_dist:
            best_dist = d
            best_node = node
    if best_node is None:
        raise ValueError("Graph has no nodes to match incident/hospital location")
    return best_node


def random_hospital(G: nx.Graph, seed: int = 42) -> Hashable:
    nodes = list(G.nodes)
    if not nodes:
        raise ValueError("Graph is empty")
    rng = random.Random(seed)
    return rng.choice(nodes)


def shortest_route(
    G: nx.Graph, source: Hashable, target: Hashable, use_priority: bool = True
) -> tuple[list[Hashable], float]:
    """
    Compute shortest route and cost using Dijkstra on selected weight attribute.
    """
    key = (id(G), source, target, use_priority)
    if key in _ROUTE_CACHE:
        path, cost = _ROUTE_CACHE[key]
        return list(path), cost

    weight = "priority_weight" if use_priority else "weight"
    path = nx.dijkstra_path(G, source=source, target=target, weight=weight)
    cost = nx.path_weight(G, path, weight=weight)
    if len(_ROUTE_CACHE) >= _CACHE_LIMIT:
        _ROUTE_CACHE.pop(next(iter(_ROUTE_CACHE)))
    _ROUTE_CACHE[key] = (tuple(path), cost)
    return path, cost


__all__ = ["nearest_node", "random_hospital", "shortest_route"]
