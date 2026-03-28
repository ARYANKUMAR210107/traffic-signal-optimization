from __future__ import annotations

import random
from functools import partial
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")

import numpy as np
import networkx as nx

from dataset_loader import delay_lookup_by_hour, load_ambulance_incident, load_traffic_data
from graph_builder import build_graph
from traffic_model import apply_traffic
from ambulance_model import nearest_node, random_hospital, shortest_route
from fitness import fitness, evaluate_solution
from ga import genetic_algorithm
from goa import grasshopper_optimization
from visualization import plot_comparison, plot_convergence, plot_delay_over_time


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAFFIC_CSV = DATA_DIR / "Metro_Interstate_Traffic_Volume.csv"
AMBULANCE_CSV = DATA_DIR / "ambulance" / "911.csv"
ROAD_DIR = DATA_DIR / "road_network"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def pick_signal_nodes(path: List, k: int) -> List:
    unique_nodes = []
    for node in path:
        if node not in unique_nodes:
            unique_nodes.append(node)
        if len(unique_nodes) >= k:
            break
    return unique_nodes


def webster_timings(flow: np.ndarray, lost_time: float = 16.0) -> np.ndarray:
    sat_flow = 1800.0
    y = np.clip(flow / sat_flow, 0.01, 0.9)
    Y = y.sum()
    C = (1.5 * lost_time + 5) / max(1e-3, 1 - Y)
    g = (y / max(Y, 1e-3)) * (C - lost_time)
    return np.clip(g, 5.0, 60.0)


def main():
    print("Loading datasets...")
    traffic_df = load_traffic_data(TRAFFIC_CSV)
    delay_lookup = delay_lookup_by_hour(traffic_df)
    incident_lat, incident_lon = load_ambulance_incident(AMBULANCE_CSV, seed=SEED)

    print("Building road network graph...")
    G = build_graph(ROAD_DIR, max_edges=4000)

    apply_traffic(G, delay_lookup, hour=8, seed=SEED)

    incident_node = nearest_node(G, incident_lat, incident_lon)

    for attempt in range(10):
        hospital_node = random_hospital(G, seed=SEED + attempt)
        try:
            base_path, _ = shortest_route(G, hospital_node, incident_node, use_priority=True)
            break
        except:
            continue

    signal_nodes = pick_signal_nodes(base_path, k=min(8, len(base_path)))
    dim = len(signal_nodes)

    rng = np.random.default_rng(SEED)

    baseline_timings = np.full(dim, 30.0)

    # 🔥 FIXED RANDOM BASELINE
    base = np.full(dim, 30.0)
    noise = rng.uniform(0.85, 1.15, size=dim)
    random_timings = np.clip(base * noise, 10.0, 50.0)
    if random_timings.sum() > 120:
        random_timings *= 120.0 / random_timings.sum()

    # Webster
    flow_guess = rng.uniform(400, 900, size=dim)
    webster = webster_timings(flow_guess)
    if webster.sum() > 120:
        webster *= 120.0 / webster.sum()

    # 🔥 CORRECT evaluate_solution usage
    baseline_fit, _, baseline_series, baseline_delay = evaluate_solution(
        G, base_path, signal_nodes, baseline_timings
    )

    random_fit, _, _, random_delay = evaluate_solution(
        G, base_path, signal_nodes, random_timings
    )

    webster_fit, _, _, webster_delay = evaluate_solution(
        G, base_path, signal_nodes, webster
    )

    fit_func = partial(fitness, G, base_path, signal_nodes)

    print("Running GA...")
    ga_best, ga_fit, ga_hist = genetic_algorithm(
        dim=dim,
        fitness_func=fit_func,
        bounds=(5, 60),
        pop_size=60,
        generations=80,
        mutation_rate=0.15,
        elite_size=2,
        seed=SEED,
    )

    if ga_best.sum() > 120:
        ga_best *= 120.0 / ga_best.sum()

    print("Running GOA (multi-start)...")
    best_goa = None
    best_goa_fit = float("inf")
    best_goa_hist = []
    best_goa_delay = float("inf")
    for restart in range(3):
        candidate, cand_fit, cand_hist = grasshopper_optimization(
            dim=dim,
            fitness_func=fit_func,
            bounds=(5, 60),
            population_size=30,
            iterations=80,
            seed=SEED + restart,
        )
        if candidate.sum() > 120:
            candidate *= 120.0 / candidate.sum()
        _, _, _, cand_delay = evaluate_solution(G, base_path, signal_nodes, candidate)
        if cand_delay < best_goa_delay:
            best_goa_delay = cand_delay
            best_goa = candidate
            best_goa_fit = cand_fit
            best_goa_hist = cand_hist

    goa_best, goa_fit, goa_hist = best_goa, best_goa_fit, best_goa_hist

    # 🔥 REAL DELAYS
    _, _, _, ga_delay = evaluate_solution(G, base_path, signal_nodes, ga_best)
    goa_delay = best_goa_delay

    # 🔥 FIXED IMPROVEMENT
    improvement = (baseline_delay - goa_delay) / baseline_delay * 100

    print("\n=== RESULTS ===")
    print(f"Normal delay: {baseline_delay:.2f}")
    print(f"Random delay: {random_delay:.2f}")
    print(f"Webster delay: {webster_delay:.2f}")
    print(f"GA delay: {ga_delay:.2f}")
    print(f"GOA delay: {goa_delay:.2f}")
    print(f"Improvement: {improvement:.2f}%")

    print(f"\nFitness vs Real ratio: {goa_fit / goa_delay:.3f}")

    # Plots
    plots_dir = Path(__file__).resolve().parent.parent / "plots"
    plots_dir.mkdir(exist_ok=True)

    plot_convergence(goa_hist, "GOA", plots_dir / "goa.png")
    plot_convergence(ga_hist, "GA", plots_dir / "ga.png")

    plot_comparison(
        {
            "Normal": baseline_delay,
            "Random": random_delay,
            "Webster": webster_delay,
            "GA": ga_delay,
            "GOA": goa_delay,
        },
        plots_dir / "comparison.png",
    )


if __name__ == "__main__":
    main()
