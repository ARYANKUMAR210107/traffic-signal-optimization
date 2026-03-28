from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def plot_convergence(history: List[float], title: str = "GOA Convergence", outfile: str | Path | None = None):
    plt.figure(figsize=(6, 4))
    plt.plot(history, label="Best fitness", color="teal")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (delay)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    if outfile:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_comparison(values: dict, outfile: str | Path | None = None):
    labels = list(values.keys())
    vals = [values[k] for k in labels]
    plt.figure(figsize=(6, 4))
    palette = ["gray", "orange", "cornflowerblue", "seagreen", "mediumpurple", "gold"]
    colors = [palette[i % len(palette)] for i in range(len(labels))]
    bars = plt.bar(labels, vals, color=colors)
    plt.ylabel("Delay")
    plt.title("Routing delay comparison")
    plt.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.1f}", ha="center", va="bottom")
    if outfile:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_delay_over_time(series: dict[str, List[float]], outfile: str | Path | None = None):
    plt.figure(figsize=(7, 4))
    for label, values in series.items():
        plt.plot(values, label=label)
    plt.xlabel("Cycle")
    plt.ylabel("Delay")
    plt.title("Delay vs time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if outfile:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


__all__ = ["plot_convergence", "plot_comparison", "plot_delay_over_time"]
