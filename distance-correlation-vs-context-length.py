"""Plots distance correlation (activation-space vs. grid) as a function
of CONTEXT LENGTH, one line per layer, for each word list -- extracting
exactly the DC values already annotated on each subplot of
morphology-grid-evolution-layers-vs-seqlen.py's PCA grid, but as a proper
line plot instead of scattered subplot titles.

Reuses that script's cached class means (results/reproduce/data/{key}/
grid_evolution_layers_vs_seqlen.npz) -- pure cache reads, no GPU / model
needed, since the class means were already computed there.
"""
import importlib.util
import os

import matplotlib.pyplot as plt
import numpy as np

from utils import Grid, setup_plotting, save_figure, compute_distance_correlation
from word_lists import WORD_LISTS

REPO = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reproduce_mod = _load_module("reproduce_01", "01_reproduce.py")
grid_evolution_mod = _load_module("grid_evolution", "morphology-grid-evolution-layers-vs-seqlen.py")

SWEEP_KEYS = grid_evolution_mod.SWEEP_KEYS
LAYERS = grid_evolution_mod.LAYERS
CHECKPOINTS = grid_evolution_mod.CHECKPOINTS


def compute_dc_curves(word_list_key):
    words = WORD_LISTS[word_list_key]
    side = round(len(words) ** 0.5)
    grid = Grid(words=words, rows=side, cols=side)
    grid_coords = reproduce_mod.get_grid_coords(grid, words)

    class_means = grid_evolution_mod.load_or_compute_class_means(None, word_list_key)
    dc_by_layer = {}
    for layer in LAYERS:
        dc_by_layer[layer] = [
            compute_distance_correlation(class_means[(layer, k)], grid_coords)
            for k in CHECKPOINTS
        ]
    return dc_by_layer


def plot_dc_vs_context_length(word_list_key, dc_by_layer, out_dir,
                               filename="distance_correlation_vs_context_length"):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for layer in LAYERS:
        ax.plot(CHECKPOINTS, dc_by_layer[layer], marker="o", markersize=4,
                 linewidth=1.3, label=f"Layer {layer}")
    ax.set_xscale("log")
    ax.set_xlabel("Context length (sequence position)")
    ax.set_ylabel("Distance correlation (activations vs. grid)")
    ax.set_title(f"Distance correlation vs. context length ({word_list_key})", fontsize=10)
    ax.legend(loc="best", frameon=True, framealpha=1.0, edgecolor="gray", fontsize=8)
    plt.tight_layout()

    save_figure(fig, out_dir, f"{filename}.pdf")
    print(f"Saved {word_list_key}/{filename}")


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False

    for key in SWEEP_KEYS:
        path = grid_evolution_mod.cache_path(key)
        if not os.path.exists(path):
            print(f"Skipping {key}: no cache at {path}")
            continue
        print(f"\n=== {key} ===")
        dc_by_layer = compute_dc_curves(key)
        for layer in LAYERS:
            values = ", ".join(f"{k}:{dc:.3f}" for k, dc in zip(CHECKPOINTS, dc_by_layer[layer]))
            print(f"  Layer {layer}: {values}")
        plot_dc_vs_context_length(key, dc_by_layer, f"results/reproduce/plots/{key}")


if __name__ == "__main__":
    main()
