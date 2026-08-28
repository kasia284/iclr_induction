"""Dirichlet energy and distance correlation of the final-layer,
contextualized per-word representations, as a function of sequence length --
to plot alongside accuracy_curve_sweep.png's accuracy(sequence length)
curves on the same log-x axis, across the same 6 morphology grid-position
permutations (MORPHOLOGY_SWEEP_KEYS).

At each checkpoint sequence length k, per-word class means are built from
each sequence's first k tokens using the same trailing-window convention
01_reproduce.py's main() uses for the complete SEQ_LEN=1400 sequence: the
last min(N_LOOKBACK, k) positions. So DE/DC at the final checkpoint
(k=SEQ_LEN) exactly reproduce morphology_final_dirichlet_energy_sweep.py /
morphology_distance_correlation_sweep.py's values -- same window, same
full sequence, just re-derived here at every checkpoint instead of only
the last one.

Unlike those two scripts, this needs a GPU: it requires the FINAL layer's
hook_resid_pre at every position of the sequence (not just the cached
tail-50 window), via one full forward pass per permutation. Results are
cached per permutation to
results/reproduce/data/{key}/de_dc_vs_seqlen_Grid_{key}.npz, so re-running
the plot alone doesn't require the model or a GPU.

Reuses build_word_token_ids / tokenize_sequences_by_id_batch / get_grid_
coords / load_cached_accuracies / MORPHOLOGY_SWEEP_KEYS / N_LOOKBACK /
SEQ_LEN from 01_reproduce.py (imported via importlib since its filename
isn't a valid module identifier) -- same random walks (same set_seed(42)
call) and same window convention as the rest of the sweep.
"""
import importlib.util
import os

import matplotlib.pyplot as plt
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import torch
import tqdm

from utils import (
    Grid, set_seed, load_model, compute_class_means_batch,
    compute_dirichlet_energy, compute_distance_correlation,
    setup_plotting, save_figure, save_plotly,
)
from word_lists import WORD_LISTS

REPO = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reproduce_mod = _load_module("reproduce_01", "01_reproduce.py")

DATA_DIR = "results/reproduce/data"
PLOTS_DIR = "results/reproduce/plots/morphology_sweep_comparison"

MORPHOLOGY_SWEEP_KEYS = reproduce_mod.MORPHOLOGY_SWEEP_KEYS
N_LOOKBACK = reproduce_mod.N_LOOKBACK
SEQ_LEN = reproduce_mod.SEQ_LEN

COLORS = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

# Log-spaced sequence-length checkpoints. Cheap to add more: the only GPU
# cost is the one full-sequence forward pass per permutation: everything
# per-checkpoint below is CPU postprocessing on the resulting cache.
N_CHECKPOINTS = 30
CHECKPOINTS = sorted(set(
    np.geomspace(N_LOOKBACK, SEQ_LEN, N_CHECKPOINTS).astype(int).tolist() + [SEQ_LEN]
))


def cache_path(word_list_key):
    return os.path.join(DATA_DIR, word_list_key, f"de_dc_vs_seqlen_Grid_{word_list_key}.npz")


def compute_de_dc_vs_seqlen(model, word_list_key):
    words = WORD_LISTS[word_list_key]
    grid = Grid(words=words, rows=4, cols=4)
    adjacency = grid.build_adjacency_matrix()
    grid_coords = reproduce_mod.get_grid_coords(grid, words)

    word_to_id = reproduce_mod.build_word_token_ids(model, words)

    # Same seed + generation call as 01_reproduce.py's main(), so this is
    # the exact same batch of random walks already used for the cached
    # accuracy curves.
    set_seed(42)
    sequences = grid.generate_batch(SEQ_LEN, len(words))

    n_layers = model.cfg.n_layers
    tokens = reproduce_mod.tokenize_sequences_by_id_batch(model, sequences, word_to_id)
    names_filter = [f"blocks.{n_layers - 1}.hook_resid_pre"]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens.to(model.cfg.device), names_filter=names_filter)
    # [batch, SEQ_LEN, d_model], BOS position removed -- full sequence, not
    # just the tail (unlike everywhere else in the sweep).
    activations = cache[f"blocks.{n_layers - 1}.hook_resid_pre"][:, 1:, :].cpu()
    del cache
    torch.cuda.empty_cache()

    de_values, dc_values = [], []
    for k in tqdm.tqdm(CHECKPOINTS, desc=f"{word_list_key}: DE/DC vs seq len"):
        window = min(N_LOOKBACK, k)
        sequences_window = [seq[k - window:k] for seq in sequences]
        activations_window = activations[:, k - window:k, :]
        class_means = compute_class_means_batch(
            activations_window, sequences_window, words, window
        ).numpy()
        de_values.append(compute_dirichlet_energy(class_means, adjacency))
        dc_values.append(compute_distance_correlation(class_means, grid_coords))

    checkpoints = np.array(CHECKPOINTS)
    de_values = np.array(de_values)
    dc_values = np.array(dc_values)

    path = cache_path(word_list_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, checkpoints=checkpoints, de_values=de_values, dc_values=dc_values)
    print(f"Cached {path}")
    return checkpoints, de_values, dc_values


def load_or_compute(model, word_list_key):
    path = cache_path(word_list_key)
    if os.path.exists(path):
        data = np.load(path)
        return data["checkpoints"], data["de_values"], data["dc_values"]
    if model is None:
        raise RuntimeError(f"No cache at {path} and no model loaded to compute it.")
    return compute_de_dc_vs_seqlen(model, word_list_key)


def plot_combined_sweep(accs_by_key, checkpoints_by_key, de_by_key, dc_by_key, out_dir,
                         filename="accuracy_de_dc_vs_seqlen_sweep"):
    """3-panel figure (accuracy, Dirichlet energy, distance correlation), all
    vs. sequence length on a shared log-x axis, one color per permutation."""
    fig, (ax_acc, ax_de, ax_dc) = plt.subplots(3, 1, figsize=(6, 9), sharex=True)

    for (key, all_accs), color in zip(accs_by_key.items(), COLORS):
        mean = all_accs.mean(axis=0)
        x_vals = np.arange(1, len(mean) + 1)
        ax_acc.plot(x_vals, mean, color=color, linewidth=1.2, label=key)

        checkpoints = checkpoints_by_key[key]
        ax_de.plot(checkpoints, de_by_key[key], color=color, linewidth=1.2, marker="o", markersize=3)
        ax_dc.plot(checkpoints, dc_by_key[key], color=color, linewidth=1.2, marker="o", markersize=3)

    ax_acc.set_xscale("log")
    ax_acc.set_ylim(0, 1)
    ax_acc.set_ylabel("Accuracy\n(grid task)")
    ax_acc.set_title("Accuracy, Dirichlet energy & distance correlation vs. sequence length\n(morphology permutations, final-layer activations)", fontsize=9)
    ax_acc.legend(loc="upper left", frameon=True, framealpha=1.0, edgecolor="gray", fontsize=6)

    ax_de.set_ylabel("Dirichlet energy")
    ax_dc.set_ylabel("Distance correlation")
    ax_dc.set_xlabel("Sequence length")

    for ax in (ax_acc, ax_de, ax_dc):
        ax.grid(alpha=0.2)

    save_figure(fig, out_dir, f"{filename}.pdf")
    print(f"Saved {filename}")

    # ── Plotly interactive (3 stacked, x-linked subplots) ────────────────────
    pfig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                          subplot_titles=("Accuracy", "Dirichlet energy", "Distance correlation"))
    for (key, all_accs), color in zip(accs_by_key.items(), COLORS):
        mean = all_accs.mean(axis=0)
        x_vals = list(range(1, len(mean) + 1))
        pfig.add_trace(go.Scatter(x=x_vals, y=mean.tolist(), mode="lines",
                                   line=dict(color=color, width=2), name=key,
                                   legendgroup=key), row=1, col=1)

        checkpoints = checkpoints_by_key[key].tolist()
        pfig.add_trace(go.Scatter(x=checkpoints, y=de_by_key[key].tolist(), mode="lines+markers",
                                   line=dict(color=color, width=2), marker=dict(size=4),
                                   name=key, legendgroup=key, showlegend=False), row=2, col=1)
        pfig.add_trace(go.Scatter(x=checkpoints, y=dc_by_key[key].tolist(), mode="lines+markers",
                                   line=dict(color=color, width=2), marker=dict(size=4),
                                   name=key, legendgroup=key, showlegend=False), row=3, col=1)

    pfig.update_xaxes(type="log", title_text="Sequence length", row=3, col=1)
    pfig.update_yaxes(title_text="Accuracy", range=[0, 1], row=1, col=1)
    pfig.update_yaxes(title_text="Dirichlet energy", row=2, col=1)
    pfig.update_yaxes(title_text="Distance correlation", row=3, col=1)
    pfig.update_layout(
        title="Accuracy, Dirichlet energy & distance correlation vs. sequence length (morphology permutations)",
        width=800, height=900,
        legend=dict(x=1.02, y=1.0),
    )
    save_plotly(pfig, out_dir, f"{filename}.html")


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False

    model = None
    if not all(os.path.exists(cache_path(k)) for k in MORPHOLOGY_SWEEP_KEYS):
        model = load_model()

    checkpoints_by_key, de_by_key, dc_by_key = {}, {}, {}
    for key in MORPHOLOGY_SWEEP_KEYS:
        checkpoints, de_values, dc_values = load_or_compute(model, key)
        checkpoints_by_key[key] = checkpoints
        de_by_key[key] = de_values
        dc_by_key[key] = dc_values

    accs_by_key = {k: reproduce_mod.load_cached_accuracies(k) for k in MORPHOLOGY_SWEEP_KEYS}
    plot_combined_sweep(accs_by_key, checkpoints_by_key, de_by_key, dc_by_key, PLOTS_DIR)


if __name__ == "__main__":
    main()
