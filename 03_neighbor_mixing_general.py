"""General, configurable neighbor-mixing experiment.


Starting-vector sources (--source):
  - random:    independent random Gaussian vectors, no relationship to any
               graph (utils.set_seed-seeded torch.randn). Same as the
               original 03_neighbor_mixing.py.
  - synthetic: points lying EXACTLY on the graph's own natural low-
               dimensional layout (Grid/Torus: (row, col); Ring: (cos, sin)
               unit-circle position), lifted into R^d_embed via a random
               orthonormal basis -- so PCA on these, before any mixing,
               recovers the graph's natural shape exactly. Generalizes
               03_neighbor_mixing_synthetic_grid_permuted.py /
               -synthetic_grid_transformed.py's construction to any graph
               type exposing word_to_row/word_to_col (not just Grid).
  - llm:       static (context-free) token embeddings read directly out of
               an LLM's W_E matrix -- no forward pass, CPU-only (reads the
               local safetensors shard directly via real-embeddings-
               nearest-neighbors.py's lightweight loader, same as
               starting_geometry_vs_accuracy_scatter.py /
               compute_accuracy_with_self.py use elsewhere in this repo).
               Every word must be a single token.

Graph structures (--graph): grid, torus (Grid with periodic boundary
conditions), or ring (utils.Grid / utils.Torus / utils.Ring).

Multi-step mixing (--rounds): a list of round counts, e.g. [1, 2, 3, 10] --
each round chains onto the PREVIOUS round's output (e[i] <- e[i] +
mean(e[neighbors of i])), not back onto the original embeddings, so
"round 10" really is 10 iterated applications. Round 0 (no mixing) is
always included for reference.

PCA dimensionality (--dims): 2, 3, or both -- one full set of plots per
requested dimensionality. Every plot ALWAYS reports each principal
component's fraction of variance explained (FVE) on its axis label,
regardless of --dims.

Usage:
    python 03_neighbor_mixing_general.py
    python 03_neighbor_mixing_general.py --source llm --graph ring --rounds 1 2 3 10 30 --dims 2 3
    python 03_neighbor_mixing_general.py --source synthetic --graph torus --words two_digit_numbers

Or call run_neighbor_mixing_experiment(...) directly from another script.
"""
import argparse
import importlib.util
import math
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from utils import (
    Grid, Torus, Ring, WORDS, set_seed, setup_plotting, save_figure,
    build_word_to_color, set_square_limits,
    pca_2d, draw_class_mean_pca_on_ax, add_pca_grid_faces_3d,
    plotly_pca_layout, plotly_pca_traces, save_plotly,
)

REPO = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = "results/neighbor_mixing_general/plots"
D_EMBED = 4096  # only used for the random/synthetic sources -- llm uses the model's own d_model


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── Starting-vector sources ──────────────────────────────────────────────────

def random_embeddings(words, d_embed=D_EMBED, seed=42):
    """Independent random Gaussian vectors, no relationship to any graph."""
    set_seed(seed)
    return torch.randn(len(words), d_embed)


def synthetic_grid_embeddings(graph, words, d_embed=D_EMBED, seed=42):
    """Points lying exactly on the graph's own natural low-dim coordinate
    layout (word_to_row/word_to_col -- (row, col) for Grid/Torus, (cos, sin)
    for Ring), lifted into R^d_embed via a random orthonormal 2D basis.
    Works for any graph exposing word_to_row/word_to_col."""
    coords = torch.tensor(
        [[graph.word_to_row[w], graph.word_to_col[w]] for w in words], dtype=torch.float32
    )
    coords = coords - coords.mean(dim=0, keepdim=True)
    set_seed(seed)
    raw = torch.randn(d_embed, 2)
    basis, _ = torch.linalg.qr(raw)  # [d_embed, 2], orthonormal columns
    return coords @ basis.T  # [n_words, d_embed]


def llm_embeddings(words):
    """Static token embeddings straight out of an LLM's W_E, no forward
    pass. CPU-only: reads the local safetensors shard directly instead of
    loading the full model."""
    nn_mod = _load_module("real_embeddings_nn", "real_embeddings_nearest_neighbors.py")
    dot_mod = _load_module("act_unembed_dot", "activation_unembedding_dot_product.py")
    model = nn_mod.load_embedding_only_model()
    token_ids = [dot_mod.to_single_token(model.tokenizer, w) for w in words]
    return model.W_E[torch.tensor(token_ids, dtype=torch.long)].float()


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph(graph_type, words):
    if graph_type in ("grid", "torus"):
        side = math.isqrt(len(words))
        if side * side != len(words):
            raise ValueError(
                f"--graph {graph_type} needs a perfect-square word count, got {len(words)}."
            )
        cls = Grid if graph_type == "grid" else Torus
        return cls(words=words, rows=side, cols=side)
    elif graph_type == "ring":
        return Ring(words)
    raise ValueError(f"Unknown graph_type: {graph_type!r}")


# ── Multi-round mixing ────────────────────────────────────────────────────────

def run_mixing_rounds(embeddings, adjacency, rounds):
    """embeddings after 0..max(rounds) rounds of e[i] <- e[i] + mean(e[neighbors
    of i]) on the given graph, keeping only round 0 plus the requested rounds.
    Each round chains onto the PREVIOUS round's output."""
    degree = adjacency.sum(dim=1, keepdim=True)
    keep = sorted(set([0] + list(rounds)))
    embs_by_round = {0: embeddings}
    curr = embeddings
    for r in range(1, max(keep) + 1):
        neighbor_sum = adjacency @ curr
        curr = curr + neighbor_sum / degree
        if r in keep:
            embs_by_round[r] = curr
    return embs_by_round


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_round(graph, embeddings, words, word_to_color, round_num, top_n, plots_dir, tag):
    """One PCA scatter (2D or 3D, per top_n) of `embeddings` at a given
    mixing round, ALWAYS labeled with each PC's fraction of variance
    explained. Saves a static PDF (+ interactive Plotly HTML for the 2D
    case). Returns the explained-variance array."""
    projected_t, var = pca_2d(embeddings, top_n=top_n)
    projected = projected_t.numpy()
    is_3d = (top_n == 3)

    if is_3d:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(projection="3d")
    else:
        fig, ax = plt.subplots(figsize=(5, 5))

    draw_class_mean_pca_on_ax(ax, graph, projected, words=words, word_to_color=word_to_color, is_3d=is_3d)
    # add_pca_grid_faces_3d needs a 2D (rows x cols) layout -- Grid/Torus
    # have one, Ring doesn't, so skip gracefully for graphs without it.
    if is_3d and hasattr(graph, "rows") and hasattr(graph, "grid"):
        add_pca_grid_faces_3d(ax, graph, words, projected)

    # Always report FVE on the axes -- this is the whole point of this
    # script vs. the older 03_neighbor_mixing*.py variants that sometimes
    # omitted it.
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    if is_3d:
        ax.set_zlabel(f"PC3 ({var[2]*100:.1f}%)")
        ax.set_box_aspect((1, 1, 1))
    else:
        set_square_limits(ax, projected[:, 0], projected[:, 1])
        ax.set_aspect("equal")

    title = "0 rounds mixing" if round_num == 0 else f"after {round_num} round{'s' if round_num != 1 else ''} of mixing"
    ax.set_title(title, fontsize=10)

    dim_suffix = "3d" if is_3d else "2d"
    filename = f"pca_round_{round_num}_{tag}_{dim_suffix}"
    save_figure(fig, plots_dir, f"{filename}.pdf")
    print(f"Saved {filename}")

    # Interactive Plotly companion, 2D only -- plotly_pca_traces draws a
    # flat 2D scatter; 3D rounds are static-only here (matching how other
    # 3D-capable scripts in this repo handle it, e.g. via a separate
    # rotatable Plotly export -- not added here to keep this script focused).
    if not is_3d:
        pfig = go.Figure(data=plotly_pca_traces(projected, graph, words=words, word_to_color=word_to_color))
        pfig.update_layout(**plotly_pca_layout(title))
        pfig.update_xaxes(title_text=f"PC1 ({var[0]*100:.1f}%)")
        pfig.update_yaxes(title_text=f"PC2 ({var[1]*100:.1f}%)")
        save_plotly(pfig, plots_dir, f"{filename}.html")

    return var


# ── Core experiment ───────────────────────────────────────────────────────────

def run_neighbor_mixing_experiment(source="random", graph_type="grid", rounds=(1, 2, 3, 10),
                                    dims=(2,), words=None, seed=42, d_embed=D_EMBED):
    """Runs one full neighbor-mixing sweep and produces the plots. Returns
    {(round_num, top_n): explained_variance} for programmatic use.

    source: "random" | "synthetic" | "llm"
    graph_type: "grid" | "torus" | "ring"
    rounds: list of mixing-round counts to plot (round 0 is always included too)
    dims: which PCA dimensionalities to plot -- (2,), (3,), or (2, 3)
    words: list of words, defaults to utils.WORDS
    """
    if words is None:
        words = WORDS

    graph = build_graph(graph_type, words)
    word_to_color = build_word_to_color(words)
    adjacency = torch.tensor(graph.build_adjacency_matrix(), dtype=torch.float32)

    if source == "random":
        embeddings = random_embeddings(words, d_embed=d_embed, seed=seed)
    elif source == "synthetic":
        embeddings = synthetic_grid_embeddings(graph, words, d_embed=d_embed, seed=seed)
    elif source == "llm":
        embeddings = llm_embeddings(words)
    else:
        raise ValueError(f"Unknown source: {source!r}")

    embs_by_round = run_mixing_rounds(embeddings, adjacency, rounds)
    tag = f"{source}_{graph_type}"
    plots_dir = os.path.join(PLOTS_DIR, tag)

    results = {}
    for round_num in sorted(embs_by_round):
        for top_n in dims:
            var = plot_round(graph, embs_by_round[round_num], words, word_to_color,
                              round_num, top_n, plots_dir, tag)
            results[(round_num, top_n)] = var
            print(f"  round {round_num:3d}, {top_n}D PCA: FVE = "
                  + ", ".join(f"{v*100:.1f}%" for v in var))

    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["random", "synthetic", "llm"], default="random",
                        help="Where the round-0 (starting) embeddings come from.")
    parser.add_argument("--graph", choices=["grid", "torus", "ring"], default="grid",
                        help="Graph structure to mix neighbors over.")
    parser.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3, 10],
                        help="Mixing-round counts to plot (round 0 is always included too).")
    parser.add_argument("--dims", type=int, nargs="+", choices=[2, 3], default=[2],
                        help="PCA dimensionalities to plot: 2, 3, or both.")
    parser.add_argument("--words", type=str, default=None,
                        help="word_lists.WORD_LISTS key (defaults to utils.WORDS, "
                             "a 16-word 4x4-compatible list).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_plotting()
    plt.rcParams['text.usetex'] = False

    if args.words is not None:
        from word_lists import WORD_LISTS
        words = WORD_LISTS[args.words]
    else:
        words = WORDS

    run_neighbor_mixing_experiment(
        source=args.source, graph_type=args.graph, rounds=args.rounds,
        dims=args.dims, words=words, seed=args.seed,
    )


if __name__ == "__main__":
    main()
