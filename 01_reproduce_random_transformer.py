import os
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plotly.graph_objects as go
import tqdm

from utils import (
    WORDS, LAYER, SEQ_LEN, WORD_TO_COLOR,
    Grid, set_seed, load_toy_model, get_activations_toy_batch, 
    compute_class_means_batch, compute_pca_directions, setup_plotting, save_figure,
    plotly_pca_layout, plotly_pca_traces, save_plotly, plot_attention_heatmaps
)

LAYER = 0
SEQ_LEN = 64
PLOTS_DIR = "results/cosine-similarity"
N_LOOKBACK = 50
BATCH_SIZE = 64

print(f"{SEQ_LEN=}, {N_LOOKBACK=}")


# ── Fig 2 Right: Class-mean PCA ───────────────────────────────────────────────

def plot_class_mean_pca(grid, class_means, pca_dirs, hp_str="", hp_title=""):
    """Scatter of class-mean centroids with grid edges."""
    projected = (class_means - class_means.mean(axis=0, keepdims=True)) @ pca_dirs.T

    fig, ax = plt.subplots(figsize=(5, 5))

    # Grid edges (gray dashed)
    A = grid.build_adjacency_matrix()
    for i in range(len(WORDS)):
        for j in range(i + 1, len(WORDS)):
            if A[i, j]:
                ax.plot(
                    [projected[i, 0].item(), projected[j, 0].item()],
                    [projected[i, 1].item(), projected[j, 1].item()],
                    color="dimgray", alpha=0.7, linestyle="--", linewidth=0.8,
                )

    # Scatter + labels
    for i, word in enumerate(WORDS):
        ax.scatter(
            projected[i, 0].item(), projected[i, 1].item(),
            color=WORD_TO_COLOR[word], s=120, marker="*",
            edgecolors="black", linewidths=0.5, zorder=5,
        )
        ax.annotate(
            word, (projected[i, 0].item(), projected[i, 1].item()),
            xytext=(5, 5), textcoords="offset points", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    # Display the hyperparams on a new line in the title
    ax.set_title(f"PCA of per-node mean activations\n{hp_title}", fontsize=10)
    ax.set_aspect("equal")
    save_figure(fig, PLOTS_DIR, f"pca_class_means_{hp_str}.pdf")
    print(f"Saved pca_class_means_{hp_str}.pdf")

    # ── Plotly interactive ───────────────────────────────────────────────────
    pfig = go.Figure(data=plotly_pca_traces(projected, grid))
    # Use HTML line break <br> for Plotly titles
    pfig.update_layout(**plotly_pca_layout(f"PCA of per-node mean activations<br><span style='font-size:12px'>{hp_title}</span>"))
    save_plotly(pfig, PLOTS_DIR, f"pca_class_means_{hp_str}.html")


# ── Fig 6: Bigram PCA ─────────────────────────────────────────────────────────

def _draw_bigram_scatter(ax, projected_all, projected_means, tail, grid, label=True):
    """Draw bigram scatter on a given axes. Shared by main plot and inset."""
    A = grid.build_adjacency_matrix()
    for i in range(len(WORDS)):
        for j in range(i + 1, len(WORDS)):
            if A[i, j]:
                ax.plot(
                    [projected_means[i, 0].item(), projected_means[j, 0].item()],
                    [projected_means[i, 1].item(), projected_means[j, 1].item()],
                    color="dimgray", alpha=0.7, linestyle="--", linewidth=0.8,
                )

    for idx in range(1, len(tail)):
        cur_word = tail[idx]
        prev_word = tail[idx - 1]
        ax.scatter(
            projected_all[idx, 0].item(), projected_all[idx, 1].item(),
            c=WORD_TO_COLOR[cur_word],
            edgecolors=WORD_TO_COLOR[prev_word],
            linewidths=1.0, s=25, alpha=1.0, zorder=3,
        )

    for i, word in enumerate(WORDS):
        ax.scatter(
            projected_means[i, 0].item(), projected_means[i, 1].item(),
            color=WORD_TO_COLOR[word], s=120, marker="*",
            edgecolors="black", linewidths=1.0, zorder=5,
        )
        if label:
            ax.annotate(
                word, (projected_means[i, 0].item(), projected_means[i, 1].item()),
                xytext=(5, 5), textcoords="offset points", fontsize=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )


def _make_bigram_legend(ax):
    """Add legend for bigram PCA plots."""
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=8, markeredgecolor="black", markeredgewidth=1.5,
               label="Fill = current token"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markersize=8, markeredgecolor="gray", markeredgewidth=1.5,
               label="Border = previous token"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
               markersize=12, markeredgecolor="black", markeredgewidth=0.8,
               label="Token centroid"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=True,
              framealpha=1.0, edgecolor="gray", fontsize=8)
    

def plot_bigram_pca(grid, sequences, activations, class_means, pca_dirs, hp_str="", hp_title=""):
    """Individual activations colored by (current token, previous token) for a batch of sequences."""
    class_means_mean = class_means.mean(axis=0, keepdims=True)
    projected_all = (activations - class_means_mean) @ pca_dirs.T
    projected_means = (class_means - class_means_mean) @ pca_dirs.T

    # ── Main bigram plot (Matplotlib) ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for b in range(len(sequences)):
        tail = sequences[b][-N_LOOKBACK:]
        _draw_bigram_scatter(ax, projected_all[b], projected_means, tail, grid)
        
    _make_bigram_legend(ax)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    # Display the hyperparams on a new line in the title
    ax.set_title(f"PCA of individual activations, labeled by bigram\n{hp_title}", fontsize=10)
    ax.set_aspect("equal")
    save_figure(fig, PLOTS_DIR, f"bigram_pca_{hp_str}.pdf")
    print(f"Saved bigram_pca_{hp_str}.pdf")

    #── Plotly interactive ───────────────────────────────────────────────────
    pfig = go.Figure(data=plotly_pca_traces(projected_means, grid))

    for word in WORDS:
        x_coords = []
        y_coords = []
        prev_words = []
        
        for b in range(len(sequences)):
            tail = sequences[b][-N_LOOKBACK:]
            idxs = [idx for idx in range(1, len(tail)) if tail[idx] == word]
            for idx in idxs:
                x_coords.append(projected_all[b, idx, 0].item())
                y_coords.append(projected_all[b, idx, 1].item())
                prev_words.append(tail[idx - 1])
                
        if not x_coords:
            continue
            
        pfig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="markers",
            marker=dict(
                size=8, color=WORD_TO_COLOR[word],
                line=dict(width=2, color=[WORD_TO_COLOR[pw] for pw in prev_words]),
            ),
            customdata=[[pw] for pw in prev_words],
            hovertemplate=(
                "current: <b>" + word + "</b><br>"
                "previous: <b>%{customdata[0]}</b>"
                "<extra></extra>"
            ),
            hoverlabel=dict(bgcolor=WORD_TO_COLOR[word], font=dict(color="black")),
            name=word, showlegend=False,
        ))

    pfig.update_layout(**plotly_pca_layout(
        f"PCA of individual activations, labeled by bigram<br><span style='font-size:12px'>{hp_title}</span>"))
    pfig.update_layout(width=1000, height=1000)

    pfig.add_annotation(
        text=(
            "● Fill = current token<br>"
            "● Border = previous token<br>"
            "★ Token centroid"
        ),
        xref="paper", yref="paper", x=0.02, y=0.98,
        showarrow=False, font=dict(size=12),
        align="left", bgcolor="rgba(255,255,255,0.9)",
        bordercolor="gray", borderwidth=1, borderpad=6,
    )

    save_plotly(pfig, PLOTS_DIR, f"bigram_pca_{hp_str}.html")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False
    grid = Grid()

    # model
    model = load_toy_model() 
    print(model.cfg)
    print(model)

    # generate data
    seed = 42
    set_seed(seed)
    sequences = grid.generate_batch(SEQ_LEN, BATCH_SIZE)
    print(f"{len(sequences)=}")

    activations_t = get_activations_toy_batch(model, sequences, LAYER, N_LOOKBACK)
    class_means_t = compute_class_means_batch(activations_t, sequences, WORDS, N_LOOKBACK)
    pca_dirs_t, _ = compute_pca_directions(class_means_t, top_n=2)

    # Conversion
    activations = activations_t.cpu().numpy()
    class_means = class_means_t.cpu().numpy()
    pca_dirs = pca_dirs_t.cpu().numpy()

    # ── Clean Hyperparameter Strings ──────────────────────────────────────────
    # hp_str is used for safe filenames without spaces
    hp_str = f"Seq{SEQ_LEN}_Look{N_LOOKBACK}_Batch{BATCH_SIZE}_Seed{seed}"
    
    # hp_title is used for the readable plot titles
    hp_title = f"Seq: {SEQ_LEN} | Lookback: {N_LOOKBACK} | Batch: {BATCH_SIZE} | Seed: {seed}"

    # ── Plotting ──────────────────────────────────────────────────────────────
    plot_class_mean_pca(grid, class_means, pca_dirs, hp_str=hp_str, hp_title=hp_title)
    # plot_bigram_pca(grid, sequences, activations, class_means, pca_dirs, hp_str=hp_str, hp_title=hp_title)

if __name__ == "__main__":
    main()