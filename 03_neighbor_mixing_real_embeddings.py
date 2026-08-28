"""Reproduce Figures 2 and 6: accuracy curve, class-mean PCA, and bigram PCA."""

import os
import json

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tqdm
import torch

from utils import (
    WORDS, LAYER, SEQ_LEN, WORD_TO_COLOR,
    Grid, set_seed, load_model, get_model_accuracies, get_activations,
    compute_class_means, compute_pca_directions, setup_plotting, save_figure,
    smooth, plotly_pca_layout, plotly_line_layout, plotly_pca_traces, save_plotly,
    draw_class_mean_pca_on_ax, draw_bigram_scatter_on_ax, make_bigram_legend,
)

DATA_DIR = "results/reproduce/data"
PLOTS_DIR = "results/reproduce/plots"
N_LOOKBACK = 200


# ── Fig 2 Left: Accuracy curve ────────────────────────────────────────────────

def plot_accuracy_curve(all_accs):
    """Average accuracy across 16 sequences with uniform starting positions."""
    mean = smooth(all_accs.mean(axis=0))
    std = smooth(all_accs.std(axis=0))

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(mean, color="black", linewidth=1.0, label="Mean accuracy")
    ax.fill_between(range(len(mean)), mean - std, mean + std,
                    alpha=0.15, color="gray", edgecolor="none", label="$\\pm$1 std")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Accuracy (grid task)")
    ax.set_title("Accuracy vs sequence length", fontsize=10)
    ax.legend(loc="upper left", frameon=True, framealpha=1.0, edgecolor="gray")

    save_figure(fig, PLOTS_DIR, "accuracy_curve.pdf")
    print("Saved accuracy_curve")

    # ── Plotly interactive ───────────────────────────────────────────────────
    pfig = go.Figure()
    x = list(range(len(mean)))
    pfig.add_trace(go.Scatter(
        x=x, y=(mean + std).tolist(), mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    pfig.add_trace(go.Scatter(
        x=x, y=(mean - std).tolist(), mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(128,128,128,0.2)",
        showlegend=False, hoverinfo="skip",
    ))
    pfig.add_trace(go.Scatter(
        x=x, y=mean.tolist(), mode="lines",
        line=dict(color="black", width=2), name="Mean accuracy",
        hovertemplate="pos: %{x}<br>accuracy: %{y:.3f}<extra></extra>",
    ))
    pfig.update_layout(**plotly_line_layout(
        "Accuracy vs sequence length", "Sequence length", "Accuracy (grid task)"))
    save_plotly(pfig, PLOTS_DIR, "accuracy_curve.html")


# ── Fig 2 Right & Neighbor Mixing: Class-mean PCA ─────────────────────────────

def plot_class_mean_pca(grid, class_means, pca_dirs, title="PCA of per-node mean activations", filename_stem="pca_class_means"):
    """Scatter of 16 class-mean centroids with grid edges (Supports 2D and 3D)."""
    projected = (class_means - class_means.mean(axis=0, keepdims=True)) @ pca_dirs.T  # [16, num_components]
    num_dims = projected.shape[1]
    is_3d = (num_dims == 3)

    # Setup the figure canvas based on dimensionality
    if is_3d:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(projection='3d')
    else:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Grid edges (gray dashed) + star centroid markers + word labels.
    draw_class_mean_pca_on_ax(ax, grid, projected, words=WORDS, word_to_color=WORD_TO_COLOR,
                               marker_size=120, label_fontsize=8, label_offset=(5, 5), is_3d=is_3d)

    # Axis Labels & Aspect Ratio
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    
    if is_3d:
        ax.set_zlabel("PC3")
        ax.set_box_aspect((1, 1, 1))
    else:
        ax.set_aspect("equal")
        
    ax.set_title(f"{'3D ' if is_3d else ''}{title}", fontsize=10)
    
    dim_suffix = "3d" if is_3d else "2d"
    out_name = f"{filename_stem}_{dim_suffix}"
    
    save_figure(fig, PLOTS_DIR, f"{out_name}.pdf")
    print(f"Saved {out_name}")

    # ── Plotly interactive ───────────────────────────────────────────────────
    pfig = go.Figure(data=plotly_pca_traces(projected, grid))
    pfig.update_layout(**plotly_pca_layout(f"{'3D ' if is_3d else ''}{title}"))
    save_plotly(pfig, PLOTS_DIR, f"{out_name}.html")

    
# ── Fig 6: Bigram PCA ─────────────────────────────────────────────────────────

def plot_bigram_pca(grid, sequence, activations, class_means, pca_dirs):
    """Individual activations colored by (current token, previous token)."""
    tail = sequence[-N_LOOKBACK:]
    class_means_mean = class_means.mean(axis=0, keepdims=True)
    projected_all = (activations - class_means_mean) @ pca_dirs.T    # [N_LOOKBACK, 2]
    projected_means = (class_means - class_means_mean) @ pca_dirs.T  # [16, 2]

    # ── Main bigram plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_bigram_scatter_on_ax(ax, projected_all, projected_means, tail, grid,
                               words=WORDS, word_to_color=WORD_TO_COLOR,
                               marker_size=120, label_fontsize=7,
                               individual_size=25, individual_linewidth=1.0, individual_alpha=1.0)
    make_bigram_legend(ax, loc="upper left", fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA of individual activations, labeled by bigram", fontsize=10)
    ax.set_aspect("equal")
    save_figure(fig, PLOTS_DIR, "bigram_pca.pdf")
    print("Saved bigram_pca")

    # ── Plotly interactive ───────────────────────────────────────────────────
    pfig = go.Figure(data=plotly_pca_traces(projected_means, grid))

    for word in WORDS:
        idxs = [idx for idx in range(1, len(tail)) if tail[idx] == word]
        if not idxs:
            continue
        pfig.add_trace(go.Scatter(
            x=[projected_all[idx, 0].item() for idx in idxs],
            y=[projected_all[idx, 1].item() for idx in idxs],
            mode="markers",
            marker=dict(
                size=8, color=WORD_TO_COLOR[word],
                line=dict(
                    width=2,
                    color=[WORD_TO_COLOR[tail[idx - 1]] for idx in idxs],
                ),
            ),
            customdata=[[tail[idx - 1]] for idx in idxs],
            hovertemplate=(
                "current: <b>" + word + "</b><br>"
                "previous: <b>%{customdata[0]}</b>"
                "<extra></extra>"
            ),
            hoverlabel=dict(bgcolor=WORD_TO_COLOR[word], font=dict(color="black")),
            name=word, showlegend=False,
        ))

    pfig.update_layout(**plotly_pca_layout(
        "PCA of individual activations, labeled by bigram"))
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

    save_plotly(pfig, PLOTS_DIR, "bigram_pca.html")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False  # Bypassing missing LaTeX binaries
    grid = Grid()

    acc_path = os.path.join(DATA_DIR, "accuracies.npz")
    pca_path = os.path.join(DATA_DIR, "pca.npz")
    seq_path = os.path.join(DATA_DIR, "sequence.json")

    # We defer loading the model unless we strictly need it. 
    # If caches are missing OR we need to extract raw embeddings, we will load it.
    model = None

    if os.path.exists(acc_path) and os.path.exists(pca_path) and os.path.exists(seq_path):
        print("Loading cached data (delete data/ to recompute)...")
        all_accs = np.load(acc_path)["all_accs"]
        pca_data = np.load(pca_path)
        activations = pca_data["activations"]
        class_means = pca_data["class_means"]
        pca_dirs = pca_data["pca_dirs"]
        with open(seq_path) as f:
            sequence = json.load(f)
    else:
        model = load_model()
        os.makedirs(DATA_DIR, exist_ok=True)

        # Accuracy data
        set_seed(42)
        sequences = grid.generate_batch(SEQ_LEN)
        all_accs = []
        for seq in tqdm.tqdm(sequences, desc="Accuracy curves"):
            all_accs.append(get_model_accuracies(model, grid, seq))
        all_accs = np.array(all_accs)
        np.savez(acc_path, all_accs=all_accs)
        print(f"Cached {acc_path}")

        # PCA data
        set_seed(42)
        sequence = grid.generate_sequence(SEQ_LEN)
        activations_t = get_activations(model, sequence, LAYER, N_LOOKBACK)
        class_means_t = compute_class_means(activations_t, sequence, WORDS, N_LOOKBACK)
        pca_dirs_t, _ = compute_pca_directions(class_means_t, top_n=3)

        activations = activations_t.cpu().numpy()
        class_means = class_means_t.cpu().numpy()
        pca_dirs = pca_dirs_t.cpu().numpy()

        np.savez(pca_path, activations=activations, class_means=class_means, pca_dirs=pca_dirs)
        with open(seq_path, "w") as f:
            json.dump(sequence, f)
        print(f"Cached {pca_path}")

    # ── Original Plotting ─────────────────────────────────────────────────────
    plot_accuracy_curve(all_accs)
    plot_class_mean_pca(grid, class_means, pca_dirs, 
                        title="PCA of per-node mean activations", 
                        filename_stem="pca_class_means")
    plot_bigram_pca(grid, sequence, activations, class_means, pca_dirs)

    # ── Embedding Neighbor Mixing Simulation ──────────────────────────────────
    print("\n--- Extracting Initial Embeddings and Simulating Neighbor Mixing ---")
    if model is None:
        print("Loading model to extract base embeddings...")
        model = load_model()
        
    try:
        # Get activations immediately following the embedding layer (layer 0)
        embed_acts_t = get_activations(model, sequence, layer=0, n_lookback=N_LOOKBACK)
        
        # Calculate centroids of the 16 tokens to isolate the core word embedding representations
        embs_round_0 = compute_class_means(embed_acts_t, sequence, WORDS, N_LOOKBACK).cpu().numpy()
        
        A_torch = torch.tensor(grid.build_adjacency_matrix(), dtype=torch.float32)
        degree = A_torch.sum(dim=1, keepdim=True)
        
        # Mixing operations
        embs_round_0_t = torch.tensor(embs_round_0)
        embs_round_1_t = embs_round_0_t + (A_torch @ embs_round_0_t) / degree
        embs_round_2_t = embs_round_1_t + (A_torch @ embs_round_1_t) / degree
        embs_round_3_t = embs_round_2_t + (A_torch @ embs_round_2_t) / degree
        embs_round_4_t = embs_round_3_t + (A_torch @ embs_round_3_t) / degree
        
        embs_round_1 = embs_round_1_t.numpy()
        embs_round_2 = embs_round_2_t.numpy()
        embs_round_3 = embs_round_3_t.numpy()
        embs_round_4 = embs_round_4_t.numpy()

        # Plot 0 rounds
        pca_dirs_0, _ = compute_pca_directions(torch.tensor(embs_round_0), top_n=2)
        pca_dirs_0 = pca_dirs_0.numpy()
        plot_class_mean_pca(grid, embs_round_0, pca_dirs_0,
                            title="PCA of Learned Embeddings\n(0 rounds mixing)",
                            filename_stem="learned_embs_mixing_0")

        # Plot 1 round
        pca_dirs_1, _ = compute_pca_directions(embs_round_1_t, top_n=2)
        pca_dirs_1 = pca_dirs_1.numpy()
        plot_class_mean_pca(grid, embs_round_1, pca_dirs_1,
                            title="PCA of Learned Embeddings\n(after 1 round of mixing)",
                            filename_stem="learned_embs_mixing_1")

        # Plot 2 rounds
        pca_dirs_2, _ = compute_pca_directions(embs_round_2_t, top_n=2)
        pca_dirs_2 = pca_dirs_2.numpy()
        plot_class_mean_pca(grid, embs_round_2, pca_dirs_2,
                            title="PCA of Learned Embeddings\n(after 2 rounds of mixing)",
                            filename_stem="learned_embs_mixing_2")

        # Plot 3 rounds
        pca_dirs_3, _ = compute_pca_directions(embs_round_3_t, top_n=2)
        pca_dirs_3 = pca_dirs_3.numpy()
        plot_class_mean_pca(grid, embs_round_3, pca_dirs_3,
                            title="PCA of Learned Embeddings\n(after 3 rounds of mixing)",
                            filename_stem="learned_embs_mixing_3")

        # Plot 4 rounds
        pca_dirs_4, _ = compute_pca_directions(embs_round_4_t, top_n=2)
        pca_dirs_4 = pca_dirs_4.numpy()
        plot_class_mean_pca(grid, embs_round_4, pca_dirs_4,
                            title="PCA of Learned Embeddings\n(after 4 rounds of mixing)",
                            filename_stem="learned_embs_mixing_4")

    except Exception as e:
        print(f"Skipping learned embedding mixing due to error: {e}")


if __name__ == "__main__":
    main()