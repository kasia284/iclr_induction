import os
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plotly.graph_objects as go
import tqdm
import torch

from utils import (
    WORDS, LAYER, SEQ_LEN, WORD_TO_COLOR,
    Grid, set_seed, load_model, get_model_accuracies, get_activations,
    compute_class_means, compute_pca_directions, setup_plotting, save_figure,
    smooth, plotly_pca_layout, plotly_line_layout, plotly_pca_traces, save_plotly,
)

from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "results/reproduce/data"
PLOTS_DIR = "results/reproduce/plots"
N_LOOKBACK = 200


def plot_cosine_similarity(cos_sim_matrix):
    """
    Plots the pairwise cosine similarity matrix using both Matplotlib and Plotly.
    """
    # ── Matplotlib Heatmap ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    
    # Display the matrix with a clear colormap ranging from -1 to 1
    cax = ax.imshow(cos_sim_matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    
    # Add a colorbar with LaTeX label formatting
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Cosine Similarity $\\cos(\\theta)$", rotation=270, labelpad=15)
    
    # Configure tick labels using the token names (WORDS)
    ax.set_xticks(range(len(WORDS)))
    ax.set_yticks(range(len(WORDS)))
    ax.set_xticklabels(WORDS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(WORDS, fontsize=8)
    
    ax.set_title("Pairwise Cosine Similarity of Learned Embeddings", fontsize=10, pad=12)
    
    # Ensure labels are not truncated or overlapping
    plt.tight_layout()
    
    save_figure(fig, PLOTS_DIR, "embeddings_cosine_similarity.pdf")
    print("Saved embeddings_cosine_similarity.pdf")

    # ── Plotly Interactive Heatmap ───────────────────────────────────────────
    pfig = go.Figure(data=go.Heatmap(
        z=cos_sim_matrix.tolist(),
        x=WORDS,
        y=WORDS,
        colorscale="RdBu",
        reversescale=True,  # Matches 'RdBu_r' where red is positive similarity
        zmin=-1.0,
        zmax=1.0,
        hovertemplate="Token A: <b>%{y}</b><br>Token B: <b>%{x}</b><br>Similarity: <b>%{z:.3f}</b><extra></extra>"
    ))
    
    pfig.update_layout(
        title=dict(text="Pairwise Cosine Similarity of Learned Embeddings", font=dict(size=14)),
        xaxis=dict(tickangle=-45, automargin=True),
        yaxis=dict(automargin=True),
        width=600,
        height=600,
    )
    
    save_plotly(pfig, PLOTS_DIR, "embeddings_cosine_similarity.html")
    print("Saved embeddings_cosine_similarity.html")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False  # Bypassing missing LaTeX binaries
    grid = Grid()

    acc_path = os.path.join(DATA_DIR, "accuracies.npz")
    pca_path = os.path.join(DATA_DIR, "pca.npz")
    seq_path = os.path.join(DATA_DIR, "sequence.json")

    model = None

    if os.path.exists(acc_path) and os.path.exists(pca_path) and os.path.exists(seq_path):
        print("Loading cached data (delete data/ to recompute)...")
        all_accs = np.load(acc_path)["all_accs"]
        pca_data = np.load(pca_path)
        activations = pca_data["activations"]
        class_means = pca_data["class_means"]
        pca_dirs = pca_data["pca_dirs"]
        # Fallback to None if old cache doesn't have it
        explained_variance = pca_data.get("explained_variance", None) 
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
        
        # Unpack both items here
        pca_dirs_t, explained_variance = compute_pca_directions(class_means_t, top_n=3)

        activations = activations_t.cpu().numpy()
        class_means = class_means_t.cpu().numpy()
        pca_dirs = pca_dirs_t.cpu().numpy()

        np.savez(pca_path, activations=activations, class_means=class_means, 
                 pca_dirs=pca_dirs, explained_variance=explained_variance)
        with open(seq_path) as f:
            json.dump(sequence, f)
        print(f"Cached {pca_path}")


    # ── Embedding Neighbor Mixing Simulation ──────────────────────────────────
    print("\n--- Extracting Initial Embeddings and Simulating Neighbor Mixing ---")
    if model is None:
        print("Loading model to extract base embeddings...")
        model = load_model()
        
    try:
        embed_acts_t = get_activations(model, sequence, layer=0, n_lookback=N_LOOKBACK)
        embs_round_0 = compute_class_means(embed_acts_t, sequence, WORDS, N_LOOKBACK).cpu().numpy()
        
        cos_sim_matrix = cosine_similarity(embs_round_0)
        
        plot_cosine_similarity(cos_sim_matrix)
        # np.save(os.path.join(DATA_DIR, "embeddings_cosine_similarity.npy"), cos_sim_np)
        # print("Computed and saved pairwise cosine similarity matrix.")

    except Exception as e:
        print(f"Skipping learned embedding mixing due to error: {e}")


if __name__ == "__main__":
    main()