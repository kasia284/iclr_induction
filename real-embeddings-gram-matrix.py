import numpy as np
import matplotlib.pyplot as plt
import torch

from utils import load_model, setup_plotting, save_figure
from word_lists import WORD_LISTS

from sklearn.metrics.pairwise import cosine_similarity

WORD_LIST_KEY = "judging_nearest_neighbors" #
WORDS = WORD_LISTS[WORD_LIST_KEY]

DATA_DIR = "results/gram-matrix"
PLOTS_DIR = "results/gram-matrix"


def _label_heatmap_ax(ax):
    ax.set_xticks(range(len(WORDS)))
    ax.set_yticks(range(len(WORDS)))
    ax.set_xticklabels(WORDS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(WORDS, fontsize=8)


def plot_cosine_similarity(cos_sim_matrix, gram_matrix, title=None):
    """
    Plots the pairwise cosine similarity matrix and the raw Gram (inner
    product) matrix side by side. Cosine similarity only captures direction;
    the Gram matrix also preserves embedding magnitude.
    """
    fig, (ax_cos, ax_gram) = plt.subplots(1, 2, figsize=(13, 5.5))

    cax = ax_cos.imshow(cos_sim_matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax, ax=ax_cos, fraction=0.046, pad=0.04, label="Cosine Similarity")
    _label_heatmap_ax(ax_cos)
    ax_cos.set_title("Cosine Similarity", fontsize=10, pad=12)

    vmax = np.abs(gram_matrix).max()
    gax = ax_gram.imshow(gram_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(gax, ax=ax_gram, fraction=0.046, pad=0.04, label="Inner Product")
    _label_heatmap_ax(ax_gram)
    ax_gram.set_title("Gram Matrix", fontsize=10, pad=12)

    fig.suptitle(f"Pairwise Similarity of Static Embeddings ({title})", fontsize=12)
    plt.tight_layout()

    save_figure(fig, PLOTS_DIR, f"embeddings_cosine_similarity_{title}.png")
    print(f"Saved embeddings_cosine_similarity_{title}.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False  # Bypassing missing LaTeX binaries

    print("Loading model to extract static vocabulary embeddings...")
    model = load_model()
        
    try:
        token_ids = []
        
        print("Mapping words directly to unique token IDs...")
        for word in WORDS:
            # Most LLM tokenizers expect a leading space for standalone words (e.g., " Paris")
            word_with_space = f" {word}"
            
            try:
                # model.to_single_token strictly enforces that the string maps to EXACTLY ONE token ID.
                # If it splits into multiple subwords, it raises an AssertionError.
                tid = model.to_single_token(word_with_space)
            except AssertionError:
                # Fallback: check if the word fits as a single token without a leading space
                try:
                    tid = model.to_single_token(word)
                except AssertionError:
                    raise ValueError(f"The word '{word}' cannot be represented as a single token in this model's vocabulary!")
            
            token_ids.append(tid)
            
        # Convert list of IDs to a PyTorch tensor
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=model.cfg.device)
        
        # Pull the weights directly from the embedding matrix (W_E) 
        # This completely bypasses the forward pass/caching logic!
        # Shape: (len(WORDS), d_model)
        static_embeddings = model.W_E[token_ids_tensor].detach().cpu().numpy()
        print(f"Retrieved embedding matrix directly from W_E. Shape: {static_embeddings.shape}")
        
        # Compute pairwise cosine similarity and raw Gram (inner product) matrices
        cos_sim_matrix = cosine_similarity(static_embeddings)
        gram_matrix = static_embeddings @ static_embeddings.T

        # Plot and save the heatmaps
        plot_cosine_similarity(cos_sim_matrix, gram_matrix, title=WORD_LIST_KEY)

    except ValueError as val_err:
        print(f"\n[Vocabulary Error]: {val_err}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    main()