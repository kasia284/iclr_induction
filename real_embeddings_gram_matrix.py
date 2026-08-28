import matplotlib.pyplot as plt
import torch

from utils import load_model, setup_plotting, compute_and_plot_similarity_panels
from word_lists import WORD_LISTS

DATA_DIR = "results/gram-matrix"
PLOTS_DIR = "results/gram-matrix"

RUN_WORD_LIST_KEYS = [
    "morphology", "morphology_permuted",
]


# ── Main ───────────────────────────────────────────────────────────────────────

def process_word_list(model, word_list_key, words):
    try:
        token_ids = []

        print(f"Mapping words directly to unique token IDs for '{word_list_key}'...")
        for word in words:
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
        # Shape: (len(words), d_model)
        static_embeddings = model.W_E[token_ids_tensor].detach().cpu().numpy()
        print(f"Retrieved embedding matrix directly from W_E. Shape: {static_embeddings.shape}")

        # Compute pairwise cosine similarity, Gram, and L2 distance matrices
        # and plot the heatmaps
        compute_and_plot_similarity_panels(
            static_embeddings, words, PLOTS_DIR,
            f"embeddings_similarity_{word_list_key}.png",
            title=f"Static Embeddings ({word_list_key})",
        )

    except ValueError as val_err:
        print(f"\n[Vocabulary Error] ({word_list_key}): {val_err}")
    except Exception as e:
        print(f"\nAn unexpected error occurred ({word_list_key}): {e}")


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False  # Bypassing missing LaTeX binaries

    print("Loading model to extract static vocabulary embeddings...")
    model = load_model()

    for word_list_key in RUN_WORD_LIST_KEYS:
        process_word_list(model, word_list_key, WORD_LISTS[word_list_key])


if __name__ == "__main__":
    main()