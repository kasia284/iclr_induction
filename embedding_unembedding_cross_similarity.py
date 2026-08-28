"""Pairwise similarity between EMBEDDING vectors (W_E, rows) and
UNEMBEDDING vectors (W_U, columns) for each word list -- three heatmaps
(cosine similarity, Gram/inner-product, L2 distance) per word list, with
row i = word i's static embedding and column j = word j's unembedding.

Unlike real_embeddings_gram_matrix.py / real_embeddings_gram_matrix_unembed.py (which each compare a vector set against ITSELF, giving
symmetric matrices), this is a CROSS comparison between two different
vector sets for the same words, so the matrices are generally asymmetric:
cell (i, j) = embedding(word i) vs. unembedding(word j) can differ from
cell (j, i) = embedding(word j) vs. unembedding(word i).

Answers: for a given word, does its own static embedding point toward its
own unembedding (diagonal) more than toward other words' unembeddings
(off-diagonal), and does grid adjacency show up as elevated off-diagonal
similarity?

W_E and W_U are both read directly from the cached safetensors shard (via
real_embeddings_nearest_neighbors.py's load_embedding_only_model and
activation_unembedding_dot_product.py's load_unembedding_matrix). Runs for
every WORD_LISTS key, skipping any list with a word that isn't a single
token. CPU-only, no GPU / forward pass needed.
"""
import importlib.util
import os

import matplotlib.pyplot as plt

from utils import setup_plotting, compute_and_plot_cross_similarity_panels
from word_lists import WORD_LISTS

REPO = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = "results/embedding-unembedding-cross-similarity"


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nn_mod = _load_module("real_embeddings_nearest_neighbors", "real_embeddings_nearest_neighbors.py")
dot_product_mod = _load_module("activation_unembedding_dot_product", "activation_unembedding_dot_product.py")

SWEEP_KEYS = list(WORD_LISTS.keys())


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False

    print("Loading static embedding (W_E) and unembedding (W_U) matrices...")
    embed_model = nn_mod.load_embedding_only_model()
    W_E, tokenizer = embed_model.W_E, embed_model.tokenizer
    W_U = dot_product_mod.load_unembedding_matrix()

    for key in SWEEP_KEYS:
        words = WORD_LISTS[key]
        try:
            token_ids = [dot_product_mod.to_single_token(tokenizer, w) for w in words]
        except ValueError as e:
            print(f"Skipping {key}: {e}")
            continue

        import torch
        token_ids_t = torch.tensor(token_ids, dtype=torch.long)
        embeddings = W_E[token_ids_t].numpy()
        unembeddings = W_U[token_ids_t].numpy()

        compute_and_plot_cross_similarity_panels(
            embeddings, unembeddings, words, PLOTS_DIR,
            f"cross_similarity_{key}.png",
            title=key, row_label="Embedding", col_label="Unembedding",
        )


if __name__ == "__main__":
    main()
