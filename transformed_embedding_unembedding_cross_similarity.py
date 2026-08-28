"""Applies the fitted ridge regression map (embedding_to_unembedding_linear_fit.py's W_ridge, b_ridge -- picked by held-out R^2 over a vocab-
wide fit, cached in results/embedding-to-unembedding-linear-fit/
fitted_maps.npz) to each word list's static embeddings, then plots the
same three pairwise-similarity panels (cosine similarity, Gram, L2
distance) as embedding_unembedding_cross_similarity.py, but between the
TRANSFORMED embeddings and the raw unembeddings, instead of raw
embeddings vs. raw unembeddings.

transformed_embedding(word i) = embedding(word i) @ W_ridge + b_ridge

Since the ridge fit only explained ~7% of held-out variance (R^2=0.072)
and didn't transfer to any individual word list (negative R^2 there), the
expectation is that this mostly won't look dramatically more diagonal
than the untransformed version -- this is the direct visual check of
that.

Requires embedding_to_unembedding_linear_fit.py to have been run first
(to produce fitted_maps.npz). W_E and W_U are read directly from the
cached safetensors shard as usual. Runs for every WORD_LISTS key, skipping
any list with a word that isn't a single token. CPU-only.
"""
import importlib.util
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import setup_plotting, compute_and_plot_cross_similarity_panels
from word_lists import WORD_LISTS

REPO = os.path.dirname(os.path.abspath(__file__))
FIT_CACHE_PATH = "results/embedding-to-unembedding-linear-fit/fitted_maps.npz"
PLOTS_DIR = "results/embedding-to-unembedding-linear-fit/transformed_cross_similarity"


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

    if not os.path.exists(FIT_CACHE_PATH):
        raise RuntimeError(
            f"No cached fit at {FIT_CACHE_PATH} -- run embedding_to_unembedding_linear_fit.py first."
        )
    fit = np.load(FIT_CACHE_PATH)
    W_ridge, b_ridge, best_lambda = fit["W_ridge"], fit["b_ridge"], float(fit["best_lambda"])
    print(f"Loaded ridge map (lambda={best_lambda:g}) from {FIT_CACHE_PATH}")

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

        token_ids_t = torch.tensor(token_ids, dtype=torch.long)
        embeddings = W_E[token_ids_t].double().numpy()
        unembeddings = W_U[token_ids_t].double().numpy()
        transformed_embeddings = embeddings @ W_ridge + b_ridge

        compute_and_plot_cross_similarity_panels(
            transformed_embeddings, unembeddings, words, PLOTS_DIR,
            f"transformed_cross_similarity_{key}.png",
            title=f"{key}, ridge-transformed",
            row_label="Ridge-transformed Embedding", col_label="Unembedding",
        )


if __name__ == "__main__":
    main()
