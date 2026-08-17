"""PCA of JUST the unembedding vectors (rows of W_U / lm_head.weight) for
each word list -- no activations, no forward pass, no cache dependency at
all. Fits PCA directly on the n raw unembedding rows (top_n=2, via
compute_pca_directions) and plots them with the same grid-edges/star-
markers/word-colors styling as every other PCA plot in the repo (via
01_reproduce.py's plot_class_mean_pca, imported via importlib), so it's
visually comparable to the activation PCA plots at a glance.

Answers: does grid adjacency show up in the readout directions ALONE,
purely from how the model's unembedding matrix is laid out, with no
context/induction involved at all? (Companion to the static W_E PCA and
to final-activations-pca-with-unembeddings.py's overlay version.)

W_U is read directly from the cached safetensors shard (activation-
unembedding-dot-product.py's load_unembedding_matrix). Runs for every
WORD_LISTS key (not just ones with a cached 01_reproduce.py run), skipping
any list with a word that isn't a single token. CPU-only.
"""
import importlib.util
import os

import matplotlib.pyplot as plt
import torch

from utils import Grid, setup_plotting, save_figure, compute_pca_directions
from word_lists import WORD_LISTS

REPO = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reproduce_mod = _load_module("reproduce_01", "01_reproduce.py")
dot_product_mod = _load_module("activation_unembedding_dot_product", "activation-unembedding-dot-product.py")

SWEEP_KEYS = list(WORD_LISTS.keys())


def plot_unembeddings_pca(unembeddings, word_list_key):
    words = WORD_LISTS[word_list_key]
    side = round(len(words) ** 0.5)
    grid = Grid(words=words, rows=side, cols=side)
    reproduce_mod.configure_for_word_list(word_list_key)

    pca_dirs_t, var = compute_pca_directions(torch.tensor(unembeddings), top_n=2)
    pca_dirs = pca_dirs_t.numpy()

    # Pass our own axes (non-standalone) so plot_class_mean_pca doesn't
    # self-save under its default "pca_class_means_2d" filename, which
    # would collide with 01_reproduce.py's own class-means PCA output in
    # the same PLOTS_DIR.
    fig, ax = plt.subplots(figsize=(5, 5))
    reproduce_mod.plot_class_mean_pca(
        grid, unembeddings, pca_dirs, ax=ax,
        title=f"PCA of unembedding vectors ({word_list_key})",
        explained_variance=var,
    )
    plots_dir = f"results/reproduce/plots/{word_list_key}"
    save_figure(fig, plots_dir, "pca_unembeddings_2d.pdf")


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False

    print("Loading unembedding matrix (lm_head.weight)...")
    W_U = dot_product_mod.load_unembedding_matrix()
    from transformers import AutoTokenizer
    from utils import MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for key in SWEEP_KEYS:
        words = WORD_LISTS[key]
        side = round(len(words) ** 0.5)
        if side * side != len(words):
            print(f"Skipping {key}: {len(words)} words is not a perfect square.")
            continue
        try:
            token_ids = torch.tensor(
                [dot_product_mod.to_single_token(tokenizer, w) for w in words], dtype=torch.long
            )
        except ValueError as e:
            print(f"Skipping {key}: {e}")
            continue
        unembeddings = W_U[token_ids].numpy()
        plot_unembeddings_pca(unembeddings, key)
        print(f"Saved {key}/pca_unembeddings_2d")


if __name__ == "__main__":
    main()
