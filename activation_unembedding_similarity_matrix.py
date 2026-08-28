"""For each grid-position permutation in the morphology sweep (see
01_reproduce.py's MORPHOLOGY_SWEEP_KEYS and word_lists.py), builds one
combined 32x32 pairwise similarity matrix (cosine similarity, Gram, and L2
distance panels) over the 16 words' final-layer activations stacked with
their 16 unembedding vectors:

    [ act . act    act . U  ]
    [  U  . act    U  . U   ]

-- i.e. activation-vs-activation, activation-vs-unembedding, and
unembedding-vs-unembedding all visible in one set of panels, to relate the
geometry of contextualized representations to the geometry of the
(permutation-independent) unembedding space.

Reuses load_unembedding_matrix/to_single_token/load_final_layer_class_means
from activation_unembedding_dot_product.py (imported via importlib,
consistent with how sibling scripts in this repo share helpers) and
utils.compute_and_plot_similarity_panels for the actual panels. Final-layer
activations are read from 01_reproduce.py's cache
(results/reproduce/data/{key}/pca_all_layers_Grid_{key}.npz), so
01_reproduce.py must be run first for each key. CPU-only otherwise.
"""
import importlib.util
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from utils import MODEL_NAME, setup_plotting, compute_and_plot_similarity_panels
from word_lists import WORD_LISTS

REPO = os.path.dirname(os.path.abspath(__file__))

spec = importlib.util.spec_from_file_location(
    "activation_unembedding_dot_product",
    os.path.join(REPO, "activation_unembedding_dot_product.py"),
)
dot_product_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dot_product_mod)

PLOTS_DIR = "results/activation-unembedding-similarity-matrix"

# Mirrors 01_reproduce.py's MORPHOLOGY_SWEEP_KEYS -- same 16 morphology
# words, 5 different assignments onto the 4x4 grid, spanning same-lemma
# grid-edge fraction from 0.50 (morphology) down to 0.00
# (morphology_permuted). See word_lists.py for the permutations themselves.
MORPHOLOGY_SWEEP_KEYS = [
    "morphology_corners", "morphology", "morphology_rand3",
    "morphology_rand2", "morphology_rand1", "morphology_permuted",
]


def process_word_list(W_U, tokenizer, word_list_key):
    words = WORD_LISTS[word_list_key]

    activations, final_layer = dot_product_mod.load_final_layer_class_means(word_list_key)

    token_ids = [dot_product_mod.to_single_token(tokenizer, w) for w in words]
    unembeddings = W_U[torch.tensor(token_ids, dtype=torch.long)].numpy()

    stacked = np.concatenate([activations, unembeddings], axis=0)  # [32, d_model]
    labels = [f"{w}·act" for w in words] + [f"{w}·U" for w in words]

    compute_and_plot_similarity_panels(
        stacked, labels, PLOTS_DIR,
        f"act_unemb_similarity_{word_list_key}.png",
        title=f"Activation (layer {final_layer}) + Unembedding similarity ({word_list_key})",
    )


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False

    print(f"Loading unembedding matrix (lm_head.weight) for {MODEL_NAME}...")
    W_U = dot_product_mod.load_unembedding_matrix()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for word_list_key in MORPHOLOGY_SWEEP_KEYS:
        print(f"\n=== {word_list_key} ===")
        process_word_list(W_U, tokenizer, word_list_key)


if __name__ == "__main__":
    main()
