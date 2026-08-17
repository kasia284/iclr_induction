"""Like activation-unembedding-dot-product.py, but instead of each token's
*mean* final-layer activation (averaged over every occurrence across
N_SEQUENCES random walks), uses the single activation vector from that
token's LAST occurrence in one cached random-walk sequence -- no averaging.

Requires a real forward pass (mean-based version could reuse cached class
means; a single occurrence's activation isn't cached anywhere), so this
needs a GPU -- run via runai exec into an interactive job, not on a
CPU-only shell.
"""
import importlib.util
import json
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import load_model, Grid, setup_plotting
from word_lists import WORD_LISTS

REPO = os.path.dirname(os.path.abspath(__file__))

# Reuse plot_dot_product_heatmap from activation-unembedding-dot-product.py
# (hyphenated filename isn't a valid import target, so load it via
# importlib, same trick used elsewhere in this repo, e.g.
# run_capital_letters_layer_pca.py / 03_neighbor_mixing-grid_likeness_check.py).
spec = importlib.util.spec_from_file_location(
    "activation_unembedding_dot_product",
    os.path.join(REPO, "activation-unembedding-dot-product.py"),
)
dot_product_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dot_product_mod)

# Redirect the imported module's output dir so this doesn't overwrite the
# mean-based heatmaps (plot_dot_product_heatmap saves via the PLOTS_DIR
# global in dot_product_mod's own namespace, resolved at call time).
dot_product_mod.PLOTS_DIR = "results/activation-unembedding-dot-product-last-occurrence"

DATA_DIR = "results/reproduce/data"


def build_word_token_ids(model, words):
    word_to_id = {}
    for w in words:
        try:
            word_to_id[w] = model.to_single_token(f" {w}")
        except AssertionError:
            word_to_id[w] = model.to_single_token(w)
    return word_to_id


def get_last_occurrence_activations(model, sequence, word_to_id, words, layer):
    """Single forward pass over `sequence`; for each word in `words`,
    return blocks.{layer}.hook_resid_pre at that word's LAST occurrence in
    the sequence (no averaging). Returns [len(words), d_model]."""
    ids = [model.tokenizer.bos_token_id] + [word_to_id[w] for w in sequence]
    tokens = torch.tensor([ids], dtype=torch.long)
    hook_name = f"blocks.{layer}.hook_resid_pre"
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens.to(model.cfg.device), names_filter=[hook_name])
    acts = cache[hook_name][0, 1:, :]  # drop BOS -> [seq_len, d_model]

    vectors = []
    for w in words:
        occurrences = [i for i, tok in enumerate(sequence) if tok == w]
        if not occurrences:
            raise ValueError(f"'{w}' never occurs in this cached sequence -- can't take its last occurrence.")
        vectors.append(acts[occurrences[-1]].cpu().numpy())
    return np.stack(vectors)


def process_word_list(model, word_list_key, final_layer):
    words = WORD_LISTS[word_list_key]
    seq_path = os.path.join(DATA_DIR, word_list_key, f"sequence_Grid_{word_list_key}.json")
    with open(seq_path) as f:
        sequence = json.load(f)

    word_to_id = build_word_token_ids(model, words)
    last_occurrence_acts = get_last_occurrence_activations(model, sequence, word_to_id, words, final_layer)

    side = round(len(words) ** 0.5)
    grid = Grid(words=words, rows=side, cols=side)
    adjacency = grid.build_adjacency_matrix()

    token_ids = torch.tensor([word_to_id[w] for w in words], dtype=torch.long)
    # TransformerLens's W_U convention is [d_model, d_vocab] (transposed
    # relative to lm_head.weight's [d_vocab, d_model]).
    unembeddings = model.W_U[:, token_ids].T.cpu().numpy()

    dot_product_mod.plot_dot_product_heatmap(
        last_occurrence_acts, unembeddings, words, adjacency, word_list_key, final_layer,
    )


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False

    print("Loading model...")
    model = load_model()
    final_layer = model.cfg.n_layers - 1

    for word_list_key in ["text_numbers", "text_numbers_permuted"]:
        print(f"\n=== {word_list_key} ===")
        process_word_list(model, word_list_key, final_layer)


if __name__ == "__main__":
    main()
