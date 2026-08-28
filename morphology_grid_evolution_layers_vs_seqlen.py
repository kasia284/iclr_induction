"""For each word list in SWEEP_KEYS (every cached grid-position permutation
of the "morphology", "text_numbers", and "two_digit_numbers" word-set
families), a grid of PCA scatter plots:
rows = model layer (0, 6, 13, 20, 26, 31), columns = sequence length (6
log-spaced checkpoints from N_LOOKBACK up to SEQ_LEN=1400), so "grid-ness"
of the class-mean geometry can be inspected visually as it evolves along
both depth and context length at once, for each word list.

Each cell's class means are built the same way as everywhere else in the
sweep: the last min(N_LOOKBACK, k) positions of each of the 16 random-walk
sequences (same set_seed(42) walks as 01_reproduce.py's main()), batched
over all 16 sequences. Each permutation's bottom-right cell (layer 31, seq
len 1400) exactly matches 01_reproduce.py's own pca_across_layers/
pca_all_layers_grid output for that permutation and layer.

Requires one full forward pass per permutation, caching hook_resid_pre at
all 6 layers for every position of the sequence (not just the tail), so
needs a GPU. Results are cached per permutation to
results/reproduce/data/{key}/grid_evolution_layers_vs_seqlen.npz, so
re-running the plots alone doesn't require the model.

Usage:
    python morphology_grid_evolution_layers_vs_seqlen.py                # all SWEEP_KEYS
    python morphology_grid_evolution_layers_vs_seqlen.py <key> [<key>...]  # only these keys

    # --post-layernorm caches ln_final.hook_normalized instead of the usual
    # LAYERS sweep of blocks.{l}.hook_resid_pre -- the residual stream after
    # ALL 32 blocks AND the model's final RMSNorm (exactly what feeds the
    # unembedding matrix), stored under the virtual layer key POST_LN_KEY
    # ("31_post_ln") in a separate cache file so it never collides with a
    # normal run's cache. Combine with distance-correlation-accuracy-phase-
    # plane.py --post-layernorm to plot it. Doesn't produce a
    # plot_grid_evolution grid (nothing to sweep -- there's only one layer).
    python morphology_grid_evolution_layers_vs_seqlen.py --post-layernorm [<key> ...]

Reuses build_word_token_ids / tokenize_sequences_by_id_batch /
configure_for_word_list / plot_class_mean_pca / MORPHOLOGY_SWEEP_KEYS from
01_reproduce.py (imported via importlib since its filename isn't a valid
module identifier) -- plot_class_mean_pca already supports drawing onto a
caller-provided axes, so each cell reuses the exact same PCA-scatter
rendering (grid edges, word colors/markers) as the rest of the repo.
"""
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from utils import Grid, set_seed, load_model, compute_class_means_batch, compute_pca_directions, setup_plotting, save_figure, compute_distance_correlation
from word_lists import WORD_LISTS

REPO = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reproduce_mod = _load_module("reproduce_01", "01_reproduce.py")

SWEEP_KEYS = [
    "text_numbers", "text_numbers_permuted",
    "text_numbers_rand1", "text_numbers_rand2", "text_numbers_rand3",
    "text_numbers_rand4", "text_numbers_rand5",
    "two_digit_numbers", "two_digit_numbers_permuted",
    "two_digit_numbers_rand1", "two_digit_numbers_rand2", "two_digit_numbers_rand3",
    "two_digit_numbers_rand4", "two_digit_numbers_rand5",
    "morphology", "morphology_permuted",
    "morphology_rand1", "morphology_rand2", "morphology_rand3", "morphology_corners",
    "morphology_rand4", "morphology_rand5",
    "original_paper", "original_paper_permuted",
    "original_paper_rand1", "original_paper_rand2", "original_paper_rand3",
    "original_paper_rand4", "original_paper_rand5", "original_paper_rand6",
    "original_paper_rand7",
]
LAYERS = ['0', '6', '13', '20', '26', '31']
N_LOOKBACK = reproduce_mod.N_LOOKBACK
SEQ_LEN = reproduce_mod.SEQ_LEN

# 6 log-spaced sequence-length checkpoints, matching len(LAYERS) for a
# square 6x6 grid. Cheap to change: the only GPU cost is the one full-
# sequence, multi-layer forward pass per permutation -- everything per-
# checkpoint below is CPU postprocessing on the resulting cache.
N_CHECKPOINTS = 6
CHECKPOINTS = sorted(set(
    np.geomspace(N_LOOKBACK, SEQ_LEN, N_CHECKPOINTS).astype(int).tolist() + [SEQ_LEN]
))[:N_CHECKPOINTS]

# Virtual "layer" label for the post-layernorm variant: the residual
# stream after ALL 32 blocks AND the model's final RMSNorm (TransformerLens
# hook "ln_final.hook_normalized") -- i.e. exactly what feeds the
# unembedding matrix, as opposed to LAYERS' "31" (blocks.31.hook_resid_pre,
# the input to the LAST block: after blocks 0-30, before block 31 runs and
# before any final normalization).
POST_LN_KEY = "31_post_ln"
POST_LN_HOOK_NAME = "ln_final.hook_normalized"


def cache_path(word_list_key, post_layernorm=False):
    suffix = "_post_ln" if post_layernorm else ""
    return os.path.join(
        "results/reproduce/data", word_list_key, f"grid_evolution_layers_vs_seqlen{suffix}.npz"
    )


def _cell_key(layer, k):
    return f"L{layer}_k{k}"


def compute_class_means_grid(model, word_list_key, post_layernorm=False):
    words = WORD_LISTS[word_list_key]
    grid = Grid(words=words, rows=4, cols=4)
    word_to_id = reproduce_mod.build_word_token_ids(model, words)

    # Same seed + generation call as 01_reproduce.py's main() for this word
    # list, so this is the exact same batch of random walks used elsewhere.
    set_seed(42)
    sequences = grid.generate_batch(SEQ_LEN, len(words))

    tokens = reproduce_mod.tokenize_sequences_by_id_batch(model, sequences, word_to_id)

    # post_layernorm has just ONE virtual "layer" (ln_final only runs once,
    # at the very end of the network) instead of the usual LAYERS sweep.
    layers = [POST_LN_KEY] if post_layernorm else LAYERS
    names_filter = [POST_LN_HOOK_NAME] if post_layernorm else [f"blocks.{l}.hook_resid_pre" for l in LAYERS]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens.to(model.cfg.device), names_filter=names_filter)
    # {layer: [batch, SEQ_LEN, d_model]}, BOS position removed -- full
    # sequence at every layer, not just the tail.
    if post_layernorm:
        layer_activations = {POST_LN_KEY: cache[POST_LN_HOOK_NAME][:, 1:, :].cpu()}
    else:
        layer_activations = {
            l: cache[f"blocks.{l}.hook_resid_pre"][:, 1:, :].cpu()
            for l in LAYERS
        }
    del cache
    torch.cuda.empty_cache()

    class_means = {}
    for layer in tqdm.tqdm(layers, desc=f"{word_list_key}: layers"):
        acts = layer_activations[layer]
        for k in CHECKPOINTS:
            window = min(N_LOOKBACK, k)
            sequences_window = [seq[k - window:k] for seq in sequences]
            activations_window = acts[:, k - window:k, :]
            class_means[(layer, k)] = compute_class_means_batch(
                activations_window, sequences_window, words, window
            ).numpy()
    del layer_activations

    path = cache_path(word_list_key, post_layernorm=post_layernorm)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, checkpoints=np.array(CHECKPOINTS), **{
        _cell_key(layer, k): class_means[(layer, k)]
        for layer in layers for k in CHECKPOINTS
    })
    print(f"Cached {path}")
    return class_means


def load_or_compute_class_means(model, word_list_key, post_layernorm=False):
    path = cache_path(word_list_key, post_layernorm=post_layernorm)
    layers = [POST_LN_KEY] if post_layernorm else LAYERS
    if os.path.exists(path):
        data = np.load(path)
        return {
            (layer, k): data[_cell_key(layer, k)]
            for layer in layers for k in CHECKPOINTS
        }
    if model is None:
        raise RuntimeError(f"No cache at {path} and no model loaded to compute it.")
    return compute_class_means_grid(model, word_list_key, post_layernorm=post_layernorm)


def plot_grid_evolution(class_means, word_list_key):
    words = WORD_LISTS[word_list_key]
    grid = Grid(words=words, rows=4, cols=4)
    grid_coords = reproduce_mod.get_grid_coords(grid, words)
    # Sets reproduce_mod.WORDS / WORD_TO_COLOR / PLOTS_DIR, read internally
    # by reproduce_mod.plot_class_mean_pca.
    reproduce_mod.configure_for_word_list(word_list_key)

    n_rows, n_cols = len(LAYERS), len(CHECKPOINTS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows))

    for i, layer in enumerate(LAYERS):
        for j, k in enumerate(CHECKPOINTS):
            cm = class_means[(layer, k)]
            pca_dirs, var = compute_pca_directions(torch.tensor(cm), top_n=2)
            dc = compute_distance_correlation(cm, grid_coords)
            reproduce_mod.plot_class_mean_pca(
                grid, cm, pca_dirs.numpy(), ax=axes[i, j],
                title=f"Layer {layer}, seq len {k}\nDC={dc:.3f}", explained_variance=var,
            )

    fig.suptitle(
        f"PCA of per-node mean activations across layers x sequence length ({word_list_key})",
        fontsize=12,
    )
    plots_dir = f"results/reproduce/plots/{word_list_key}"
    save_figure(fig, plots_dir, "grid_evolution_layers_vs_seqlen_2d.pdf")
    print(f"Saved {word_list_key}/grid_evolution_layers_vs_seqlen_2d")


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False

    argv = sys.argv[1:]
    post_layernorm = "--post-layernorm" in argv
    requested = [a for a in argv if a != "--post-layernorm"]
    if requested:
        unknown = [k for k in requested if k not in SWEEP_KEYS]
        if unknown:
            raise SystemExit(f"Not in SWEEP_KEYS: {unknown}")
        keys = requested
    else:
        keys = SWEEP_KEYS

    model = None
    if not all(os.path.exists(cache_path(k, post_layernorm=post_layernorm)) for k in keys):
        model = load_model()

    for word_list_key in keys:
        print(f"\n=== {word_list_key} ===")
        class_means = load_or_compute_class_means(model, word_list_key, post_layernorm=post_layernorm)
        # plot_grid_evolution sweeps LAYERS as a grid row per layer, which
        # doesn't apply to the single post-ln "layer" -- that comparison is
        # made via distance_correlation_accuracy_phase_plane.py --post-layernorm
        # instead.
        if not post_layernorm:
            plot_grid_evolution(class_means, word_list_key)


if __name__ == "__main__":
    main()
