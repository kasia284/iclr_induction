"""PCA scatter of each word's final-layer class-mean activation (as in
01_reproduce.py's Fig 2 right / pca_all_layers_grid), with each word's
UNEMBEDDING vector (a row of W_U / lm_head.weight) projected onto the SAME
two PCA directions and overlaid on the same axes -- to see whether the
model's readout direction for a word points toward that word's own
grid-neighbor cluster, or somewhere else in the same geometric space the
activations occupy.

The cached class means are `hook_resid_pre` (pre-final-LayerNorm residual
stream), which lives at a totally different scale than raw W_U rows -- the
model never dots those two things together directly. What it actually
computes is logits = RMSNorm(resid, weight=model.norm.weight) @ W_U^T, so
before PCA/projection each class mean is passed through that same RMSNorm
(load_final_norm_weight reads model.norm.weight + rms_norm_eps straight
from the cached safetensors shard / config.json, mirroring
load_unembedding_matrix's loader), putting activations and unembeddings in
the same space the model's own logit computation uses. (Caveat: LAYERS'
"final" entry is blocks.31.hook_resid_pre, i.e. the input to the last
block, not its output -- one block short of the literal pre-ln_final
residual stream -- but it's the same "final-layer" cache used everywhere
else in the repo, e.g. the Dirichlet-energy/distance-correlation sweeps.)

PCA directions are fit on the RMSNorm'd class means (top_n=2, via
compute_pca_directions); unembedding vectors are projected using those
same directions and the same centering:

    (W_U[token_id] - normed_class_means.mean(0)) @ pca_dirs.T

A thin dashed line connects each word's activation (star) to its own
unembedding (diamond), same color, so the "readout offset" is visible per
word.

Base scatter (grid edges, star markers, word labels/colors, axis labels)
is drawn via 01_reproduce.py's plot_class_mean_pca, imported via
importlib, reused with a caller-provided axes so it's pixel-identical to
every other PCA plot in the repo -- only the unembedding overlay is new.

Final-layer class means come from 01_reproduce.py's cache
(pca_all_layers_Grid_{key}.npz); W_U and model.norm.weight are read
directly from the cached safetensors shard (activation-unembedding-dot-
product.py's load_unembedding_matrix, and this script's
load_final_norm_weight). CPU-only, no GPU / forward pass needed.

Produces one plot per word list in SWEEP_KEYS (every WORD_LISTS key with a
cached final-layer activation file), saved to
results/reproduce/plots/{key}/pca_class_means_with_unembeddings_2d.png.
"""
import glob
import importlib.util
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import Grid, setup_plotting, save_figure, compute_pca_directions, MODEL_NAME
from word_lists import WORD_LISTS

REPO = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reproduce_mod = _load_module("reproduce_01", "01_reproduce.py")
dot_product_mod = _load_module("activation_unembedding_dot_product", "activation-unembedding-dot-product.py")


def discover_sweep_keys():
    """Every WORD_LISTS key with a cached final-layer class-means file."""
    keys = []
    for key in WORD_LISTS:
        path = f"results/reproduce/data/{key}/pca_all_layers_Grid_{key}.npz"
        if os.path.exists(path):
            keys.append(key)
    return keys


SWEEP_KEYS = discover_sweep_keys()


def load_final_norm_weight():
    """Read model.norm.weight (final RMSNorm gain, [d_model]) and
    rms_norm_eps directly out of the cached safetensors shard / config.json,
    without loading the rest of the model."""
    hf_home = os.environ.get("HF_HOME")
    cache_root = os.path.join(hf_home, "hub") if hf_home else os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = os.path.join(cache_root, f"models--{MODEL_NAME.replace('/', '--')}")
    snapshot_dirs = glob.glob(os.path.join(model_dir, "snapshots", "*"))
    if not snapshot_dirs:
        raise FileNotFoundError(f"No local snapshot found for {MODEL_NAME} under {model_dir}")
    snapshot_dir = snapshot_dirs[0]

    with open(os.path.join(snapshot_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    shard_path = os.path.join(snapshot_dir, weight_map["model.norm.weight"])

    from safetensors import safe_open
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        norm_weight = f.get_tensor("model.norm.weight").float()

    with open(os.path.join(snapshot_dir, "config.json")) as f:
        eps = json.load(f)["rms_norm_eps"]
    return norm_weight, eps


def apply_rmsnorm(x, weight, eps):
    """x: [n, d_model] numpy. Standard Llama RMSNorm: no mean subtraction,
    divide by RMS over the last dim, then scale by the learned weight."""
    x_t = torch.tensor(x)
    variance = x_t.pow(2).mean(dim=-1, keepdim=True)
    normed = x_t * torch.rsqrt(variance + eps)
    return (normed * weight).numpy()


def plot_pca_with_unembeddings(class_means, unembeddings, word_list_key):
    """class_means, unembeddings: [n_words, d_model] numpy, same word order
    as WORD_LISTS[word_list_key]."""
    words = WORD_LISTS[word_list_key]
    side = round(len(words) ** 0.5)
    grid = Grid(words=words, rows=side, cols=side)
    reproduce_mod.configure_for_word_list(word_list_key)

    pca_dirs_t, var = compute_pca_directions(torch.tensor(class_means), top_n=2)
    pca_dirs = pca_dirs_t.numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    reproduce_mod.plot_class_mean_pca(
        grid, class_means, pca_dirs, ax=ax,
        title=f"RMSNorm'd final-layer activations + unembeddings ({word_list_key})",
        explained_variance=var,
    )

    class_means_mean = class_means.mean(axis=0, keepdims=True)
    proj_act = (class_means - class_means_mean) @ pca_dirs.T  # [n, 2]

    # Unembedding rows have ~150x smaller norm than the RMSNorm'd
    # activations, but are also ~150x smaller than the activation
    # CENTROID itself -- so centering unembeddings on class_means_mean (as
    # if both clouds shared one origin) would make that huge centroid
    # dominate every projected point, swamping the actual per-word
    # variation we care about. Center unembeddings on their OWN mean
    # instead, isolating the inter-word signal, then rescale that (much
    # smaller) cloud by one shared scalar so its average radius matches
    # the activation cloud's -- preserves each point's direction/relative
    # position within its own cloud exactly; only the overall magnitude is
    # stretched for visibility (labeled explicitly below; not physically
    # to scale, and not directly comparable in *absolute* position to the
    # activations -- only in shape/relative layout).
    unembed_mean = unembeddings.mean(axis=0, keepdims=True)
    proj_unembed = (unembeddings - unembed_mean) @ pca_dirs.T  # [n, 2]

    act_radius = np.linalg.norm(proj_act, axis=1).mean()
    unembed_radius = np.linalg.norm(proj_unembed, axis=1).mean()
    rescale = act_radius / unembed_radius if unembed_radius > 0 else 1.0
    proj_unembed_vis = proj_unembed * rescale

    for i, word in enumerate(words):
        color = reproduce_mod.WORD_TO_COLOR[word]
        ax.plot(
            [proj_act[i, 0], proj_unembed_vis[i, 0]], [proj_act[i, 1], proj_unembed_vis[i, 1]],
            color=color, alpha=0.5, linestyle="--", linewidth=0.7, zorder=4,
        )
        ax.scatter(
            proj_unembed_vis[i, 0], proj_unembed_vis[i, 1],
            color=color, s=80, marker="D", edgecolors="black", linewidths=0.5, zorder=5,
        )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=11, label="RMSNorm'd final-layer activation"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=8,
               label=f"Unembedding vector (own-mean centered, rescaled {rescale:.0f}x for visibility)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=6.5,
              frameon=True, framealpha=0.9, edgecolor="gray")

    plots_dir = f"results/reproduce/plots/{word_list_key}"
    save_figure(fig, plots_dir, "pca_class_means_with_unembeddings_2d.pdf")
    print(f"Saved {word_list_key}/pca_class_means_with_unembeddings_2d")


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False

    if not SWEEP_KEYS:
        raise RuntimeError(
            "No word lists with cached pca_all_layers_Grid_{key}.npz found -- "
            "run 01_reproduce.py for at least one word list first."
        )
    print(f"Found {len(SWEEP_KEYS)} word lists with cached final-layer activations.")

    print("Loading unembedding matrix (lm_head.weight) and final RMSNorm weight...")
    W_U = dot_product_mod.load_unembedding_matrix()
    norm_weight, eps = load_final_norm_weight()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for key in SWEEP_KEYS:
        words = WORD_LISTS[key]
        class_means, final_layer = dot_product_mod.load_final_layer_class_means(key)
        normed_class_means = apply_rmsnorm(class_means, norm_weight, eps)
        token_ids = torch.tensor(
            [dot_product_mod.to_single_token(tokenizer, w) for w in words], dtype=torch.long
        )
        unembeddings = W_U[token_ids].numpy()
        plot_pca_with_unembeddings(normed_class_means, unembeddings, key)


if __name__ == "__main__":
    main()
