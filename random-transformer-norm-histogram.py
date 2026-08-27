import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import (
    WORDS, Grid, set_seed, load_toy_model, save_figure,
)


def layer_norm_normalized_hook(activation, hook, eps=1e-6):
    """Rescale LayerNorm's output (norm sqrt(d)) down to the unit
    hypersphere (norm 1), so attention is computed on unit-norm vectors."""
    norm = activation.norm(dim=-1, keepdim=True)
    return activation / (norm + eps)


def get_activation_norms(model, sequence, hook_name, fwd_hooks=[]):
    """Per-position norm of the activation at `hook_name` for a single
    sequence. Returns [seq_len]."""
    word_to_id = {word: i for i, word in enumerate(WORDS)}
    tokens = torch.tensor([[word_to_id[w] for w in sequence]], device=model.cfg.device)
    with model.hooks(fwd_hooks=fwd_hooks):
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
    return cache[hook_name][0].norm(dim=-1).cpu().numpy()  # [seq_len]


def plot_norm_histogram(seq_len=1000, layer=0, seed=0, n_bins=60):
    """Histograms of activation norms for no LayerNorm, LayerNorm, and
    LayerNorm with an extra unit-hypersphere renormalization, all evaluated
    on the same random-walk sequence."""
    set_seed(seed)
    grid = Grid()
    sequence = grid.generate_sequence(seq_len)

    # (label, normalization_type passed to load_toy_model, hook to read norms
    # from, extra fwd_hooks)
    configs = [
        ("before LayerNorm", None,
         f"blocks.{layer}.hook_resid_pre", []),
        ("after LayerNorm", "LN",
         f"blocks.{layer}.ln1.hook_normalized", []),
        ("after LayerNorm, normalized to unit sphere", "LN",
         f"blocks.{layer}.ln1.hook_normalized",
         [(f"blocks.{layer}.ln1.hook_normalized", layer_norm_normalized_hook)]),
    ]

    norms = {}
    for label, norm_type, hook_name, hooks in configs:
        model = load_toy_model(normalization_type=norm_type, seed=seed, n_ctx=seq_len)
        model.W_pos.data.zero_()
        norms[label] = get_activation_norms(model, sequence, hook_name, fwd_hooks=hooks)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (label, vals) in zip(axes, norms.items()):
        vals = vals.astype(np.float64)
        # unit-sphere renorm collapses the range to float32-eps width, which
        # breaks matplotlib's bin-edge computation, so fall back to 1 bin.
        nb = n_bins if (vals.max() - vals.min()) > 1e-4 else 1
        ax.hist(vals, bins=nb, color="tab:blue", edgecolor="black", alpha=0.8)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("activation norm")
        ax.set_ylabel("count")
        mean, std = vals.mean(), vals.std()
        ax.axvline(mean, color="red", linestyle="--", linewidth=1)
        ax.text(0.97, 0.95, f"mean={mean:.3f}\nstd={std:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8)

    fig.suptitle(
        f"Activation norm histograms (layer {layer}, seq_len={seq_len}): "
        "before LN vs. after LN vs. after LN normalized to unit sphere"
    )
    fig.tight_layout()
    save_figure(fig, "results/random_transformer/plots", "activation_norm_histogram_ln_variants.png")


if __name__ == "__main__":
    plot_norm_histogram()
