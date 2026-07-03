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


def get_attn_patterns(model, sequence, layer=0, fwd_hooks=[]):
    """Attention pattern for every head at `layer` on a single sequence.
    Returns [n_heads, seq_len, seq_len]."""
    word_to_id = {word: i for i, word in enumerate(WORDS)}
    tokens = torch.tensor([[word_to_id[w] for w in sequence]], device=model.cfg.device)
    hook_name = f"blocks.{layer}.attn.hook_pattern"
    with model.hooks(fwd_hooks=fwd_hooks):
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
    return cache[hook_name][0].cpu().numpy()  # [n_heads, seq_len, seq_len]


def plot_attn_comparison(seq_len=16, layer=0, seed=0):
    """Attention score matrices for no LayerNorm, LayerNorm, and LayerNorm
    with an extra unit-hypersphere renormalization, one row each, one column
    per head, sharing the same random-walk sequence and the same (seeded)
    weight initialization."""
    set_seed(seed)
    grid = Grid()
    sequence = grid.generate_sequence(seq_len)

    # (label, normalization_type passed to load_toy_model, extra fwd_hooks)
    configs = [
        ("without LayerNorm", None, []),
        ("with LayerNorm", "LN", []),
        ("LayerNorm, normalized to unit sphere", "LN",
         [(f"blocks.{layer}.ln1.hook_normalized", layer_norm_normalized_hook)]),
    ]
    models = {
        label: load_toy_model(normalization_type=norm_type, seed=seed, n_ctx=seq_len)
        for label, norm_type, _ in configs
    }
    for model in models.values():
        model.W_pos.data.zero_()

    patterns = {
        label: get_attn_patterns(models[label], sequence, layer, fwd_hooks=hooks)
        for label, _, hooks in configs
    }
    n_heads = next(iter(patterns.values())).shape[0]

    fig, axes = plt.subplots(3, n_heads, figsize=(3.2 * n_heads, 9.6))
    for row, (label, pat) in enumerate(patterns.items()):
        for h in range(n_heads):
            ax = axes[row, h]
            im = ax.imshow(pat[h], cmap="viridis", vmin=0, vmax=1, aspect="equal")
            ax.set_title(f"{label} | head {h}", fontsize=9)
            ax.set_xticks(range(seq_len))
            ax.set_yticks(range(seq_len))
            ax.set_xticklabels(sequence, fontsize=6, rotation=90)
            ax.set_yticklabels(sequence, fontsize=6)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Attention patterns (layer {layer}): no LN vs. LN vs. LN normalized to unit sphere")
    fig.tight_layout()
    save_figure(fig, "results/random_transformer/plots", "attn_patterns_ln_variants.png")


if __name__ == "__main__":
    plot_attn_comparison()
