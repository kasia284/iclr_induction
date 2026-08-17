"""Exploratory: plot the L2 norm of the final-layer residual stream
(blocks.{n_layers-1}.hook_resid_post) as a function of token position and
layer, for one cached grid-task sequence per word list in RUN_WORD_LIST_KEYS
(plus an average over 16 fresh random-walk sequences per word list). Informs
the design of the norm-matching step in 01_reproduce.py's
run_iterative_forward_passes (per-position rescale to a single target vs. a
single global scalar that preserves relative spread).

Reuses the cached sequence from results/reproduce/data/{word_list_key}/ for
each word list, so results are directly comparable to the existing
01_reproduce.py analyses. Requires a GPU (loads the full 8B-parameter
model) -- run via runai exec into an interactive job, not on a CPU-only
shell.
"""
import gc
import json
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from utils import load_model, setup_plotting, save_figure, build_word_to_color, Grid, set_seed
from word_lists import WORD_LISTS

WORD_LIST_KEY = "original_paper"
WORDS = WORD_LISTS[WORD_LIST_KEY]
WORD_TO_COLOR = build_word_to_color(WORDS)
SEQ_PATH = f"results/reproduce/data/{WORD_LIST_KEY}/sequence_Grid_{WORD_LIST_KEY}.json"
OUT_DIR = "results/exploratory"


def configure_for_word_list(word_list_key):
    """Reassign the word-list-dependent globals so main() can be run for a
    different WORD_LISTS vocabulary without editing module-level constants
    by hand (mirrors 01_reproduce.py's configure_for_word_list)."""
    global WORD_LIST_KEY, WORDS, WORD_TO_COLOR, SEQ_PATH
    WORD_LIST_KEY = word_list_key
    WORDS = WORD_LISTS[WORD_LIST_KEY]
    WORD_TO_COLOR = build_word_to_color(WORDS)
    SEQ_PATH = f"results/reproduce/data/{WORD_LIST_KEY}/sequence_Grid_{WORD_LIST_KEY}.json"


def build_word_token_ids(model, words):
    word_to_id = {}
    for w in words:
        try:
            word_to_id[w] = model.to_single_token(f" {w}")
        except AssertionError:
            word_to_id[w] = model.to_single_token(w)
    return word_to_id


def main(model):
    print(f"\n=== Running for WORD_LIST_KEY={WORD_LIST_KEY!r} ===")
    with open(SEQ_PATH) as f:
        sequence = json.load(f)
    print(f"Loaded sequence: {len(sequence)} tokens")

    n_layers = model.cfg.n_layers
    word_to_id = build_word_token_ids(model, WORDS)

    ids = [model.tokenizer.bos_token_id] + [word_to_id[w] for w in sequence]
    tokens = torch.tensor([ids], dtype=torch.long)

    final_hook = f"blocks.{n_layers - 1}.hook_resid_post"
    embed_hook = "blocks.0.hook_resid_pre"
    resid_pre_hooks = [f"blocks.{l}.hook_resid_pre" for l in range(n_layers)]
    print("Running forward pass...")
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=resid_pre_hooks + [final_hook])

    final_resid = cache[final_hook][0, 1:, :]   # drop BOS -> [seq_len, d_model]
    embed_resid = cache[embed_hook][0, 1:, :]   # drop BOS -> [seq_len, d_model]

    final_norms = final_resid.norm(dim=-1).cpu().numpy()
    embed_norms = embed_resid.norm(dim=-1).cpu().numpy()

    # [n_layers + 1, seq_len]: row l (l < n_layers) is blocks.{l}.hook_resid_pre's
    # norm (row 0 = raw token embeddings), last row is the final block's
    # hook_resid_post -- the residual stream right before ln_final/unembed.
    layer_norms = np.stack(
        [cache[h][0, 1:, :].norm(dim=-1).cpu().numpy() for h in resid_pre_hooks]
        + [final_norms],
        axis=0,
    )

    # Full-vocab embedding norm reference, from the raw embedding matrix.
    vocab_embed_norm_mean = model.W_E.norm(dim=-1).mean().item()

    positions = np.arange(1, len(final_norms) + 1)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(
        os.path.join(OUT_DIR, f"final_layer_norms_{WORD_LIST_KEY}.npz"),
        positions=positions, final_norms=final_norms, embed_norms=embed_norms,
        sequence=np.array(sequence),
        vocab_embed_norm_mean=vocab_embed_norm_mean,
        layer_norms=layer_norms,
    )

    # ── 16 random-walk sequences, one per starting grid word (mirrors the
    # accuracy-curve batch in 01_reproduce.py's main()) ─────────────────────
    side = round(len(WORDS) ** 0.5)
    assert side * side == len(WORDS), f"{len(WORDS)} words don't form a square grid"
    grid = Grid(words=WORDS, rows=side, cols=side)
    n_sequences = len(WORDS)
    set_seed(42)
    walk_sequences = grid.generate_batch(len(sequence), n_sequences)
    walk_tokens = torch.tensor(
        [[model.tokenizer.bos_token_id] + [word_to_id[w] for w in seq] for seq in walk_sequences],
        dtype=torch.long,
    )
    print(f"Running batched forward pass over {n_sequences} random-walk sequences...")
    with torch.no_grad():
        _, walk_cache = model.run_with_cache(walk_tokens, names_filter=[final_hook])
    walk_final_norms = walk_cache[final_hook][:, 1:, :].norm(dim=-1).cpu().numpy()  # [n_sequences, seq_len]

    np.savez(
        os.path.join(OUT_DIR, f"final_layer_norms_multi_seq_{WORD_LIST_KEY}.npz"),
        positions=positions, walk_final_norms=walk_final_norms,
        sequences=np.array(walk_sequences),
    )

    setup_plotting()
    plt.rcParams['text.usetex'] = False  # no latex binary on this pod

    # ── Plot 1: norm vs. position, full sequence ────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(positions, final_norms, linewidth=0.7, color="black", label="Final-layer resid_post norm")
    ax.axhline(embed_norms.mean(), color="tab:blue", linestyle="--", linewidth=1,
               label=f"Mean embedding norm (this seq's tokens) = {embed_norms.mean():.1f}")
    ax.axhline(vocab_embed_norm_mean, color="tab:orange", linestyle=":", linewidth=1,
               label=f"Mean embedding norm (full vocab W_E) = {vocab_embed_norm_mean:.1f}")
    ax.set_xlabel("Token position in sequence")
    ax.set_ylabel("L2 norm")
    ax.set_title(f"Final-layer activation norm vs. token position ({WORD_LIST_KEY})")
    ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=1.0, edgecolor="gray")
    save_figure(fig, OUT_DIR, f"final_layer_norm_vs_position_{WORD_LIST_KEY}.pdf")
    fig.savefig(os.path.join(OUT_DIR, f"final_layer_norm_vs_position_{WORD_LIST_KEY}.png"), dpi=150)
    print(f"Saved {OUT_DIR}/final_layer_norm_vs_position_{WORD_LIST_KEY}.png")

    # ── Plot 2: colored by uncontextualized embedding norm of current token ─
    # Raw W_E row norm per word -- looked up directly from the embedding
    # matrix, not from blocks.0.hook_resid_pre of any particular sequence
    # position (those are equal for this RoPE model with no processing, but
    # this makes the "uncontextualized" lookup explicit and sequence-
    # independent).
    word_embed_norm = {w: model.W_E[word_to_id[w]].norm().item() for w in WORDS}
    color_vals = [word_embed_norm[w] for w in sequence]

    fig2, ax2 = plt.subplots(figsize=(18, 4))
    ax2.plot(positions, final_norms, linewidth=0.4, color="gray", alpha=0.4, zorder=1)
    sc2 = ax2.scatter(positions, final_norms, c=color_vals, cmap="viridis", s=6, zorder=2, edgecolors="none")
    ax2.set_xlabel("Token position in sequence")
    ax2.set_ylabel("L2 norm")
    ax2.set_title(f"Final-layer norm, all {len(positions)} positions, "
                  f"colored by current token's embedding norm ({WORD_LIST_KEY})")
    cbar2 = fig2.colorbar(sc2, ax=ax2)
    cbar2.set_label("Embedding norm $\\|W_E[\\mathrm{token}]\\|$ (uncontextualized)")
    save_figure(fig2, OUT_DIR, f"final_layer_norm_by_word_{WORD_LIST_KEY}.pdf")
    fig2.savefig(os.path.join(OUT_DIR, f"final_layer_norm_by_word_{WORD_LIST_KEY}.png"), dpi=150)
    print(f"Saved {OUT_DIR}/final_layer_norm_by_word_{WORD_LIST_KEY}.png")

    # ── Plot 3: per-word distribution (boxplot-ish via scatter+mean) ───────
    word_to_norms = {w: [] for w in WORDS}
    for pos_idx, w in enumerate(sequence):
        word_to_norms[w].append(final_norms[pos_idx])

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    means, stds = [], []
    for i, w in enumerate(WORDS):
        vals = np.array(word_to_norms[w])
        means.append(vals.mean())
        stds.append(vals.std())
        jitter = (np.random.rand(len(vals)) - 0.5) * 0.3
        ax3.scatter(np.full(len(vals), i) + jitter, vals, s=8, alpha=0.5, color=WORD_TO_COLOR[w])
    ax3.errorbar(range(len(WORDS)), means, yerr=stds, fmt="_", color="black", capsize=3, markersize=15, linewidth=1)
    ax3.set_xticks(range(len(WORDS)))
    ax3.set_xticklabels(WORDS, rotation=45, ha="right", fontsize=8)
    ax3.set_ylabel("L2 norm (final layer)")
    ax3.set_title(f"Final-layer norm by current-token identity ({WORD_LIST_KEY})")
    save_figure(fig3, OUT_DIR, f"final_layer_norm_per_word_{WORD_LIST_KEY}.pdf")
    fig3.savefig(os.path.join(OUT_DIR, f"final_layer_norm_per_word_{WORD_LIST_KEY}.png"), dpi=150)
    print(f"Saved {OUT_DIR}/final_layer_norm_per_word_{WORD_LIST_KEY}.png")

    # ── Plot 4: heatmap, norm across (token position x layer) ──────────────
    # Log-scale color: the residual stream grows ~2 orders of magnitude from
    # layer 0 to the final layer (see growth-factor print below), so a linear
    # scale would make every early layer look uniformly near-zero.
    n_rows = layer_norms.shape[0]
    fig4, ax4 = plt.subplots(figsize=(18, 6))
    im = ax4.imshow(
        layer_norms, aspect="auto", origin="lower",
        norm=LogNorm(vmin=layer_norms.min(), vmax=layer_norms.max()),
        cmap="viridis",
        extent=[positions[0] - 0.5, positions[-1] + 0.5, -0.5, n_rows - 0.5],
    )
    ax4.set_xlabel("Token position in sequence")
    ax4.set_ylabel("Layer (resid_pre); top row = final resid_post")
    ax4.set_yticks(list(range(0, n_rows - 1, 4)) + [n_rows - 1])
    ax4.set_yticklabels([str(l) for l in range(0, n_rows - 1, 4)] + ["final"])
    ax4.set_title(f"L2 norm across token position and layer ({WORD_LIST_KEY})")
    cbar = fig4.colorbar(im, ax=ax4)
    cbar.set_label("L2 norm (log scale)")
    save_figure(fig4, OUT_DIR, f"final_layer_norm_heatmap_{WORD_LIST_KEY}.pdf")
    fig4.savefig(os.path.join(OUT_DIR, f"final_layer_norm_heatmap_{WORD_LIST_KEY}.png"), dpi=150)
    print(f"Saved {OUT_DIR}/final_layer_norm_heatmap_{WORD_LIST_KEY}.png")

    # ── Plot 5: average norm vs. position, across n_sequences random walks ─
    mean_norm = walk_final_norms.mean(axis=0)
    std_norm = walk_final_norms.std(axis=0)

    fig5, ax5 = plt.subplots(figsize=(10, 4))
    for row in walk_final_norms:
        ax5.plot(positions, row, linewidth=0.3, color="gray", alpha=0.3)
    ax5.plot(positions, mean_norm, color="black", linewidth=1.2, label="Mean final-layer norm")
    ax5.fill_between(positions, mean_norm - std_norm, mean_norm + std_norm,
                      alpha=0.15, color="gray", edgecolor="none", label="$\\pm$1 std")
    ax5.axhline(vocab_embed_norm_mean, color="tab:orange", linestyle=":", linewidth=1,
                label=f"Mean embedding norm (full vocab W_E) = {vocab_embed_norm_mean:.1f}")
    ax5.set_xlabel("Token position in sequence")
    ax5.set_ylabel("L2 norm")
    ax5.set_title(f"Final-layer norm vs. position, averaged over {n_sequences} random walks ({WORD_LIST_KEY})")
    ax5.legend(loc="upper left", fontsize=8, frameon=True, framealpha=1.0, edgecolor="gray")
    save_figure(fig5, OUT_DIR, f"final_layer_norm_vs_position_avg{n_sequences}_{WORD_LIST_KEY}.pdf")
    fig5.savefig(os.path.join(OUT_DIR, f"final_layer_norm_vs_position_avg{n_sequences}_{WORD_LIST_KEY}.png"), dpi=150)
    print(f"Saved {OUT_DIR}/final_layer_norm_vs_position_avg{n_sequences}_{WORD_LIST_KEY}.png")

    print(f"\nfinal_norms: mean={final_norms.mean():.2f} std={final_norms.std():.2f} "
          f"min={final_norms.min():.2f} max={final_norms.max():.2f} "
          f"(max/min={final_norms.max()/final_norms.min():.2f})")
    print(f"embed_norms (this seq): mean={embed_norms.mean():.2f} std={embed_norms.std():.4f}")
    print(f"embed_norm (full vocab W_E): mean={vocab_embed_norm_mean:.2f}")
    print(f"growth factor (final/embed): {final_norms.mean()/embed_norms.mean():.2f}x")

    # Drop references to this run's large GPU tensors before the next word
    # list starts -- otherwise per-iteration ActivationCache dicts (33 hook
    # points x 1400 positions x 4096 dims, plus the 16-sequence batch's
    # attention-pattern scratch buffers) accumulate across loop iterations
    # instead of being freed promptly, and a later iteration OOMs even
    # though each individual iteration's peak usage is roughly constant.
    del cache, walk_cache, final_resid, embed_resid, tokens, walk_tokens
    gc.collect()
    torch.cuda.empty_cache()


RUN_WORD_LIST_KEYS = [
    "original_paper",
    "text_numbers", "text_numbers_permuted",
    "two_digit_numbers", "two_digit_numbers_permuted",
    "grammar_vs_concrete_nouns",
]

if __name__ == "__main__":
    print("Loading model (this may take a few minutes)...")
    _model = load_model()
    for _key in RUN_WORD_LIST_KEYS:
        configure_for_word_list(_key)
        main(_model)
