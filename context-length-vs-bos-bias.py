"""Checks whether the model's next-token distribution over a word list's
own N candidate words, after a LONG random-walk context, is still biased
toward the BOS-only (zero-context) prior from bos_only_next_token_
distribution.py -- i.e. does the model keep favoring the same "default"
words it would guess with no context at all, or does the in-context/
induction mechanism override that prior as context accumulates?

Not answerable from any existing cache: 01_reproduce.py's cached
"accuracy" is the SUM of predicted probability over each position's
VALID grid neighbors only (a single scalar per position) -- it never
keeps each of the N words' probabilities separately, which is what's
needed here. This computes and caches that full per-word breakdown via a
fresh forward pass (same per-sequence loop / tokenization as
get_model_accuracies_by_id, just keeping every word's probability instead
of summing over valid neighbors).

Method: same 16 set_seed(42) random-walk sequences (SEQ_LEN=1400) as
01_reproduce.py. At each of several log-spaced context-length checkpoints
k (same convention as morphology-grid-evolution-layers-vs-seqlen.py),
average the model's predicted probability on each of the N words over the
last min(N_LOOKBACK, k) query positions and all 16 sequences -- an
"empirical next-word marginal" vector, analogous to how class means
average ACTIVATIONS over the same window, just for output PROBABILITIES
instead. Compare this N-dim vector to the BOS-only prior vector (reused
from bos_only_next_token_distribution.py's cache) via both Pearson
correlation and cosine similarity, and plot both against context length.

If bias toward the BOS prior persists, these should stay high even at
long context; if the induction mechanism fully takes over, they should
decay toward 0 (or something unrelated to the prior) as context grows.

Requires a CUDA GPU with >= 48 GB memory (Llama-3.1-8B via
TransformerLens) -- this repo session has no GPU, so this must be run on
a GPU machine; bring results/reproduce/data/{key}/bos_bias_vs_seqlen.npz
back to replot without the model.
"""
import importlib.util
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from utils import Grid, set_seed, load_model, setup_plotting, save_figure
from word_lists import WORD_LISTS

REPO = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reproduce_mod = _load_module("reproduce_01", "01_reproduce.py")
bos_mod = _load_module("bos_only_next_token_distribution", "bos_only_next_token_distribution.py")

SWEEP_KEYS = [
    "text_numbers", "text_numbers_permuted",
    "two_digit_numbers", "two_digit_numbers_permuted",
    "morphology", "morphology_permuted",
]
N_LOOKBACK = reproduce_mod.N_LOOKBACK
SEQ_LEN = reproduce_mod.SEQ_LEN

N_CHECKPOINTS = 8
CHECKPOINTS = sorted(set(
    np.geomspace(N_LOOKBACK, SEQ_LEN, N_CHECKPOINTS).astype(int).tolist() + [SEQ_LEN]
))


def cache_path(word_list_key):
    return f"results/reproduce/data/{word_list_key}/bos_bias_vs_seqlen.npz"


def compute_per_word_probs(model, sequence, word_to_id, words):
    """[seq_len, N] softmax probability of each of the N words at every
    query position, mirroring get_model_accuracies_by_id's tokenization/
    forward pass but keeping every word's probability instead of summing
    over just the valid-neighbor subset."""
    tokens = reproduce_mod.tokenize_sequence_by_id(model, sequence, word_to_id)
    logits = model.run_with_hooks(tokens.to(model.cfg.device))
    probs = torch.softmax(logits, dim=-1)[0, 1:, :]  # remove BOS position
    word_ids = torch.tensor([word_to_id[w] for w in words])
    return probs[:, word_ids].cpu().numpy()  # [seq_len, N]


def compute_bias_curve(model, word_list_key):
    words = WORD_LISTS[word_list_key]
    grid = Grid(words=words, rows=int(round(len(words) ** 0.5)), cols=int(round(len(words) ** 0.5)))
    word_to_id = reproduce_mod.build_word_token_ids(model, words)

    set_seed(42)
    sequences = grid.generate_batch(SEQ_LEN, len(words))

    all_probs = []  # list of [seq_len, N]
    for seq in tqdm.tqdm(sequences, desc=f"{word_list_key}: sequences"):
        all_probs.append(compute_per_word_probs(model, seq, word_to_id, words))
    all_probs = np.stack(all_probs, axis=0)  # [n_seqs, seq_len, N]

    bos_words, bos_probs_all = bos_mod.load_or_compute(model, word_list_key)
    assert bos_words == words
    bos_probs = bos_probs_all  # [N]

    pearson_by_k, cosine_by_k, word_probs_by_k = [], [], []
    for k in CHECKPOINTS:
        window = min(N_LOOKBACK, k)
        window_probs = all_probs[:, k - window:k, :].mean(axis=(0, 1))  # [N]
        word_probs_by_k.append(window_probs)
        pearson_by_k.append(float(np.corrcoef(window_probs, bos_probs)[0, 1]))
        cosine_by_k.append(float(
            np.dot(window_probs, bos_probs)
            / (np.linalg.norm(window_probs) * np.linalg.norm(bos_probs) + 1e-12)
        ))
    word_probs_by_k = np.stack(word_probs_by_k, axis=0)  # [n_checkpoints, N]

    path = cache_path(word_list_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, checkpoints=np.array(CHECKPOINTS), words=np.array(words),
             pearson=np.array(pearson_by_k), cosine=np.array(cosine_by_k),
             word_probs_by_checkpoint=word_probs_by_k)
    print(f"Cached {path}")
    return np.array(CHECKPOINTS), np.array(pearson_by_k), np.array(cosine_by_k), words, word_probs_by_k


def load_or_compute_bias_curve(model, word_list_key):
    path = cache_path(word_list_key)
    if os.path.exists(path) and "word_probs_by_checkpoint" in np.load(path).files:
        data = np.load(path, allow_pickle=True)
        return (data["checkpoints"], data["pearson"], data["cosine"],
                list(data["words"]), data["word_probs_by_checkpoint"])
    if model is None:
        raise RuntimeError(
            f"No up-to-date cache at {path} (missing word_probs_by_checkpoint -- "
            "regenerate on a GPU machine) and no model loaded to compute it."
        )
    return compute_bias_curve(model, word_list_key)


def plot_bias_curves(curves_by_key, out_dir, filename="context_length_vs_bos_bias"):
    fig, (ax_p, ax_c) = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, (checkpoints, pearson, cosine) in curves_by_key.items():
        ax_p.plot(checkpoints, pearson, marker="o", markersize=3, linewidth=1.2, label=key)
        ax_c.plot(checkpoints, cosine, marker="o", markersize=3, linewidth=1.2, label=key)

    for ax, title in [(ax_p, "Pearson correlation"), (ax_c, "Cosine similarity")]:
        ax.set_xscale("log")
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.set_xlabel("Context length (window end position)")
        ax.set_ylabel(f"{title} to BOS-only prior")
        ax.set_title(title, fontsize=10)
    ax_p.legend(loc="best", frameon=True, framealpha=1.0, edgecolor="gray", fontsize=7)

    fig.suptitle("Does the long-context next-word distribution stay biased toward the BOS-only prior?", fontsize=11)
    plt.tight_layout()
    save_figure(fig, out_dir, f"{filename}.pdf")
    print(f"Saved {filename}")


def plot_average_distribution(word_list_key, words, long_context_probs, out_dir,
                               filename="average_next_token_distribution"):
    """Average next-token probability at the longest cached context
    length (averaged over the 16 random-walk sequences and the trailing
    N_LOOKBACK-position window), one bar per word. BOS-only prior is
    NOT plotted alongside -- its probabilities are orders of magnitude
    smaller (a handful of candidate words vs. the full ~128k vocab with
    no context to narrow it down), so on a shared axis it renders as
    indistinguishable from zero and just wastes a legend entry; the
    bos-bias correlation curves already give the proper (scale-invariant)
    comparison to the BOS prior."""
    total_mass = float(long_context_probs.sum())
    order = np.argsort(-long_context_probs)
    sorted_words = [words[i] for i in order]
    sorted_long = long_context_probs[order]

    print(f"Total probability mass on the {len(words)} '{word_list_key}' words (long context): {total_mass:.3e}")

    x = np.arange(len(words))
    fig, ax = plt.subplots(figsize=(max(6.5, 0.45 * len(words)), 4.2))
    ax.bar(x, sorted_long, color="#377eb8")
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_words, rotation=45, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title(
        f"Average next-token distribution, long context ({word_list_key})\n"
        f"(total mass on these {len(words)} words: {total_mass:.3e})",
        fontsize=10,
    )
    plt.tight_layout()

    save_figure(fig, out_dir, f"{filename}.pdf")
    print(f"Saved {word_list_key}/{filename}")


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False

    def is_up_to_date(key):
        path = cache_path(key)
        return os.path.exists(path) and "word_probs_by_checkpoint" in np.load(path).files

    model = None
    if not all(is_up_to_date(k) for k in SWEEP_KEYS):
        model = load_model()

    curves_by_key = {}
    for key in SWEEP_KEYS:
        print(f"\n=== {key} ===")
        checkpoints, pearson, cosine, words, word_probs_by_k = load_or_compute_bias_curve(model, key)
        curves_by_key[key] = (checkpoints, pearson, cosine)
        for k, p, c in zip(checkpoints, pearson, cosine):
            print(f"  context={k:5d}  pearson={p:+.4f}  cosine={c:+.4f}")

        plot_average_distribution(
            key, words, word_probs_by_k[-1],  # last checkpoint = longest context
            f"results/reproduce/plots/{key}",
        )

    plot_bias_curves(curves_by_key, "results/reproduce/plots/bos_bias_comparison")


if __name__ == "__main__":
    main()
