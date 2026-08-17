"""Checks the model's next-token probability distribution over each word
list's own words, when the input sequence is JUST the BOS token -- the
pure unconditional prior, zero context, "sequence length 0" limit of
every accuracy curve elsewhere in the repo.

Not answerable from any existing cache: every cached accuracy/activation
file starts from the first REAL word in the random walk (accuracy is
defined per-position starting after the first word of context), not the
bare BOS position -- so this needs a fresh forward pass on [BOS] alone.
Uses build_word_token_ids (01_reproduce.py, imported via importlib since
its filename isn't a valid module identifier) for the same word-to-token
alignment used everywhere else in the repo.

Loops over every WORD_LISTS key (SWEEP_KEYS), loading the model ONCE and
reusing it across all lists -- the BOS-only forward pass is identical
regardless of word list (input is just [BOS]), so only the token-id
lookup and probability readout differ per key. Skips any word list with a
word that isn't a single token (e.g. multilingual_father, chemical_
elements), same convention as elsewhere in the repo. Per-key caching
means it's safe to interrupt/re-run -- only missing keys trigger a model
load.

Requires a CUDA GPU with >= 48 GB memory (Llama-3.1-8B via
TransformerLens) -- this repo session has no GPU available, so this must
be run on a GPU machine; bring results/reproduce/data/{key}/
bos_only_next_token_probs.npz back to replot without the model.
"""
import importlib.util
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import load_model, setup_plotting, save_figure
from word_lists import WORD_LISTS

REPO = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reproduce_mod = _load_module("reproduce_01", "01_reproduce.py")

SWEEP_KEYS = list(WORD_LISTS.keys())

# BOS-only logits don't depend on word_list_key at all (input is just
# [BOS] regardless), so compute them ONCE and reuse across every key.
_BOS_PROBS_CACHE = {}


def cache_path(word_list_key):
    return f"results/reproduce/data/{word_list_key}/bos_only_next_token_probs.npz"


def get_bos_probs(model):
    """Full-vocab softmax probability distribution after just [BOS],
    computed once per process and reused for every word list."""
    if "probs" not in _BOS_PROBS_CACHE:
        bos_id = model.tokenizer.bos_token_id
        tokens = torch.tensor([[bos_id]], dtype=torch.long).to(model.cfg.device)
        with torch.no_grad():
            logits = model(tokens)  # [1, 1, d_vocab]
        _BOS_PROBS_CACHE["probs"] = torch.softmax(logits[0, -1, :].float(), dim=-1).cpu().numpy()
    return _BOS_PROBS_CACHE["probs"]


def compute_bos_only_distribution(model, word_list_key):
    words = WORD_LISTS[word_list_key]
    word_to_id = reproduce_mod.build_word_token_ids(model, words)  # raises ValueError if not single-token
    probs = get_bos_probs(model)
    word_probs = np.array([probs[word_to_id[w]] for w in words])

    path = cache_path(word_list_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, words=np.array(words), probs=word_probs)
    print(f"Cached {path}")
    return words, word_probs


def load_or_compute(model, word_list_key):
    path = cache_path(word_list_key)
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        return list(data["words"]), data["probs"]
    if model is None:
        raise RuntimeError(f"No cache at {path} and no model loaded to compute it.")
    return compute_bos_only_distribution(model, word_list_key)


def plot_distribution(words, word_probs, word_list_key):
    total_mass = float(word_probs.sum())
    order = np.argsort(-word_probs)
    sorted_words = [words[i] for i in order]
    sorted_probs = word_probs[order]

    print(f"Total probability mass on the {len(words)} '{word_list_key}' words: {total_mass:.3e}")
    for w, p in zip(sorted_words, sorted_probs):
        print(f"  {w:12s} {p:.3e}")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(sorted_words, sorted_probs, color="#377eb8")
    ax.set_ylabel("P(token | BOS)")
    ax.set_title(
        f"Next-token distribution over '{word_list_key}' words, BOS-only input\n"
        f"(total mass on these {len(words)} words: {total_mass:.3e})",
        fontsize=10,
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plots_dir = f"results/reproduce/plots/{word_list_key}"
    save_figure(fig, plots_dir, "bos_only_next_token_distribution.pdf")
    print(f"Saved {word_list_key}/bos_only_next_token_distribution")


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False

    model = None
    if not all(os.path.exists(cache_path(k)) for k in SWEEP_KEYS):
        model = load_model()

    succeeded, skipped = [], []
    for i, key in enumerate(SWEEP_KEYS, 1):
        print(f"\n=== [{i}/{len(SWEEP_KEYS)}] {key} ===")
        try:
            words, word_probs = load_or_compute(model, key)
        except ValueError as e:
            print(f"Skipping {key}: {e}")
            skipped.append(key)
            continue
        plot_distribution(words, word_probs, key)
        succeeded.append(key)

    print(f"\nDone. {len(succeeded)} succeeded, {len(skipped)} skipped.")
    if skipped:
        print("Skipped:", ", ".join(skipped))


if __name__ == "__main__":
    main()
