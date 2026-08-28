"""For each word list in RING_SWEEP_KEYS (the days_of_week family), caches
per-node mean activations at layer 26 (the only layer 01_reproduce_ring.py
tracks -- unlike the grid task's 6-layer sweep) at each of several
log-spaced sequence-length checkpoints, so ring-geometry metrics
(distance correlation, Dirichlet energy) can be tracked as they evolve
with context length. Ring analog of morphology_grid_evolution_layers_vs_seqlen.py, minus the layer dimension.

Uses the exact same N_LOOKBACK/SEQ_LEN (50 / 1400) as 01_reproduce_ring.py,
so CHECKPOINTS works out identical to the grid task's
[50, 97, 189, 369, 718, 1400] -- directly comparable context lengths
across experiments.

Requires one full forward pass per word list, caching hook_resid_pre at
layer 26 for every position of the sequence (not just the tail), so needs
a GPU. Results cached per word list to results/reproduce_ring/data/{key}/
ring_evolution_seqlen.npz, so re-running the plots alone doesn't require
the model.

Usage:
    python ring_evolution_seqlen.py                   # all RING_SWEEP_KEYS
    python ring_evolution_seqlen.py <key> [<key> ...]  # only these keys
"""
import importlib.util
import os
import sys

import matplotlib.pyplot as plt  # noqa: F401 -- unused, but must import before
# torch: torch bundles its own libstdc++ that otherwise shadows the system
# one and breaks matplotlib's C extension (CXXABI mismatch) if torch loads first.
import numpy as np
import torch
import tqdm

from utils import Ring, set_seed, load_model, compute_class_means_batch
from word_lists import WORD_LISTS

REPO = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reproduce_mod = _load_module("reproduce_01", "01_reproduce.py")
ring_mod = _load_module("reproduce_ring", "01_reproduce_ring.py")

RING_SWEEP_KEYS = ring_mod.RING_SWEEP_KEYS
N_LOOKBACK = ring_mod.N_LOOKBACK
SEQ_LEN = ring_mod.SEQ_LEN
LAYER = ring_mod.LAYER

N_CHECKPOINTS = 6
CHECKPOINTS = sorted(set(
    np.geomspace(N_LOOKBACK, SEQ_LEN, N_CHECKPOINTS).astype(int).tolist() + [SEQ_LEN]
))[:N_CHECKPOINTS]


def cache_path(word_list_key):
    return os.path.join("results/reproduce_ring/data", word_list_key, "ring_evolution_seqlen.npz")


def _cell_key(k):
    return f"k{k}"


def compute_class_means_checkpoints(model, word_list_key):
    words = WORD_LISTS[word_list_key]
    ring = Ring(words)
    word_to_id = reproduce_mod.build_word_token_ids(model, words)

    # Same seed + generation call as 01_reproduce_ring.py's main() for this
    # word list, so this is the exact same batch of random walks used
    # elsewhere (just cached with the full sequence, not only the tail).
    set_seed(42)
    sequences = ring.generate_batch(SEQ_LEN, len(words))

    tokens = reproduce_mod.tokenize_sequences_by_id_batch(model, sequences, word_to_id)
    name = f"blocks.{LAYER}.hook_resid_pre"
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens.to(model.cfg.device), names_filter=[name])
    acts = cache[name][:, 1:, :].cpu()  # [batch, SEQ_LEN, d_model], BOS position removed
    del cache
    torch.cuda.empty_cache()

    class_means = {}
    for k in tqdm.tqdm(CHECKPOINTS, desc=f"{word_list_key}: checkpoints"):
        window = min(N_LOOKBACK, k)
        sequences_window = [seq[k - window:k] for seq in sequences]
        activations_window = acts[:, k - window:k, :]
        class_means[k] = compute_class_means_batch(
            activations_window, sequences_window, words, window
        ).numpy()
    del acts

    path = cache_path(word_list_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, checkpoints=np.array(CHECKPOINTS), **{_cell_key(k): class_means[k] for k in CHECKPOINTS})
    print(f"Cached {path}")
    return class_means


def load_or_compute_class_means(model, word_list_key):
    path = cache_path(word_list_key)
    if os.path.exists(path):
        data = np.load(path)
        return {k: data[_cell_key(k)] for k in CHECKPOINTS}
    if model is None:
        raise RuntimeError(f"No cache at {path} and no model loaded to compute it.")
    return compute_class_means_checkpoints(model, word_list_key)


def main():
    requested = sys.argv[1:]
    if requested:
        unknown = [k for k in requested if k not in RING_SWEEP_KEYS]
        if unknown:
            raise SystemExit(f"Not in RING_SWEEP_KEYS: {unknown}")
        keys = requested
    else:
        keys = RING_SWEEP_KEYS

    model = None
    if not all(os.path.exists(cache_path(k)) for k in keys):
        model = load_model()

    for word_list_key in keys:
        print(f"\n=== {word_list_key} ===")
        load_or_compute_class_means(model, word_list_key)


if __name__ == "__main__":
    main()
