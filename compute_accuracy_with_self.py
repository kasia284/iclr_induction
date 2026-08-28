"""Recomputes real accuracy for the grid/ring word lists swept by
starting_geometry_vs_accuracy_scatter.py / -ring.py, under an ALTERNATIVE
definition of "correct": probability mass on the CURRENT token itself now
also counts, not just probability mass on its graph neighbors (utils.
get_model_accuracies's / 01_reproduce.py's get_model_accuracies_by_id's new
include_self=True option -- "staying put" counts as a correct prediction,
in addition to moving to a neighbor).

Regenerates the EXACT SAME random-walk sequences the original
accuracies_Grid_{key}.npz / accuracies_Ring_{key}.npz caches were built
from (set_seed(42) + {grid,ring}.generate_batch(SEQ_LEN, n_sequences),
byte-for-byte matching 01_reproduce.py's / 01_reproduce_ring.py's own
pipelines), so the two accuracy definitions are directly comparable
position-by-position and sequence-by-sequence -- only the probability-mass
SUM per position differs, never which walk was taken.

Writes to SEPARATE cache files (accuracies_Grid_{key}_with_self.npz /
accuracies_Ring_{key}_with_self.npz) so the original "neighbors only"
caches -- which many other scripts in this repo read from -- are never
touched or overwritten.

Requires a CUDA GPU with >=48GB memory (loads meta-llama/Llama-3.1-8B via
utils.load_model) -- same requirement as 01_reproduce.py / 01_reproduce_ring.py.
This machine has no GPU, so this script is written but NOT run here; run it
on a GPU machine, then re-run starting_geometry_vs_accuracy_scatter.py
--with-self / starting_geometry_vs_accuracy_scatter_ring.py --with-self
(which just read these new cache files) wherever's convenient afterward.

Usage:
    python compute_accuracy_with_self.py                # both grid + ring word lists
    python compute_accuracy_with_self.py --grid-only
    python compute_accuracy_with_self.py --ring-only
"""
import importlib.util
import os
import sys

import numpy as np
import tqdm

from utils import Grid, Ring, set_seed, load_model, get_model_accuracies
from word_lists import WORD_LISTS

REPO = os.path.dirname(os.path.abspath(__file__))
SEQ_LEN = 1400  # matches 01_reproduce.py / 01_reproduce_ring.py

GRID_DATA_DIR = "results/reproduce/data"
RING_DATA_DIR = "results/reproduce_ring/data"


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reproduce_mod = _load_module("reproduce_01", "01_reproduce.py")
ring_mod = _load_module("reproduce_ring", "01_reproduce_ring.py")
scatter_mod = _load_module("starting_geometry_scatter", "starting_geometry_vs_accuracy_scatter.py")

RING_SWEEP_KEYS = ring_mod.RING_SWEEP_KEYS


def grid_keys():
    """The same word lists starting_geometry_vs_accuracy_scatter.py sweeps
    (morphology/text_numbers/two_digit_numbers families + permutations),
    restricted to ones with an existing "neighbors only" accuracy cache."""
    keys = []
    for key in scatter_mod.family_keys():
        acc_path = os.path.join(GRID_DATA_DIR, key, f"accuracies_Grid_{key}.npz")
        if os.path.exists(acc_path):
            keys.append(key)
    return keys


def compute_grid_accuracy_with_self(model, word_list_key):
    words = WORD_LISTS[word_list_key]
    side = int(round(len(words) ** 0.5))
    grid = Grid(words=words, rows=side, cols=side)
    word_to_id = reproduce_mod.build_word_token_ids(model, words)

    set_seed(42)
    sequences = grid.generate_batch(SEQ_LEN, len(words))  # matches 01_reproduce.py's N_SEQUENCES = len(WORDS)

    all_accs = []
    for seq in tqdm.tqdm(sequences, desc=f"{word_list_key} (grid, with-self)"):
        all_accs.append(reproduce_mod.get_model_accuracies_by_id(
            model, grid, seq, word_to_id, include_self=True))
    return np.array(all_accs)


def compute_ring_accuracy_with_self(model, word_list_key):
    words = WORD_LISTS[word_list_key]
    ring = Ring(words)
    base_key = ring_mod.get_base_word_list_key(word_list_key)
    canonical_ring = Ring(WORD_LISTS[base_key])

    set_seed(42)
    sequences = ring.generate_batch(SEQ_LEN, len(words))

    graph_accs, semantic_accs = [], []
    for seq in tqdm.tqdm(sequences, desc=f"{word_list_key} (ring, with-self)"):
        graph_accs.append(get_model_accuracies(model, ring, seq, include_self=True))
        semantic_accs.append(get_model_accuracies(model, canonical_ring, seq, include_self=True))
    return np.array(graph_accs), np.array(semantic_accs)


def main():
    argv = sys.argv[1:]
    do_grid = "--ring-only" not in argv
    do_ring = "--grid-only" not in argv

    g_keys = grid_keys() if do_grid else []
    r_keys = ([k for k in RING_SWEEP_KEYS
               if os.path.exists(os.path.join(RING_DATA_DIR, k, f"accuracies_Ring_{k}.npz"))]
              if do_ring else [])

    print(f"Grid word lists: {len(g_keys)}. Ring word lists: {len(r_keys)}.")
    if not g_keys and not r_keys:
        raise RuntimeError("Nothing to do -- no word lists with an existing accuracy cache found.")

    print("Loading model (meta-llama/Llama-3.1-8B)...")
    model = load_model()

    for key in g_keys:
        out_path = os.path.join(GRID_DATA_DIR, key, f"accuracies_Grid_{key}_with_self.npz")
        if os.path.exists(out_path):
            print(f"Skipping {key} (grid): already cached at {out_path}")
            continue
        all_accs = compute_grid_accuracy_with_self(model, key)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez(out_path, all_accs=all_accs)
        print(f"Cached {out_path}")

    for key in r_keys:
        out_path = os.path.join(RING_DATA_DIR, key, f"accuracies_Ring_{key}_with_self.npz")
        if os.path.exists(out_path):
            print(f"Skipping {key} (ring): already cached at {out_path}")
            continue
        graph_accs, semantic_accs = compute_ring_accuracy_with_self(model, key)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez(out_path, graph_accs=graph_accs, semantic_accs=semantic_accs)
        print(f"Cached {out_path}")


if __name__ == "__main__":
    main()
