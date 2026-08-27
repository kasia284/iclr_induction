"""Quantifies how "geometrically similar" different grid-position
permutations of the same word set are to the canonical (unpermuted) layout,
using the SAME Dirichlet energy / distance correlation machinery
(utils.compute_dirichlet_energy / compute_distance_correlation) normally
used to compare model activations against the graph -- except here both
sides are graph structure, no model/activations involved:

  - Dirichlet energy: canonical-layout coordinates vs. the PERMUTED layout's
    adjacency matrix, reindexed into canonical word order. Low = tokens that
    are adjacent under the permuted layout were already close together
    under the canonical layout (structurally similar graphs); high = the
    permutation connects tokens that used to be far apart.

  - Distance correlation: graph shortest-path (hop-count) distance in the
    canonical graph vs. graph shortest-path distance in the permuted graph,
    both reindexed into canonical word order (utils.
    compute_distance_correlation_graph). Graph distance on BOTH sides --
    never Euclidean/Manhattan on embedded coordinates. High = the
    permutation preserves overall pairwise hop-distances between tokens,
    not just direct adjacency.

Also reports, per word list, the distance correlation between each word's
STATIC (context-free) embedding -- straight out of the model's embedding
matrix W_E, no forward pass -- and that word list's OWN grid coordinates
(i.e. the graph its random-walk sequences are actually generated on).
Unlike the two metrics above, this one DOES involve the model: how much do
the raw pretrained embeddings already align with a given permutation's
layout, before any in-context adaptation? Uses the standard
(Euclidean-vs-Manhattan) utils.compute_distance_correlation, matching every
other activation-vs-graph DC computation in this repo (e.g.
gridness-vs-accuracy-scatter.py's compute_final_dc), just with W_E instead
of final-layer activations. Reuses real-embeddings-nearest-neighbors.py's
load_embedding_only_model, so this only reads W_E off disk -- CPU-only, no
GPU / forward pass.

Word-list "families" (canonical + its permuted/rand variants, e.g.
text_numbers / text_numbers_permuted / text_numbers_rand1-5) are the same
groups gridness-vs-accuracy-scatter.py's assign_families operates on:
same underlying word SET, different grid-position assignment. The canonical
member of a family is its shortest key name (matches assign_families'
convention).
"""
import importlib.util
import json
import math
import os

import numpy as np
import torch

from utils import Grid, compute_dirichlet_energy, compute_distance_correlation, compute_distance_correlation_graph
from word_lists import WORD_LISTS

REPO = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "results/reproduce/data/permutation_graph_similarity"

BASE_KEY = "text_numbers"


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reproduce_mod = _load_module("reproduce_01", "01_reproduce.py")  # reuse get_grid_coords
nn_mod = _load_module("real_embeddings_nearest_neighbors", "real-embeddings-nearest-neighbors.py")  # load_embedding_only_model
dot_product_mod = _load_module("activation_unembedding_dot_product", "activation-unembedding-dot-product.py")  # to_single_token


def family_members(base_key):
    """Canonical key (shortest name) + sorted list of its permuted/rand
    siblings -- every WORD_LISTS key with the exact same word SET."""
    word_set = frozenset(WORD_LISTS[base_key])
    members = [k for k in WORD_LISTS if frozenset(WORD_LISTS[k]) == word_set]
    canonical = min(members, key=len)
    variants = sorted(k for k in members if k != canonical)
    return canonical, variants


def make_grid(word_list_key):
    words = WORD_LISTS[word_list_key]
    side = math.isqrt(len(words))
    if side * side != len(words):
        raise ValueError(f"{word_list_key!r} has {len(words)} words, not a perfect square.")
    return Grid(words=words, rows=side, cols=side)


def adjacency_in_order(grid, words):
    """Like reproduce_mod.get_grid_coords, but for the adjacency matrix:
    an [n, n] binary matrix with row/col i corresponding to `words[i]`,
    regardless of `grid`'s own internal word ordering. Built off
    get_valid_next_words (word-identity based), so it works for any Grid /
    Torus / Ring instance."""
    index = {w: i for i, w in enumerate(words)}
    n = len(words)
    A = np.zeros((n, n))
    for w in words:
        i = index[w]
        for neighbor in grid.get_valid_next_words(w):
            A[i, index[neighbor]] = 1
    return A


def compute_static_embedding_dc(model, word_list_key):
    """Distance correlation between each word's static W_E embedding and
    its OWN grid coordinates -- the graph THIS word list's random walks are
    generated on, not the family's canonical layout."""
    words = WORD_LISTS[word_list_key]
    grid = make_grid(word_list_key)
    grid_coords = reproduce_mod.get_grid_coords(grid, words)
    token_ids = torch.tensor(
        [dot_product_mod.to_single_token(model.tokenizer, w) for w in words], dtype=torch.long
    )
    static_embeddings = model.W_E[token_ids].numpy()
    return compute_distance_correlation(static_embeddings, grid_coords)


def main():
    canonical_key, variant_keys = family_members(BASE_KEY)
    canonical_words = WORD_LISTS[canonical_key]
    canonical_grid = make_grid(canonical_key)
    canonical_coords = reproduce_mod.get_grid_coords(canonical_grid, canonical_words)
    canonical_adjacency = canonical_grid.build_adjacency_matrix()  # already in canonical_words order

    print(f"Family of {canonical_key!r}: {len(variant_keys)} variant(s) -- {variant_keys}\n")

    print("Loading W_E (embedding-only, CPU)...")
    embed_model = nn_mod.load_embedding_only_model()

    results = {}

    # Identity reference row: canonical graph against itself. Distance
    # correlation is exactly 1.0 here (both sides are the identical
    # shortest-path distance matrix) -- unlike the Euclidean-vs-Manhattan
    # version, this is a real ceiling, confirming the metric is well-formed.
    results[canonical_key] = dict(
        dirichlet_energy=compute_dirichlet_energy(canonical_coords, canonical_adjacency),
        distance_correlation=compute_distance_correlation_graph(canonical_adjacency, canonical_adjacency),
        static_embedding_distance_correlation=compute_static_embedding_dc(embed_model, canonical_key),
    )

    for key in variant_keys:
        variant_grid = make_grid(key)
        variant_adjacency = adjacency_in_order(variant_grid, canonical_words)
        results[key] = dict(
            dirichlet_energy=compute_dirichlet_energy(canonical_coords, variant_adjacency),
            distance_correlation=compute_distance_correlation_graph(canonical_adjacency, variant_adjacency),
            static_embedding_distance_correlation=compute_static_embedding_dc(embed_model, key),
        )

    print(f"{'word list':<28} {'dirichlet energy':>18} {'distance correlation':>22} {'static embedding DC':>22}")
    print("-" * 94)
    for key, r in results.items():
        marker = " (canonical)" if key == canonical_key else ""
        print(f"{key:<28} {r['dirichlet_energy']:>18.4f} {r['distance_correlation']:>22.4f} "
              f"{r['static_embedding_distance_correlation']:>22.4f}{marker}")

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, f"{canonical_key}.json")
    with open(out_path, "w") as f:
        json.dump(dict(canonical_key=canonical_key, results=results), f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
