"""Empirically checks the claim from the synthetic-grid neighbor-mixing
analysis: for a SINGLE, fixed permutation of tokens onto graph nodes (a
uniformly random permutation, verified below to NOT be a graph automorphism
of the 4x4 grid -- i.e. it does not preserve the grid's adjacency structure),
does the post-mixing PCA shape ever re-converge to something grid-like, at
any round?

"Grid-like" is measured via Procrustes disparity between the round-k PCA
projection (points ordered by TRUE (row, col) label) and the plain 4x4
integer grid, after the optimal rotation/reflection/uniform-scale/
translation alignment (scipy.spatial.procrustes). Disparity is in [0, 1];
0 = identical shape up to similarity transform, ~1 = no shape correlation.

Reuses the construction helpers from
03_neighbor_mixing-synthetic_grid_transformed.py rather than duplicating
them (loaded via importlib since that module's filename isn't a valid
Python identifier).
"""
import importlib.util
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial import procrustes

REPO = os.path.dirname(os.path.abspath(__file__))

spec = importlib.util.spec_from_file_location(
    "synth_mod", os.path.join(REPO, "03_neighbor_mixing-synthetic_grid_transformed.py")
)
synth_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(synth_mod)

from utils import Grid, setup_plotting, save_figure

PLOTS_DIR = "results/neighbor-mixing-synthetic-grid-transformed/"
MAX_ROUND = 60
SEED = 0


def random_permutation_non_automorphism(n, adjacency, seed):
    """A uniformly random permutation of range(n), verified to NOT be a
    graph automorphism of `adjacency` (i.e. A[i, j] != A[perm[i], perm[j]]
    for at least one pair). Automorphisms of a 4x4 grid graph are just the
    8 symmetries of the square (identity, 3 rotations, 4 reflections) out of
    16! total permutations, so a random draw essentially never collides --
    this just makes that guarantee explicit rather than assumed."""
    rng = np.random.default_rng(seed)
    attempt = 0
    while True:
        perm = rng.permutation(n)
        is_automorphism = all(
            adjacency[i, j] == adjacency[perm[i], perm[j]]
            for i in range(n) for j in range(n)
        )
        if not is_automorphism:
            return perm.tolist()
        attempt += 1
        print(f"[seed {seed}, attempt {attempt}] drew an automorphism by chance, redrawing...")
        seed += 1


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    setup_plotting()
    plt.rcParams['text.usetex'] = False

    # ── Same construction as the transformed-mixing script ───────────────────
    points_natural = synth_mod.make_grid_points_2d(synth_mod.GRID_ROWS, synth_mod.GRID_COLS)
    points_transformed = synth_mod.transform_2d(
        points_natural,
        rotate_degrees=synth_mod.ROTATE_DEGREES,
        stretch=synth_mod.STRETCH,
        translate=synth_mod.TRANSLATE,
    )
    embeddings_natural = synth_mod.lift_to_dmodel(points_transformed, synth_mod.D_EMBED, seed=42)

    node_ids = [str(i) for i in range(synth_mod.N_POINTS)]
    grid = Grid(words=node_ids, rows=synth_mod.GRID_ROWS, cols=synth_mod.GRID_COLS)
    adjacency_np = grid.build_adjacency_matrix()
    adjacency = torch.tensor(adjacency_np, dtype=torch.float32)

    # The single, fixed, verified-non-automorphism RANDOM permutation under test.
    permuted_order = random_permutation_non_automorphism(synth_mod.N_POINTS, adjacency_np, seed=SEED)
    print(f"random permutation (true idx at each node): {permuted_order}")
    embeddings_permuted = embeddings_natural[torch.tensor(permuted_order)]
    # true_label_at_node[node] = true flat index whose point sits at that node
    true_label_at_node = permuted_order
    # node_holding_true_label[true_idx] = which node holds that true point
    node_holding_true_label = [0] * synth_mod.N_POINTS
    for node, true_idx in enumerate(true_label_at_node):
        node_holding_true_label[true_idx] = node

    rounds = list(range(0, MAX_ROUND + 1))
    embs_by_round = synth_mod.run_mixing_rounds(embeddings_permuted, adjacency, rounds)

    # Reference shape: the plain, untransformed 4x4 integer grid, in true
    # (row, col) order -- Procrustes handles rotation/scale/translation, so
    # this is a fair "does it have grid topology at all" reference.
    reference = synth_mod.make_grid_points_2d(synth_mod.GRID_ROWS, synth_mod.GRID_COLS).numpy()

    disparities = []
    for r in rounds:
        projected = synth_mod.pca_2d(embs_by_round[r]).numpy()  # indexed by node
        # Reorder to true (row, col) label order: point at true index i sits
        # at node_holding_true_label[i].
        by_true_label = projected[node_holding_true_label]
        _, _, disparity = procrustes(reference, by_true_label)
        disparities.append(disparity)

    for r, d in zip(rounds, disparities):
        print(f"round {r:3d}: procrustes disparity = {d:.4f}")

    # ── PCA scatter panels for this same permutation, at rounds spanning the
    #    disparity curve's transition (0 -> ~30) plus a late-round check. ────
    panel_rounds = [0, 1, 2, 3, 5, 10, 15, 20, 30, 60]
    permuted_labels = [synth_mod.TRUE_LABELS[i] for i in permuted_order]
    n_rows, n_cols = 2, 5
    panel_fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4.2 * n_rows))
    axes_flat = axes.flatten()
    for ax, r in zip(axes_flat, panel_rounds):
        projected = synth_mod.pca_2d(embs_by_round[r]).numpy()
        title = "0 rounds mixing" if r == 0 else f"after {r} round{'s' if r != 1 else ''}\n(permuted graph)"
        synth_mod.plot_round_on_ax(ax, projected, grid, permuted_labels, title)
        synth_mod.save_round_plotly(projected, grid, permuted_labels, r)
    for ax in axes_flat[len(panel_rounds):]:
        ax.axis("off")
    panel_fig.suptitle(
        "PCA of the SAME non-automorphism-permutation mixing across rounds\n"
        "(random non-automorphism permutation; rounds chosen to span the grid-likeness transition)"
    )
    save_figure(panel_fig, PLOTS_DIR, "grid_likeness_pca_panels.png")
    print("Saved grid_likeness_pca_panels")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rounds, disparities, marker="o", markersize=3, color="black", linewidth=1.0)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="chance-level shape correlation")
    ax.set_xlabel("mixing round")
    ax.set_ylabel("Procrustes disparity to true grid\n(0 = identical shape, higher = less grid-like)")
    ax.set_title(
        "Grid-likeness of the permuted-graph mixed shape across rounds\n"
        "(single fixed non-automorphism permutation: uniformly random)",
        fontsize=10,
    )
    ax.legend(loc="best", fontsize=8)
    save_figure(fig, PLOTS_DIR, "grid_likeness_vs_round.png")
    print("\nSaved grid_likeness_vs_round")


if __name__ == "__main__":
    main()
