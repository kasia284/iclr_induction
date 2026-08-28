"""Same scatter as gridness_vs_accuracy_scatter.py (final-layer distance
correlation vs. real accuracy, one point per word list), but with each
point's actual final-layer PCA scatter (grid edges + colored word
markers, no axes/title -- same rendering as everywhere else in the repo
via 01_reproduce.py's plot_class_mean_pca) embedded as a small thumbnail
image next to it, connected by a thin line -- so you can see at a glance
whether a high/low distance-correlation point's underlying geometry
actually looks grid-like or scrambled, instead of just trusting the
scalar summary.

Thumbnails for all 42 cached word lists are included in one very large
figure (an explicit tradeoff: a curated subset would avoid overlap but
leave out data -- some overlap between nearby thumbnails in dense regions
of the scatter is expected here). Offset direction is cycled through 4
diagonal directions by point index to reduce (not eliminate) systematic
overlap between neighboring points.

Pure cache reads (final-layer class means + accuracy curves from
01_reproduce.py's cache), no GPU / model needed. Reuses discover_sweep_
keys / compute_final_dc / load_real_full_context_accuracy / assign_
families / make_grid from gridness_vs_accuracy_scatter.py (imported via
importlib, consistent with how sibling scripts in this repo share helpers).
"""
import importlib.util
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import torch

from utils import setup_plotting, save_figure, compute_pca_directions
from word_lists import WORD_LISTS

REPO = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scatter_mod = _load_module("gridness_vs_accuracy_scatter", "gridness_vs_accuracy_scatter.py")
reproduce_mod = scatter_mod.reproduce_mod
dot_product_mod = scatter_mod.dot_product_mod

PLOTS_DIR = "results/reproduce/plots/gridness_vs_accuracy"
THUMB_DIR = os.path.join(PLOTS_DIR, "thumbnails")
THUMB_SIZE_IN = 1.5    # inches, individual thumbnail PNG figure size
THUMB_ZOOM = 0.55       # OffsetImage zoom factor in the composite figure
THUMB_OFFSETS = [(60, 45), (60, -45), (-60, 45), (-60, -45)]  # cycled by index


def render_thumbnail(word_list_key):
    """Render this word list's final-layer PCA scatter (grid edges + word
    markers, no axes/title/legend) to a small PNG and return its path."""
    grid = scatter_mod.make_grid(word_list_key)
    class_means, _ = dot_product_mod.load_final_layer_class_means(word_list_key)
    pca_dirs, _ = compute_pca_directions(torch.tensor(class_means), top_n=2)

    reproduce_mod.configure_for_word_list(word_list_key)
    fig, ax = plt.subplots(figsize=(THUMB_SIZE_IN, THUMB_SIZE_IN))
    reproduce_mod.plot_class_mean_pca(grid, class_means, pca_dirs.numpy(), ax=ax, title=" ")
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    os.makedirs(THUMB_DIR, exist_ok=True)
    path = os.path.join(THUMB_DIR, f"{word_list_key}.png")
    fig.savefig(path, dpi=110, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False

    keys = scatter_mod.SWEEP_KEYS
    print(f"Rendering {len(keys)} thumbnails...")
    dc_by_key, acc_by_key, thumb_paths = {}, {}, {}
    for key in keys:
        dc_by_key[key], _ = scatter_mod.compute_final_dc(key)
        acc_by_key[key] = scatter_mod.load_real_full_context_accuracy(key)
        thumb_paths[key] = render_thumbnail(key)
        print(f"  {key}: DC={dc_by_key[key]:.3f}, acc={acc_by_key[key]:.3f}")

    xs = np.array([dc_by_key[k] for k in keys])
    ys = np.array([acc_by_key[k] for k in keys])
    r = float(np.corrcoef(xs, ys)[0, 1])

    key_to_family, n_families = scatter_mod.assign_families(keys)
    cmap = cm.get_cmap("tab20", max(n_families, 1))
    colors = [cmap(key_to_family[k][1]) for k in keys]

    fig, ax = plt.subplots(figsize=(22, 18))
    ax.scatter(xs, ys, c=colors, s=55, zorder=5, edgecolors="black", linewidths=0.4)
    for x, y, k in zip(xs, ys, keys):
        ax.annotate(k, (x, y), fontsize=6.5, alpha=0.85,
                    xytext=(4, -10), textcoords="offset points", zorder=6)

    for i, (x, y, k) in enumerate(zip(xs, ys, keys)):
        img = plt.imread(thumb_paths[k])
        imagebox = OffsetImage(img, zoom=THUMB_ZOOM)
        offset = THUMB_OFFSETS[i % len(THUMB_OFFSETS)]
        ab = AnnotationBbox(
            imagebox, (x, y), xybox=offset, xycoords="data",
            boxcoords="offset points", frameon=True, pad=0.15,
            bboxprops=dict(edgecolor="gray", linewidth=0.5),
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.6),
            zorder=4,
        )
        ax.add_artist(ab)

    ax.set_xlabel("Distance correlation (final-layer activations vs. grid)", fontsize=13)
    ax.set_ylabel("Real accuracy (full context, seq len 1400)", fontsize=13)
    ax.set_title(
        f"Final-layer gridness vs. real accuracy, with PCA thumbnails (n={len(keys)}, r={r:.3f})",
        fontsize=15,
    )

    save_figure(fig, PLOTS_DIR, "gridness_vs_accuracy_with_thumbnails.pdf")
    print("Saved gridness_vs_accuracy_with_thumbnails")


if __name__ == "__main__":
    main()
