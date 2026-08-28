"""Same interactive scatter as gridness_vs_accuracy_scatter_interactive_thumbnails.py (final-layer distance correlation vs. real accuracy, hover
or click a point to reveal its PCA thumbnail), but restricted to just the
three word-set families that show the most convincing negative
correlation: "morphology", "text_numbers", and "two_digit_numbers" --
i.e. each family's base arrangement plus all of its grid-position
permutations (_permuted, _rand*, and morphology's _corners). Families are
identified by frozenset(words) so this doesn't need to be kept in sync by
hand as more _randN permutations are added.

Excludes "two_digit_numbers_2468" / "two_digit_numbers_4to7" -- those are
different vocabularies (different numbers), not permutations of
"two_digit_numbers".

Reuses everything from gridness_vs_accuracy_scatter_interactive_thumbnails.py (imported via importlib,
consistent with how sibling scripts in this repo share helpers) -- only the key list and output filename differ. Pure
cache reads + thumbnail rendering, no GPU/model needed.
"""
import importlib.util
import os

import numpy as np
import plotly.graph_objects as go

REPO = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


interactive_mod = _load_module(
    "gridness_vs_accuracy_interactive_thumbnails",
    "gridness_vs_accuracy_scatter_interactive_thumbnails.py",
)
scatter_mod = interactive_mod.scatter_mod
thumb_mod = interactive_mod.thumb_mod

from word_lists import WORD_LISTS

PLOTS_DIR = "results/reproduce/plots/gridness_vs_accuracy"
FIG_WIDTH, FIG_HEIGHT = 1400, 1000
FAMILY_BASE_KEYS = ["morphology", "text_numbers", "two_digit_numbers"]


def family_keys():
    """All cached SWEEP_KEYS whose word set matches one of FAMILY_BASE_KEYS
    (base arrangement + every grid-position permutation of it)."""
    family_word_sets = {frozenset(WORD_LISTS[base]) for base in FAMILY_BASE_KEYS}
    return [k for k in scatter_mod.SWEEP_KEYS if frozenset(WORD_LISTS[k]) in family_word_sets]


def main():
    keys = family_keys()
    print(f"Building interactive plot for {len(keys)} word lists (morphology / "
          f"text_numbers / two_digit_numbers families only)...")
    for k in keys:
        print(f"  - {k}")

    dc_by_key, acc_by_key = {}, {}
    for key in keys:
        dc_by_key[key], _ = scatter_mod.compute_final_dc(key)
        acc_by_key[key] = scatter_mod.load_real_full_context_accuracy(key)

    xs = np.array([dc_by_key[k] for k in keys])
    ys = np.array([acc_by_key[k] for k in keys])
    r = float(np.corrcoef(xs, ys)[0, 1])

    key_to_family, n_families = scatter_mod.assign_families(keys)
    import matplotlib.cm as cm
    cmap = cm.get_cmap("tab20", max(n_families, 1))
    colors = [
        "rgb({},{},{})".format(*(int(c * 255) for c in cmap(key_to_family[k][1])[:3]))
        for k in keys
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs.tolist(), y=ys.tolist(), mode="markers+text",
        marker=dict(color=colors, size=11, line=dict(width=1, color="black")),
        text=keys, textposition="top center", textfont=dict(size=7),
        hovertemplate="%{text}<br>final DC: %{x:.3f}<br>real accuracy: %{y:.3f}<extra></extra>",
    ))

    print("Preparing thumbnail images...")
    x_range = float(xs.max() - xs.min())
    y_range = float(ys.max() - ys.min())
    sizex = 0.12 * x_range
    sizey = sizex * (FIG_HEIGHT / FIG_WIDTH)
    # Anchor each thumbnail on whichever side of its point has more room
    # (below for points in the lower half of the y-range, above for points
    # in the upper half) so it doesn't get clipped by the plot's top/bottom
    # edge -- a fixed yanchor="bottom" cuts off thumbnails for points near
    # the top of the axis range.
    y_mid = (float(ys.max()) + float(ys.min())) / 2
    images = []
    for key, x, y in zip(keys, xs, ys):
        uri = interactive_mod.thumbnail_data_uri(key)
        yanchor = "top" if y >= y_mid else "bottom"
        images.append(dict(
            source=uri, xref="x", yref="y", x=float(x), y=float(y),
            xanchor="center", yanchor=yanchor,
            sizex=sizex, sizey=sizey,
            sizing="contain", opacity=0.0, layer="above",
        ))

    fig.update_layout(
        title=(
            "Final-layer gridness vs. real accuracy -- morphology / "
            f"text_numbers / two_digit_numbers only -- hover or click a "
            f"point for its PCA plot (n={len(keys)}, r={r:.3f})"
        ),
        xaxis_title="Distance correlation (final-layer activations vs. grid)",
        yaxis_title="Real accuracy (full context, seq len 1400)",
        template="plotly_white",
        images=images,
        width=FIG_WIDTH, height=FIG_HEIGHT,
    )

    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "gridness_vs_accuracy_interactive_thumbnails_morphology_numbers.html")
    fig.write_html(path, config={"responsive": False}, include_plotlyjs="cdn", post_script=interactive_mod.HOVER_JS)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
