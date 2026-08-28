"""Interactive scatter: distance correlation (layer 26 activations vs.
ring position -- the only layer cached by 01_reproduce_ring.py, unlike
the grid task's multi-layer sweep) vs. real accuracy, one point per RING
word list (currently the "days_of_week" and "months_of_year" families,
each the natural cyclic order plus its systematic and random
permutations). Analogous to gridness_vs_accuracy_scatter_interactive_thumbnails.py for the Grid task -- hover or click a point to reveal its
PCA thumbnail (utils.Ring edges + word markers, same rendering as
01_reproduce_ring.py's plot_class_mean_pca).

Points are colored by family (frozenset(words), via assign_families --
reused from gridness_vs_accuracy_scatter.py, which is topology-agnostic)
so days_of_week and months_of_year read as two distinct groups instead of
one undifferentiated set of individually-colored points.

Reads 01_reproduce_ring.py's cache (results/reproduce_ring/data/{key}/
accuracies_Ring_{key}.npz + pca_Ring_{key}.npz) -- pure cache reads, no
GPU/model needed.
"""
import base64
import importlib.util
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import plotly.graph_objects as go
import torch

from utils import Ring, build_word_to_color, compute_distance_correlation, compute_pca_directions, setup_plotting
from word_lists import WORD_LISTS

REPO = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "results/reproduce_ring/data"
PLOTS_DIR = "results/reproduce_ring/plots/gridness_vs_accuracy"
THUMB_DIR = os.path.join(PLOTS_DIR, "thumbnails")


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ring_mod = _load_module("reproduce_ring", "01_reproduce_ring.py")
scatter_mod = _load_module("gridness_vs_accuracy_scatter", "gridness_vs_accuracy_scatter.py")
RING_SWEEP_KEYS = ring_mod.RING_SWEEP_KEYS


def get_ring_coords(ring, words):
    return np.array([[ring.word_to_row[w], ring.word_to_col[w]] for w in words])


def load_class_means(word_list_key):
    path = os.path.join(DATA_DIR, word_list_key, f"pca_Ring_{word_list_key}.npz")
    return np.load(path)["class_means"]


def load_real_full_context_accuracy(word_list_key):
    path = os.path.join(DATA_DIR, word_list_key, f"accuracies_Ring_{word_list_key}.npz")
    acc_data = np.load(path)
    acc_key = "graph_accs" if "graph_accs" in acc_data else "all_accs"
    all_accs = acc_data[acc_key]
    return float(all_accs[:, -1].mean())


def compute_dc(word_list_key):
    words = WORD_LISTS[word_list_key]
    ring = Ring(words)
    ring_coords = get_ring_coords(ring, words)
    class_means = load_class_means(word_list_key)
    return compute_distance_correlation(class_means, ring_coords)


def render_thumbnail(word_list_key):
    """Render (or reuse a cached) small PCA-scatter PNG: ring edges +
    word colors/markers, no axes/title -- same rendering as
    01_reproduce_ring.py's plot_class_mean_pca, just stripped down."""
    path = os.path.join(THUMB_DIR, f"{word_list_key}.png")
    if os.path.exists(path):
        return path

    words = WORD_LISTS[word_list_key]
    ring = Ring(words)
    word_to_color = build_word_to_color(words)
    class_means = load_class_means(word_list_key)
    pca_dirs, _ = compute_pca_directions(torch.tensor(class_means), top_n=2)
    projected = (class_means - class_means.mean(axis=0, keepdims=True)) @ pca_dirs.numpy().T

    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    A = ring.build_adjacency_matrix()
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if A[i, j]:
                ax.plot(
                    [projected[i, 0], projected[j, 0]], [projected[i, 1], projected[j, 1]],
                    color="dimgray", alpha=0.7, linestyle="--", linewidth=0.8,
                )
    for i, word in enumerate(words):
        ax.scatter(
            projected[i, 0], projected[i, 1], color=word_to_color[word],
            s=120, marker="*", edgecolors="black", linewidths=0.5, zorder=5,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    os.makedirs(THUMB_DIR, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def thumbnail_data_uri(word_list_key):
    path = render_thumbnail(word_list_key)
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def make_hover_js(trace_offsets):
    """image index = TRACE_OFFSETS[curveNumber] + pointIndex -- needed
    because with one trace per family (for the legend/coloring), trace
    sizes vary, so a plain curveNumber*n_per_trace multiplication (fine
    when every trace has the same length) doesn't work here."""
    offsets_js = "[" + ",".join(str(o) for o in trace_offsets) + "]"
    return """
    var gd = document.getElementsByClassName('plotly-graph-div')[0];
    var pinned = new Set();
    var TRACE_OFFSETS = %s;
    function setOpacity(idx, val) {
        var upd = {};
        upd['images[' + idx + '].opacity'] = val;
        Plotly.relayout(gd, upd);
    }
    gd.on('plotly_hover', function(data) {
        var pt = data.points[0];
        var idx = TRACE_OFFSETS[pt.curveNumber] + pt.pointIndex;
        setOpacity(idx, 1);
    });
    gd.on('plotly_unhover', function(data) {
        var pt = data.points[0];
        var idx = TRACE_OFFSETS[pt.curveNumber] + pt.pointIndex;
        if (!pinned.has(idx)) { setOpacity(idx, 0); }
    });
    gd.on('plotly_click', function(data) {
        var pt = data.points[0];
        var idx = TRACE_OFFSETS[pt.curveNumber] + pt.pointIndex;
        if (pinned.has(idx)) { pinned.delete(idx); setOpacity(idx, 0); }
        else { pinned.add(idx); setOpacity(idx, 1); }
    });
    """ % offsets_js


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False

    keys = [k for k in RING_SWEEP_KEYS
            if os.path.exists(os.path.join(DATA_DIR, k, f"accuracies_Ring_{k}.npz"))
            and os.path.exists(os.path.join(DATA_DIR, k, f"pca_Ring_{k}.npz"))]
    missing = [k for k in RING_SWEEP_KEYS if k not in keys]
    if missing:
        print(f"Skipping (no cache yet, run 01_reproduce_ring.py): {missing}")
    print(f"Building interactive plot for {len(keys)} ring word lists...")

    dc_by_key = {k: compute_dc(k) for k in keys}
    acc_by_key = {k: load_real_full_context_accuracy(k) for k in keys}

    xs_all = np.array([dc_by_key[k] for k in keys])
    ys_all = np.array([acc_by_key[k] for k in keys])
    r = float(np.corrcoef(xs_all, ys_all)[0, 1]) if len(keys) > 1 else float("nan")

    key_to_family, n_families = scatter_mod.assign_families(keys)
    cmap = cm.get_cmap("tab10", max(n_families, 1))
    family_labels_seen = sorted({key_to_family[k] for k in keys}, key=lambda t: t[1])

    print("Preparing thumbnail images...")
    FIG_WIDTH, FIG_HEIGHT = 900, 700
    x_range = float(xs_all.max() - xs_all.min()) or 1.0
    sizex = 0.18 * x_range
    sizey = sizex * (FIG_HEIGHT / FIG_WIDTH)
    # Anchor each thumbnail on whichever side of its point has more room
    # (below for points in the lower half of the y-range, above for points
    # in the upper half) so it doesn't get clipped by the plot's top/bottom
    # edge -- a fixed yanchor="bottom" cuts off thumbnails for points near
    # the top of the axis range.
    y_mid = (float(ys_all.max()) + float(ys_all.min())) / 2

    fig = go.Figure()
    images = []
    trace_offsets = []
    for label, idx in family_labels_seen:
        family_keys = [k for k in keys if key_to_family[k][0] == label]
        xs = np.array([dc_by_key[k] for k in family_keys])
        ys = np.array([acc_by_key[k] for k in family_keys])
        rgba = cmap(idx)
        color = "rgb({},{},{})".format(*(int(c * 255) for c in rgba[:3]))
        fig.add_trace(go.Scatter(
            x=xs.tolist(), y=ys.tolist(), mode="markers+text",
            marker=dict(color=color, size=14, line=dict(width=1, color="black")),
            text=family_keys, textposition="top center", textfont=dict(size=9),
            name=label,
            hovertemplate="%{text}<br>DC: %{x:.3f}<br>real accuracy: %{y:.3f}<extra></extra>",
        ))

        trace_offsets.append(len(images))
        for key, x, y in zip(family_keys, xs, ys):
            uri = thumbnail_data_uri(key)
            yanchor = "top" if y >= y_mid else "bottom"
            images.append(dict(
                source=uri, xref="x", yref="y", x=float(x), y=float(y),
                xanchor="center", yanchor=yanchor,
                sizex=sizex, sizey=sizey,
                sizing="contain", opacity=0.0, layer="above",
            ))

    fig.update_layout(
        title=(
            f"Ring gridness vs. real accuracy -- hover or click a point "
            f"for its PCA plot (n={len(keys)}, r={r:.3f})"
        ),
        xaxis_title="Distance correlation (layer 26 activations vs. ring)",
        yaxis_title="Real accuracy (full context, seq len 1400)",
        template="plotly_white",
        legend_title_text="Word-set family",
        images=images,
        width=FIG_WIDTH, height=FIG_HEIGHT,
    )

    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "gridness_vs_accuracy_ring_interactive_thumbnails.html")
    fig.write_html(path, config={"responsive": False}, include_plotlyjs="cdn",
                    post_script=make_hover_js(trace_offsets))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
