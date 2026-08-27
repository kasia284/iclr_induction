"""Phase-plane plot relating a ring-geometry metric (distance correlation,
or Dirichlet energy) to real accuracy, with CONTEXT LENGTH as the implicit
parameter tracing out a curve for each word list -- the ring analog of
distance-correlation-accuracy-phase-plane.py. For a fixed word list, as
context length k grows through the log-spaced CHECKPOINTS (ring-evolution-
seqlen.py), we get a sequence of points (metric(k), accuracy(k));
connecting them in order of increasing k shows the joint trajectory of
representation geometry and task performance.

Two geometry metrics, both computed from layer-26 class means (the only
layer 01_reproduce_ring.py / ring-evolution-seqlen.py track) at each
checkpoint, against utils.Ring's circle-embedding coordinates / adjacency:
  - distance correlation (utils.compute_distance_correlation): HIGH =
    more ring-like.
  - Dirichlet energy (utils.compute_dirichlet_energy): LOW = more
    ring-like.
The static matplotlib output is one PNG per metric. The interactive
Plotly output is a single HTML with buttons to switch which metric is on
the x axis, and hover/click a point to reveal its PCA thumbnail at that
context length (hover text always shows both metrics regardless of which
is plotted).

Covers the days_of_week family (5 word lists so far), colored
individually since there's only one word-set family.

Only plots word lists with BOTH a ring_evolution_seqlen.npz cache (needs
ring-evolution-seqlen.py, a GPU run) and an accuracies_Ring_*.npz cache
(01_reproduce_ring.py) -- any missing member is skipped with a printed
note.

Pure cache reads, no GPU / model needed once those caches exist.
"""
import base64
import importlib.util
import os
import textwrap

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import plotly.graph_objects as go
import torch

from utils import (
    Ring, build_word_to_color, setup_plotting, save_figure,
    compute_distance_correlation, compute_dirichlet_energy, compute_pca_directions,
)
from word_lists import WORD_LISTS

REPO = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = "results/reproduce_ring/plots/gridness_vs_accuracy"
THUMB_DIR = os.path.join(PLOTS_DIR, "thumbnails_phase_plane")


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ring_evo_mod = _load_module("ring_evolution", "ring-evolution-seqlen.py")
scatter_mod = _load_module("gridness_vs_accuracy_scatter", "gridness-vs-accuracy-scatter.py")
RING_SWEEP_KEYS = ring_evo_mod.RING_SWEEP_KEYS
CHECKPOINTS = ring_evo_mod.CHECKPOINTS
LAYER = ring_evo_mod.LAYER

METRICS = {
    "dc": dict(
        key="dc", label="Distance correlation",
        axis_label=f"Distance correlation (layer {LAYER} activations vs. ring, higher = more ring-like)",
        filename="distance_correlation_accuracy_phase_plane_ring",
        title="Ring accuracy vs. distance correlation phase plane",
    ),
    "energy": dict(
        key="energy", label="Dirichlet energy",
        axis_label=f"Dirichlet energy (layer {LAYER} activations vs. ring, lower = more ring-like)",
        filename="dirichlet_energy_accuracy_phase_plane_ring",
        title="Ring accuracy vs. Dirichlet energy phase plane",
    ),
}


def get_ring_coords(ring, words):
    return np.array([[ring.word_to_row[w], ring.word_to_col[w]] for w in words])


def load_phase_curve(word_list_key):
    """Returns (dc_values, energy_values, acc_values, class_means) aligned
    to CHECKPOINTS, or None if the required caches aren't present."""
    if not os.path.exists(ring_evo_mod.cache_path(word_list_key)):
        return None
    acc_path = f"results/reproduce_ring/data/{word_list_key}/accuracies_Ring_{word_list_key}.npz"
    if not os.path.exists(acc_path):
        return None

    words = WORD_LISTS[word_list_key]
    ring = Ring(words)
    ring_coords = get_ring_coords(ring, words)
    adjacency = ring.build_adjacency_matrix()

    class_means = ring_evo_mod.load_or_compute_class_means(None, word_list_key)
    dc_values = np.array([compute_distance_correlation(class_means[k], ring_coords) for k in CHECKPOINTS])
    energy_values = np.array([compute_dirichlet_energy(class_means[k], adjacency) for k in CHECKPOINTS])

    acc_data = np.load(acc_path)
    acc_key = "graph_accs" if "graph_accs" in acc_data else "all_accs"
    mean_acc = acc_data[acc_key].mean(axis=0)
    acc_values = np.array([mean_acc[k - 1] for k in CHECKPOINTS])

    return dc_values, energy_values, acc_values, class_means


def render_point_thumbnail(word_list_key, checkpoint, class_means_at_checkpoint):
    path = os.path.join(THUMB_DIR, f"{word_list_key}_k{checkpoint}.png")
    if os.path.exists(path):
        return path

    words = WORD_LISTS[word_list_key]
    ring = Ring(words)
    word_to_color = build_word_to_color(words)
    pca_dirs, _ = compute_pca_directions(torch.tensor(class_means_at_checkpoint), top_n=2)
    projected = (class_means_at_checkpoint - class_means_at_checkpoint.mean(axis=0, keepdims=True)) \
        @ pca_dirs.numpy().T

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


def thumbnail_data_uri(word_list_key, checkpoint, class_means_at_checkpoint):
    path = render_point_thumbnail(word_list_key, checkpoint, class_means_at_checkpoint)
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_static(metric, curves, plot_keys, colors_rgb, cmap, family_label, n_families, description):
    metric_idx = 0 if metric["key"] == "dc" else 1

    fig, ax = plt.subplots(figsize=(6.5, 6.3))
    for key in plot_keys:
        x_values = curves[key][metric_idx]
        acc_values = curves[key][2]
        color = colors_rgb[key]
        ax.plot(x_values, acc_values, color=color, linewidth=1.2, alpha=0.8,
                 marker="o", markersize=4, zorder=3)
        ax.annotate(
            "", xy=(x_values[-1], acc_values[-1]), xytext=(x_values[-2], acc_values[-2]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, alpha=0.9),
        )
    legend_handles = [
        plt.Line2D([0], [0], color=cmap(i)[:3], lw=1.5, marker="o", markersize=4)
        for i in range(n_families)
    ]
    legend_labels = [family_label[i] for i in range(n_families)]
    ax.legend(legend_handles, legend_labels, loc="best", frameon=True,
              framealpha=1.0, edgecolor="gray", fontsize=7, title="Word-set family")
    ax.set_xlabel(metric["axis_label"])
    ax.set_ylabel("Real accuracy (ring task)")
    ax.set_title(
        f"{metric['title']}\n(context length as parameter; arrow = increasing context length)",
        fontsize=10,
    )
    fig.text(
        0.5, 0.01,
        "\n".join(textwrap.wrap(description, 95)),
        ha="center", va="bottom", fontsize=7, color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.tight_layout = lambda *args, **kwargs: None
    save_figure(fig, PLOTS_DIR, f"{metric['filename']}.pdf")
    print(f"Saved {metric['filename']}.png")


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False

    curves = {}
    for key in RING_SWEEP_KEYS:
        result = load_phase_curve(key)
        if result is None:
            print(f"Skipping {key}: missing ring_evolution_seqlen.npz "
                  f"(run ring-evolution-seqlen.py, on a GPU) or accuracies cache.")
            continue
        curves[key] = result

    plot_keys = list(curves.keys())
    print(f"Plotting phase curves for {len(plot_keys)}/{len(RING_SWEEP_KEYS)} ring word lists:")
    for k in plot_keys:
        print(f"  - {k}")

    key_to_family, n_families = scatter_mod.assign_families(plot_keys)
    cmap = cm.get_cmap("tab10", max(n_families, 1))
    colors_rgb = {k: cmap(key_to_family[k][1])[:3] for k in plot_keys}
    family_label = {key_to_family[k][1]: key_to_family[k][0] for k in plot_keys}

    checkpoints_str = ", ".join(str(c) for c in CHECKPOINTS)
    description = (
        f"Each curve traces one ring word list through (geometry metric, accuracy) "
        f"space as context length grows -- one dot per curve at each of the "
        f"{len(CHECKPOINTS)} context lengths sampled: {checkpoints_str} tokens. "
        f"Dot size and the trailing arrowhead both grow with context length."
    )

    for metric in METRICS.values():
        render_static(metric, curves, plot_keys, colors_rgb, cmap, family_label, n_families, description)

    # ── Plotly (interactive) ─────────────────────────────────────────────
    FIG_WIDTH, FIG_HEIGHT = 1000, 850
    n_checkpoints = len(CHECKPOINTS)

    print("Rendering per-point PCA thumbnails...")
    pfig = go.Figure()
    images = []
    legend_shown = set()
    for key in plot_keys:
        dc_values, energy_values, acc_values, class_means = curves[key]
        r, g, b = colors_rgb[key]
        rgb_str = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        family_idx = key_to_family[key][1]
        pfig.add_trace(go.Scatter(
            x=dc_values.tolist(), y=acc_values.tolist(),
            mode="lines+markers",
            line=dict(color=rgb_str, width=2),
            marker=dict(size=[6 + 2 * i for i in range(n_checkpoints)],
                        color=rgb_str, line=dict(width=1, color="black")),
            name=family_label[family_idx],
            legendgroup=family_label[family_idx],
            showlegend=family_idx not in legend_shown,
            customdata=[[c, dc, en] for c, dc, en in zip(CHECKPOINTS, dc_values, energy_values)],
            hovertemplate=(
                f"{key}<br>context length: %{{customdata[0]}}"
                "<br>distance correlation: %{customdata[1]:.3f}"
                "<br>Dirichlet energy: %{customdata[2]:.3f}"
                "<br>accuracy: %{y:.3f}<extra></extra>"
            ),
        ))
        legend_shown.add(family_idx)
        for i, k in enumerate(CHECKPOINTS):
            uri = thumbnail_data_uri(key, k, class_means[k])
            images.append(dict(
                source=uri, xref="x", yref="y",
                x=float(dc_values[i]), y=float(acc_values[i]),
                xanchor="center", yanchor="bottom",  # fixed up below
                sizex=0, sizey=0,  # filled in below, per metric
                sizing="contain", opacity=0.0, layer="above",
            ))

    acc_all = np.concatenate([curves[key][2] for key in plot_keys])
    y_mid = (float(acc_all.max()) + float(acc_all.min())) / 2
    for img in images:
        img["yanchor"] = "top" if img["y"] >= y_mid else "bottom"

    dc_all = np.concatenate([curves[key][0] for key in plot_keys])
    energy_all = np.concatenate([curves[key][1] for key in plot_keys])
    dc_range = float(dc_all.max() - dc_all.min()) or 1.0
    energy_range = float(energy_all.max() - energy_all.min()) or 1.0
    dc_sizex = 0.12 * dc_range
    dc_sizey = dc_sizex * (FIG_HEIGHT / FIG_WIDTH)
    energy_sizex = 0.12 * energy_range
    energy_sizey = energy_sizex * (FIG_HEIGHT / FIG_WIDTH)
    for img in images:
        img["sizex"] = dc_sizex
        img["sizey"] = dc_sizey

    dc_x_by_trace = [curves[key][0].tolist() for key in plot_keys]
    energy_x_by_trace = [curves[key][1].tolist() for key in plot_keys]
    dc_image_x = [x for key in plot_keys for x in curves[key][0].tolist()]
    energy_image_x = [x for key in plot_keys for x in curves[key][1].tolist()]

    dc_layout_update = {
        "xaxis.title.text": METRICS["dc"]["axis_label"], "xaxis.autorange": True,
        "title.text": f"{METRICS['dc']['title']} -- context length as parameter "
                       f"(marker size grows with context length; hover or click a point for its PCA plot)",
    }
    energy_layout_update = {
        "xaxis.title.text": METRICS["energy"]["axis_label"], "xaxis.autorange": True,
        "title.text": f"{METRICS['energy']['title']} -- context length as parameter "
                       f"(marker size grows with context length; hover or click a point for its PCA plot)",
    }
    for i in range(len(images)):
        dc_layout_update[f"images[{i}].x"] = dc_image_x[i]
        dc_layout_update[f"images[{i}].sizex"] = dc_sizex
        dc_layout_update[f"images[{i}].sizey"] = dc_sizey
        energy_layout_update[f"images[{i}].x"] = energy_image_x[i]
        energy_layout_update[f"images[{i}].sizex"] = energy_sizex
        energy_layout_update[f"images[{i}].sizey"] = energy_sizey

    updatemenus = [dict(
        type="buttons", direction="left", showactive=True,
        x=0.5, xanchor="center", y=1.1, yanchor="top", pad=dict(t=5, b=5),
        buttons=[
            dict(label="Distance correlation", method="update", args=[{"x": dc_x_by_trace}, dc_layout_update]),
            dict(label="Dirichlet energy", method="update", args=[{"x": energy_x_by_trace}, energy_layout_update]),
        ],
    )]

    pfig.update_layout(
        title=dc_layout_update["title.text"],
        xaxis_title=METRICS["dc"]["axis_label"],
        yaxis_title="Real accuracy (ring task)",
        template="plotly_white",
        width=FIG_WIDTH, height=FIG_HEIGHT,
        margin=dict(b=110, t=110),
        updatemenus=updatemenus,
        images=images,
        annotations=[dict(
            text="<br>".join(textwrap.wrap(description, 110)),
            xref="paper", yref="paper", x=0.5, y=-0.16,
            showarrow=False, align="center",
            font=dict(size=11, color="dimgray"),
        )],
    )

    hover_js = """
    var gd = document.getElementsByClassName('plotly-graph-div')[0];
    var pinned = new Set();
    var N_CHECKPOINTS = %d;
    function setOpacity(idx, val) {
        var upd = {};
        upd['images[' + idx + '].opacity'] = val;
        Plotly.relayout(gd, upd);
    }
    gd.on('plotly_hover', function(data) {
        var pt = data.points[0];
        var idx = pt.curveNumber * N_CHECKPOINTS + pt.pointIndex;
        setOpacity(idx, 1);
    });
    gd.on('plotly_unhover', function(data) {
        var pt = data.points[0];
        var idx = pt.curveNumber * N_CHECKPOINTS + pt.pointIndex;
        if (!pinned.has(idx)) { setOpacity(idx, 0); }
    });
    gd.on('plotly_click', function(data) {
        var pt = data.points[0];
        var idx = pt.curveNumber * N_CHECKPOINTS + pt.pointIndex;
        if (pinned.has(idx)) { pinned.delete(idx); setOpacity(idx, 0); }
        else { pinned.add(idx); setOpacity(idx, 1); }
    });
    """ % n_checkpoints

    path = os.path.join(PLOTS_DIR, "distance_correlation_accuracy_phase_plane_ring.html")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    pfig.write_html(path, config={"responsive": False}, include_plotlyjs="cdn", post_script=hover_js)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
