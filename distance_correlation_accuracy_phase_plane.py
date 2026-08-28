"""Phase-plane plot relating a final-layer grid-geometry metric (distance
correlation, or Dirichlet energy) to real accuracy, with CONTEXT LENGTH as
the implicit parameter tracing out a curve for each word list -- i.e. for
a fixed word list, as context length k grows through the log-spaced
CHECKPOINTS (morphology_grid_evolution_layers_vs_seqlen.py), we get a
sequence of points (metric(k), accuracy(k)); connecting them in order of
increasing k shows the joint trajectory of representation geometry and
task performance, rather than just their final (k=SEQ_LEN) values as in
gridness_vs_accuracy_scatter.py's static scatter.

Two geometry metrics, both computed from per-layer class means at each
checkpoint (LAYERS, from morphology_grid_evolution_layers_vs_seqlen.py --
by default '0','6','13','20','26','31'):
  - distance correlation (utils.compute_distance_correlation): HIGH =
    more grid-like.
  - Dirichlet energy (utils.compute_dirichlet_energy): LOW = more
    grid-like.
The static matplotlib output is one PNG per metric, at the final layer
only. The interactive Plotly output is a single HTML with two dropdowns
above the plot -- "Metric" (distance correlation / Dirichlet energy) and
"Layer" (any of LAYERS) -- that can be combined freely; hover text always
shows both metrics, plus accuracy and context length, regardless of which
metric/layer is plotted.

y axis (both, all layers): mean real accuracy at context length k
(position k-1 of accuracies_Grid_{key}.npz's all_accs, averaged over the
16 walks) -- same quantity as 01_reproduce.py's accuracy_curve, just read
off at the same checkpoints as the geometry metrics instead of the full
curve. Accuracy is a property of the model's output, not of any one
layer, so it doesn't change when the Layer dropdown is changed -- only
the x axis (and the PCA thumbnails) do.

Restricted to the "morphology" / "text_numbers" / "two_digit_numbers"
families (see gridness_vs_accuracy_scatter_interactive_thumbnails_morphology_numbers.py) -- where the negative final-DC-vs-accuracy
correlation is clearest -- to see whether that relationship is already
present early in the sequence/shallow layers or only emerges at long
context / late layers.

Only plots word lists with BOTH a grid_evolution_layers_vs_seqlen.npz
cache (needs morphology_grid_evolution_layers_vs_seqlen.py's SWEEP_KEYS
to include them + a GPU run) and an accuracies_Grid_*.npz cache -- any
missing member is skipped with a printed note rather than silently
omitted from the family.

Usage:
    python distance_correlation_accuracy_phase_plane.py

    # Same, but the x axis is ln_final.hook_normalized (the residual
    # stream after ALL blocks AND the model's final RMSNorm -- what
    # actually feeds the unembedding matrix) instead of the usual LAYERS
    # sweep of blocks.{l}.hook_resid_pre. Requires a
    # grid_evolution_layers_vs_seqlen_post_ln.npz cache per word list (run
    # morphology_grid_evolution_layers_vs_seqlen.py --post-layernorm first,
    # on a GPU). Static PNG/PDF only (saved with a "_post_ln" filename
    # suffix so it never overwrites the regular final-layer plot) -- skips
    # the interactive Plotly output, which is keyed to the fixed LAYERS
    # sweep and isn't worth extending for one extra virtual layer.
    python distance_correlation_accuracy_phase_plane.py --post-layernorm

Pure cache reads, no GPU / model needed (reuses whatever's already been
computed by 01_reproduce.py and morphology_grid_evolution_layers_vs_seqlen.py). Reuses load_cached_accuracies from 01_reproduce.py, Grid /
get_grid_coords conventions from utils.py / 01_reproduce.py, and
assign_families from gridness_vs_accuracy_scatter.py (both imported via
importlib, consistent with how sibling scripts in this repo share helpers).
"""
import base64
import importlib.util
import json
import os
import sys
import textwrap

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import plotly.graph_objects as go
import torch

from utils import (
    Grid, setup_plotting, save_figure, compute_distance_correlation, compute_dirichlet_energy,
    compute_pca_directions,
)
from word_lists import WORD_LISTS

REPO = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = "results/reproduce/plots/gridness_vs_accuracy"
THUMB_DIR = os.path.join(PLOTS_DIR, "thumbnails_phase_plane")
FAMILY_BASE_KEYS = ["morphology", "text_numbers", "two_digit_numbers", "original_paper"]


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reproduce_mod = _load_module("reproduce_01", "01_reproduce.py")
grid_evolution_mod = _load_module("grid_evolution", "morphology_grid_evolution_layers_vs_seqlen.py")
scatter_mod = _load_module("gridness_vs_accuracy_scatter", "gridness_vs_accuracy_scatter.py")

CHECKPOINTS = grid_evolution_mod.CHECKPOINTS
LAYERS = grid_evolution_mod.LAYERS  # e.g. ['0', '6', '13', '20', '26', '31']
FINAL_LAYER = LAYERS[-1]  # used for the static PNG/PDF output, matches "final DC" elsewhere in the repo

METRICS = {
    "dc": dict(
        key="dc", label="Distance correlation",
        axis_label_fn=lambda layer: f"Distance correlation (layer {layer} activations vs. grid, higher = more grid-like)",
        filename="distance_correlation_accuracy_phase_plane",
        title="Accuracy vs. distance correlation phase plane",
    ),
    "energy": dict(
        key="energy", label="Dirichlet energy",
        axis_label_fn=lambda layer: f"Dirichlet energy (layer {layer} activations vs. grid, lower = more grid-like)",
        filename="dirichlet_energy_accuracy_phase_plane",
        title="Accuracy vs. Dirichlet energy phase plane",
    ),
}


def family_keys():
    family_word_sets = {frozenset(WORD_LISTS[base]) for base in FAMILY_BASE_KEYS}
    return [k for k in WORD_LISTS if frozenset(WORD_LISTS[k]) in family_word_sets]


def load_phase_curve(word_list_key, post_layernorm=False):
    """Returns (dc_by_layer, energy_by_layer, acc_values, class_means) or
    None if the required caches aren't present. dc_by_layer/energy_by_layer
    are {layer: [len(CHECKPOINTS)] array} dicts, one entry per LAYERS (or,
    with post_layernorm=True, a single entry keyed by
    grid_evolution_mod.POST_LN_KEY -- ln_final.hook_normalized, the residual
    stream after all blocks AND the model's final RMSNorm, instead of the
    usual blocks.{l}.hook_resid_pre sweep). class_means is the {(layer,
    checkpoint): [16, d_model]} dict (reused below to render per-point PCA
    thumbnails without recomputing)."""
    layers = [grid_evolution_mod.POST_LN_KEY] if post_layernorm else LAYERS
    if not os.path.exists(grid_evolution_mod.cache_path(word_list_key, post_layernorm=post_layernorm)):
        return None
    acc_path = f"results/reproduce/data/{word_list_key}/accuracies_Grid_{word_list_key}.npz"
    if not os.path.exists(acc_path):
        return None

    words = WORD_LISTS[word_list_key]
    side = round(len(words) ** 0.5)
    grid = Grid(words=words, rows=side, cols=side)
    grid_coords = reproduce_mod.get_grid_coords(grid, words)
    adjacency = grid.build_adjacency_matrix()

    class_means = grid_evolution_mod.load_or_compute_class_means(
        None, word_list_key, post_layernorm=post_layernorm
    )
    dc_by_layer = {
        layer: np.array([
            compute_distance_correlation(class_means[(layer, k)], grid_coords)
            for k in CHECKPOINTS
        ])
        for layer in layers
    }
    energy_by_layer = {
        layer: np.array([
            compute_dirichlet_energy(class_means[(layer, k)], adjacency)
            for k in CHECKPOINTS
        ])
        for layer in layers
    }

    mean_acc = reproduce_mod.load_cached_accuracies(word_list_key).mean(axis=0)
    acc_values = np.array([mean_acc[k - 1] for k in CHECKPOINTS])

    return dc_by_layer, energy_by_layer, acc_values, class_means


def render_point_thumbnail(word_list_key, layer, checkpoint, class_means_at_checkpoint):
    """Render (or reuse a cached) small PCA-scatter PNG for one word list
    at one layer and context-length checkpoint -- same rendering (grid
    edges, word colors/markers, no axes/title) as gridness-vs-accuracy-
    scatter-with-thumbnails.py's render_thumbnail, just parameterized by
    layer and checkpoint instead of always using the final layer / final
    (k=SEQ_LEN) class means."""
    path = os.path.join(THUMB_DIR, f"{word_list_key}_L{layer}_k{checkpoint}.png")
    if os.path.exists(path):
        return path

    grid = scatter_mod.make_grid(word_list_key)
    pca_dirs, _ = compute_pca_directions(torch.tensor(class_means_at_checkpoint), top_n=2)

    reproduce_mod.configure_for_word_list(word_list_key)
    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    reproduce_mod.plot_class_mean_pca(grid, class_means_at_checkpoint, pca_dirs.numpy(), ax=ax, title=" ")
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    os.makedirs(THUMB_DIR, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def thumbnail_data_uri(word_list_key, layer, checkpoint, class_means_at_checkpoint):
    path = render_point_thumbnail(word_list_key, layer, checkpoint, class_means_at_checkpoint)
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_static(metric, curves, plot_keys, colors_rgb, cmap, family_label, n_families, description, layer,
                   layer_label=None, filename_suffix=""):
    """layer is the dict key curves are indexed by (e.g. FINAL_LAYER or
    grid_evolution_mod.POST_LN_KEY); layer_label is what's shown in the axis
    label/title (defaults to layer itself) -- kept separate so a raw key
    like "31_post_ln" can be indexed correctly while still displaying a
    human-readable "31 (post-LN)"."""
    if layer_label is None:
        layer_label = layer
    metric_idx = 0 if metric["key"] == "dc" else 1

    fig, ax = plt.subplots(figsize=(6.5, 6.3))
    for key in plot_keys:
        x_values = curves[key][metric_idx][layer]
        acc_values = curves[key][2]
        color = colors_rgb[key]
        ax.plot(x_values, acc_values, color=color, linewidth=1.2, alpha=0.8,
                 marker="o", markersize=4, zorder=3)
        # Arrowhead on the final segment shows the direction of increasing
        # context length along the curve.
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
              framealpha=1.0, edgecolor="gray", fontsize=7, title="Word list")
    ax.set_xlabel(metric["axis_label_fn"](layer_label))
    ax.set_ylabel("Real accuracy (grid task)")
    ax.set_title(
        f"{metric['title']}\n(context length as parameter; arrow = increasing context length)",
        fontsize=10,
    )
    fig.text(
        0.5, 0.01,
        "\n".join(textwrap.wrap(description, 95)),
        ha="center", va="bottom", fontsize=7, color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))  # reserve bottom 8% of the canvas for the caption
    # save_figure() calls fig.tight_layout() again internally with no rect,
    # which would collapse the margin just reserved above -- neutralize
    # that second call for this figure only.
    fig.tight_layout = lambda *args, **kwargs: None
    filename = f"{metric['filename']}{filename_suffix}"
    save_figure(fig, PLOTS_DIR, f"{filename}.pdf")
    print(f"Saved {filename}.png")


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False

    post_layernorm = "--post-layernorm" in sys.argv[1:]

    keys = family_keys()
    curves = {}
    for key in keys:
        result = load_phase_curve(key, post_layernorm=post_layernorm)
        if result is None:
            cache_name = ("grid_evolution_layers_vs_seqlen_post_ln.npz" if post_layernorm
                          else "grid_evolution_layers_vs_seqlen.npz")
            post_ln_flag = " --post-layernorm" if post_layernorm else ""
            print(f"Skipping {key}: missing {cache_name} (run "
                  f"morphology_grid_evolution_layers_vs_seqlen.py{post_ln_flag} with this "
                  f"key added to SWEEP_KEYS, on a GPU) or accuracies cache.")
            continue
        curves[key] = result

    plot_keys = list(curves.keys())
    print(f"Plotting phase curves for {len(plot_keys)}/{len(keys)} family members:")
    for k in plot_keys:
        print(f"  - {k}")

    key_to_family, n_families = scatter_mod.assign_families(plot_keys)
    cmap = cm.get_cmap("tab20", max(n_families, 1))
    colors_rgb = {k: cmap(key_to_family[k][1])[:3] for k in plot_keys}
    # Reference word list name for each family's color (its shortest member,
    # e.g. "morphology", "text_numbers", "two_digit_numbers").
    family_label = {key_to_family[k][1]: key_to_family[k][0] for k in plot_keys}
    checkpoints_str = ", ".join(str(c) for c in CHECKPOINTS)
    description = (
        f"Each curve traces one word list through (geometry metric, accuracy) "
        f"space as context length grows -- one dot per curve at each of the "
        f"{len(CHECKPOINTS)} context lengths sampled: {checkpoints_str} tokens. "
        f"Dot size and the trailing arrowhead both grow with context length."
    )

    if post_layernorm:
        # Static PNG/PDF only, at the single post-ln "layer" -- the
        # interactive Plotly output below is keyed to the fixed LAYERS
        # sweep (layer dropdown + per-layer thumbnail cache) and isn't
        # worth extending for one extra virtual layer.
        for metric in METRICS.values():
            render_static(metric, curves, plot_keys, colors_rgb, cmap, family_label, n_families, description,
                          grid_evolution_mod.POST_LN_KEY, layer_label="31 (post-LN)", filename_suffix="_post_ln")
        return

    # ── Matplotlib (static) ── one PNG per metric, final layer only ────────
    for metric in METRICS.values():
        render_static(metric, curves, plot_keys, colors_rgb, cmap, family_label, n_families, description, FINAL_LAYER)

    # ── Plotly (interactive) ── one HTML, two dropdowns (metric x layer)
    # switch what's on the x axis and which layer's PCA thumbnails are
    # shown; hover over a point reveals its PCA thumbnail at that layer /
    # context length ──
    FIG_WIDTH, FIG_HEIGHT = 1100, 900
    n_checkpoints = len(CHECKPOINTS)

    print("Rendering per-point PCA thumbnails (one per word list x layer x checkpoint)...")
    # images_by_layer[layer]: list of dicts, key-major/checkpoint-minor
    # order (matches trace order below), holding both possible x positions
    # (one per metric) so the JS layer/metric switcher can reposition them
    # without re-rendering.
    images_by_layer = {}
    acc_all = np.concatenate([curves[key][2] for key in plot_keys])
    y_mid = (float(acc_all.max()) + float(acc_all.min())) / 2
    for layer in LAYERS:
        imgs = []
        for key in plot_keys:
            dc_values, energy_values, acc_values, class_means = curves[key]
            for i, k in enumerate(CHECKPOINTS):
                uri = thumbnail_data_uri(key, layer, k, class_means[(layer, k)])
                y = float(acc_values[i])
                imgs.append(dict(
                    source=uri,
                    dcX=float(dc_values[layer][i]), energyX=float(energy_values[layer][i]),
                    y=y, xanchor="center",
                    # Anchor on whichever side of the point has more room (below
                    # for points in the lower half of the accuracy range, above
                    # for points in the upper half) so it doesn't get clipped by
                    # the plot's top/bottom edge. Accuracy is always the y axis
                    # regardless of which metric/layer is active, so this is
                    # computed once, layer-independent.
                    yanchor="top" if y >= y_mid else "bottom",
                ))
        images_by_layer[layer] = imgs

    # Size thumbnails relative to each (layer, metric) combo's own x-axis
    # data range (distance correlation and Dirichlet energy live on very
    # different scales, and so can different layers' values of the same
    # metric).
    size_by_layer_metric = {}
    for layer in LAYERS:
        dc_all = np.concatenate([curves[key][0][layer] for key in plot_keys])
        energy_all = np.concatenate([curves[key][1][layer] for key in plot_keys])
        dc_range = max(float(dc_all.max() - dc_all.min()), 1e-6)
        energy_range = max(float(energy_all.max() - energy_all.min()), 1e-6)
        dc_sizex = 0.09 * dc_range
        energy_sizex = 0.09 * energy_range
        size_by_layer_metric[layer] = {
            "dc": {"x": dc_sizex, "y": dc_sizex * (FIG_HEIGHT / FIG_WIDTH)},
            "energy": {"x": energy_sizex, "y": energy_sizex * (FIG_HEIGHT / FIG_WIDTH)},
        }

    trace_x = {
        layer: {
            "dc": [curves[key][0][layer].tolist() for key in plot_keys],
            "energy": [curves[key][1][layer].tolist() for key in plot_keys],
        }
        for layer in LAYERS
    }
    axis_label_by_layer_metric = {
        layer: {mkey: METRICS[mkey]["axis_label_fn"](layer) for mkey in METRICS}
        for layer in LAYERS
    }
    title_by_layer_metric = {
        layer: {
            mkey: (
                f"{METRICS[mkey]['title']} -- context length as parameter "
                f"(layer {layer} activations; marker size grows with context length; "
                f"hover or click a point for its PCA plot)"
            )
            for mkey in METRICS
        }
        for layer in LAYERS
    }

    DEFAULT_LAYER, DEFAULT_METRIC = FINAL_LAYER, "dc"

    pfig = go.Figure()
    legend_shown = set()
    for key in plot_keys:
        dc_values = curves[key][0][DEFAULT_LAYER]
        energy_values = curves[key][1][DEFAULT_LAYER]
        acc_values = curves[key][2]
        r, g, b = colors_rgb[key]
        rgb_str = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        family_idx = key_to_family[key][1]
        pfig.add_trace(go.Scatter(
            x=dc_values.tolist(), y=acc_values.tolist(),  # DC, final layer shown by default
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

    def materialize_images(layer, metric_key):
        size = size_by_layer_metric[layer][metric_key]
        out = []
        for img in images_by_layer[layer]:
            out.append(dict(
                source=img["source"], xref="x", yref="y",
                x=img["dcX"] if metric_key == "dc" else img["energyX"],
                y=img["y"], xanchor=img["xanchor"], yanchor=img["yanchor"],
                sizex=size["x"], sizey=size["y"],
                sizing="contain", opacity=0.0, layer="above",
            ))
        return out

    pfig.update_layout(
        title=title_by_layer_metric[DEFAULT_LAYER][DEFAULT_METRIC],
        xaxis_title=axis_label_by_layer_metric[DEFAULT_LAYER][DEFAULT_METRIC],
        yaxis_title="Real accuracy (grid task)",
        template="plotly_white",
        width=FIG_WIDTH, height=FIG_HEIGHT,
        margin=dict(b=110, t=110),
        images=materialize_images(DEFAULT_LAYER, DEFAULT_METRIC),
        annotations=[dict(
            text="<br>".join(textwrap.wrap(description, 120)),
            xref="paper", yref="paper", x=0.5, y=-0.14,
            showarrow=False, align="center",
            font=dict(size=11, color="dimgray"),
        )],
    )

    # Metric + layer selectors (plain HTML <select>s inserted above the
    # plot) and hover/click thumbnail reveal, all driven by one small state
    # object so the two dropdowns can be changed independently -- Plotly's
    # own updatemenus can't do this cleanly since each button's args are
    # static and so can't depend on which option is currently active in a
    # different menu.
    control_js = """
    var gd = document.getElementsByClassName('plotly-graph-div')[0];
    var LAYERS = %(layers_json)s;
    var TRACE_X = %(trace_x_json)s;
    var IMAGES = %(images_json)s;
    var SIZE = %(size_json)s;
    var AXIS_LABEL = %(axis_label_json)s;
    var TITLE = %(title_json)s;
    var N_CHECKPOINTS = %(n_checkpoints)d;

    var state = {layer: %(default_layer_json)s, metric: %(default_metric_json)s};
    var pinned = new Set();

    function buildImages() {
        var size = SIZE[state.layer][state.metric];
        return IMAGES[state.layer].map(function(img) {
            return {
                source: img.source, xref: "x", yref: "y",
                x: state.metric === "dc" ? img.dcX : img.energyX,
                y: img.y, xanchor: img.xanchor, yanchor: img.yanchor,
                sizex: size.x, sizey: size.y,
                sizing: "contain", opacity: 0.0, layer: "above",
            };
        });
    }

    function applyState() {
        pinned.clear();
        Plotly.update(gd, {x: TRACE_X[state.layer][state.metric]}, {
            "xaxis.title.text": AXIS_LABEL[state.layer][state.metric],
            "xaxis.autorange": true,
            "title.text": TITLE[state.layer][state.metric],
            images: buildImages(),
        });
    }

    function makeSelect(labelText, options, onChange) {
        var wrap = document.createElement("span");
        wrap.style.marginRight = "18px";
        var label = document.createElement("label");
        label.textContent = labelText + ": ";
        var select = document.createElement("select");
        options.forEach(function(opt) {
            var o = document.createElement("option");
            o.value = opt.value; o.textContent = opt.label;
            if (opt.selected) o.selected = true;
            select.appendChild(o);
        });
        select.addEventListener("change", function(e) { onChange(e.target.value); });
        label.appendChild(select);
        wrap.appendChild(label);
        return wrap;
    }

    var bar = document.createElement("div");
    bar.style.cssText = "text-align:center;margin-bottom:8px;font-family:sans-serif;font-size:13px;";
    bar.appendChild(makeSelect("Metric", [
        {value: "dc", label: "Distance correlation", selected: state.metric === "dc"},
        {value: "energy", label: "Dirichlet energy", selected: state.metric === "energy"},
    ], function(v) { state.metric = v; applyState(); }));
    bar.appendChild(makeSelect("Layer", LAYERS.map(function(l) {
        return {value: l, label: "Layer " + l, selected: l === state.layer};
    }), function(v) { state.layer = v; applyState(); }));
    gd.parentNode.insertBefore(bar, gd);

    // Hover/click reveal, image index = curveNumber * N_CHECKPOINTS +
    // pointIndex, since traces and images are both key-major/checkpoint-
    // minor in the same order, for every layer.
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
    """ % dict(
        layers_json=json.dumps(LAYERS),
        trace_x_json=json.dumps(trace_x),
        images_json=json.dumps(images_by_layer),
        size_json=json.dumps(size_by_layer_metric),
        axis_label_json=json.dumps(axis_label_by_layer_metric),
        title_json=json.dumps(title_by_layer_metric),
        n_checkpoints=n_checkpoints,
        default_layer_json=json.dumps(DEFAULT_LAYER),
        default_metric_json=json.dumps(DEFAULT_METRIC),
    )

    path = os.path.join(PLOTS_DIR, "distance_correlation_accuracy_phase_plane.html")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    pfig.write_html(path, config={"responsive": False}, include_plotlyjs="cdn", post_script=control_js)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
