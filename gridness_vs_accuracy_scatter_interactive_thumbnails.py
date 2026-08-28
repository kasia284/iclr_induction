"""Interactive version of gridness_vs_accuracy_scatter_with_thumbnails.py:
same scatter (final-layer distance correlation vs. real accuracy, one
point per word list, colored by word-set family), but instead of showing
all 53 PCA thumbnails at once (crowded/overlapping in the static PNG),
each thumbnail is HIDDEN by default and only revealed by hovering over or
clicking its point:

  - hover a point  -> its thumbnail appears (temporarily, hides again on
    mouseout, unless pinned)
  - click a point   -> its thumbnail is "pinned" (stays visible even
    after the mouse moves away); click again to unpin

This needs no server/Dash -- it's a single self-contained HTML file.
Thumbnails are embedded as base64 data URIs in Plotly's layout.images
(one per point, opacity 0 by default), and a small vanilla-JS snippet
(injected via fig.write_html's post_script, since utils.save_plotly
doesn't expose that) listens for Plotly's plotly_hover / plotly_unhover /
plotly_click browser events and toggles each image's opacity via
Plotly.relayout -- all client-side, works when just opened in a browser.

Reuses thumbnail PNGs already rendered by gridness_vs_accuracy_scatter_with_thumbnails.py (results/reproduce/plots/gridness_vs_accuracy/
thumbnails/{key}.png), rendering any that are missing. Both companion
scripts are imported via importlib, consistent with how sibling scripts
in this repo share helpers. Pure cache reads + thumbnail rendering, no GPU/model
needed (beyond what the thumbnail renderer itself needs, which is also
CPU-only / cache-only).
"""
import base64
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


scatter_mod = _load_module("gridness_vs_accuracy_scatter", "gridness_vs_accuracy_scatter.py")
thumb_mod = _load_module("gridness_vs_accuracy_thumbnails", "gridness_vs_accuracy_scatter_with_thumbnails.py")

PLOTS_DIR = "results/reproduce/plots/gridness_vs_accuracy"
FIG_WIDTH, FIG_HEIGHT = 1400, 1000

HOVER_JS = """
var gd = document.getElementsByClassName('plotly-graph-div')[0];
var pinned = new Set();
function setOpacity(idx, val) {
    var upd = {};
    upd['images[' + idx + '].opacity'] = val;
    Plotly.relayout(gd, upd);
}
gd.on('plotly_hover', function(data) {
    var idx = data.points[0].pointIndex;
    setOpacity(idx, 1);
});
gd.on('plotly_unhover', function(data) {
    var idx = data.points[0].pointIndex;
    if (!pinned.has(idx)) { setOpacity(idx, 0); }
});
gd.on('plotly_click', function(data) {
    var idx = data.points[0].pointIndex;
    if (pinned.has(idx)) { pinned.delete(idx); setOpacity(idx, 0); }
    else { pinned.add(idx); setOpacity(idx, 1); }
});
"""


def thumbnail_data_uri(key):
    path = os.path.join(thumb_mod.THUMB_DIR, f"{key}.png")
    if not os.path.exists(path):
        path = thumb_mod.render_thumbnail(key)
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def main():
    keys = scatter_mod.SWEEP_KEYS
    print(f"Building interactive plot for {len(keys)} word lists...")

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
    # This plotly version doesn't support xsizemode/ysizemode="pixel" (fixed
    # pixel size regardless of zoom), so size images in data-axis units
    # instead: chosen so sizex/x_range and sizey/y_range roughly match the
    # figure's width/height ratio, i.e. thumbnails render close to square.
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
        uri = thumbnail_data_uri(key)
        yanchor = "top" if y >= y_mid else "bottom"
        images.append(dict(
            source=uri, xref="x", yref="y", x=float(x), y=float(y),
            xanchor="center", yanchor=yanchor,
            sizex=sizex, sizey=sizey,
            sizing="contain", opacity=0.0, layer="above",
        ))

    fig.update_layout(
        title=f"Final-layer gridness vs. real accuracy -- hover or click a point for its PCA plot (n={len(keys)}, r={r:.3f})",
        xaxis_title="Distance correlation (final-layer activations vs. grid)",
        yaxis_title="Real accuracy (full context, seq len 1400)",
        template="plotly_white",
        images=images,
        width=FIG_WIDTH, height=FIG_HEIGHT,
    )

    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "gridness_vs_accuracy_interactive_thumbnails.html")
    fig.write_html(path, config={"responsive": False}, include_plotlyjs="cdn", post_script=HOVER_JS)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
