"""Fits two kinds of linear maps from embedding vectors (W_E) to
unembedding vectors (W_U), and compares their held-out fit quality:

  1. Ridge regression -- a general affine map Y ~ X @ W + b, fit by
     minimizing ||Xc @ W - Yc||^2 + lambda||W||^2 (Xc/Yc = mean-centered
     X/Y), swept over several lambda and picked by held-out R^2. Maximum
     flexibility (d_model^2 free parameters in W): can stretch/shear
     different directions differently.

  2. Orthogonal Procrustes (Umeyama's algorithm) -- a similarity
     transform Y ~ s * (X @ R) + b, where R is constrained ORTHOGONAL
     (R^T R = I) and s is a single global scale. Preserves all pairwise
     angles/relative distances between embedding vectors exactly; only
     reorients + uniformly rescales the point cloud. Far fewer effective
     degrees of freedom than ridge.

If Procrustes fits nearly as well as ridge, unembeddings are basically a
rotated+rescaled copy of embeddings (same relative geometry). If ridge
does much better, the relationship involves real distortion that a rigid
transform can't capture.

Both are fit on a large candidate pool of vocabulary tokens (NOT the
16-64 words in any one word list -- with d_model=4096 that's wildly
underdetermined: fewer equations than unknowns, would just memorize),
built via real-embeddings-nearest-neighbors.py's build_candidate_pool
(ranks tokens by embedding norm, drops special/byte-fallback tokens),
then split 80/20 into a fit set and a held-out set. Reports mean cosine
similarity, mean L2 distance, and R^2 on: the fit set, the held-out pool,
and each of EVAL_WORD_LISTS' own words (the thing we actually care about).

CPU-only, no GPU / forward pass. Compute cost is dominated by one
[n_train, d_model] x [n_train, d_model] -> [d_model, d_model] matmul (for
Xc^T Xc and Xc^T Yc) and one [d_model, d_model] SVD -- expect a few
minutes with POOL_SIZE=30000.
"""
import importlib.util
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import setup_plotting, save_figure, set_seed
from word_lists import WORD_LISTS

REPO = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nn_mod = _load_module("real_embeddings_nearest_neighbors", "real-embeddings-nearest-neighbors.py")
dot_product_mod = _load_module("activation_unembedding_dot_product", "activation-unembedding-dot-product.py")

PLOTS_DIR = "results/embedding-to-unembedding-linear-fit"
FIT_CACHE_PATH = os.path.join(PLOTS_DIR, "fitted_maps.npz")
POOL_SIZE = 30000
TEST_FRACTION = 0.2
LAMBDAS = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
EVAL_WORD_LISTS = ["text_numbers", "morphology", "morphology_permuted", "world_cities"]


def ridge_precompute(X_train, Y_train):
    """Xc^T Xc and Xc^T Yc, computed once and reused across the whole
    lambda sweep (and by Procrustes, which needs the same Xc^T Yc)."""
    X_mean = X_train.mean(axis=0, keepdims=True)
    Y_mean = Y_train.mean(axis=0, keepdims=True)
    Xc = X_train - X_mean
    Yc = Y_train - Y_mean
    A = Xc.T @ Xc
    B = Xc.T @ Yc
    return A, B, X_mean, Y_mean


def ridge_solve(A, B, X_mean, Y_mean, lam):
    d = A.shape[0]
    W = np.linalg.solve(A + lam * np.eye(d), B)
    b = Y_mean - X_mean @ W
    return W, b


def fit_procrustes(A, B, X_mean, Y_mean):
    """Orthogonal Procrustes + optimal uniform scale (Umeyama's formula):
    R = U V^T from SVD of B = Xc^T Yc = U S V^T; scale = trace(S) / ||Xc||_F^2
    = trace(S) / trace(A)."""
    U, S, Vt = np.linalg.svd(B)
    R = U @ Vt
    scale = S.sum() / np.trace(A)
    W = scale * R
    b = Y_mean - X_mean @ W
    return W, b, scale


def evaluate(X, Y, W, b):
    Y_pred = X @ W + b
    diffs = Y - Y_pred
    l2 = np.linalg.norm(diffs, axis=1)
    cos = np.sum(Y_pred * Y, axis=1) / (
        np.linalg.norm(Y_pred, axis=1) * np.linalg.norm(Y, axis=1) + 1e-12
    )
    ss_res = (diffs ** 2).sum()
    ss_tot = ((Y - Y.mean(axis=0, keepdims=True)) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return dict(mean_cos=float(cos.mean()), median_cos=float(np.median(cos)),
                mean_l2=float(l2.mean()), r2=float(r2)), cos


def main():
    setup_plotting()
    plt.rcParams['text.usetex'] = False
    set_seed(42)

    print("Loading embedding (W_E) and unembedding (W_U) matrices...")
    embed_model = nn_mod.load_embedding_only_model()
    W_E, tokenizer = embed_model.W_E, embed_model.tokenizer
    W_U = dot_product_mod.load_unembedding_matrix()

    print(f"Building candidate pool of {POOL_SIZE} tokens "
          f"(ranked by embedding norm, excluding special/byte-fallback tokens)...")
    pool_ids = nn_mod.build_candidate_pool(embed_model, pool_size=POOL_SIZE).numpy()

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(pool_ids))
    n_test = int(len(pool_ids) * TEST_FRACTION)
    test_ids = pool_ids[perm[:n_test]]
    train_ids = pool_ids[perm[n_test:]]
    print(f"Fit set: {len(train_ids)} tokens, held-out set: {len(test_ids)} tokens")

    X_train = W_E[train_ids].double().numpy()
    Y_train = W_U[train_ids].double().numpy()
    X_test = W_E[test_ids].double().numpy()
    Y_test = W_U[test_ids].double().numpy()

    A, B, X_mean, Y_mean = ridge_precompute(X_train, Y_train)

    # ── Ridge: sweep lambda, pick best by held-out R^2 ──────────────────────
    print("\nFitting ridge regression (general linear map), sweeping lambda...")
    best_ridge = None
    for lam in LAMBDAS:
        W_r, b_r = ridge_solve(A, B, X_mean, Y_mean, lam)
        train_metrics, _ = evaluate(X_train, Y_train, W_r, b_r)
        test_metrics, _ = evaluate(X_test, Y_test, W_r, b_r)
        print(f"  lambda={lam:>10.1f}  train R^2={train_metrics['r2']:.4f}  "
              f"held-out R^2={test_metrics['r2']:.4f}  held-out cos={test_metrics['mean_cos']:.4f}")
        if best_ridge is None or test_metrics['r2'] > best_ridge[3]['r2']:
            best_ridge = (lam, W_r, b_r, test_metrics, train_metrics)
    best_lam, W_ridge, b_ridge, ridge_test_metrics, ridge_train_metrics = best_ridge
    print(f"Best ridge lambda (by held-out R^2): {best_lam}")

    # ── Procrustes: rotation + uniform scale ────────────────────────────────
    print("\nFitting orthogonal Procrustes (rotation + scale)...")
    W_proc, b_proc, scale = fit_procrustes(A, B, X_mean, Y_mean)
    proc_train_metrics, _ = evaluate(X_train, Y_train, W_proc, b_proc)
    proc_test_metrics, _ = evaluate(X_test, Y_test, W_proc, b_proc)
    print(f"  scale={scale:.4f}  train R^2={proc_train_metrics['r2']:.4f}  "
          f"held-out R^2={proc_test_metrics['r2']:.4f}  held-out cos={proc_test_metrics['mean_cos']:.4f}")

    # ── Null baseline: always predict the mean unembedding ──────────────────
    W_null = np.zeros_like(W_ridge)
    b_null = Y_mean
    null_test_metrics, _ = evaluate(X_test, Y_test, W_null, b_null)

    print("\n=== Held-out pool summary ===")
    print(f"{'Method':<25}{'mean cos':>10}{'median cos':>12}{'mean L2':>10}{'R^2':>10}")
    for name, m in [("Ridge (best lambda)", ridge_test_metrics),
                    ("Procrustes (rot+scale)", proc_test_metrics),
                    ("Null (mean unembedding)", null_test_metrics)]:
        print(f"{name:<25}{m['mean_cos']:>10.4f}{m['median_cos']:>12.4f}{m['mean_l2']:>10.4f}{m['r2']:>10.4f}")

    # ── Per-word-list evaluation ─────────────────────────────────────────────
    print("\n=== Per-word-list held-out evaluation ===")
    print(f"{'Word list':<30}{'Ridge cos':>12}{'Procrustes cos':>16}{'Ridge R^2':>12}{'Procrustes R^2':>16}")
    for key in EVAL_WORD_LISTS:
        words = WORD_LISTS[key]
        try:
            token_ids = [dot_product_mod.to_single_token(tokenizer, w) for w in words]
        except ValueError as e:
            print(f"Skipping {key}: {e}")
            continue
        ids_t = torch.tensor(token_ids, dtype=torch.long)
        X_list = W_E[ids_t].double().numpy()
        Y_list = W_U[ids_t].double().numpy()
        ridge_m, _ = evaluate(X_list, Y_list, W_ridge, b_ridge)
        proc_m, _ = evaluate(X_list, Y_list, W_proc, b_proc)
        print(f"{key:<30}{ridge_m['mean_cos']:>12.4f}{proc_m['mean_cos']:>16.4f}"
              f"{ridge_m['r2']:>12.4f}{proc_m['r2']:>16.4f}")

    # ── Plot: held-out cosine similarity distributions ───────────────────────
    _, cos_ridge = evaluate(X_test, Y_test, W_ridge, b_ridge)
    _, cos_proc = evaluate(X_test, Y_test, W_proc, b_proc)
    _, cos_null = evaluate(X_test, Y_test, W_null, b_null)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bins = np.linspace(-1, 1, 60)
    ax.hist(cos_null, bins=bins, alpha=0.4, label="Null (mean unembedding)", color="gray")
    ax.hist(cos_ridge, bins=bins, alpha=0.55, label=f"Ridge (lambda={best_lam:g})", color="#377eb8")
    ax.hist(cos_proc, bins=bins, alpha=0.55, label="Procrustes (rotation+scale)", color="#e41a1c")
    ax.set_xlabel("Cosine similarity (predicted vs. true unembedding)")
    ax.set_ylabel("Count (held-out tokens)")
    ax.set_title(f"Held-out fit quality: ridge vs. Procrustes (n={len(test_ids)})", fontsize=10)
    ax.legend(loc="upper left", frameon=True, framealpha=1.0, edgecolor="gray", fontsize=8)
    save_figure(fig, PLOTS_DIR, "held_out_cosine_similarity_hist.pdf")
    print("\nSaved held_out_cosine_similarity_hist")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    np.savez(
        FIT_CACHE_PATH,
        W_ridge=W_ridge, b_ridge=b_ridge, best_lambda=best_lam,
        W_proc=W_proc, b_proc=b_proc, scale=scale,
    )
    print(f"Saved fitted maps to {FIT_CACHE_PATH}")


if __name__ == "__main__":
    main()
