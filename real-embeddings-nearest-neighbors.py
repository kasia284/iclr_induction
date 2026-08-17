"""Find a set of 16 token embeddings from Llama's embedding matrix (W_E)
whose pairwise Gram matrix (raw, unnormalized inner products) has entries
that are as high as possible -- i.e. a "clique" of mutually similar,
high-magnitude embeddings, rather than just the neighbors of one fixed query
token.

Metric: raw inner product w_i . w_j (the literal Gram matrix entry), not
cosine similarity -- consistent with the "Gram Matrix" panel in
real-embeddings-gram-matrix.py, as opposed to its separate "Cosine
Similarity" panel. Because this is unnormalized, both magnitude and
direction matter: an entry w_i . w_j is large only when both vectors are
large AND well-aligned (Cauchy-Schwarz: w_i . w_j <= |w_i| |w_j|).
"""
import glob
import json
import os
import re

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from utils import MODEL_NAME, set_seed

K = 16  # size of the clique to find
CANDIDATE_POOL_SIZE = 3000  # tokens considered, ranked by embedding norm
N_SWAP_PASSES = 3  # local-search refinement passes after the greedy build

# Byte-fallback tokens (e.g. "<0x0A>") and other non-printable / reserved
# tokens pollute a norm-based candidate pool (some reserved slots have
# outlier norms) without being interpretable "words", so they're excluded
# up front.
_BYTE_FALLBACK_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


class _EmbeddingOnlyModel:
    """Minimal stand-in for utils.load_model()'s HookedTransformer, exposing
    just .W_E and .tokenizer. Loading the full 8B-parameter model (all 32
    layers) is unnecessary for a Gram-matrix search over token embeddings
    and OOMs on memory-constrained (e.g. CPU-only, sandboxed) machines; this
    reads only the embedding matrix (~1GB in bf16) directly out of its
    safetensors shard instead."""

    def __init__(self, W_E, tokenizer):
        self.W_E = W_E
        self.tokenizer = tokenizer


def load_embedding_only_model():
    hf_home = os.environ.get("HF_HOME")
    cache_root = os.path.join(hf_home, "hub") if hf_home else os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = os.path.join(cache_root, f"models--{MODEL_NAME.replace('/', '--')}")
    snapshot_dirs = glob.glob(os.path.join(model_dir, "snapshots", "*"))
    if not snapshot_dirs:
        raise FileNotFoundError(f"No local snapshot found for {MODEL_NAME} under {model_dir}")
    snapshot_dir = snapshot_dirs[0]

    with open(os.path.join(snapshot_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    shard_path = os.path.join(snapshot_dir, weight_map["model.embed_tokens.weight"])

    with safe_open(shard_path, framework="pt", device="cpu") as f:
        W_E = f.get_tensor("model.embed_tokens.weight").float()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _EmbeddingOnlyModel(W_E, tokenizer)


def build_candidate_pool(model, pool_size=CANDIDATE_POOL_SIZE):
    """Rank all vocab tokens by raw embedding norm (a necessary condition
    for a large raw inner product, by Cauchy-Schwarz) and return the top
    `pool_size` token ids, after dropping special/byte-fallback tokens."""
    W_E = model.W_E  # [d_vocab, d_model], raw (unnormalized) embeddings
    norms = W_E.norm(dim=-1)

    special_ids = set(model.tokenizer.all_special_ids)
    d_vocab = W_E.shape[0]
    valid_mask = torch.ones(d_vocab, dtype=torch.bool)
    for tid in range(d_vocab):
        if tid in special_ids:
            valid_mask[tid] = False
            continue
        s = model.tokenizer.decode([tid])
        if not s.strip() or _BYTE_FALLBACK_RE.match(s.strip()) or "reserved_special_token" in s:
            valid_mask[tid] = False

    masked_norms = norms.clone()
    masked_norms[~valid_mask] = -float("inf")
    pool_ids = torch.topk(masked_norms, pool_size).indices
    return pool_ids


def find_high_gram_clique(model, k=K, pool_size=CANDIDATE_POOL_SIZE, n_swap_passes=N_SWAP_PASSES):
    """Greedily build, then locally refine, a k-token subset maximizing the
    sum of off-diagonal Gram matrix entries (raw inner products).

    Returns: (chosen_ids: list[int], gram: [k, k] tensor over chosen_ids,
    pool_ids: the candidate pool the search ran over).
    """
    pool_ids = build_candidate_pool(model, pool_size)
    pool_embeds = model.W_E[pool_ids]  # [pool_size, d_model]
    gram_pool = pool_embeds @ pool_embeds.T  # [pool_size, pool_size], raw inner products
    n = gram_pool.shape[0]

    # ── Greedy construction ──────────────────────────────────────────────
    # Seed with the single best-scoring pair, then repeatedly add whichever
    # remaining candidate maximizes the sum of its inner products with the
    # tokens already selected (i.e. the marginal gain in total off-diagonal
    # Gram mass).
    off_diag = gram_pool.clone()
    off_diag.fill_diagonal_(-float("inf"))
    flat_idx = torch.argmax(off_diag)
    i0, j0 = divmod(flat_idx.item(), n)
    selected = [i0, j0]
    remaining = set(range(n)) - set(selected)

    while len(selected) < k:
        sel_tensor = torch.tensor(selected)
        gains = gram_pool[sel_tensor][:, list(remaining)].sum(dim=0)  # [len(remaining)]
        remaining_list = list(remaining)
        best_local = torch.argmax(gains).item()
        best_candidate = remaining_list[best_local]
        selected.append(best_candidate)
        remaining.remove(best_candidate)

    # ── Local-search refinement ──────────────────────────────────────────
    # A handful of passes trying to swap out each member of the current set
    # for the best available outside candidate, keeping the swap only if it
    # increases the total off-diagonal Gram mass. Each position's gain over
    # all remaining candidates is computed as one vectorized op (rather than
    # a Python loop over ~pool_size candidates), so this stays cheap.
    for _ in range(n_swap_passes):
        improved = False
        for pos in range(len(selected)):
            others = selected[:pos] + selected[pos + 1:]
            others_t = torch.tensor(others)
            current_contrib = gram_pool[selected[pos], others_t].sum()

            remaining = [c for c in range(n) if c not in selected]
            remaining_t = torch.tensor(remaining)
            cand_contribs = gram_pool[remaining_t][:, others_t].sum(dim=1)

            best_local = torch.argmax(cand_contribs).item()
            best_gain = (cand_contribs[best_local] - current_contrib).item()
            if best_gain > 1e-6:
                selected[pos] = remaining[best_local]
                improved = True
        if not improved:
            break

    chosen_ids = pool_ids[torch.tensor(selected)].tolist()
    chosen_embeds = model.W_E[torch.tensor(chosen_ids)]
    gram = chosen_embeds @ chosen_embeds.T
    return chosen_ids, gram, pool_ids


def main():
    set_seed(42)
    model = load_embedding_only_model()

    chosen_ids, gram, _ = find_high_gram_clique(model)

    off_diag = gram.clone()
    off_diag.fill_diagonal_(0.0)
    mean_off_diag = off_diag.sum().item() / (gram.shape[0] * (gram.shape[0] - 1))

    print(f"Selected {len(chosen_ids)} tokens (mean off-diagonal Gram entry = {mean_off_diag:.4f}):")
    for rank, tid in enumerate(chosen_ids, start=1):
        s = model.tokenizer.decode([tid])
        print(f"  {rank:2d}. id={tid:<8} norm={model.W_E[tid].norm().item():.2f}  repr={s!r}")

    print("\nGram matrix (rows/cols in selection order):")
    print(gram)


if __name__ == "__main__":
    main()
