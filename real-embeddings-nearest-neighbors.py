"""Pick a random token embedding from Llama's embedding matrix (W_E) and
find its 15 nearest neighbors by cosine similarity."""

import torch

from utils import load_model, set_seed


def find_nearest_neighbors(model, token_id, k=15):
    """Return the k nearest neighbors (by cosine similarity) to
    W_E[token_id], excluding the token itself.

    Returns: (neighbor_ids, similarities), each a list of length k.
    """
    W_E = model.W_E  # [d_vocab, d_model]
    query = W_E[token_id]

    W_E_norm = W_E / W_E.norm(dim=-1, keepdim=True)
    query_norm = query / query.norm()

    sims = W_E_norm @ query_norm  # [d_vocab]
    sims[token_id] = -float("inf")  # exclude the query token itself

    top_sims, top_ids = torch.topk(sims, k)
    return top_ids.tolist(), top_sims.tolist()


def main():
    set_seed(42)
    model = load_model()

    d_vocab = model.cfg.d_vocab
    query_id = torch.randint(0, d_vocab, (1,)).item()
    query_str = model.tokenizer.decode([query_id])
    print(f"Query token: id={query_id}  repr={query_str!r}")

    neighbor_ids, sims = find_nearest_neighbors(model, query_id, k=15)

    print("\n15 nearest neighbors (cosine similarity):")
    for rank, (nid, sim) in enumerate(zip(neighbor_ids, sims), start=1):
        nstr = model.tokenizer.decode([nid])
        print(f"  {rank:2d}. id={nid:<8} sim={sim:.4f}  repr={nstr!r}")


if __name__ == "__main__":
    main()
