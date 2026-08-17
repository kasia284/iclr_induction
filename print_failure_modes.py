"""Prints concrete failure modes of the model on the grid task: for every
position where the model's TOP predicted next token is NOT one of the
current word's valid grid neighbors, prints the context, the valid
answer(s), and what the model predicted instead (with probabilities) --
for actually reading and understanding wrong predictions, rather than
just looking at the aggregate accuracy number.

Requires a fresh forward pass per sequence (same tokenization as
01_reproduce.py's get_model_accuracies_by_id), since no existing cache
keeps each position's ARGMAX prediction or the identity of what token the
model guessed -- only the summed probability on the valid-neighbor subset
("accuracy") is cached anywhere in the repo.

Runs on all N_SEQUENCES=16 random-walk sequences (same set_seed(42) walks
as everywhere else) for WORD_LIST_KEY. Every failure (there can be
thousands, particularly at short context lengths, since accuracy climbs
with context) is written to results/reproduce/data/{key}/
failure_modes.csv; only MAX_PRINT of them are echoed to the console,
evenly spaced across the full (context-length-ordered) list of failures
-- not just the first few, which would be almost all short-context/
harder positions -- so both short- and long-context failures show up in
what gets printed.

Requires a CUDA GPU with >= 48 GB memory (Llama-3.1-8B via
TransformerLens) -- this repo session has no GPU, so this must be run on
a GPU machine; bring results/reproduce/data/{key}/failure_modes.csv back
to reprint/reanalyze without the model.
"""
import csv
import importlib.util
import os
from collections import Counter

import numpy as np
import torch
import tqdm

from utils import Grid, set_seed, load_model
from word_lists import WORD_LISTS

REPO = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reproduce_mod = _load_module("reproduce_01", "01_reproduce.py")

WORD_LIST_KEY = "morphology"
SEQ_LEN = reproduce_mod.SEQ_LEN
MAX_PRINT = 40

FIELDNAMES = ["seq_idx", "position", "context_length", "current_word",
              "valid_next", "predicted_word", "predicted_prob", "correct_prob"]


def cache_path(word_list_key):
    return f"results/reproduce/data/{word_list_key}/failure_modes.csv"


def find_failures(model, word_list_key):
    words = WORD_LISTS[word_list_key]
    side = int(round(len(words) ** 0.5))
    grid = Grid(words=words, rows=side, cols=side)
    word_to_id = reproduce_mod.build_word_token_ids(model, words)
    id_to_word = {v: k for k, v in word_to_id.items()}

    set_seed(42)
    sequences = grid.generate_batch(SEQ_LEN, len(words))

    failures = []
    for seq_idx, seq in enumerate(tqdm.tqdm(sequences, desc=f"{word_list_key}: sequences")):
        tokens = reproduce_mod.tokenize_sequence_by_id(model, seq, word_to_id)
        logits = model.run_with_hooks(tokens.to(model.cfg.device))
        probs = torch.softmax(logits, dim=-1)[0, 1:, :]  # remove BOS position

        for i in range(len(seq)):
            valid_next = grid.get_valid_next_words(seq[i])
            valid_ids = [word_to_id[w] for w in valid_next]
            top_id = int(probs[i].argmax().item())
            if top_id in valid_ids:
                continue  # correct -- not a failure

            top_word = id_to_word.get(top_id, model.tokenizer.decode([top_id]))
            top_prob = float(probs[i, top_id].item())
            correct_prob = float(probs[i][torch.tensor(valid_ids)].sum().item())
            failures.append(dict(
                seq_idx=seq_idx, position=i, context_length=i + 1,
                current_word=seq[i], valid_next="|".join(valid_next),
                predicted_word=top_word, predicted_prob=top_prob,
                correct_prob=correct_prob,
            ))

    path = cache_path(word_list_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(failures)
    print(f"Cached {len(failures)} failures to {path}")
    return failures


def load_or_find_failures(model, word_list_key):
    path = cache_path(word_list_key)
    if os.path.exists(path):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            failures = list(reader)
        for row in failures:
            row["seq_idx"] = int(row["seq_idx"])
            row["position"] = int(row["position"])
            row["context_length"] = int(row["context_length"])
            row["predicted_prob"] = float(row["predicted_prob"])
            row["correct_prob"] = float(row["correct_prob"])
        return failures
    if model is None:
        raise RuntimeError(f"No cache at {path} and no model loaded to compute it.")
    return find_failures(model, word_list_key)


def print_sample(failures, word_list_key, max_print=MAX_PRINT):
    n_total = len(failures)
    print(f"\n{word_list_key}: {n_total} total failures")
    if n_total == 0:
        return

    idxs = sorted(set(np.linspace(0, n_total - 1, min(max_print, n_total)).astype(int).tolist()))
    for i in idxs:
        f = failures[i]
        print(
            f"[seq {f['seq_idx']:2d}, context_len {f['context_length']:4d}] "
            f"current='{f['current_word']}' valid_next={f['valid_next'].split('|')}\n"
            f"    model predicted '{f['predicted_word']}' (p={f['predicted_prob']:.4f}) "
            f"| total prob on correct answers: {f['correct_prob']:.4f}"
        )

    wrong_word_counts = Counter(f["predicted_word"] for f in failures)
    print(f"\nMost common wrong predictions for '{word_list_key}':")
    for word, count in wrong_word_counts.most_common(10):
        print(f"  {word!r:15s} {count:5d}  ({100 * count / n_total:.1f}% of failures)")


def main():
    model = None
    if not os.path.exists(cache_path(WORD_LIST_KEY)):
        model = load_model()

    failures = load_or_find_failures(model, WORD_LIST_KEY)
    print_sample(failures, WORD_LIST_KEY)


if __name__ == "__main__":
    main()
