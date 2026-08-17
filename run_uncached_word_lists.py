"""Runs 01_reproduce.py's main() for every WORD_LISTS key that doesn't
yet have a cached accuracy curve + final-layer activation file --
in particular the 48 new grid-position permutations added by
generate_extra_grid_permutations.py, plus any other never-run word list.

Requires a CUDA GPU with >= 48 GB memory (Llama-3.1-8B via
TransformerLens) and a Hugging Face login with access to the gated
meta-llama/Llama-3.1-8B checkpoint -- see README.md's Setup section.
Run this on a GPU machine; this repo session has no GPU available.

Safe to re-run / interrupt: 01_reproduce.py's main() checks its own cache
files per word list and skips model computation entirely for any key
that's already fully cached, so this only ever does work for keys that
are actually missing data.

Usage:
    python run_uncached_word_lists.py
    python run_uncached_word_lists.py <key> [<key> ...]   # only these keys
"""
import importlib.util
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reproduce_mod = _load_module("reproduce_01", "01_reproduce.py")

from word_lists import WORD_LISTS


def is_cached(key):
    acc_path = f"results/reproduce/data/{key}/accuracies_Grid_{key}.npz"
    pca_all_layers_path = f"results/reproduce/data/{key}/pca_all_layers_Grid_{key}.npz"
    return os.path.exists(acc_path) and os.path.exists(pca_all_layers_path)


def main():
    requested = sys.argv[1:]
    if requested:
        unknown = [k for k in requested if k not in WORD_LISTS]
        if unknown:
            raise SystemExit(f"Unknown word list key(s): {unknown}")
        already_cached = [k for k in requested if is_cached(k)]
        if already_cached:
            print(f"Already cached, skipping: {already_cached}")
        todo = [k for k in requested if not is_cached(k)]
        print(f"{len(requested)} keys requested, {len(todo)} missing cached data.")
    else:
        todo = [k for k in WORD_LISTS if not is_cached(k)]
        print(f"{len(WORD_LISTS)} word lists total, {len(todo)} missing cached data.")

    succeeded, failed = [], []
    for i, key in enumerate(todo, 1):
        print(f"\n=== [{i}/{len(todo)}] {key} ===")
        try:
            reproduce_mod.configure_for_word_list(key)
            reproduce_mod.main()
            succeeded.append(key)
        except ValueError as e:
            # e.g. a word that isn't a single token (multilingual_father,
            # chemical_elements, and their _rand variants) -- skip and
            # keep going rather than aborting the whole run.
            print(f"Skipping {key}: {e}")
            failed.append((key, str(e)))

    print(f"\nDone. {len(succeeded)} succeeded, {len(failed)} skipped.")
    if failed:
        print("Skipped:")
        for key, reason in failed:
            print(f"  {key}: {reason}")


if __name__ == "__main__":
    main()
