"""Layer-sweep PCA (layers 0, 6, 13, 20, 26, 31) and the accuracy-vs-context-
length overlay for capital_letters and capital_letters_permuted.

Unlike the word lists this has already been run for (text_numbers,
two_digit_numbers, grammar_vs_concrete_nouns), no activation/accuracy cache
exists yet for these two, so this does real forward passes through the full
Llama-3.1-8B model, then plots via 01_reproduce.py's plot_pca_across_layers
and plot_accuracy_comparison (same functions used for the other pairs).

Requires a GPU (loads the full 8B-parameter model) -- run via runai submit,
not on a CPU-only shell.
"""
import importlib.util
import os

import numpy as np
import tqdm

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)

spec = importlib.util.spec_from_file_location("reproduce01", os.path.join(REPO, "01_reproduce.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mod.setup_plotting()
mod.plt.rcParams["text.usetex"] = False

model = mod.load_model()

for key in ["capital_letters", "capital_letters_permuted"]:
    print(f"=== {key} ===")
    mod.configure_for_word_list(key)
    os.makedirs(mod.DATA_DIR, exist_ok=True)

    word_to_id = mod.build_word_token_ids(model, mod.WORDS)
    grid = mod.Grid(words=mod.WORDS, rows=mod.GRID_ROWS, cols=mod.GRID_COLS)
    tag = f"Grid_{mod.WORD_LIST_KEY}"

    mod.set_seed(42)
    sequence = grid.generate_sequence(mod.SEQ_LEN)

    layer_class_means = {}
    for layer in mod.LAYERS:
        acts = mod.get_activations_by_id(model, sequence, word_to_id, layer, mod.N_LOOKBACK)
        layer_class_means[layer] = mod.compute_class_means(
            acts, sequence, mod.WORDS, mod.N_LOOKBACK
        ).cpu().numpy()

    layers_path = os.path.join(mod.DATA_DIR, f"pca_layers_{tag}.npz")
    np.savez(layers_path, **{str(l): layer_class_means[l] for l in mod.LAYERS})
    print(f"Cached {layers_path}")

    with open(os.path.join(mod.DATA_DIR, f"sequence_{tag}.json"), "w") as f:
        import json
        json.dump(sequence, f)

    mod.plot_pca_across_layers(grid, layer_class_means, mod.LAYERS, suffix=f"_{tag}", top_n=2)
    mod.plot_pca_across_layers(grid, layer_class_means, mod.LAYERS, suffix=f"_{tag}", top_n=3)

    acc_path = os.path.join(mod.DATA_DIR, f"accuracies_{tag}.npz")
    mod.set_seed(42)
    sequences = grid.generate_batch(mod.SEQ_LEN, mod.N_SEQUENCES)
    all_accs = []
    for seq in tqdm.tqdm(sequences, desc=f"Accuracy curves ({key})"):
        all_accs.append(mod.get_model_accuracies_by_id(model, grid, seq, word_to_id))
    all_accs = np.array(all_accs)
    np.savez(acc_path, all_accs=all_accs)
    print(f"Cached {acc_path}")

comparison_keys = ["capital_letters", "capital_letters_permuted"]
accs_by_key = {k: mod.load_cached_accuracies(k) for k in comparison_keys}
mod.plot_accuracy_comparison(
    accs_by_key, out_dir="results/reproduce/plots/capital_letters_comparison",
)

print("Done.")
