import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

model_id = "meta-llama/Llama-3.1-8B"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
model.eval()

E = model.get_input_embeddings().weight.detach().numpy()  # (vocab_size, hidden_dim)

# --- Filter to "real" single tokens: alphabetic, decodes cleanly, reasonable length ---
vocab_size = E.shape[0]
candidate_ids = []
candidate_strs = []
for tid in range(vocab_size):
    s = tok.decode([tid])
    stripped = s.strip()
    if stripped.isalpha() and 2 <= len(stripped) <= 12 and s.startswith(" "):
        candidate_ids.append(tid)
        candidate_strs.append(stripped)

print(f"Filtered candidate pool: {len(candidate_ids)} tokens")

cand_E = E[candidate_ids]

# --- Global PCA over the candidate pool to find top-2 directions ---
pca_global = PCA(n_components=2)
proj_global = pca_global.fit_transform(cand_E)

# --- Define a 4x4 grid spanning the 5th-95th percentile range (avoid outliers) ---
x_lo, x_hi = np.percentile(proj_global[:, 0], [5, 95])
y_lo, y_hi = np.percentile(proj_global[:, 1], [5, 95])
grid_x = np.linspace(x_lo, x_hi, 4)
grid_y = np.linspace(y_lo, y_hi, 4)
target_points = np.array([[gx, gy] for gx in grid_x for gy in grid_y])

# --- Greedily assign nearest unused token to each grid point ---
selected_idx = []
used = set()
for tx, ty in target_points:
    dists = np.linalg.norm(proj_global - np.array([tx, ty]), axis=1)
    order = np.argsort(dists)
    for idx in order:
        if idx not in used:
            used.add(idx)
            selected_idx.append(idx)
            break

selected_ids = [candidate_ids[i] for i in selected_idx]
selected_strs = [candidate_strs[i] for i in selected_idx]
print("Selected tokens:", selected_strs)

# --- Verify: PCA on JUST these 16 real embeddings ---
final_E = E[selected_ids]
pca_final = PCA(n_components=2)
final_proj = pca_final.fit_transform(final_E)
print("Explained variance ratio (top 2):", pca_final.explained_variance_ratio_)

plt.figure(figsize=(7, 7))
for (x, y), s in zip(final_proj, selected_strs):
    plt.scatter(x, y)
    plt.annotate(s, (x, y), fontsize=9)
plt.title("PCA of 16 real Llama-3.1-8B token embeddings")
plt.savefig("grid_check.png", dpi=150)
plt.close()  # frees memory, avoids leftover figure state if you loop/re-run