"""Shared utilities for reproducing 'In-context learning of representations
can be explained by induction circuits.'"""

import os
import random
import functools

from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors
import torch
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import einops
import math
from scipy.sparse.csgraph import shortest_path


from torch import Tensor
from jaxtyping import Float
from transformer_lens import HookedTransformer, HookedTransformerConfig, utils
from transformer_lens.hook_points import HookPoint
from typing import Tuple
import plotly.graph_objects as go


torch.set_grad_enabled(False)

# ── Constants ──────────────────────────────────────────────────────────────────

# WORDS = [
#     "apple", "bird", "car", "egg",
#     "house", "milk", "plane", "opera",
#     "box", "sand", "sun", "mango",
#     "rock", "math", "code", "phone",
# ]

# WORDS = [
#     "Paris", "London", "Berlin", "Rome",
#     "Madrid", "Tokyo", "Beijing", "Seoul",
#     "Cairo", "Lima", "Austin", "Boston",
#     "Denver", "Miami", "Phoenix", "Chicago",
# ]

WORDS = [
    "one", "two", "three", "four", 
    "five", "six", "seven", "eight", 
    "nine", "ten", "eleven", "twelve", 
    "thirteen", "fourteen", "fifteen", "sixteen"
]

# WORDS = [
#     "Paris", "London", "Berlin", "Rome",
#     "Madrid", "Vienna", "Dublin", "Prague",
#     "Athens", "Brussels", "Lisbon", "Warsaw",
#     "Milan", "Munich", "Zurich", "Geneva"
# ]

# WORDS = [
#     "Washington", "Jefferson", "Madison", "Jackson",
#     "Lincoln", "Grant", "Truman", "Kennedy",
#     "Nixon", "Ford", "Carter", "Reagan",
#     "Bush", "Clinton", "Obama", "Trump"
# ]

# WORDS = [
#     "and", "stone", "run", "yellow",
#     "carefully", "circle", "heavy", "computer",
#     "ancient", "stomach", "ocean", "music",
#     "ghost", "oxygen", "market", "building"
# ]


# WORDS = [
#     "George", "John", "Thomas", "James",
#     "Andrew", "Martin", "William", "Franklin",
#     "Harry", "Richard", "Gerald", "Jimmy",
#     "Ronald", "Bill", "Donald", "Joe"
# ]


# WORDS = [
#     # Group 1: Heavy grammar / structural tokens
#     "the", "and", "of", "to", 
#     "with", "it", "that", "is",
    
#     # Group 2: Ultra-specific concrete nouns
#     "dinosaur", "galaxy", "concrete", "submarine", 
#     "microscope", "volcano", "oxygen", "glacier"
# ]

GRID_ROWS = 4
GRID_COLS = 4
MODEL_NAME = "meta-llama/Llama-3.1-8B"
LAYER = 26
SEQ_LEN = 64
N_SEQUENCES = 16
SMOOTHING_WINDOW = 30

COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#ffff33", "#a65628", "#f781bf",
    "#999999", "#66c2a5", "#fc8d62", "#8da0cb",
    "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
]

def build_word_to_color(words):
    """Map words to colors by grid position, so e.g. position 0 is always
    the same color regardless of which WORD_LISTS vocabulary is in use.

    Falls back to a generated colormap when there are more words than the
    fixed 16-color palette covers (e.g. an 8x8 grid)."""
    if len(words) <= len(COLORS):
        return {word: color for word, color in zip(words, COLORS)}
    cmap = plt.get_cmap("hsv", len(words))
    colors = [mcolors.to_hex(cmap(i)) for i in range(len(words))]
    return {word: color for word, color in zip(words, colors)}


WORD_TO_COLOR = build_word_to_color(WORDS)


# ── Grid ───────────────────────────────────────────────────────────────────────

class Grid:
    def __init__(self, words=WORDS, rows=GRID_ROWS, cols=GRID_COLS):
        if rows * cols != len(words):
            raise ValueError(
                f"Grid dimensions ({rows}x{cols}={rows * cols}) "
                f"do not match number of words ({len(words)})"
            )
        self.words = words
        self.rows = rows
        self.cols = cols
        self.grid = np.array(words).reshape(rows, cols).tolist()
        self.word_to_row = {w: i // cols for i, w in enumerate(words)}
        self.word_to_col = {w: i % cols for i, w in enumerate(words)}

    # ── sequence generation ────────────────────────────────────────────────

    def generate_sequence(self, seq_len, start_word=None):
        """Random walk on the grid.  Optionally fix the starting word."""
        if start_word is not None:
            row, col = self.word_to_row[start_word], self.word_to_col[start_word]
        else:
            row, col = np.random.randint(0, self.rows), np.random.randint(0, self.cols)

        sequence = [self.grid[row][col]]
        while len(sequence) < seq_len:
            moves = self._valid_moves(row, col)
            direction = np.random.choice(moves)
            if direction == "up":    row -= 1
            elif direction == "down":  row += 1
            elif direction == "left":  col -= 1
            elif direction == "right": col += 1
            sequence.append(self.grid[row][col])
        return sequence

    # def generate_batch(self, seq_len):
    #     """16 sequences, each starting at a different grid word."""
    #     return [self.generate_sequence(seq_len, start_word=w) for w in self.words]

    def generate_batch(self, seq_len, n_sequences):
        """
        Generate a batch of random-walk sequences.

        Start words are assigned cyclically so that all grid words
        appear approximately equally often as starting points.
        """
        sequences = []

        for i in range(n_sequences):
            start_word = self.words[i % len(self.words)]
            sequences.append(
                self.generate_sequence(seq_len, start_word=start_word)
            )

        return sequences
    

    # ── adjacency ──────────────────────────────────────────────────────────

    def get_valid_next_words(self, word):
        row, col = self.word_to_row[word], self.word_to_col[word]
        next_words = []
        for move in self._valid_moves(row, col):
            if move == "up":    next_words.append(self.grid[row - 1][col])
            elif move == "down":  next_words.append(self.grid[row + 1][col])
            elif move == "left":  next_words.append(self.grid[row][col - 1])
            elif move == "right": next_words.append(self.grid[row][col + 1])
        return next_words

    def build_adjacency_matrix(self):
        """Return a 16x16 binary adjacency matrix (symmetric)."""
        n = len(self.words)
        A = np.zeros((n, n))
        for i, word in enumerate(self.words):
            for neighbor in self.get_valid_next_words(word):
                j = self.words.index(neighbor)
                A[i, j] = 1
        return A

    # ── internals ──────────────────────────────────────────────────────────

    def _valid_moves(self, row, col):
        moves = []
        if row > 0:              moves.append("up")
        if row < self.rows - 1:  moves.append("down")
        if col > 0:              moves.append("left")
        if col < self.cols - 1:  moves.append("right")
        return moves


# ── Torus ──────────────────────────────────────────────────────────────────────

class Torus:
    """Same word grid as Grid, but with periodic boundary conditions: moving
    off one edge wraps around to the opposite edge, so every cell has all
    four neighbors."""

    def __init__(self, words=WORDS, rows=GRID_ROWS, cols=GRID_COLS):
        if rows * cols != len(words):
            raise ValueError(
                f"Grid dimensions ({rows}x{cols}={rows * cols}) "
                f"do not match number of words ({len(words)})"
            )
        self.words = words
        self.rows = rows
        self.cols = cols
        self.grid = np.array(words).reshape(rows, cols).tolist()
        self.word_to_row = {w: i // cols for i, w in enumerate(words)}
        self.word_to_col = {w: i % cols for i, w in enumerate(words)}

    # ── sequence generation ────────────────────────────────────────────────

    def generate_sequence(self, seq_len, start_word=None):
        """Random walk on the torus.  Optionally fix the starting word."""
        if start_word is not None:
            row, col = self.word_to_row[start_word], self.word_to_col[start_word]
        else:
            row, col = np.random.randint(0, self.rows), np.random.randint(0, self.cols)

        sequence = [self.grid[row][col]]
        while len(sequence) < seq_len:
            moves = self._valid_moves(row, col)
            direction = np.random.choice(moves)
            if direction == "up":    row = (row - 1) % self.rows
            elif direction == "down":  row = (row + 1) % self.rows
            elif direction == "left":  col = (col - 1) % self.cols
            elif direction == "right": col = (col + 1) % self.cols
            sequence.append(self.grid[row][col])
        return sequence

    def generate_batch(self, seq_len, n_sequences):
        """
        Generate a batch of random-walk sequences.

        Start words are assigned cyclically so that all grid words
        appear approximately equally often as starting points.
        """
        sequences = []

        for i in range(n_sequences):
            start_word = self.words[i % len(self.words)]
            sequences.append(
                self.generate_sequence(seq_len, start_word=start_word)
            )

        return sequences

    # ── adjacency ──────────────────────────────────────────────────────────

    def get_valid_next_words(self, word):
        row, col = self.word_to_row[word], self.word_to_col[word]
        next_words = []
        for move in self._valid_moves(row, col):
            if move == "up":    next_words.append(self.grid[(row - 1) % self.rows][col])
            elif move == "down":  next_words.append(self.grid[(row + 1) % self.rows][col])
            elif move == "left":  next_words.append(self.grid[row][(col - 1) % self.cols])
            elif move == "right": next_words.append(self.grid[row][(col + 1) % self.cols])
        return next_words

    def build_adjacency_matrix(self):
        """Return a 16x16 binary adjacency matrix (symmetric)."""
        n = len(self.words)
        A = np.zeros((n, n))
        for i, word in enumerate(self.words):
            for neighbor in self.get_valid_next_words(word):
                j = self.words.index(neighbor)
                A[i, j] = 1
        return A

    # ── internals ──────────────────────────────────────────────────────────

    def _valid_moves(self, row, col):
        # Every cell has all four neighbors under periodic boundary conditions.
        return ["up", "down", "left", "right"]


# ── Ring ───────────────────────────────────────────────────────────────────────

class Ring:
    """Words arranged in a cycle: word order in `words` IS the ring order,
    i.e. position i is adjacent only to positions i-1 and i+1 (mod n).
    Unlike Grid/Torus there's no rows/cols layout -- any number of words
    forms a valid ring, not just perfect squares.

    Exposes word_to_row / word_to_col like Grid/Torus so the same
    downstream analyses (get_grid_coords, compute_distance_correlation,
    ...) work unmodified: each word's "coordinates" are its position on a
    unit circle (cos, sin), so Euclidean/L1 distance between two words
    smoothly reflects ring-adjacency (including wraparound, e.g. Sunday
    and Monday end up close together) -- unlike the raw position index,
    which would treat the two ends of the list as maximally far apart.
    """

    def __init__(self, words):
        self.words = words
        self.n = len(words)
        self.word_to_pos = {w: i for i, w in enumerate(words)}
        angles = 2 * np.pi * np.arange(self.n) / self.n
        self.word_to_row = {w: float(np.cos(angles[i])) for i, w in enumerate(words)}
        self.word_to_col = {w: float(np.sin(angles[i])) for i, w in enumerate(words)}

    # ── sequence generation ────────────────────────────────────────────────

    def generate_sequence(self, seq_len, start_word=None):
        """Random walk on the ring: at each step, move one position
        clockwise or counterclockwise with equal probability."""
        if start_word is not None:
            pos = self.word_to_pos[start_word]
        else:
            pos = np.random.randint(0, self.n)

        sequence = [self.words[pos]]
        while len(sequence) < seq_len:
            direction = np.random.choice(["cw", "ccw"])
            if direction == "cw":
                pos = (pos + 1) % self.n
            else:
                pos = (pos - 1) % self.n
            sequence.append(self.words[pos])
        return sequence

    def generate_batch(self, seq_len, n_sequences):
        """
        Generate a batch of random-walk sequences.

        Start words are assigned cyclically so that all ring words
        appear approximately equally often as starting points.
        """
        sequences = []

        for i in range(n_sequences):
            start_word = self.words[i % self.n]
            sequences.append(
                self.generate_sequence(seq_len, start_word=start_word)
            )

        return sequences

    # ── adjacency ──────────────────────────────────────────────────────────

    def get_valid_next_words(self, word):
        pos = self.word_to_pos[word]
        return [self.words[(pos + 1) % self.n], self.words[(pos - 1) % self.n]]

    def build_adjacency_matrix(self):
        """Return an nxn binary adjacency matrix (symmetric)."""
        n = self.n
        A = np.zeros((n, n))
        for i in range(n):
            A[i, (i + 1) % n] = 1
            A[i, (i - 1) % n] = 1
        return A


# ── Seeding ────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


# def set_seed(seed, model_seed):
#     if seed:
#         random.seed(seed)
#         np.random.seed(seed)
#     if model_seed:
#         torch.manual_seed(model_seed)
#         torch.cuda.manual_seed_all(model_seed)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(cache_dir=None, device=None):
    if cache_dir is None:
        cache_dir = os.environ.get("HF_HOME", None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return HookedTransformer.from_pretrained_no_processing(
        MODEL_NAME, device=device, cache_dir=cache_dir,
    )


def load_toy_model(normalization_type=None, seed=42, device=None, n_ctx=None, n_layers=1, zero_out_pos_emb=False):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if n_ctx is None:
        n_ctx = SEQ_LEN

    # n_layers-layer, 4-head attention-only causal transformer
    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        n_heads=4,
        d_model=128,
        d_head=32,
        n_ctx=n_ctx,
        d_vocab=len(WORDS),
        act_fn=None,
        attention_dir="causal",
        attn_only=True,
        normalization_type=normalization_type,
        device=device,
        seed=seed,
    )
    model = HookedTransformer(cfg)

    if zero_out_pos_emb:
        model.W_pos.data.zero_()
        
    return model


# ── Tokenization ──────────────────────────────────────────────────────────────

def tokenize_sequence(model, sequence):
    """Tokenize a word sequence. Returns input_ids tensor."""
    text = " " + " ".join(sequence)
    return model.tokenizer(text, return_tensors="pt").input_ids


def simple_tokenize(sequence, vocab):
    # Map words to IDs, default to 0 if word is unknown
    tokens = [vocab.get(word, 0) for word in sequence.split()]
    return torch.tensor([tokens], dtype=torch.long)


# ── Accuracy ───────────────────────────────────────────────────────────────────

def get_model_accuracies(model, grid, sequence, fwd_hooks=[]):
    """Per-position probability assigned to valid next tokens."""
    tokens = tokenize_sequence(model, sequence)
    logits = model.run_with_hooks(tokens.to(model.cfg.device), fwd_hooks=fwd_hooks)
    probs = torch.softmax(logits, dim=-1)
    probs = probs[0, 1:, :]  # remove BOS position

    accuracies = []
    for i in range(len(sequence)):
        valid_next = grid.get_valid_next_words(sequence[i])
        token_ids = torch.tensor(
            [model.tokenizer.encode(" " + w, add_special_tokens=False) for w in valid_next]
        ).squeeze()
        accuracies.append(probs[i, token_ids].sum().item())
    return accuracies


# ── Activations ────────────────────────────────────────────────────────────────

def get_activations(model, sequence, layer, n_lookback, fwd_hooks=[]):
    """Return last n_lookback residual-stream activations at `layer`.

    Shape: [n_lookback, d_model]
    """
    tokens = tokenize_sequence(model, sequence)
    names_filter = [f"blocks.{layer}.hook_resid_pre"]

    with model.hooks(fwd_hooks=fwd_hooks):
        _, cache = model.run_with_cache(
            tokens.to(model.cfg.device), names_filter=names_filter,
        )

    acts = cache[f"blocks.{layer}.hook_resid_pre"]  # [1, seq+1, d_model]
    # Remove BOS then take last n_lookback
    acts = acts[0, 1:, :]  # [seq_len, d_model]
    return acts[-n_lookback:, :]  # [n_lookback, d_model]



def get_activations_toy_batch(model, sequences, layer, n_lookback, act_type="attn_out", fwd_hooks=[]):
    """
    Returns the last n_lookback activations for a BATCH of sequences.
    Output shape: [batch_size, n_lookback, d_model]
    """
    word_to_id = {word: i for i, word in enumerate(WORDS)}

    # Map all sequences in the batch to token IDs
    token_ids = [[word_to_id[word] for word in seq] for seq in sequences]
    tokens = torch.tensor(token_ids, dtype=torch.long).to(model.cfg.device)

    # Define the hook point
    hook_name = utils.get_act_name(act_type, layer)

    # Run with cache
    with model.hooks(fwd_hooks=fwd_hooks):
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
    acts = cache[hook_name] # Shape: [batch_size, seq_len, d_model]
    
    # Return the last n_lookback positions for ALL sequences in the batch
    # We drop the hardcoded acts[0] so we don't lose the batch dimension
    return acts[:, -n_lookback:, :]


# ── PCA helpers ────────────────────────────────────────────────────────────────

def compute_class_means(
    activations: Float[Tensor, "n_lookback d_model"],
    sequence: list[str],
    words: list[str],
    n_lookback: int,
) -> Float[Tensor, "n_words d_model"]:
    """Mean activation per word over the last n_lookback positions."""
    tail = sequence[-n_lookback:]
    means = []
    for word in words:
        idxs = [i for i, w in enumerate(tail) if w == word]
        if idxs:
            means.append(activations[idxs].mean(dim=0))
        else:
            means.append(torch.zeros(activations.shape[-1], device=activations.device))
    return torch.stack(means)  # [16, d_model]


def compute_class_means_batch(activations_t, sequences, words, n_lookback):
    """
    Computes the mean activation vector for each word, averaging over all 
    occurrences across the lookback window AND the entire batch.
    
    activations_t: Tensor of shape [batch_size, n_lookback, d_model]
    sequences: List of lists of words, shape [batch_size, seq_len]
    """
    batch_size, n_look, d_model = activations_t.shape
    
    # Dictionary to collect activation vectors for each word
    word_vectors = {word: [] for word in words}
    
    # Loop over every sequence in the batch
    for b in range(batch_size):
        # Get the last n_lookback words for this specific sequence
        lookback_words = sequences[b][-n_look:] 
        
        # Match each lookback word with its corresponding activation vector
        for l in range(n_look):
            word = lookback_words[l]
            if word in word_vectors:
                word_vectors[word].append(activations_t[b, l, :])
                
    # Compute the mean vector for each word across all batch occurrences
    class_means = []
    for word in words:
        vectors = word_vectors[word]
        if len(vectors) > 0:
            # Stack all collected occurrences and average them
            mean_vec = torch.stack(vectors).mean(dim=0)
        else:
            # Fallback tensor if a word never appeared in the lookback windows
            mean_vec = torch.zeros(d_model, device=activations_t.device)
        class_means.append(mean_vec)
        
    return torch.stack(class_means) # Shape: [len(words), d_model]


def compute_pca_directions(
    class_means: Float[Tensor, "n_words d_model"],
    top_n: int = 2,
) -> Tuple[Float[Tensor, "top_n d_model"], np.ndarray]:
    """PCA on the 16 class-mean vectors. Returns top_n right singular vectors and variance ratio."""
    centered = class_means - class_means.mean(dim=0, keepdim=True)
    U, S, V = torch.svd(centered)
    
    # Calculate fraction of variance explained
    variance_explained = (S ** 2) / torch.sum(S ** 2)
    explained_variance_ratio = variance_explained[:top_n].cpu().numpy()
    
    pca_dirs = einops.rearrange(V, "d_model n -> n d_model")[:top_n, :]
    return pca_dirs, explained_variance_ratio


def compute_dirichlet_energy(class_means: np.ndarray, adjacency: np.ndarray) -> float:
    """Normalized Dirichlet energy: sum of squared representation distances
    over state pairs adjacent in `adjacency`, divided by the sum of squared
    distances over ALL state pairs. Low = adjacent states are close in
    representation space relative to the overall spread (grid-like);
    high = adjacency carries no representational signal.

    class_means: [n_words, d_model]. adjacency: [n_words, n_words] binary.
    """
    diffs = class_means[:, None, :] - class_means[None, :, :]
    sq_dists = np.sum(diffs ** 2, axis=-1)  # [n_words, n_words]
    total_energy = sq_dists.sum()
    adjacent_energy = (adjacency * sq_dists).sum()
    return float(adjacent_energy / total_energy)


def compute_distance_correlation(class_means: np.ndarray, grid_coords: np.ndarray) -> float:
    """Pearson correlation between representation-space Euclidean distances
    and state-space Manhattan (L1) distances, over all i != j state pairs.
    High = representation geometry mirrors the state space's layout.

    class_means: [n_words, d_model]. grid_coords: [n_words, 2] (row, col)
    per state, in the same word order as class_means.
    """
    n = class_means.shape[0]
    rep_dists = np.linalg.norm(class_means[:, None, :] - class_means[None, :, :], axis=-1)
    grid_dists = np.abs(grid_coords[:, None, :] - grid_coords[None, :, :]).sum(axis=-1)
    iu = np.triu_indices(n, k=1)
    return float(np.corrcoef(rep_dists[iu], grid_dists[iu])[0, 1])


def compute_graph_shortest_path_distances(adjacency: np.ndarray) -> np.ndarray:
    """All-pairs shortest-path distance matrix (hop count) for an unweighted,
    undirected binary adjacency matrix."""
    return shortest_path(adjacency, method="D", directed=False, unweighted=True)


def compute_distance_correlation_graph_embedding(embeddings: np.ndarray, adjacency: np.ndarray) -> float:
    """Like compute_distance_correlation, but against a graph's TOPOLOGY
    instead of a 2D coordinate layout: Pearson correlation between
    embedding-space Euclidean distances and the graph's shortest-path
    (hop-count) distances, over all i != j node pairs. Graph distance --
    never Euclidean/Manhattan on embedded coordinates -- so this only
    depends on graph topology and works for any adjacency matrix, not just
    ones with a natural 2D embedding (grid/ring).

    embeddings: [n, d]. adjacency: [n, n] binary, same node order as embeddings.
    """
    n = embeddings.shape[0]
    rep_dists = np.linalg.norm(embeddings[:, None, :] - embeddings[None, :, :], axis=-1)
    graph_dists = compute_graph_shortest_path_distances(adjacency)
    iu = np.triu_indices(n, k=1)
    return float(np.corrcoef(rep_dists[iu], graph_dists[iu])[0, 1])


def set_square_limits(ax, xs, ys, pad_frac=0.15):
    """Force equal-width x/y limits (centered on the data) so that, combined
    with ax.set_aspect('equal'), every subplot renders at the same physical
    size regardless of how spread out its own data happens to be."""
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    half_span = max(x_max - x_min, y_max - y_min, 1e-6) / 2 * (1 + pad_frac)
    ax.set_xlim(x_center - half_span, x_center + half_span)
    ax.set_ylim(y_center - half_span, y_center + half_span)


# ── Ablation hooks ─────────────────────────────────────────────────────────────

def head_ablation_hook(
    activation: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head: int,
) -> Float[Tensor, "batch seq n_heads d_head"]:
    activation[:, :, head, :] = 0.0
    return activation


def make_ablation_hooks(heads_to_ablate):
    """Return a list of (name, hook_fn) pairs that zero-ablate the given heads.

    heads_to_ablate: list of (layer, head) tuples.
    """
    return [
        (
            utils.get_act_name("z", layer),
            functools.partial(head_ablation_hook, head=head),
        )
        for layer, head in heads_to_ablate
    ]


# ── Plotting ───────────────────────────────────────────────────────────────────

def setup_plotting():
    plt.style.use("science")
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.top": False,
        "ytick.right": False,
        "axes.grid": True,
        "grid.alpha": 0.12,
        "grid.color": "gray",
        "grid.linewidth": 0.4,
        "savefig.bbox": "standard",  # override scienceplots' tight default
    })


def save_figure(fig, directory, filename):
    os.makedirs(directory, exist_ok=True)
    stem = os.path.splitext(filename)[0]
    fig.tight_layout()
    # fig.savefig(os.path.join(directory, stem + ".pdf"))
    fig.savefig(os.path.join(directory, stem + ".png"), dpi=300)
    plt.close(fig)


def smooth(arr, window=SMOOTHING_WINDOW):
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


# ── Plotly helpers ────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    plot_bgcolor="white",
    autosize=False,
    margin=dict(l=60, r=40, t=60, b=60),
)

PLOTLY_AXIS = dict(
    showgrid=True, gridcolor="rgba(128,128,128,0.12)", gridwidth=0.4,
    zeroline=False, showline=True, linecolor="black", linewidth=1,
    ticks="outside", tickcolor="black",
)


def plotly_pca_layout(title):
    """Standard layout for PCA scatter plots in plotly."""
    return dict(
        title=title,
        width=600, height=600,
        xaxis=dict(title="PC1", scaleanchor="y", scaleratio=1, **PLOTLY_AXIS),
        yaxis=dict(title="PC2", **PLOTLY_AXIS),
        **PLOTLY_LAYOUT,
    )


def plotly_line_layout(title, xlabel, ylabel):
    """Standard layout for line plots in plotly."""
    return dict(
        title=title,
        width=700, height=400,
        xaxis=dict(title=xlabel, type="log", **PLOTLY_AXIS),
        yaxis=dict(title=ylabel, **PLOTLY_AXIS),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="gray", borderwidth=1),
        **PLOTLY_LAYOUT,
    )


def save_plotly(fig, directory, filename):
    """Save a plotly figure as HTML with fixed dimensions."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    fig.write_html(path, config={"responsive": False}, include_plotlyjs="cdn")
    print(f"Saved {path}")


def plotly_pca_traces(projected, grid, words=WORDS, word_to_color=WORD_TO_COLOR):
    """Return list of plotly traces for a PCA scatter: edges + star centroids."""
    traces = []
    A = grid.build_adjacency_matrix()
    # Collect all edges into a single trace with None breaks
    edge_x, edge_y = [], []
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if A[i, j]:
                edge_x += [projected[i, 0].item(), projected[j, 0].item(), None]
                edge_y += [projected[i, 1].item(), projected[j, 1].item(), None]
    traces.append(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="dimgray", width=1.2, dash="dash"),
        opacity=0.8, showlegend=False, hoverinfo="skip",
    ))
    # Centroids
    for i, word in enumerate(words):
        traces.append(go.Scatter(
            x=[projected[i, 0].item()], y=[projected[i, 1].item()],
            mode="markers+text",
            marker=dict(size=16, symbol="star", color=word_to_color[word],
                        line=dict(width=1, color="black")),
            text=word, textposition="top right", textfont=dict(size=10),
            hovertemplate=f"centroid: <b>{word}</b><extra></extra>",
            hoverlabel=dict(bgcolor=word_to_color[word], font=dict(color="black")),
            showlegend=False,
        ))
    return traces



def _label_heatmap_ax(ax, words):
    ax.set_xticks(range(len(words)))
    ax.set_yticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(words, fontsize=8)


def plot_similarity_panels(cos_sim_matrix, gram_matrix, l2_dist_matrix, words,
                            plots_dir, filename, title=None):
    """
    Plots the pairwise cosine similarity matrix, the raw Gram (inner
    product) matrix, and the pairwise L2 (Euclidean) distance matrix side by
    side. Cosine similarity only captures direction; the Gram matrix also
    preserves embedding magnitude; L2 distance captures absolute separation
    in embedding space (unlike the other two, smaller = more similar).
    """
    fig, (ax_cos, ax_gram, ax_l2) = plt.subplots(1, 3, figsize=(19, 5.5))

    cax = ax_cos.imshow(cos_sim_matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax, ax=ax_cos, fraction=0.046, pad=0.04, label="Cosine Similarity")
    _label_heatmap_ax(ax_cos, words)
    ax_cos.set_title("Cosine Similarity", fontsize=10, pad=12)

    vmax = np.abs(gram_matrix).max()
    gax = ax_gram.imshow(gram_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(gax, ax=ax_gram, fraction=0.046, pad=0.04, label="Inner Product")
    _label_heatmap_ax(ax_gram, words)
    ax_gram.set_title("Gram Matrix", fontsize=10, pad=12)

    lax = ax_l2.imshow(l2_dist_matrix, cmap="viridis", vmin=0.0, vmax=l2_dist_matrix.max())
    fig.colorbar(lax, ax=ax_l2, fraction=0.046, pad=0.04, label="L2 Distance")
    _label_heatmap_ax(ax_l2, words)
    ax_l2.set_title("Pairwise L2 Distance", fontsize=10, pad=12)

    fig.suptitle(f"Pairwise Similarity{f' ({title})' if title else ''}", fontsize=12)
    plt.tight_layout()

    save_figure(fig, plots_dir, filename)
    print(f"Saved {filename}")


def compute_and_plot_similarity_panels(embeddings, words, plots_dir, filename, title=None):
    """Compute cosine similarity, Gram, and pairwise L2 distance matrices for
    a [n_words, d_model] array of embeddings/activations, then plot all three
    side by side via plot_similarity_panels."""
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    cos_sim_matrix = normalized @ normalized.T
    gram_matrix = embeddings @ embeddings.T
    diffs = embeddings[:, None, :] - embeddings[None, :, :]
    l2_dist_matrix = np.linalg.norm(diffs, axis=-1)
    plot_similarity_panels(cos_sim_matrix, gram_matrix, l2_dist_matrix, words,
                            plots_dir, filename, title=title)


def plot_cross_similarity_panels(cos_sim_matrix, gram_matrix, l2_dist_matrix, words,
                                  plots_dir, filename, title=None,
                                  row_label="Embedding", col_label="Unembedding"):
    """Like plot_similarity_panels, but for CROSS matrices between two
    different vector sets for the same words (e.g. embedding rows on one
    axis, unembedding rows on the other) -- generally not symmetric, so
    both axes are explicitly labeled with which side is which.

    Cross cosine similarities between embeddings and unembeddings tend to
    be tiny (e.g. +/-0.05) since the two live in nearly orthogonal
    subspaces, unlike a self-similarity matrix's diagonal of 1s -- so
    unlike plot_similarity_panels' fixed [-1, 1] scale, this auto-scales
    to the matrix's own actual range (like the Gram panel already does),
    or the true structure is invisible against a near-white background."""
    fig, (ax_cos, ax_gram, ax_l2) = plt.subplots(1, 3, figsize=(19, 5.5))

    cos_vmax = np.abs(cos_sim_matrix).max()
    cax = ax_cos.imshow(cos_sim_matrix, cmap="RdBu_r", vmin=-cos_vmax, vmax=cos_vmax)
    fig.colorbar(cax, ax=ax_cos, fraction=0.046, pad=0.04, label="Cosine Similarity")
    _label_heatmap_ax(ax_cos, words)
    ax_cos.set_xlabel(f"{col_label} of token j")
    ax_cos.set_ylabel(f"{row_label} of token i")
    ax_cos.set_title("Cosine Similarity", fontsize=10, pad=12)

    vmax = np.abs(gram_matrix).max()
    gax = ax_gram.imshow(gram_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(gax, ax=ax_gram, fraction=0.046, pad=0.04, label="Inner Product")
    _label_heatmap_ax(ax_gram, words)
    ax_gram.set_xlabel(f"{col_label} of token j")
    ax_gram.set_ylabel(f"{row_label} of token i")
    ax_gram.set_title("Gram Matrix", fontsize=10, pad=12)

    lax = ax_l2.imshow(l2_dist_matrix, cmap="viridis", vmin=0.0, vmax=l2_dist_matrix.max())
    fig.colorbar(lax, ax=ax_l2, fraction=0.046, pad=0.04, label="L2 Distance")
    _label_heatmap_ax(ax_l2, words)
    ax_l2.set_xlabel(f"{col_label} of token j")
    ax_l2.set_ylabel(f"{row_label} of token i")
    ax_l2.set_title("Pairwise L2 Distance", fontsize=10, pad=12)

    fig.suptitle(
        f"Pairwise {row_label} vs. {col_label} Similarity{f' ({title})' if title else ''}",
        fontsize=12,
    )
    plt.tight_layout()

    save_figure(fig, plots_dir, filename)
    print(f"Saved {filename}")


def compute_and_plot_cross_similarity_panels(row_vectors, col_vectors, words, plots_dir, filename,
                                              title=None, row_label="Embedding", col_label="Unembedding"):
    """Compute cosine similarity, Gram (inner product), and pairwise L2
    distance matrices BETWEEN two different [n_words, d_model] vector sets
    for the same words in the same order (e.g. row_vectors=W_E rows,
    col_vectors=W_U rows), then plot all three side by side via
    plot_cross_similarity_panels. Unlike compute_and_plot_similarity_panels,
    these matrices are generally NOT symmetric."""
    row_normalized = row_vectors / np.linalg.norm(row_vectors, axis=1, keepdims=True)
    col_normalized = col_vectors / np.linalg.norm(col_vectors, axis=1, keepdims=True)
    cos_sim_matrix = row_normalized @ col_normalized.T
    gram_matrix = row_vectors @ col_vectors.T
    diffs = row_vectors[:, None, :] - col_vectors[None, :, :]
    l2_dist_matrix = np.linalg.norm(diffs, axis=-1)
    plot_cross_similarity_panels(cos_sim_matrix, gram_matrix, l2_dist_matrix, words,
                                  plots_dir, filename, title=title,
                                  row_label=row_label, col_label=col_label)


def plot_attention_heatmaps(model, sequences, layer: int = 0, seq_index: int = 0,
                                cmap: str = 'viridis', vmin=None, vmax=None,
                                use_log_scale: bool = False, log_eps: float = 1e-12,
                                figsize=(20,16), save_path: str = None, show: bool = False):
    """
    Extracts and plots attention heatmaps for every head in a chosen layer 
    for our toy word-based transformer setup.
    
    sequences: List of lists of strings (e.g., [['The', 'cat', 'sat'], ...])
    """
    # 1. Convert string sequences to token IDs for the model
    word_to_id = {word: i for i, word in enumerate(WORDS)}
    token_ids = [[word_to_id[word] for word in seq] for seq in sequences]
    tokens = torch.tensor(token_ids, dtype=torch.long).to(model.cfg.device)
    
    # 2. Extract attention patterns from TransformerLens cache
    # hook_pattern shape is [batch, n_heads, query_pos, key_pos]
    hook_name = f"blocks.{layer}.attn.hook_pattern"
    _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
    
    # Convert to numpy and select our specific sequence index
    layer_w = cache[hook_name].cpu().numpy()  # [batch, n_heads, T, T]
    
    batch, n_heads, T, _ = layer_w.shape
    print(f"{seq_index=}")
    if seq_index < 0 or seq_index >= batch:
        raise IndexError(f"seq_index out of range (0..{batch-1})")
        
    seq_w = layer_w[seq_index]  # [n_heads, T, T]
    
    # 3. Use the original string words as labels for the axes!
    labels = sequences[seq_index]

    # Grid layout calculations
    cols = min(4, n_heads) # Kept at 4 max for cleaner display of word labels
    rows = int(math.ceil(n_heads / cols))
    if figsize is None:
        figsize = (cols * 4.5, rows * 4.5)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    # Ensure axes is always a 2D array even if it's 1x1 or 1xN
    if n_heads == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = np_atleast_2d_flat = np.atleast_2d(axes) 
    else:
        axes = np.atleast_2d(axes)

    # Log scaling calculations
    if use_log_scale:
        pos = seq_w[seq_w > 0]
        if pos.size > 0:
            auto_vmin = float(pos.min())
            auto_vmax = float(seq_w.max())
            vmin_used = (vmin if (vmin is not None and vmin > 0) else auto_vmin)
            vmax_used = (vmax if (vmax is not None and vmax > 0) else auto_vmax)
            norm = LogNorm(vmin=max(vmin_used, log_eps), vmax=max(vmax_used, log_eps))
        else:
            norm = None
    else:
        norm = None

    # Plot each attention head
    for h in range(n_heads):
        r = h // cols
        c = h % cols
        ax = axes[r, c]
        
        imshow_kwargs = dict(aspect='equal', cmap=cmap, interpolation='nearest')
        if norm is not None:
            imshow_kwargs['norm'] = norm
        else:
            if vmin is not None: imshow_kwargs['vmin'] = vmin
            if vmax is not None: imshow_kwargs['vmax'] = vmax

        im = ax.imshow(seq_w[h], **imshow_kwargs)
        ax.set_title(f"Layer {layer} | Head {h}", fontsize=11, fontweight='bold')
        
        # Apply the actual word string labels to the axes
        ax.set_xticks(np.arange(T))
        ax.set_yticks(np.arange(T))
        ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
        ax.set_yticklabels(labels, fontsize=9)
        
        ax.set_xlabel('Key Token (What it attends to)', fontsize=9)
        ax.set_ylabel('Query Token (What is looking)', fontsize=9)
            
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Turn off any empty subplots in the grid
    for h in range(n_heads, rows * cols):
        r = h // cols
        c = h % cols
        axes[r, c].axis('off')

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    return fig