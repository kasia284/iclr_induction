import numpy as np

verified = {
    "Oslo": (60, 11), "Dublin": (53, -6), "Paris": (49, 2), "Rome": (42, 12),
    "Madrid": (40, -4), "Athens": (38, 24), "Cairo": (30, 31), "Delhi": (29, 77),
    "Dubai": (25, 55), "Miami": (26, -80), "Dallas": (33, -97), "Denver": (40, -105),
    "Seattle": (47, -122), "Toronto": (44, -79), "Chicago": (42, -88), "Boston": (42, -71),
    "Tokyo": (36, 140), "Seoul": (37, 127), "Beijing": (40, 116), "Manila": (15, 121),
    "Bangkok": (14, 101), "Jakarta": (-6, 107), "Sydney": (-34, 151), "Perth": (-32, 116),
    "Lagos": (6, 3), "Nairobi": (-1, 37), "Lima": (-12, -77),
    "Santiago": (-33, -71), "Vienna": (48, 16), "Berlin": (52, 13),
    "Moscow": (56, 38), "Helsinki": (60, 25), "Lisbon": (39, -9), "Warsaw": (52, 21),
    "Zurich": (47, 8), "Munich": (48, 12), "Milan": (45, 9), "Vancouver": (49, -123),
}

names = list(verified.keys())
coords = np.array([verified[n] for n in names], dtype=float)

# normalize lat/lon to comparable scales before computing distance
# (lon spans ~360, lat spans ~180, so raw distance would overweight lon)
coords_norm = (coords - coords.mean(0)) / coords.std(0)

def farthest_point_sampling(points, k, start_idx=0):
    n = len(points)
    selected = [start_idx]
    dists = np.linalg.norm(points - points[start_idx], axis=1)
    for _ in range(k - 1):
        next_idx = np.argmax(dists)
        selected.append(next_idx)
        new_dists = np.linalg.norm(points - points[next_idx], axis=1)
        dists = np.minimum(dists, new_dists)
    return selected

# start from the point closest to the centroid, for a more "balanced" spread
start = np.argmin(np.linalg.norm(coords_norm, axis=1))
idxs = farthest_point_sampling(coords_norm, 16, start_idx=start)

selected = [names[i] for i in idxs]
print(selected)
for n in selected:
    print(f"  {n:12s} {verified[n]}")