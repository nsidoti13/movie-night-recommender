"""
recommender.py — Recommendation engine (K-means clustering).

Movies are clustered into N_CLUSTERS groups at load time using their
sentence-transformer embeddings. User taste is represented as a weight
distribution over those clusters (fraction of liked movies per cluster).
Recommendations score each unseen movie by its cluster's weight, breaking
ties with cosine similarity to the cluster centroid.

Public API:
    next_card(liked, disliked)                         → row index
    joint_recommendations(liked_a, liked_b, ...)       → list of row indices
    reset_seeds()
"""

import json
import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR  = "data"
N_CLUSTERS = 30

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
_df: pd.DataFrame | None = None
_matrix: np.ndarray | None = None        # (N, 384) sentence embeddings
_movie_index: dict | None = None
_cluster_labels: np.ndarray | None = None  # (N,) int  — cluster id per movie
_centroid_sims: np.ndarray | None = None   # (N,) float — cosine sim to own centroid


def load_data():
    global _df, _matrix, _movie_index, _cluster_labels, _centroid_sims
    if _df is None:
        _df = pd.read_parquet(f"{DATA_DIR}/movies.parquet")
        _matrix = np.load(f"{DATA_DIR}/embeddings.npy")
        with open(f"{DATA_DIR}/movie_index.json") as f:
            _movie_index = json.load(f)

        km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10, max_iter=300)
        _cluster_labels = km.fit_predict(_matrix)

        # Pre-compute each movie's cosine similarity to its own cluster centroid
        all_sims = cosine_similarity(_matrix, km.cluster_centers_)  # (N, K)
        _centroid_sims = all_sims[np.arange(len(_df)), _cluster_labels]  # (N,)

    return _df, _matrix, _movie_index


# ---------------------------------------------------------------------------
# Seed movies — cold-start rotation
# ---------------------------------------------------------------------------
SEED_GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance",
               "Science Fiction", "Animation", "Thriller", "Adventure", "Fantasy"]


def _genre_seeds(df: pd.DataFrame, seen: set) -> list[int]:
    import ast

    def has_genre(genres_val, genre_name):
        try:
            items = json.loads(genres_val) if isinstance(genres_val, str) else genres_val
        except Exception:
            try:
                items = ast.literal_eval(genres_val)
            except Exception:
                items = []
        return any(g.get("name") == genre_name for g in items if isinstance(g, dict))

    seeds = []
    for genre in SEED_GENRES:
        candidates = df[df["genres"].apply(lambda g: has_genre(g, genre))].index.tolist()
        candidates = [c for c in candidates if c not in seen]
        if candidates:
            seeds.append(random.choice(candidates))
    return seeds


_seed_queue: list[int] = []
_seed_seen: set[int] = set()


def reset_seeds():
    global _seed_queue, _seed_seen
    _seed_queue = []
    _seed_seen = set()


def _get_next_seed(df: pd.DataFrame, seen: set) -> int:
    global _seed_queue, _seed_seen
    available = [s for s in _seed_queue if s not in seen]
    if not available:
        _seed_queue = _genre_seeds(df, seen)
        available = [s for s in _seed_queue if s not in seen]
    if available:
        idx = available[0]
        _seed_queue.remove(idx)
        return idx
    all_idx = set(range(len(df))) - seen
    return random.choice(list(all_idx)) if all_idx else 0


def _most_popular_recent(df: pd.DataFrame) -> int:
    recent = df[df["year"] >= 2010]
    if recent.empty:
        recent = df
    return int(recent["popularity"].fillna(0).idxmax())


# ---------------------------------------------------------------------------
# Cluster scoring
# ---------------------------------------------------------------------------

def _cluster_weights(liked: list[int]) -> np.ndarray:
    """Weight for each cluster = fraction of liked movies that belong to it."""
    weights = np.zeros(N_CLUSTERS)
    for idx in liked:
        weights[_cluster_labels[idx]] += 1
    total = weights.sum()
    return weights / total if total > 0 else weights


def _score_movies(weights: np.ndarray, seen: set) -> np.ndarray:
    """
    Score every movie by its cluster weight.
    Ties broken by cosine similarity to the cluster centroid (scaled small
    so cluster weight always dominates).
    """
    scores = weights[_cluster_labels] + 0.01 * _centroid_sims
    for idx in seen:
        scores[idx] = -np.inf
    return scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def next_card(liked: list[int], disliked: list[int]) -> int:
    """Return the row index of the best next movie to show."""
    df, _, _ = load_data()
    seen = set(liked) | set(disliked)

    if not liked:
        if not seen:
            return _most_popular_recent(df)
        return _get_next_seed(df, seen)

    weights = _cluster_weights(liked)
    scores  = _score_movies(weights, seen)
    return int(np.argmax(scores))


def joint_recommendations(
    liked_a: list[int],
    liked_b: list[int],
    disliked_a: list[int] = None,
    disliked_b: list[int] = None,
    blend: float = 0.5,
    n: int = 10,
) -> list[int]:
    """
    Return top-n movie indices for a joint audience using blended cluster weights.
    blend=0.5 weights both users equally; 0.0 = fully user A, 1.0 = fully user B.
    """
    load_data()
    disliked_a = disliked_a or []
    disliked_b = disliked_b or []
    seen = set(liked_a) | set(liked_b) | set(disliked_a) | set(disliked_b)

    if not liked_a and not liked_b:
        return []

    weights_a = _cluster_weights(liked_a) if liked_a else np.ones(N_CLUSTERS) / N_CLUSTERS
    weights_b = _cluster_weights(liked_b) if liked_b else np.ones(N_CLUSTERS) / N_CLUSTERS

    blended = (1 - blend) * weights_a + blend * weights_b
    scores  = _score_movies(blended, seen)

    top = np.argsort(scores)[::-1]
    return [int(i) for i in top if np.isfinite(scores[i])][:n]
