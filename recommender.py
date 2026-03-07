"""
recommender.py — Recommendation engine.
Loads precomputed data once and exposes two functions:
    next_card(liked, disliked, df, matrix) → row index
    joint_recommendations(liked_a, liked_b, df, matrix, blend=0.5, n=10) → list of row indices
"""

import json
import random
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "data"

# ---------------------------------------------------------------------------
# Data loading (cached at module level so it only happens once per session)
# ---------------------------------------------------------------------------
_df: pd.DataFrame | None = None
_matrix = None
_movie_index: dict | None = None


def load_data():
    global _df, _matrix, _movie_index
    if _df is None:
        _df = pd.read_parquet(f"{DATA_DIR}/movies.parquet")
        _matrix = sparse.load_npz(f"{DATA_DIR}/tfidf_matrix.npz")
        with open(f"{DATA_DIR}/movie_index.json") as f:
            _movie_index = json.load(f)
    return _df, _matrix, _movie_index


# ---------------------------------------------------------------------------
# Seed movies — one per major genre to bootstrap cold-start
# ---------------------------------------------------------------------------
SEED_GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance",
               "Science Fiction", "Animation", "Thriller", "Adventure", "Fantasy"]


def _genre_seeds(df: pd.DataFrame, seen: set) -> list[int]:
    """Return one random movie index per major genre, excluding already-seen ones."""
    import json, ast

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
    # Replenish queue if empty or all used
    available = [s for s in _seed_queue if s not in seen]
    if not available:
        _seed_queue = _genre_seeds(df, seen)
        available = [s for s in _seed_queue if s not in seen]
    if available:
        idx = available[0]
        _seed_queue.remove(idx)
        return idx
    # Absolute fallback: random unseen movie
    all_idx = set(range(len(df))) - seen
    return random.choice(list(all_idx)) if all_idx else 0


# ---------------------------------------------------------------------------
# Taste profile
# ---------------------------------------------------------------------------
def _taste_profile(liked: list[int], matrix) -> np.ndarray:
    """Average TF-IDF vector of liked movies."""
    if not liked:
        return None
    vectors = matrix[liked]
    return np.asarray(vectors.mean(axis=0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _most_popular_recent(df: pd.DataFrame) -> int:
    """Return the index of the highest-popularity movie from 2010 onwards."""
    recent = df[df["year"] >= 2010]
    if recent.empty:
        recent = df
    return int(recent["popularity"].fillna(0).idxmax())


def next_card(liked: list[int], disliked: list[int]) -> int:
    """Return the row index of the best next movie to show."""
    df, matrix, _ = load_data()
    seen = set(liked) | set(disliked)

    if not liked:
        if not seen:  # very first card — show the most popular recent movie
            return _most_popular_recent(df)
        return _get_next_seed(df, seen)

    profile = _taste_profile(liked, matrix)
    sims = cosine_similarity(profile, matrix)[0]
    # Zero out seen movies
    for idx in seen:
        sims[idx] = -1.0
    return int(np.argmax(sims))


def joint_recommendations(
    liked_a: list[int],
    liked_b: list[int],
    disliked_a: list[int] = None,
    disliked_b: list[int] = None,
    blend: float = 0.5,
    n: int = 10,
) -> list[int]:
    """
    Return top-n movie indices for a joint audience.
    blend=0.5 weights both users equally; 0.0 = fully user A, 1.0 = fully user B.
    """
    df, matrix, _ = load_data()
    disliked_a = disliked_a or []
    disliked_b = disliked_b or []
    seen = set(liked_a) | set(liked_b) | set(disliked_a) | set(disliked_b)

    profile_a = _taste_profile(liked_a, matrix)
    profile_b = _taste_profile(liked_b, matrix)

    if profile_a is None and profile_b is None:
        return []
    if profile_a is None:
        joint = profile_b
    elif profile_b is None:
        joint = profile_a
    else:
        joint = (1 - blend) * profile_a + blend * profile_b

    sims = cosine_similarity(joint, matrix)[0]
    for idx in seen:
        sims[idx] = -1.0

    top_indices = np.argsort(sims)[::-1]
    return [int(i) for i in top_indices if sims[i] >= 0][:n]
