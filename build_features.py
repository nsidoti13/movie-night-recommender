"""
build_features.py — One-time data prep script.
Run once before launching the app:
    python build_features.py
Outputs:
    data/movies.parquet   (includes poster_url column)
    data/tfidf_matrix.npz
    data/movie_index.json
"""

import json
import ast
import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import kagglehub

DATA_DIR = "data"


def safe_parse(value):
    """Parse a JSON/Python-literal string into a Python object, return [] on failure."""
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except Exception:
        try:
            return ast.literal_eval(value)
        except Exception:
            return []


def extract_names(value, limit=None):
    """Return a list of 'name' fields from a list of dicts."""
    items = safe_parse(value)
    names = [item.get("name", "") for item in items if isinstance(item, dict)]
    return names[:limit] if limit else names


def extract_director(crew_value):
    """Return the director's name from a crew list, or empty string."""
    crew = safe_parse(crew_value)
    for member in crew:
        if isinstance(member, dict) and member.get("job") == "Director":
            return member.get("name", "")
    return ""


def build_embed_text(row):
    """Build a rich natural-language description for embedding."""
    parts = [str(row.get("title", "")).strip()]

    overview = str(row.get("overview", "")).strip()
    if overview and overview != "nan":
        parts.append(overview)

    genres = extract_names(row.get("genres", "[]"))
    if genres:
        parts.append("Genres: " + ", ".join(genres))

    keywords = extract_names(row.get("keywords", "[]"), limit=10)
    if keywords:
        parts.append("Keywords: " + ", ".join(keywords))

    cast = extract_names(row.get("cast", "[]"), limit=3)
    if cast:
        parts.append("Starring: " + ", ".join(cast))

    director = extract_director(row.get("crew", "[]"))
    if director:
        parts.append("Directed by: " + director)

    return ". ".join(parts)


def main():
    print("Downloading TMDB dataset from Hugging Face…")
    ds = load_dataset("AiresPucrs/tmdb-5000-movies", split="train")
    df = ds.to_pandas()

    print(f"Loaded {len(df)} movies. Cleaning…")

    # Keep only useful columns
    keep = ["id", "title", "genres", "keywords", "cast", "crew",
            "overview", "vote_average", "popularity", "release_date"]
    existing = [c for c in keep if c in df.columns]
    df = df[existing].copy()

    # Drop rows missing title
    df.dropna(subset=["title"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Extract year
    if "release_date" in df.columns:
        df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)

    # Build title → index lookup (lowercase keys for search)
    movie_index = {title.lower(): int(idx) for idx, title in enumerate(df["title"])}

    # Build embedding text and encode
    print("Building embedding text…")
    df["embed_text"] = df.apply(build_embed_text, axis=1)

    print("Loading sentence transformer model (all-MiniLM-L6-v2)…")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding movies (this may take a minute)…")
    embeddings = model.encode(
        df["embed_text"].tolist(),
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,
    ).astype("float32")

    # ------------------------------------------------------------------
    # Poster URLs from Kaggle: sakshisemalti/movies-dataset-with-posters
    # Matches by title (case-insensitive). URLs are public TMDB CDN links.
    # ------------------------------------------------------------------
    print("Downloading poster dataset from Kaggle…")
    kaggle_path = kagglehub.dataset_download("sakshisemalti/movies-dataset-with-posters")
    poster_df = pd.read_csv(f"{kaggle_path}/poster.csv")

    # Build lowercase title → URL lookup
    poster_map = {
        str(row["title"]).lower(): str(row["poster"])
        for _, row in poster_df.iterrows()
        if pd.notna(row.get("poster"))
    }
    df["poster_url"] = df["title"].str.lower().map(poster_map).fillna("")
    matched = (df["poster_url"] != "").sum()
    print(f"Matched {matched}/{len(df)} movies with poster URLs.")

    # Save
    os.makedirs(DATA_DIR, exist_ok=True)

    # Drop embed_text before saving parquet (not needed at runtime)
    df.drop(columns=["embed_text"], inplace=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_parquet(f"{DATA_DIR}/movies.parquet", index=False)
    np.save(f"{DATA_DIR}/embeddings.npy", embeddings)
    with open(f"{DATA_DIR}/movie_index.json", "w") as f:
        json.dump(movie_index, f)

    print(f"Done! Saved {len(df)} movies.")
    print(f"  data/movies.parquet  ({df.memory_usage(deep=True).sum() // 1024} KB in memory)")
    print(f"  data/embeddings.npy  shape={embeddings.shape}")
    print(f"  data/movie_index.json  ({len(movie_index)} entries)")


if __name__ == "__main__":
    main()
