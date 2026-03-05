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
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
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


def build_soup(row):
    """Combine genres, keywords, top cast, and director into a single string."""
    genres = extract_names(row.get("genres", "[]"))
    keywords = extract_names(row.get("keywords", "[]"))
    cast = extract_names(row.get("cast", "[]"), limit=3)
    director = extract_director(row.get("crew", "[]"))
    parts = genres + keywords + cast
    if director:
        parts.append(director)
    return " ".join(parts).lower()


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

    # Build soup for TF-IDF
    print("Building text soup…")
    df["soup"] = df.apply(build_soup, axis=1)

    # TF-IDF
    print("Fitting TF-IDF vectoriser…")
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(df["soup"])

    # Build title → index lookup (lowercase keys for search)
    movie_index = {title.lower(): int(idx) for idx, title in enumerate(df["title"])}

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

    df.to_parquet(f"{DATA_DIR}/movies.parquet", index=False)
    sparse.save_npz(f"{DATA_DIR}/tfidf_matrix.npz", matrix)
    with open(f"{DATA_DIR}/movie_index.json", "w") as f:
        json.dump(movie_index, f)

    print(f"Done! Saved {len(df)} movies.")
    print(f"  data/movies.parquet  ({df.memory_usage(deep=True).sum() // 1024} KB in memory)")
    print(f"  data/tfidf_matrix.npz  shape={matrix.shape}")
    print(f"  data/movie_index.json  ({len(movie_index)} entries)")


if __name__ == "__main__":
    main()
