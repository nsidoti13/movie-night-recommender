"""
llm_search.py — AI-powered movie search using Claude.

Interprets queries like:
  - "marvel movies"         → franchise/keyword search
  - "Nolan films"           → director search
  - "avenjers" (typo)       → fuzzy title match
  - "space war with aliens" → thematic search

Results are cached per query string so each unique query only calls the API once.
Requires ANTHROPIC_API_KEY in Streamlit secrets or as an environment variable.
"""

import json
import difflib
import streamlit as st
import pandas as pd
import anthropic


@st.cache_resource
def _get_client():
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if api_key:
            return anthropic.Anthropic(api_key=api_key)
    except Exception:
        pass
    # Falls back to ANTHROPIC_API_KEY env var
    return anthropic.Anthropic()


def _api_available() -> bool:
    """Return True if we can reach the API (key configured)."""
    try:
        _get_client()
        return True
    except Exception:
        return False


@st.cache_data(ttl=3600, show_spinner=False)
def _llm_interpret(query: str) -> dict:
    """
    Ask Claude to interpret the search query.
    Returns: {type, keywords, suggested_titles}
    Cached per query string for 1 hour.
    """
    try:
        client = _get_client()
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": (
                    f'A user searched a movie database for: "{query}"\n\n'
                    "Return a JSON object with exactly these keys:\n"
                    '- "type": "title" if searching for a specific movie (possibly misspelled/paraphrased),'
                    ' or "description" if describing a genre/theme/franchise/director/actor\n'
                    '- "keywords": list of 4-8 lowercase terms (genres, themes, character names, '
                    "director names, actor names, franchise names) that would appear in a movie's "
                    "description or keywords — useful for matching movies in a database\n"
                    '- "suggested_titles": if type is "title", up to 5 movie titles the user probably meant;'
                    " if type is \"description\", up to 12 real movie titles that fit the description\n\n"
                    "Return only valid JSON. No explanation, no markdown."
                )
            }]
        )
        return json.loads(response.content[0].text)
    except Exception:
        return {
            "type": "description",
            "keywords": query.lower().split(),
            "suggested_titles": [],
        }


def _fuzzy_title_match(query: str, movie_index: dict, limit: int = 6) -> list[int]:
    """Return row indices of movies whose titles closely match the query string."""
    q = query.lower().strip()
    titles = list(movie_index.keys())

    # Exact contains match first
    contains = [t for t in titles if q in t]
    contains.sort(key=lambda t: (not t.startswith(q), len(t)))

    # Fuzzy (edit-distance) match
    fuzzy = difflib.get_close_matches(q, titles, n=limit, cutoff=0.55)

    seen = set()
    results = []
    for t in contains[:limit] + fuzzy:
        idx = movie_index[t]
        if idx not in seen:
            seen.add(idx)
            results.append(idx)
    return results[:limit]


def _keyword_search(keywords: list[str], df: pd.DataFrame, limit: int = 20) -> list[int]:
    """
    Search the precomputed 'soup' column for keyword hits.
    Ranks by number of keyword matches, then by popularity.
    """
    scores: dict[int, float] = {}
    for idx, row in df.iterrows():
        soup = str(row.get("soup", "")).lower()
        hits = sum(1 for kw in keywords if kw.lower() in soup)
        if hits > 0:
            popularity = float(row.get("popularity", 0) or 0)
            scores[idx] = hits * 1000 + popularity

    return sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:limit]


def smart_search(query: str, movie_index: dict, df: pd.DataFrame) -> list[int]:
    """
    Full AI-powered search pipeline:
      1. Claude interprets the query → type + keywords + suggested_titles
      2. Suggested titles → looked up / fuzzy-matched in movie_index
      3. Fuzzy title match on the raw query (for title searches)
      4. Keyword search on the soup column using Claude's extracted keywords
    Returns up to 20 deduplicated row indices, best matches first.
    """
    interp = _llm_interpret(query)
    q_type = interp.get("type", "description")
    keywords = interp.get("keywords", [])
    suggested = interp.get("suggested_titles", [])

    seen: set[int] = set()
    results: list[int] = []

    def add(idx: int):
        if idx not in seen:
            seen.add(idx)
            results.append(idx)

    # 1. Claude's suggested titles → direct lookup + fuzzy fallback
    for title in suggested:
        t = title.lower()
        if t in movie_index:
            add(movie_index[t])
        else:
            for idx in _fuzzy_title_match(t, movie_index, limit=2):
                add(idx)

    # 2. Fuzzy match on the raw query (especially useful for typos)
    if q_type == "title":
        for idx in _fuzzy_title_match(query, movie_index, limit=5):
            add(idx)

    # 3. Keyword search using Claude's extracted terms
    for idx in _keyword_search(keywords, df, limit=20):
        add(idx)

    return results[:20]
