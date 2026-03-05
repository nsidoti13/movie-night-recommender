"""
storage.py — Persist ratings across sessions.

Strategy (works both locally and on Streamlit Cloud):
- Primary:   module-level dict shared across all browser sessions in the same
             running process (survives page refreshes, handles multi-device).
- Secondary: data/ratings.json on disk (survives app restarts locally;
             on Streamlit Cloud the file may reset on redeploy, but the
             in-memory store covers the active session).
"""

import json
import os

RATINGS_FILE = "data/ratings.json"

_DEFAULTS = {
    "liked": [],
    "disliked": [],
    "unseen": [],
    "done": False,
    "current_card": None,
}

# In-memory store — shared across ALL sessions in the same running process.
_store: dict | None = None


def _default_store() -> dict:
    return {
        "regan":   dict(_DEFAULTS),
        "nicholas": dict(_DEFAULTS),
    }


def _ensure_keys(data: dict) -> dict:
    for user in ("regan", "nicholas"):
        data.setdefault(user, {})
        for k, v in _DEFAULTS.items():
            data[user].setdefault(k, v)
    return data


def load() -> dict:
    """Return the full ratings dict (in-memory, falling back to disk)."""
    global _store
    if _store is not None:
        return _store
    # Try loading from disk
    if os.path.exists(RATINGS_FILE):
        try:
            with open(RATINGS_FILE) as f:
                _store = _ensure_keys(json.load(f))
            return _store
        except Exception:
            pass
    _store = _default_store()
    return _store


def save(data: dict):
    """Write to the in-memory store and attempt to persist to disk."""
    global _store
    _store = data
    try:
        os.makedirs("data", exist_ok=True)
        with open(RATINGS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass  # Disk write optional — in-memory store is authoritative


def load_user(user: str) -> dict:
    return load()[user.lower()]


def save_user(user: str, user_data: dict):
    data = load()
    data[user.lower()] = user_data
    save(data)


def clear():
    """Reset everything."""
    global _store
    _store = _default_store()
    try:
        if os.path.exists(RATINGS_FILE):
            os.remove(RATINGS_FILE)
    except Exception:
        pass
