"""
storage.py — Persist ratings to PostgreSQL (primary) or JSON file (fallback).

PostgreSQL is used when DATABASE_URL is set in the environment or .env file.
Falls back to in-memory + JSON when no database is configured (e.g. Streamlit Cloud
without a hosted DB).
"""

import json
import os

# ── Optional dependencies ──────────────────────────────────────────────────
try:
    import psycopg2
    import psycopg2.extras
    _HAS_PSYCOPG2 = True
except ImportError:
    _HAS_PSYCOPG2 = False

# Load .env file if present (local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DATABASE_URL = os.environ.get("DATABASE_URL", "")
RATINGS_FILE = "data/ratings.json"

_DEFAULTS = {
    "liked": [],
    "disliked": [],
    "unseen": [],
    "done": False,
    "current_card": None,
}


def _use_postgres() -> bool:
    return _HAS_PSYCOPG2 and bool(DATABASE_URL)


def _get_conn():
    return psycopg2.connect(DATABASE_URL)


def _pg_migrate():
    """Create tables if they don't exist (runs once on startup)."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ratings (
                    user_name   TEXT    NOT NULL,
                    movie_idx   INTEGER NOT NULL,
                    status      TEXT    NOT NULL CHECK (status IN ('liked', 'disliked', 'unseen')),
                    rated_at    TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (user_name, movie_idx)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_state (
                    user_name    TEXT    PRIMARY KEY,
                    done         BOOLEAN DEFAULT FALSE,
                    current_card INTEGER DEFAULT NULL,
                    updated_at   TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE OR REPLACE VIEW rating_summary AS
                SELECT
                    user_name,
                    COUNT(*) FILTER (WHERE status = 'liked')    AS liked,
                    COUNT(*) FILTER (WHERE status = 'disliked') AS disliked,
                    COUNT(*) FILTER (WHERE status = 'unseen')   AS unseen,
                    COUNT(*)                                     AS total
                FROM ratings
                GROUP BY user_name
            """)
        conn.commit()


_migrated = False


# ── PostgreSQL implementation ──────────────────────────────────────────────

def _pg_load_user(user: str) -> dict:
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Ratings
            cur.execute(
                "SELECT movie_idx, status FROM ratings WHERE user_name = %s",
                (user.lower(),)
            )
            rows = cur.fetchall()
            liked    = [r["movie_idx"] for r in rows if r["status"] == "liked"]
            disliked = [r["movie_idx"] for r in rows if r["status"] == "disliked"]
            unseen   = [r["movie_idx"] for r in rows if r["status"] == "unseen"]

            # State
            cur.execute(
                "SELECT done, current_card FROM user_state WHERE user_name = %s",
                (user.lower(),)
            )
            state = cur.fetchone()
            done         = state["done"]         if state else False
            current_card = state["current_card"] if state else None

    return {
        "liked": liked,
        "disliked": disliked,
        "unseen": unseen,
        "done": done,
        "current_card": current_card,
    }


def _pg_save_user(user: str, data: dict):
    u = user.lower()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            # Upsert each rated movie
            all_rated = (
                [(u, idx, "liked")    for idx in data["liked"]]
                + [(u, idx, "disliked") for idx in data["disliked"]]
                + [(u, idx, "unseen")   for idx in data["unseen"]]
            )

            # Remove ratings no longer present
            cur.execute("DELETE FROM ratings WHERE user_name = %s", (u,))

            if all_rated:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO ratings (user_name, movie_idx, status)
                    VALUES %s
                    ON CONFLICT (user_name, movie_idx)
                    DO UPDATE SET status = EXCLUDED.status, rated_at = NOW()
                    """,
                    all_rated,
                )

            # Upsert user state
            cur.execute(
                """
                INSERT INTO user_state (user_name, done, current_card, updated_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (user_name)
                DO UPDATE SET done = EXCLUDED.done,
                              current_card = EXCLUDED.current_card,
                              updated_at = NOW()
                """,
                (u, data["done"], data["current_card"]),
            )
        conn.commit()


def _pg_load() -> dict:
    return {
        "regan":   _pg_load_user("regan"),
        "nicholas": _pg_load_user("nicholas"),
    }


def _pg_clear():
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM ratings")
            cur.execute("DELETE FROM user_state")
        conn.commit()


# ── JSON / in-memory fallback ──────────────────────────────────────────────

_store: dict | None = None


def _default_store() -> dict:
    return {"regan": dict(_DEFAULTS), "nicholas": dict(_DEFAULTS)}


def _ensure_keys(data: dict) -> dict:
    for user in ("regan", "nicholas"):
        data.setdefault(user, {})
        for k, v in _DEFAULTS.items():
            data[user].setdefault(k, v)
    return data


def _file_load() -> dict:
    global _store
    if _store is not None:
        return _store
    if os.path.exists(RATINGS_FILE):
        try:
            with open(RATINGS_FILE) as f:
                _store = _ensure_keys(json.load(f))
            return _store
        except Exception:
            pass
    _store = _default_store()
    return _store


def _file_save(data: dict):
    global _store
    _store = data
    try:
        os.makedirs("data", exist_ok=True)
        with open(RATINGS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def _file_clear():
    global _store
    _store = _default_store()
    try:
        if os.path.exists(RATINGS_FILE):
            os.remove(RATINGS_FILE)
    except Exception:
        pass


# ── Public API ─────────────────────────────────────────────────────────────

def _ensure_migrated():
    global _migrated
    if not _migrated and _use_postgres():
        try:
            _pg_migrate()
            _migrated = True
        except Exception:
            pass


def load() -> dict:
    _ensure_migrated()
    if _use_postgres():
        try:
            return _pg_load()
        except Exception:
            pass
    return _file_load()


def save(data: dict):
    if _use_postgres():
        try:
            _pg_save_user("regan",   data["regan"])
            _pg_save_user("nicholas", data["nicholas"])
            return
        except Exception:
            pass
    _file_save(data)


def load_user(user: str) -> dict:
    _ensure_migrated()
    if _use_postgres():
        try:
            return _pg_load_user(user)
        except Exception:
            pass
    return _file_load()[user.lower()]


def save_user(user: str, user_data: dict):
    if _use_postgres():
        try:
            _pg_save_user(user, user_data)
            return
        except Exception:
            pass
    data = _file_load()
    data[user.lower()] = user_data
    _file_save(data)


def clear():
    if _use_postgres():
        try:
            _pg_clear()
            return
        except Exception:
            pass
    _file_clear()


def backend() -> str:
    """Return a string describing the active storage backend."""
    return "PostgreSQL" if _use_postgres() else "JSON file"
