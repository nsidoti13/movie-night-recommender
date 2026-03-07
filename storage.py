"""
storage.py — Persist ratings to PostgreSQL (primary) or JSON file (fallback).

PostgreSQL is used when DATABASE_URL is set in the environment or .env file.
Falls back to JSON file (local) or empty state (cloud) when no DB is configured.
"""

import hashlib
import json
import os
import secrets
from contextlib import contextmanager

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

DATABASE_URL  = os.environ.get("DATABASE_URL", "")
RATINGS_FILE  = "data/ratings.json"
ACCOUNTS_FILE = "data/accounts.json"
FRIENDS_FILE  = "data/friends.json"

_DEFAULTS = {
    "liked": [],
    "disliked": [],
    "unseen": [],
    "done": False,
    "current_card": None,
}


def _use_postgres() -> bool:
    return _HAS_PSYCOPG2 and bool(DATABASE_URL) and _pg_available is not False


@contextmanager
def _db():
    """Open a connection, yield it, then always close it."""
    url = DATABASE_URL
    # Supabase requires SSL — add sslmode if not already present
    if "sslmode" not in url:
        sep = "&" if "?" in url else "?"
        url += f"{sep}sslmode=require"
    conn = psycopg2.connect(url, connect_timeout=10)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema migration (runs once per process) ──────────────────────────────

_migrated = False
_pg_available: bool | None = None  # None = untested, True = working, False = failed


def _pg_migrate():
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username      TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    salt          TEXT NOT NULL,
                    created_at    TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS friendships (
                    user_a     TEXT NOT NULL,
                    user_b     TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (user_a, user_b)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ratings (
                    user_name   TEXT    NOT NULL,
                    movie_idx   INTEGER NOT NULL,
                    status      TEXT    NOT NULL
                                CHECK (status IN ('liked', 'disliked', 'unseen')),
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


def _ensure_migrated():
    global _migrated, _pg_available
    if _migrated or not (_HAS_PSYCOPG2 and bool(DATABASE_URL)):
        return
    try:
        _pg_migrate()
        _pg_available = True
        _migrated = True
    except Exception as exc:
        _pg_available = False
        print(f"[storage] PostgreSQL unavailable ({exc}); falling back to JSON file.")


# ── PostgreSQL read / write ────────────────────────────────────────────────

# ── Password helpers ───────────────────────────────────────────────────────

def _new_salt() -> str:
    return secrets.token_hex(16)


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()


# ── Account storage (PostgreSQL) ───────────────────────────────────────────

def _pg_create_user(username: str, password: str) -> bool:
    """Returns False if username already taken."""
    salt = _new_salt()
    pw_hash = _hash_password(password, salt)
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM users WHERE username = %s", (username,))
            if cur.fetchone():
                return False
            cur.execute(
                "INSERT INTO users (username, password_hash, salt) VALUES (%s, %s, %s)",
                (username, pw_hash, salt),
            )
    return True


def _pg_authenticate(username: str, password: str) -> bool:
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT password_hash, salt FROM users WHERE username = %s", (username,)
            )
            row = cur.fetchone()
    if not row:
        return False
    stored_hash, salt = row
    return _hash_password(password, salt) == stored_hash


# ── Account storage (JSON file) ────────────────────────────────────────────

def _accounts_load() -> dict:
    if os.path.exists(ACCOUNTS_FILE):
        try:
            with open(ACCOUNTS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _accounts_save(data: dict):
    os.makedirs("data", exist_ok=True)
    with open(ACCOUNTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── Friends storage (PostgreSQL) ───────────────────────────────────────────

def _pg_add_friend(user: str, friend: str):
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO friendships (user_a, user_b) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (user, friend),
            )
            cur.execute(
                "INSERT INTO friendships (user_a, user_b) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (friend, user),
            )


def _pg_get_friends(user: str) -> list[str]:
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_b FROM friendships WHERE user_a = %s ORDER BY user_b",
                (user,),
            )
            return [r[0] for r in cur.fetchall()]


def _pg_user_exists(username: str) -> bool:
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM users WHERE username = %s", (username,))
            return cur.fetchone() is not None


# ── Friends storage (JSON file) ─────────────────────────────────────────────

def _friends_load() -> dict:
    if os.path.exists(FRIENDS_FILE):
        try:
            with open(FRIENDS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _friends_save(data: dict):
    os.makedirs("data", exist_ok=True)
    with open(FRIENDS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _pg_list_users() -> list[str]:
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT user_name FROM user_state ORDER BY user_name")
            return [r[0] for r in cur.fetchall()]


def _pg_load_user(user: str) -> dict:
    u = user.lower()
    with _db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT movie_idx, status FROM ratings WHERE user_name = %s ORDER BY rated_at",
                (u,)
            )
            rows = cur.fetchall()
            liked    = [r["movie_idx"] for r in rows if r["status"] == "liked"]
            disliked = [r["movie_idx"] for r in rows if r["status"] == "disliked"]
            unseen   = [r["movie_idx"] for r in rows if r["status"] == "unseen"]

            cur.execute(
                "SELECT done, current_card FROM user_state WHERE user_name = %s",
                (u,)
            )
            state = cur.fetchone()

    return {
        "liked":        liked,
        "disliked":     disliked,
        "unseen":       unseen,
        "done":         bool(state["done"])         if state else False,
        "current_card": state["current_card"] if state else None,
    }


def _pg_save_user(user: str, data: dict):
    u = user.lower()
    all_rated = (
        [(u, idx, "liked")    for idx in data["liked"]]
        + [(u, idx, "disliked") for idx in data["disliked"]]
        + [(u, idx, "unseen")   for idx in data["unseen"]]
    )
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM ratings WHERE user_name = %s", (u,))
            if all_rated:
                psycopg2.extras.execute_values(
                    cur,
                    "INSERT INTO ratings (user_name, movie_idx, status) VALUES %s",
                    all_rated,
                )
            cur.execute(
                """
                INSERT INTO user_state (user_name, done, current_card, updated_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (user_name) DO UPDATE
                    SET done         = EXCLUDED.done,
                        current_card = EXCLUDED.current_card,
                        updated_at   = NOW()
                """,
                (u, data["done"], data["current_card"]),
            )


def _pg_clear():
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM ratings")
            cur.execute("DELETE FROM user_state")


# ── JSON / in-memory fallback ──────────────────────────────────────────────

_store: dict | None = None


def _default_user() -> dict:
    return {k: (list(v) if isinstance(v, list) else v) for k, v in _DEFAULTS.items()}


def _default_store() -> dict:
    return {}


def _ensure_keys(data: dict) -> dict:
    for user in list(data.keys()):
        for k, v in _DEFAULTS.items():
            data[user].setdefault(k, list(v) if isinstance(v, list) else v)
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

def user_exists(username: str) -> bool:
    _ensure_migrated()
    u = username.lower()
    if _use_postgres():
        try:
            return _pg_user_exists(u)
        except Exception as exc:
            print(f"[storage] user_exists() DB error ({exc}); using file fallback.")
    return u in _accounts_load()


def add_friend(user: str, friend: str) -> str:
    """Add a mutual friendship. Returns 'ok', 'not_found', 'already_friends', or 'self'."""
    u = user.lower()
    f = friend.lower()
    if u == f:
        return "self"
    if not user_exists(f):
        return "not_found"
    if f in get_friends(u):
        return "already_friends"
    _ensure_migrated()
    if _use_postgres():
        try:
            _pg_add_friend(u, f)
            return "ok"
        except Exception as exc:
            print(f"[storage] add_friend() DB error ({exc}); using file fallback.")
    data = _friends_load()
    data.setdefault(u, [])
    data.setdefault(f, [])
    if f not in data[u]:
        data[u].append(f)
    if u not in data[f]:
        data[f].append(u)
    _friends_save(data)
    return "ok"


def get_friends(user: str) -> list[str]:
    """Return sorted list of friend usernames for the given user."""
    _ensure_migrated()
    u = user.lower()
    if _use_postgres():
        try:
            return _pg_get_friends(u)
        except Exception as exc:
            print(f"[storage] get_friends() DB error ({exc}); using file fallback.")
    return sorted(_friends_load().get(u, []))


def create_user(username: str, password: str) -> bool:
    """Register a new account. Returns False if username already taken."""
    _ensure_migrated()
    u = username.lower()
    if _use_postgres():
        try:
            return _pg_create_user(u, password)
        except Exception as exc:
            print(f"[storage] create_user() DB error ({exc}); using file fallback.")
    accounts = _accounts_load()
    if u in accounts:
        return False
    salt = _new_salt()
    accounts[u] = {"password_hash": _hash_password(password, salt), "salt": salt}
    _accounts_save(accounts)
    return True


def authenticate(username: str, password: str) -> bool:
    """Return True if the credentials are valid."""
    _ensure_migrated()
    u = username.lower()
    if _use_postgres():
        try:
            return _pg_authenticate(u, password)
        except Exception as exc:
            print(f"[storage] authenticate() DB error ({exc}); using file fallback.")
    accounts = _accounts_load()
    if u not in accounts:
        return False
    entry = accounts[u]
    return _hash_password(password, entry["salt"]) == entry["password_hash"]


def list_users() -> list[str]:
    """Return sorted list of all known user names (lowercase)."""
    _ensure_migrated()
    if _use_postgres():
        try:
            return _pg_list_users()
        except Exception as exc:
            print(f"[storage] list_users() DB error ({exc}); using file fallback.")
    return sorted(_file_load().keys())


def load() -> dict:
    _ensure_migrated()
    if _use_postgres():
        try:
            users = _pg_list_users()
            return {u: _pg_load_user(u) for u in users}
        except Exception as exc:
            print(f"[storage] load() DB error ({exc}); using file fallback.")
    return _file_load()


def load_user(user: str) -> dict:
    _ensure_migrated()
    if _use_postgres():
        try:
            return _pg_load_user(user)
        except Exception as exc:
            print(f"[storage] load_user() DB error ({exc}); using file fallback.")
    u = user.lower()
    return _file_load().get(u, _default_user())


def save_user(user: str, user_data: dict):
    _ensure_migrated()
    if _use_postgres():
        try:
            _pg_save_user(user, user_data)
            return
        except Exception as exc:
            print(f"[storage] save_user() DB error ({exc}); using file fallback.")
    data = _file_load()
    data[user.lower()] = user_data
    _file_save(data)


def clear():
    _ensure_migrated()
    if _use_postgres():
        try:
            _pg_clear()
            return
        except Exception as exc:
            print(f"[storage] clear() DB error ({exc}); using file fallback.")
    _file_clear()


def backend() -> str:
    return "PostgreSQL" if _use_postgres() else "JSON file"
