"""
app.py — Movie Night Recommender (Streamlit)
Run via:  python start.py   (starts ngrok + streamlit)
     or:  streamlit run app.py  (local only)
"""

import json
import ast
import streamlit as st
import pandas as pd

import storage
from recommender import load_data, next_card, joint_recommendations, reset_seeds

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MIN_LIKES = 5
PLACEHOLDER_IMG = "https://via.placeholder.com/300x450?text=No+Poster"

st.set_page_config(page_title="Movie Night 🎬", layout="centered")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_list_field(value, key="name", limit=None):
    if isinstance(value, list):
        items = value
    else:
        try:
            items = json.loads(value)
        except Exception:
            try:
                items = ast.literal_eval(str(value))
            except Exception:
                return []
    result = [item.get(key, "") for item in items if isinstance(item, dict)]
    return result[:limit] if limit else result


def genre_tags(genres_val) -> str:
    names = parse_list_field(genres_val)
    return "  ·  ".join(names[:4]) if names else "—"


def get_poster(row: dict) -> str:
    url = row.get("poster_url", "")
    if url and str(url) not in ("", "nan", "None"):
        return str(url)
    return PLACEHOLDER_IMG


def render_card(row, row_idx: int):
    title = row.get("title", "Unknown")
    year = int(row.get("year", 0)) if row.get("year") else "?"
    genres = genre_tags(row.get("genres", "[]"))
    overview = str(row.get("overview", ""))
    overview_short = overview[:250] + ("…" if len(overview) > 250 else "")

    st.image(get_poster(row), width=240)
    st.markdown(f"## {title} ({year})")
    st.caption(genres)
    st.write(overview_short)


def liked_sidebar(user: str, liked: list, df: pd.DataFrame):
    with st.sidebar:
        st.markdown(f"**{user}'s Likes ({len(liked)})**")
        for idx in liked[-15:]:
            st.write(f"👍 {df.iloc[idx]['title']}")


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _load_into_session(user: str):
    """Read this user's saved ratings from disk into session state."""
    saved = storage.load_user(user)
    u = user.lower()
    st.session_state[f"{u}_liked"]   = saved["liked"]
    st.session_state[f"{u}_disliked"] = saved["disliked"]
    st.session_state[f"{u}_unseen"]  = saved["unseen"]
    st.session_state[f"{u}_done"]    = saved["done"]
    card = saved["current_card"]
    st.session_state[f"current_card_{u}"] = card


def _save_from_session(user: str):
    """Write this user's session state ratings back to disk."""
    u = user.lower()
    storage.save_user(user, {
        "liked":        st.session_state[f"{u}_liked"],
        "disliked":     st.session_state[f"{u}_disliked"],
        "unseen":       st.session_state[f"{u}_unseen"],
        "done":         st.session_state[f"{u}_done"],
        "current_card": st.session_state.get(f"current_card_{u}"),
    })


def _advance_card(user: str):
    u = user.lower()
    liked    = st.session_state[f"{u}_liked"]
    disliked = st.session_state[f"{u}_disliked"]
    unseen   = st.session_state[f"{u}_unseen"]
    st.session_state[f"current_card_{u}"] = next_card(liked, disliked + unseen)


# ---------------------------------------------------------------------------
# Session init
# ---------------------------------------------------------------------------

def init_state():
    load_data()  # warm cache
    if "phase" not in st.session_state:
        st.session_state.phase = "login"
    if "active_user" not in st.session_state:
        st.session_state.active_user = None

    # Load both users' saved ratings on first run of each session
    if "session_loaded" not in st.session_state:
        for user in ("Regan", "Nicholas"):
            _load_into_session(user)
            # If they had a saved card, keep it; otherwise we'll assign when they log in
        st.session_state.session_loaded = True

        # If both users are done, jump straight to results
        if st.session_state.regan_done and st.session_state.nicholas_done:
            st.session_state.phase = "results"


def reset_all():
    storage.clear()
    reset_seeds()
    for k in list(st.session_state.keys()):
        del st.session_state[k]


# ---------------------------------------------------------------------------
# Phase: Login
# ---------------------------------------------------------------------------

def login_phase():
    st.title("🎬 Movie Night")
    st.markdown("### Who are you?")
    st.markdown(" ")

    # Show status of each user
    ratings = storage.load()
    for user in ("Regan", "Nicholas"):
        u = user.lower()
        n_likes = len(ratings[u]["liked"])
        done = ratings[u]["done"]
        status = "✅ Done" if done else (f"{n_likes} likes so far" if n_likes else "Not started")
        st.caption(f"{user}: {status}")

    st.markdown(" ")
    col1, col2 = st.columns(2)

    if col1.button("👩 I'm Regan", use_container_width=True):
        _load_into_session("Regan")
        if st.session_state.current_card_regan is None:
            _advance_card("Regan")
            _save_from_session("Regan")
        st.session_state.active_user = "Regan"
        st.session_state.phase = "rating"
        st.rerun()

    if col2.button("👨 I'm Nicholas", use_container_width=True):
        _load_into_session("Nicholas")
        if st.session_state.current_card_nicholas is None:
            _advance_card("Nicholas")
            _save_from_session("Nicholas")
        st.session_state.active_user = "Nicholas"
        st.session_state.phase = "rating"
        st.rerun()


# ---------------------------------------------------------------------------
# Phase: Rating
# ---------------------------------------------------------------------------

def rating_phase():
    df, _, _ = load_data()
    user = st.session_state.active_user
    other = "Nicholas" if user == "Regan" else "Regan"
    u = user.lower()

    liked    = st.session_state[f"{u}_liked"]
    disliked = st.session_state[f"{u}_disliked"]
    card_key = f"current_card_{u}"

    st.title(f"🎬 {user}'s Turn")
    like_count = len(liked)
    st.progress(
        min(like_count / MIN_LIKES, 1.0),
        text=f"{like_count}/{MIN_LIKES} likes"
        if like_count < MIN_LIKES else f"{like_count} likes ✅"
    )

    card_idx = st.session_state[card_key]
    row = df.iloc[card_idx].to_dict()
    render_card(row, card_idx)
    st.markdown("---")

    b1, b2, b3 = st.columns(3)

    if b1.button("👍 Like", use_container_width=True, key="btn_like"):
        liked.append(card_idx)
        _advance_card(user)
        _save_from_session(user)
        st.rerun()

    if b2.button("👎 Dislike", use_container_width=True, key="btn_dislike"):
        disliked.append(card_idx)
        _advance_card(user)
        _save_from_session(user)
        st.rerun()

    if b3.button("🤷 Haven't Seen", use_container_width=True, key="btn_unseen"):
        st.session_state[f"{u}_unseen"].append(card_idx)
        _advance_card(user)
        _save_from_session(user)
        st.rerun()

    if like_count >= MIN_LIKES:
        st.markdown(" ")
        if st.button(f"✅ I'm Done Rating", use_container_width=True):
            st.session_state[f"{u}_done"] = True
            _save_from_session(user)
            # Check if the other user is also done (read from disk, not just session)
            other_data = storage.load_user(other)
            if other_data["done"]:
                st.session_state.phase = "results"
            else:
                st.session_state.phase = "login"
            st.rerun()

    liked_sidebar(user, liked, df)

    with st.sidebar:
        st.markdown("---")
        if st.button("← Back"):
            st.session_state.phase = "login"
            st.rerun()


# ---------------------------------------------------------------------------
# Phase: Results
# ---------------------------------------------------------------------------

def results_phase():
    df, _, _ = load_data()

    st.title("🎉 Movies You'll Both Love")

    blend = st.slider(
        "Taste Blend", 0.0, 1.0, 0.5, 0.05,
        help="0 = Regan's taste only · 1 = Nicholas's taste only"
    )
    st.caption("← Regan " + "─" * 18 + " Nicholas →")

    # Always read from disk so both devices see the same results
    ratings = storage.load()
    recs = joint_recommendations(
        ratings["regan"]["liked"],
        ratings["nicholas"]["liked"],
        ratings["regan"]["disliked"],
        ratings["nicholas"]["disliked"],
        blend=blend,
        n=10,
    )

    if not recs:
        st.warning("Not enough data. Go back and rate more movies!")
    else:
        cols = st.columns(2)
        for i, idx in enumerate(recs):
            row = df.iloc[idx].to_dict()
            with cols[i % 2]:
                render_card(row, idx)
                st.markdown("---")

    st.markdown(" ")
    if st.button("🔄 Start Over"):
        reset_all()
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

init_state()

if st.session_state.phase == "login":
    login_phase()
elif st.session_state.phase == "rating":
    rating_phase()
else:
    results_phase()
