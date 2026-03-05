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
from llm_search import smart_search, _api_available

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


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _load_into_session(user: str):
    saved = storage.load_user(user)
    u = user.lower()
    st.session_state[f"{u}_liked"]    = saved["liked"]
    st.session_state[f"{u}_disliked"] = saved["disliked"]
    st.session_state[f"{u}_unseen"]   = saved["unseen"]
    st.session_state[f"{u}_done"]     = saved["done"]
    st.session_state[f"current_card_{u}"] = saved["current_card"]


def _save_from_session(user: str):
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
    liked   = st.session_state[f"{u}_liked"]
    disliked = st.session_state[f"{u}_disliked"]
    unseen  = st.session_state[f"{u}_unseen"]
    st.session_state[f"current_card_{u}"] = next_card(liked, disliked + unseen)


def _rerate(user: str, idx: int, new_status: str):
    """Move a movie index from whatever list it's in to new_status (liked/disliked/unseen/none)."""
    u = user.lower()
    for lst in ("liked", "disliked", "unseen"):
        key = f"{u}_{lst}"
        if idx in st.session_state[key]:
            st.session_state[key].remove(idx)
    if new_status in ("liked", "disliked", "unseen"):
        st.session_state[f"{u}_{new_status}"].append(idx)
    _save_from_session(user)


def _current_status(user: str, idx: int) -> str | None:
    u = user.lower()
    for lst in ("liked", "disliked", "unseen"):
        if idx in st.session_state[f"{u}_{lst}"]:
            return lst
    return None


# ---------------------------------------------------------------------------
# Session init
# ---------------------------------------------------------------------------

def init_state():
    load_data()
    if "phase" not in st.session_state:
        st.session_state.phase = "login"
    if "active_user" not in st.session_state:
        st.session_state.active_user = None

    if "session_loaded" not in st.session_state:
        for user in ("Regan", "Nicholas"):
            _load_into_session(user)
        st.session_state.session_loaded = True
        if st.session_state.regan_done and st.session_state.nicholas_done:
            st.session_state.phase = "results"


def reset_all():
    storage.clear()
    reset_seeds()
    for k in list(st.session_state.keys()):
        del st.session_state[k]


# ---------------------------------------------------------------------------
# Sidebar: search + review
# ---------------------------------------------------------------------------

def _render_search_results(indices: list[int], user: str, df: pd.DataFrame, key_prefix: str):
    """Render a selectbox + rate buttons for a list of movie row indices."""
    if not indices:
        st.caption("No matches found.")
        return

    status_labels = {"liked": "👍 Liked", "disliked": "👎 Disliked", "unseen": "🤷 Haven't Seen"}
    options = {
        f"{df.iloc[idx]['title']} ({int(df.iloc[idx].get('year', 0) or 0)})": idx
        for idx in indices
    }
    chosen_label = st.selectbox("Results", list(options.keys()), key=f"{key_prefix}_select")
    chosen_idx = options[chosen_label]

    status = _current_status(user, chosen_idx)
    if status:
        st.caption(f"Currently: {status_labels[status]}")

    sa, sb, sc = st.columns(3)
    if sa.button("👍", key=f"{key_prefix}_like", use_container_width=True, help="Like"):
        _rerate(user, chosen_idx, "liked")
        st.rerun()
    if sb.button("👎", key=f"{key_prefix}_dislike", use_container_width=True, help="Dislike"):
        _rerate(user, chosen_idx, "disliked")
        st.rerun()
    if sc.button("🤷", key=f"{key_prefix}_unseen", use_container_width=True, help="Haven't Seen"):
        _rerate(user, chosen_idx, "unseen")
        st.rerun()


def rating_sidebar(user: str, df: pd.DataFrame, movie_index: dict):
    u = user.lower()
    liked    = st.session_state[f"{u}_liked"]
    disliked = st.session_state[f"{u}_disliked"]
    unseen   = st.session_state[f"{u}_unseen"]

    with st.sidebar:
        # ── Search & Add ──────────────────────────────────────────────────
        st.markdown("### 🔍 Search & Add")

        query = st.text_input(
            "Search movies",
            key="search_query",
            placeholder="e.g. Inception, marvel movies, Nolan films…"
        )

        if query.strip():
            q = query.strip().lower()

            # ── Basic contains search (instant) ───────────────────────────
            basic = [idx for title, idx in movie_index.items() if q in title]
            basic.sort(key=lambda idx: (
                not df.iloc[idx]['title'].lower().startswith(q),
                -float(df.iloc[idx].get('popularity', 0) or 0)
            ))
            basic = basic[:10]

            # ── AI Smart Search ───────────────────────────────────────────
            # Triggered by button; result stored in session state so it
            # persists across reruns without re-calling the API.
            ai_key = f"ai_results_{q}"
            ai_results = st.session_state.get(ai_key, None)

            if _api_available():
                if st.button("✨ AI Search", use_container_width=True, key="ai_search_btn",
                             help="Uses Claude to match descriptions, franchises, typos, and more"):
                    with st.spinner("Thinking…"):
                        st.session_state[ai_key] = smart_search(query, movie_index, df)
                    ai_results = st.session_state[ai_key]
                    st.rerun()

            # Show AI results if available, otherwise basic results
            if ai_results is not None:
                st.caption(f'✨ AI results for "{query}"')
                _render_search_results(ai_results, user, df, key_prefix="ai")
                if st.button("Clear AI results", key="clear_ai"):
                    del st.session_state[ai_key]
                    st.rerun()
            elif basic:
                st.caption("Basic results (use ✨ AI Search for descriptions/franchises/typos)")
                _render_search_results(basic, user, df, key_prefix="basic")
            else:
                st.caption("No basic matches. Try ✨ AI Search.")

        st.markdown("---")

        # ── Review Ratings ────────────────────────────────────────────────
        st.markdown("### 📋 My Ratings")

        for lst_key, label, icon in [("liked", "Liked", "👍"), ("disliked", "Disliked", "👎"), ("unseen", "Haven't Seen", "🤷")]:
            indices = st.session_state[f"{u}_{lst_key}"]
            with st.expander(f"{icon} {label} ({len(indices)})", expanded=(lst_key == "liked")):
                if not indices:
                    st.caption("None yet.")
                else:
                    for idx in reversed(indices):  # most recent first
                        row = df.iloc[idx]
                        title = row["title"]
                        year = int(row.get("year", 0) or 0)
                        st.markdown(f"**{title}** ({year})")
                        ca, cb, cc, cd = st.columns(4)
                        if ca.button("👍", key=f"re_{lst_key}_{idx}_like", help="Like"):
                            _rerate(user, idx, "liked")
                            st.rerun()
                        if cb.button("👎", key=f"re_{lst_key}_{idx}_dislike", help="Dislike"):
                            _rerate(user, idx, "disliked")
                            st.rerun()
                        if cc.button("🤷", key=f"re_{lst_key}_{idx}_unseen", help="Haven't Seen"):
                            _rerate(user, idx, "unseen")
                            st.rerun()
                        if cd.button("✕", key=f"re_{lst_key}_{idx}_remove", help="Remove"):
                            _rerate(user, idx, "none")
                            st.rerun()

        st.markdown("---")
        if st.button("← Back", use_container_width=True):
            st.session_state.phase = "login"
            st.rerun()


# ---------------------------------------------------------------------------
# Phase: Login
# ---------------------------------------------------------------------------

def login_phase():
    st.title("🎬 Movie Night")
    st.markdown("### Who are you?")
    st.markdown(" ")

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
    df, _, movie_index = load_data()
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
        if st.button("✅ I'm Done Rating", use_container_width=True):
            st.session_state[f"{u}_done"] = True
            _save_from_session(user)
            other_data = storage.load_user(other)
            if other_data["done"]:
                st.session_state.phase = "results"
            else:
                st.session_state.phase = "login"
            st.rerun()

    # Sidebar with search + review
    rating_sidebar(user, df, movie_index)


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
