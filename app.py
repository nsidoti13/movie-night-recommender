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


def inject_css():
    st.markdown("""
    <style>
    /* ── Genre + rating pill badges ─────────────────────────────────── */
    .genre-pill {
        display: inline-block;
        background: rgba(229, 9, 20, 0.12);
        color: #ff6b6b;
        border: 1px solid rgba(229, 9, 20, 0.25);
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.72rem;
        font-weight: 500;
        margin: 2px 2px 2px 0;
        letter-spacing: 0.02em;
    }
    .vote-badge {
        display: inline-block;
        background: rgba(255, 200, 0, 0.12);
        color: #ffc800;
        border: 1px solid rgba(255, 200, 0, 0.25);
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.72rem;
        font-weight: 600;
        margin: 2px 4px 2px 0;
    }

    /* ── Movie poster image ──────────────────────────────────────────── */
    [data-testid="stImage"] img {
        border-radius: 10px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.6);
        width: 100%;
    }

    /* ── Action buttons ──────────────────────────────────────────────── */
    .stButton > button {
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 600;
        padding: 0.55rem 1rem;
        transition: transform 0.12s ease, box-shadow 0.12s ease;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ── Progress bar ────────────────────────────────────────────────── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #E50914, #ff4444);
    }

    /* ── Sidebar ─────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* ── Page headings ───────────────────────────────────────────────── */
    h1 { font-weight: 800; letter-spacing: -0.5px; }
    h3 { font-weight: 700; margin-bottom: 0.25rem; }

    /* ── Tabs ────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)


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


def _badge_html(row: dict) -> str:
    """Build HTML string of vote badge + genre pills for a movie row."""
    parts = []
    vote = row.get("vote_average")
    if vote and float(vote) > 0:
        parts.append(f'<span class="vote-badge">⭐ {float(vote):.1f}</span>')
    for g in parse_list_field(row.get("genres", "[]"))[:4]:
        parts.append(f'<span class="genre-pill">{g}</span>')
    return " ".join(parts)


def render_card(row, row_idx: int):
    """Full movie card: poster left, info right."""
    title    = row.get("title", "Unknown")
    year     = int(row.get("year", 0)) if row.get("year") else "?"
    overview = str(row.get("overview", ""))
    overview_short = overview[:300] + ("…" if len(overview) > 300 else "")

    col_poster, col_info = st.columns([1, 2])
    with col_poster:
        st.image(get_poster(row), use_container_width=True)
    with col_info:
        st.markdown(f"### {title} ({year})")
        badges = _badge_html(row)
        if badges:
            st.markdown(badges, unsafe_allow_html=True)
            st.markdown(" ")
        st.write(overview_short)


def render_mini_card(row, idx: int):
    """Compact card for the results grid: poster + title + badges."""
    title = row.get("title", "Unknown")
    year  = int(row.get("year", 0)) if row.get("year") else "?"
    st.image(get_poster(row), use_container_width=True)
    st.markdown(f"**{title}** ({year})")
    badges = _badge_html(row)
    if badges:
        st.markdown(badges, unsafe_allow_html=True)


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
        # ── Tonight's Picks ───────────────────────────────────────────────
        st.markdown("### 🎬 Tonight's Picks")
        friends = storage.get_friends(u)
        if friends:
            selected_friend = st.selectbox(
                "Who are you watching with tonight?",
                [f.capitalize() for f in friends],
                key="watching_with",
            )
            if st.button("Get Picks", use_container_width=True, key="get_picks_btn"):
                st.session_state.compare_user1 = user.capitalize()
                st.session_state.compare_user2 = selected_friend
                st.session_state.phase = "results"
                st.rerun()
        else:
            st.caption("Add a friend below to get movie picks together.")

        st.markdown("---")

        # ── Friends ───────────────────────────────────────────────────────
        st.markdown("### 👥 Friends")
        if friends:
            for f in friends:
                st.markdown(f"• {f.capitalize()}")
            st.markdown(" ")

        add_input = st.text_input(
            "Add by username", placeholder="Their username…", key="add_friend_input"
        )
        if st.button("Add Friend", use_container_width=True, key="add_friend_btn"):
            result = storage.add_friend(u, add_input.strip())
            if result == "ok":
                st.success(f"Added {add_input.strip().capitalize()}!")
                st.rerun()
            elif result == "not_found":
                st.error("No account with that username.")
            elif result == "already_friends":
                st.info("Already friends!")
            elif result == "self":
                st.error("You can't add yourself.")

        st.markdown("---")

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

def _enter_as(uname: str):
    """Load user into session and go to rating phase."""
    _load_into_session(uname)
    if st.session_state.get(f"current_card_{uname}") is None:
        _advance_card(uname)
        _save_from_session(uname)
    st.session_state.active_user = uname
    st.session_state.phase = "rating"
    st.rerun()


def login_phase():
    st.markdown("""
    <div style="text-align:center; padding:2.5rem 0 1.5rem">
        <div style="font-size:3.5rem; line-height:1">🎬</div>
        <h1 style="font-size:2.6rem; font-weight:900; margin:0.4rem 0 0.2rem">Movie Night</h1>
        <p style="color:#888; font-size:1rem; margin:0">Discover movies you'll both love</p>
    </div>
    """, unsafe_allow_html=True)

    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    with tab_login:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", use_container_width=True, key="login_btn"):
            u = username.strip().lower()
            if not u or not password:
                st.error("Please enter your username and password.")
            elif storage.authenticate(u, password):
                _enter_as(u)
            else:
                st.error("Invalid username or password.")

    with tab_signup:
        new_username = st.text_input("Choose a username", key="signup_username")
        new_password = st.text_input("Choose a password", type="password", key="signup_password")
        confirm_pw   = st.text_input("Confirm password",  type="password", key="signup_confirm")
        if st.button("Create Account", use_container_width=True, key="signup_btn"):
            u = new_username.strip().lower()
            if not u or not new_password:
                st.error("Please fill in all fields.")
            elif new_password != confirm_pw:
                st.error("Passwords don't match.")
            elif not storage.create_user(u, new_password):
                st.error("That username is already taken.")
            else:
                st.success(f"Account created! Switch to the Login tab to sign in.")



# ---------------------------------------------------------------------------
# Phase: Rating
# ---------------------------------------------------------------------------

def rating_phase():
    df, _, movie_index = load_data()
    user = st.session_state.active_user
    u = user.lower()

    liked    = st.session_state[f"{u}_liked"]
    disliked = st.session_state[f"{u}_disliked"]
    card_key = f"current_card_{u}"

    st.markdown(f"## 🎬 {user.capitalize()}'s Queue")
    like_count = len(liked)
    progress_val = min(like_count / MIN_LIKES, 1.0)
    progress_text = (
        f"{like_count} / {MIN_LIKES} likes to unlock picks"
        if like_count < MIN_LIKES
        else f"✅  {like_count} likes — you can get picks anytime!"
    )
    st.progress(progress_val, text=progress_text)

    card_idx = st.session_state[card_key]
    row = df.iloc[card_idx].to_dict()
    render_card(row, card_idx)
    st.markdown("---")

    st.markdown(" ")
    b1, b2, b3 = st.columns(3)

    if b1.button("👍  Love it", use_container_width=True, key="btn_like", type="primary"):
        liked.append(card_idx)
        _advance_card(user)
        _save_from_session(user)
        st.rerun()

    if b2.button("👎  Not for me", use_container_width=True, key="btn_dislike"):
        disliked.append(card_idx)
        _advance_card(user)
        _save_from_session(user)
        st.rerun()

    if b3.button("🤷  Haven't Seen", use_container_width=True, key="btn_unseen"):
        st.session_state[f"{u}_unseen"].append(card_idx)
        _advance_card(user)
        _save_from_session(user)
        st.rerun()

    # Sidebar with search + review
    rating_sidebar(user, df, movie_index)


# ---------------------------------------------------------------------------
# Phase: Results
# ---------------------------------------------------------------------------

def results_phase():
    df, _, _ = load_data()

    st.markdown("""
    <div style="text-align:center; padding:1rem 0 0.5rem">
        <h1 style="font-size:2.2rem; font-weight:900; margin:0">🎉 Tonight's Picks</h1>
        <p style="color:#888; margin:0.3rem 0 0">Movies you'll both love</p>
    </div>
    """, unsafe_allow_html=True)

    users = storage.list_users()
    if len(users) < 2:
        st.warning("Need at least 2 users who have rated movies.")
        if st.button("← Back"):
            st.session_state.phase = "rating" if st.session_state.active_user else "login"
            st.rerun()
        return

    display_names = [u.capitalize() for u in users]

    col1, col2 = st.columns(2)
    with col1:
        p1_display = st.selectbox("Person 1", display_names, index=0, key="compare_user1")
    with col2:
        p2_display = st.selectbox("Person 2", display_names, index=1, key="compare_user2")

    p1 = p1_display.lower()
    p2 = p2_display.lower()

    if p1 == p2:
        st.warning("Please select two different people.")
        return

    blend = st.slider(
        "Taste Blend", 0.0, 1.0, 0.5, 0.05,
        help=f"0 = {p1_display}'s taste only · 1 = {p2_display}'s taste only"
    )
    st.caption(f"← {p1_display} " + "─" * 18 + f" {p2_display} →")

    r1 = storage.load_user(p1)
    r2 = storage.load_user(p2)
    recs = joint_recommendations(
        r1["liked"], r2["liked"],
        r1["disliked"], r2["disliked"],
        blend=blend,
        n=10,
    )

    if not recs:
        st.warning("Not enough data. Go back and rate more movies!")
    else:
        st.markdown(" ")
        cols = st.columns(3)
        for i, idx in enumerate(recs):
            row = df.iloc[idx].to_dict()
            with cols[i % 3]:
                render_mini_card(row, idx)
                st.markdown(" ")

    st.markdown(" ")
    if st.button("← Back to Rating", use_container_width=True):
        st.session_state.phase = "rating"
        st.rerun()
    if st.button("🔄 Reset All Data", use_container_width=True):
        reset_all()
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

inject_css()
init_state()

if st.session_state.phase == "login":
    login_phase()
elif st.session_state.phase == "rating":
    rating_phase()
else:
    results_phase()
