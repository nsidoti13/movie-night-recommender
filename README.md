# 🎬 Movie Night Recommender

A Streamlit web app for Regan and Nicholas to find movies they'll both enjoy. Each person rates movies independently, and the app blends their taste profiles to generate joint recommendations.

---

## How It Works

1. **Login** — Choose who you are (Regan or Nicholas)
2. **Rate movies** — Like, Dislike, or mark Haven't Seen on movie cards
3. **Search** — Find specific movies or use AI search for descriptions like *"marvel movies"* or *"Nolan films"*
4. **Hand off** — Once you've liked 5+ movies and click Done, the other person takes their turn
5. **Results** — After both are done, get 10 joint recommendations with a blend slider to weight each person's taste

Ratings are saved automatically and persist across sessions.

---

## Features

- 🎴 Adaptive card selection — each card is chosen based on your taste so far
- 👍 👎 🤷 Like / Dislike / Haven't Seen buttons
- 🔍 Search by title with instant results
- ✨ AI Search — powered by Claude, handles:
  - Franchise queries (*"marvel movies"*, *"star wars"*)
  - Director/actor queries (*"Nolan films"*, *"Tom Hanks movies"*)
  - Thematic queries (*"movies about space"*, *"80s horror"*)
  - Typos and paraphrasing (*"avenjers"* → The Avengers)
- 📋 Review and re-rate any previously rated movie from the sidebar
- 🎨 Movie posters loaded from TMDB
- 🎚️ Blend slider on results to shift recommendations toward either person's taste
- 💾 Ratings saved to disk — safe to close and come back later

---

## Project Structure

```
movie-recommender/
├── app.py                  # Main Streamlit app
├── recommender.py          # Recommendation engine (TF-IDF + cosine similarity)
├── llm_search.py           # AI-powered search via Claude API
├── storage.py              # Ratings persistence (in-memory + JSON file)
├── build_features.py       # One-time data prep script
├── start.py                # Local launcher with Cloudflare public URL
├── requirements.txt
└── data/
    ├── movies.parquet      # Cleaned TMDB metadata + poster URLs
    ├── tfidf_matrix.npz    # Precomputed TF-IDF feature matrix
    └── movie_index.json    # Lowercase title → row index lookup
```

---

## Running Locally

### First time setup

```bash
cd movie-recommender
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python build_features.py   # downloads dataset + builds data/ folder
```

### Start the app (with shareable public URL)

```bash
venv/bin/python start.py
```

This launches Streamlit and opens a Cloudflare tunnel. The terminal will print a public URL you can send to Regan — no account or API key required.

### Start locally only

```bash
venv/bin/streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## Streamlit Cloud Deployment

The app is deployed at a permanent URL via [Streamlit Community Cloud](https://share.streamlit.io).

To redeploy after changes:

```bash
git add .
git commit -m "your message"
git push
```

Streamlit Cloud auto-deploys on every push to `main`.

### Required Secret

Add your Anthropic API key in **Streamlit Cloud → Settings → Secrets**:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

Without this, the ✨ AI Search button will not appear. All other features work without it.

---

## Data Sources

| Dataset | Source | Used for |
|---|---|---|
| TMDB 5000 Movies | `AiresPucrs/tmdb-5000-movies` (Hugging Face) | Movie metadata, genres, cast, keywords |
| Movie Posters | `sakshisemalti/movies-dataset-with-posters` (Kaggle) | Poster image URLs (TMDB CDN) |

~4,800 movies total. ~2,850 have poster images. All data is bundled — no runtime API calls needed for the core app.

---

## How Recommendations Work

Each movie is represented as a TF-IDF vector built from its genres, keywords, top 3 cast members, and director. When you like a movie, your taste profile is updated to the average of all your liked movie vectors. The next card shown is always the most similar unseen movie to your current profile.

For joint recommendations, both profiles are blended (50/50 by default, adjustable with the slider) and the top 10 most similar movies that neither person has rated are returned.
