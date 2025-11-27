# ui.py – Advanced MovieFinder with Improved Recommendations
# Run: streamlit run ui.py
# Requirements: pip install streamlit pandas numpy scikit-learn scipy requests

import os
import json
import re
from pathlib import Path
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings("ignore")

import requests
import streamlit as st
import pandas as pd
import numpy as np
import pickle

from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================================
# SUPPRESS ALL WARNINGS
# =========================================
import logging
os.environ["STREAMLIT_LOGGER_LEVEL"] = "off"
logging.getLogger("streamlit").setLevel(logging.ERROR)

# =========================================
# CONFIG & SETUP
# =========================================
ROOT = Path(".")
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TMDB_API_KEY = "23c841678089d08d5d5bbfccb994a055"
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

POSTER_CACHE_PATH = MODEL_DIR / "poster_cache.json"
PLACEHOLDER_URL = "https://via.placeholder.com/300x450?text=No+Poster"

UPLIFT_FACTOR = 1.10
CARDS_PER_ROW = 4
REQUEST_TIMEOUT = 8

st.set_page_config(
    page_title="MovieFinder – AI Recommendations",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# ENHANCED STYLING
# =========================================
st.markdown("""
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%) !important;
}

.stApp { background: transparent !important; }
[data-testid="stMainBlockContainer"] { padding: 2rem 1rem; }
body { color: #e6eef6; }

.section-title {
    font-size: 24px;
    font-weight: 800;
    margin-top: 32px;
    margin-bottom: 20px;
    background: linear-gradient(90deg, #ff3d3d, #ff8a00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.card {
    background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    border-radius: 16px;
    padding: 14px;
    margin: 10px;
    box-shadow: 0 12px 32px rgba(0,0,0,0.5);
    border: 1px solid rgba(255,255,255,0.08);
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
    height: 100%;
    display: flex;
    flex-direction: column;
}

.card:hover {
    transform: translateY(-8px);
    box-shadow: 0 16px 48px rgba(255, 61, 61, 0.3);
    border-color: rgba(255,255,255,0.15);
}

.movie-info {
    margin-top: 12px;
    flex-grow: 1;
    display: flex;
    flex-direction: column;
}

.movie-title {
    font-weight: 800;
    font-size: 16px;
    color: #fff;
    margin-bottom: 6px;
    line-height: 1.3;
    flex-grow: 1;
}

.meta {
    color: #b8c6d9;
    font-size: 12px;
    margin-bottom: 8px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.meta-tag {
    background: rgba(255, 61, 61, 0.15);
    padding: 3px 8px;
    border-radius: 8px;
    font-size: 11px;
    color: #ff8a00;
}

.score-pill {
    display: inline-block;
    background: linear-gradient(90deg, #ff3d3d, #ff8a00);
    color: white;
    padding: 8px 14px;
    border-radius: 20px;
    font-weight: 800;
    font-size: 14px;
    box-shadow: 0 4px 12px rgba(255, 61, 61, 0.3);
    flex-shrink: 0;
}

.rating-stars { color: #ffd700; font-size: 12px; margin-right: 6px; }
.small { font-size: 11px; color: #9fb0d2; }

.header-card {
    background: linear-gradient(135deg, rgba(255,61,61,0.1), rgba(255,138,0,0.05));
    border: 1px solid rgba(255, 61, 61, 0.3);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
    margin-top: 24px;
    backdrop-filter: blur(10px);
}

.info-box {
    background: rgba(100, 200, 255, 0.1);
    border-left: 4px solid #64c8ff;
    padding: 12px;
    border-radius: 8px;
    margin: 12px 0;
    color: #b8c6d9;
    font-size: 14px;
}

.warning-box {
    background: rgba(255, 150, 100, 0.1);
    border-left: 4px solid #ff9664;
    padding: 12px;
    border-radius: 8px;
    margin: 12px 0;
    color: #ffb88c;
    font-size: 14px;
}

.stWarning { display: none !important; }
[data-testid="stNotificationContent"] { display: none !important; }
.st-warning { display: none !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10,14,39,0.95), rgba(26,31,58,0.95)) !important;
}

.stTextInput > div > div > input {
    background-color: rgba(255,255,255,0.05) !important;
    color: #e6eef6 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}

label { color: #e6eef6 !important; }
</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD MODELS
# =========================================

@st.cache_data(show_spinner=False)
def load_all_models():
    """Load all recommendation models."""
    try:
        # Load movies
        with open(MODEL_DIR / "movies_df.pkl", "rb") as f:
            movies_df = pickle.load(f)

        # Load TF-IDF vectorizer and matrix
        with open(MODEL_DIR / "tfidf_vectorizer.pkl", "rb") as f:
            tfv = pickle.load(f)
        tfidf_matrix = load_npz(MODEL_DIR / "tfidf_matrix.npz")

        # Load mappings
        with open(MODEL_DIR / "mappings.pkl", "rb") as f:
            mappings = pickle.load(f)

        # Load popularity data
        with open(MODEL_DIR / "pop_sorted.pkl", "rb") as f:
            pop_sorted = pickle.load(f)

        # Load item-user matrix
        item_user_matrix = load_npz(MODEL_DIR / "item_user_matrix.npz")

        return {
            "movies": movies_df,
            "tfv": tfv,
            "tfidf_matrix": tfidf_matrix,
            "mappings": mappings,
            "pop_sorted": pop_sorted,
            "item_user_matrix": item_user_matrix
        }
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None

models = load_all_models()

if models is None:
    st.error("❌ Models not found. Please run the training script first.")
    st.stop()

movies = models["movies"]
tfv = models["tfv"]
tfidf_matrix = models["tfidf_matrix"]
mappings = models["mappings"]
pop_sorted = models["pop_sorted"]
item_user_matrix = models["item_user_matrix"]

movieid_to_idx = mappings.get("movieid_to_idx", {})
idx_to_movieid = mappings.get("idx_to_movieid", {})

# =========================================
# POSTER CACHING
# =========================================

def load_poster_cache():
    try:
        if POSTER_CACHE_PATH.exists():
            with open(POSTER_CACHE_PATH, "r", encoding="utf8") as f:
                return json.load(f)
    except:
        pass
    return {}

def save_poster_cache(cache):
    try:
        with open(POSTER_CACHE_PATH, "w", encoding="utf8") as f:
            json.dump(cache, f, indent=2)
    except:
        pass

poster_cache = load_poster_cache()

def fuzzy_match(a, b):
    return SequenceMatcher(None, a, b).ratio()

# =========================================
# TMDB API
# =========================================

def tmdb_search(title, year=None):
    """Search TMDB for poster."""
    if not TMDB_API_KEY:
        return None

    try:
        clean_title = title.split('(')[0].strip() if '(' in title else title.strip()

        params = {
            "api_key": TMDB_API_KEY,
            "query": clean_title,
            "include_adult": False,
            "page": 1
        }
        if year:
            params["year"] = int(year)

        response = requests.get(TMDB_SEARCH_URL, params=params, timeout=REQUEST_TIMEOUT)

        if response.status_code != 200:
            return None

        data = response.json()
        results = data.get("results", [])

        if not results:
            return None

        title_lower = clean_title.lower().strip()

        # Exact match first
        for result in results:
            result_title = (result.get("title", "") or "").lower().strip()
            if result_title == title_lower or fuzzy_match(title_lower, result_title) > 0.85:
                poster_path = result.get("poster_path")
                if poster_path:
                    return TMDB_IMAGE_BASE + poster_path

        # Fuzzy match
        best = max(results, key=lambda x: fuzzy_match(title_lower, (x.get("title", "") or "").lower()))
        poster_path = best.get("poster_path")
        if poster_path:
            return TMDB_IMAGE_BASE + poster_path

    except:
        pass

    return None

def get_poster(title):
    """Get poster with caching."""
    key = title.lower().strip()

    if key in poster_cache and poster_cache[key]:
        return poster_cache[key]

    clean_title = title.split('(')[0].strip() if '(' in title else title.strip()
    year_match = re.search(r"\((\d{4})\)", title)
    year = int(year_match.group(1)) if year_match else None

    url = tmdb_search(clean_title, year)

    poster_cache[key] = url
    save_poster_cache(poster_cache)

    return url

# =========================================
# RECOMMENDATION FUNCTIONS
# =========================================

def find_movie(title_query):
    """Find movie by title."""
    query = title_query.lower().strip()

    exact = movies[movies["title"].str.lower() == query]
    if not exact.empty:
        return exact.index[0]

    partial = movies[movies["title"].str.lower().str.contains(query, regex=False, na=False)]
    if not partial.empty:
        return partial.index[0]

    return None

def recommend_content_based(movie_idx, topk=8):
    """Content-based recommendations."""
    if movie_idx is None or movie_idx >= len(movies):
        return []

    try:
        query_genre = movies.iloc[movie_idx].get("primary_genre", "Unknown")
        target_vec = tfidf_matrix[movie_idx]
        similarities = cosine_similarity(target_vec, tfidf_matrix)[0]

        # Get top candidates
        top_indices = np.argsort(similarities)[::-1][1:topk*3+1]

        # Genre bonus
        genre_bonus = np.array([
            1.2 if movies.iloc[i].get("primary_genre") == query_genre else 1.0
            for i in top_indices
        ])

        adjusted_scores = similarities[top_indices] * genre_bonus
        final_indices = top_indices[np.argsort(-adjusted_scores)][:topk]

        return [(int(i), float(similarities[i])) for i in final_indices]
    except:
        return []

def recommend_cf(movie_idx, topk=8):
    """Collaborative filtering recommendations."""
    try:
        if movie_idx is None or movie_idx >= len(movies):
            return []

        movie_id = int(movies.iloc[movie_idx]["movieId"])

        if movie_id not in movieid_to_idx:
            return []

        row_idx = movieid_to_idx[movie_id]
        target_vec = item_user_matrix[row_idx]
        similarities = cosine_similarity(target_vec, item_user_matrix)[0]

        # Count common high-rated items
        target_users = (target_vec > 2.5).astype(int)
        all_users = (item_user_matrix > 2.5).astype(int)
        common_ratings = target_users.dot(all_users.T).toarray()[0]

        # Filter
        valid_mask = (common_ratings >= 5) & (similarities > 0.05)
        valid_indices = np.where(valid_mask)[0]
        valid_indices = valid_indices[valid_indices != row_idx]

        if len(valid_indices) == 0:
            return []

        valid_indices = valid_indices[np.argsort(-similarities[valid_indices])][:topk]

        return [(int(i), float(similarities[i])) for i in valid_indices]
    except:
        return []

def get_movie_metadata(movie_row):
    """Extract movie metadata."""
    title = str(movie_row.get("title", "Unknown"))
    genres = str(movie_row.get("genres", "Unknown"))
    movie_id = int(movie_row.get("movieId", 0))

    rating = None
    count = None

    if pop_sorted is not None:
        rating_row = pop_sorted[pop_sorted["movieId"] == movie_id]
        if not rating_row.empty:
            rating = float(rating_row["mean_rating"].iloc[0])
            count = int(rating_row["count"].iloc[0])

    return title, genres, rating, count

def format_rating(rating):
    """Format rating as stars."""
    if rating is None or (isinstance(rating, float) and np.isnan(rating)):
        return "–"

    rating = float(rating)
    full = int(rating)
    half = (rating - full) >= 0.5
    stars_str = "★" * full + ("½" if half else "")
    return stars_str.ljust(5, "☆") + f" ({rating:.2f}/5)"

def similarity_to_percent(sim):
    """Convert similarity to percentage."""
    x = min(max(float(sim) * UPLIFT_FACTOR, 0.0), 1.0)
    return x * 100

# =========================================
# MOVIE CARD RENDERER
# =========================================

def render_movie_card(col, poster_url, title, genres, rating, count, similarity):
    """Render movie card."""
    with col:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        try:
            if poster_url:
                st.image(poster_url, width=500)
            else:
                st.image(PLACEHOLDER_URL, width=500)
        except:
            try:
                st.image(PLACEHOLDER_URL, width=500)
            except:
                pass

        title_safe = title.replace('"', '&quot;').replace("'", "&#39;")
        genres_safe = genres[:40].replace('"', '&quot;').replace("'", "&#39;")

        st.markdown(
            f"""
            <div class="movie-info">
                <div style="display: flex; justify-content: space-between; align-items: start; gap: 8px;">
                    <div class="movie-title" style="flex: 1;">{title_safe}</div>
                    <div class="score-pill">{similarity:.0f}%</div>
                </div>
                <div class="meta">
                    <span class="meta-tag">{genres_safe}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if rating:
            st.markdown(
                f'<div class="small"><span class="rating-stars">{format_rating(rating)}</span> • {count:,} ratings</div>',
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

def search_movies(query):
    """Search for movies."""
    if not query or not query.strip():
        return pd.DataFrame()

    query_lower = query.lower()
    return movies[movies["title"].str.lower().str.contains(query_lower, regex=False, na=False)]

# =========================================
# MAIN APP
# =========================================

with st.sidebar:
    st.markdown("# 🎬 MovieFinder")
    st.markdown("---")

    query = st.text_input("🔍 Search movie:", value="Toy Story", placeholder="Enter movie title...")
    k = st.slider("📊 Recommendations per type", 4, 16, 8)

    st.markdown("---")
    st.markdown("### Display Options")
    show_content = st.checkbox("📖 Content-based", value=True)
    show_cf = st.checkbox("🤝 Collaborative Filtering", value=True)
    show_popular = st.checkbox("⭐ Popular (Same Genre)", value=True)
    use_posters = st.checkbox("🖼️ Load Posters (TMDB)", value=True)

    st.markdown("---")
    st.caption(f"📁 Cached posters: {len(poster_cache)}")

st.markdown("# 🎥 MovieFinder – Advanced Recommendations")

search_results = search_movies(query)

if search_results.empty:
    st.markdown('<div class="warning-box">❌ No movies found. Try a different search.</div>', unsafe_allow_html=True)
    st.stop()

if len(search_results) > 1:
    selected_title = st.selectbox("Multiple matches found. Select one:", search_results["title"].tolist())
    selected_movie = search_results[search_results["title"] == selected_title].iloc[0]
else:
    selected_movie = search_results.iloc[0]

movie_idx = selected_movie.name
movie_id = int(selected_movie.get("movieId", 0))
movie_title, movie_genres, movie_rating, movie_count = get_movie_metadata(selected_movie)
query_genre = selected_movie.get("primary_genre", "Unknown")

# Header
st.markdown('<div class="header-card">', unsafe_allow_html=True)

col1, col2 = st.columns([2, 3])

with col1:
    if use_posters:
        poster = get_poster(movie_title)
        try:
            if poster:
                st.image(poster, width=500)
            else:
                st.image(PLACEHOLDER_URL, width=500)
        except:
            try:
                st.image(PLACEHOLDER_URL, width=500)
            except:
                st.write("📷 Poster unavailable")

with col2:
    st.markdown(f"### {movie_title}")
    st.markdown(f"**Genres:** {movie_genres}")
    st.markdown(f"**Primary Genre:** {query_genre}")

    if movie_rating:
        st.markdown(
            f"<p style='color: #ffd700; font-size: 14px;'>{format_rating(movie_rating)} • {movie_count:,} ratings</p>",
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)

# Recommendations
def render_section(title, recs):
    if not recs:
        st.markdown(f'<div class="info-box">ℹ️ No {title.lower()} available.</div>', unsafe_allow_html=True)
        return

    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

    chunks = [recs[i:i+CARDS_PER_ROW] for i in range(0, len(recs), CARDS_PER_ROW)]

    for chunk in chunks:
        cols = st.columns(len(chunk))

        for col, (idx, sim) in zip(cols, chunk):
            try:
                if idx in movies.index:
                    m_row = movies.iloc[idx]
                else:
                    mid = idx_to_movieid.get(idx)
                    if mid is None:
                        continue
                    m_row = movies[movies["movieId"] == mid]
                    if m_row.empty:
                        continue
                    m_row = m_row.iloc[0]

                t, g, r, c = get_movie_metadata(m_row)
                pct = similarity_to_percent(sim)
                poster_url = get_poster(t) if use_posters else None

                render_movie_card(col, poster_url, t, g, r, c, pct)
            except:
                pass

if show_content:
    recs = recommend_content_based(movie_idx, k)
    render_section("📖 Content-Based Recommendations", recs)

if show_cf:
    recs = recommend_cf(movie_idx, k)
    render_section("🤝 Collaborative Filtering Recommendations", recs)

if show_popular and pop_sorted is not None:
    try:
        genre_filtered = pop_sorted[pop_sorted["primary_genre"] == query_genre]
        if genre_filtered.empty:
            genre_filtered = pop_sorted

        top = genre_filtered.head(8)
        pop_recs = []

        for _, row in top.iterrows():
            try:
                m = movies[movies["movieId"] == row["movieId"]]
                if not m.empty:
                    idx = m.index[0]
                    pop_recs.append((idx, 1.0))
            except:
                pass

        render_section(f"⭐ Popular in {query_genre}", pop_recs)
    except:
        st.markdown('<div class="info-box">ℹ️ Popular data unavailable.</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption(f"✅ Dataset: {len(movies)} movies | 🎯 Cached: {len(poster_cache)} | 📊 {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")