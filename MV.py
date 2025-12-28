# 🎬 Movie Recommender — Full Streamlit Project (Final Fixed + Improved Night Mode)
import streamlit as st
import numpy as np
import pandas as pd
import os
import json
import pickle
from datetime import datetime
import requests

# -----------------------------
# File Paths
# -----------------------------
MOVIES_PICKLE = r'C:\Users\HP\OneDrive\Desktop\Project-ML\project-1\pklfi\movies_dict.pkl'
SIMILARITY_PICKLE = r'C:\Users\HP\OneDrive\Desktop\Project-ML\project-1\pklfi\similarity.pkl'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEWS_FILE = os.path.join(BASE_DIR, "reviews.json")
WATCHLIST_FILE = os.path.join(BASE_DIR, "watchlist.json")
FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback.json")

# -----------------------------
# TMDB API
# -----------------------------
TMDB_API_KEY = "3eaecefc1663c0bef46343c7e031bc00"

def fetch_poster(movie_id=None, movie_title=None):
    if not TMDB_API_KEY:
        return "https://via.placeholder.com/500x750?text=No+API+Key"
    try:
        if movie_id:
            url = f"https://api.themoviedb.org/3/movie/{int(movie_id)}?api_key={TMDB_API_KEY}&language=en-US"
            r = requests.get(url, timeout=6)
            if r.status_code == 200:
                d = r.json()
                if d.get("poster_path"):
                    return f"https://image.tmdb.org/t/p/w500{d['poster_path']}"
        if movie_title:
            s_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
            r = requests.get(s_url, timeout=6)
            if r.status_code == 200:
                res = r.json().get("results")
                if res and res[0].get("poster_path"):
                    return f"https://image.tmdb.org/t/p/w500{res[0]['poster_path']}"
    except Exception:
        pass
    return "https://via.placeholder.com/500x750?text=No+Image"

# -----------------------------
# JSON Helpers
# -----------------------------
def _convert_json(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (pd.Timestamp, datetime)): return obj.strftime("%Y-%m-%d %H:%M:%S")
    return str(obj)

def ensure_json(path):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f: f.write("[]")

def load_json(path):
    ensure_json(path)
    try:
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return []

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_convert_json)

# -----------------------------
# Load Movie Data
# -----------------------------
def load_pickles():
    try:
        movies_dict = pickle.load(open(MOVIES_PICKLE, "rb"))
        movies = pd.DataFrame(movies_dict)
        similarity = pickle.load(open(SIMILARITY_PICKLE, "rb"))
        return movies, similarity, None
    except Exception as e:
        return None, None, f"❌ Failed to load files: {e}"

movies, similarity, load_error = load_pickles()

# -----------------------------
# Recommendation Logic
# -----------------------------
def recommend(movie_title, top_n=5):
    if movies is None or similarity is None: return [], []
    if movie_title not in movies["title"].values: return [], []
    idx = movies[movies["title"] == movie_title].index[0]
    sims = sorted(list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True)
    names, posters = [], []
    for i, _ in sims[1: top_n + 1]:
        row = movies.iloc[i]
        movie_id = row.get("movie_id") or row.get("id") or row.get("tmdbId")
        title = row.get("title", "Untitled")
        names.append(title)
        posters.append(fetch_poster(movie_id, title))
    return names, posters

# -----------------------------
# Watchlist, Reviews, Feedback
# -----------------------------
def add_to_watchlist(title, movie_id):
    w = load_json(WATCHLIST_FILE)
    if any(item["title"] == title for item in w): return False
    w.append({"title": title, "movie_id": int(movie_id), "added_on": datetime.utcnow()})
    save_json(WATCHLIST_FILE, w); return True

def remove_from_watchlist(title):
    w = [item for item in load_json(WATCHLIST_FILE) if item["title"] != title]
    save_json(WATCHLIST_FILE, w)

def add_review(movie, rating, review, user="Anonymous"):
    r = load_json(REVIEWS_FILE)
    r.append({"movie": movie, "rating": int(rating), "review": review, "user": user, "timestamp": datetime.utcnow()})
    save_json(REVIEWS_FILE, r)

def get_reviews(movie): return [r for r in load_json(REVIEWS_FILE) if r["movie"] == movie]

def add_feedback(name, email, message):
    fb = load_json(FEEDBACK_FILE)
    fb.append({"name": name, "email": email, "message": message, "timestamp": datetime.utcnow()})
    save_json(FEEDBACK_FILE, fb)

# -----------------------------
# Streamlit UI + Theme
# -----------------------------
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")  # <-- You can rename here
mode = st.sidebar.radio("Choose Theme", ["🌞 Day Mode", "🌙 Night Mode"])

if mode == "🌙 Night Mode":
    bg, txt, box, input_bg = "#0f1724", "#e6eef8", "#1f2a37", "#2d3a4f"
else:
    bg, txt, box, input_bg = "#ffffff", "#111827", "#f1f5f9", "#ffffff"

st.markdown(f"""
<style>
.stApp {{ background-color: {bg}; color: {txt}; }}
.stButton>button {{
    background-color: {box}; color: {txt};
    border-radius: 10px; padding: 0.5em 1em;
}}
input, textarea, select {{
    background-color: {input_bg} !important;
    color: {txt} !important;
    border: 1px solid #555 !important;
    border-radius: 8px !important;
}}
div[data-baseweb="input"] > div > input {{
    color: {txt} !important;
    background-color: {input_bg} !important;
}}
div[data-baseweb="textarea"] > textarea {{
    color: {txt} !important;
    background-color: {input_bg} !important;
}}
div[data-baseweb="select"] > div {{
    color: {txt} !important;
    background-color: {input_bg} !important;
}}
</style>
""", unsafe_allow_html=True)

st.title("🎬 Movie Recommender")  # <-- You can change your project name here

if load_error:
    st.error(load_error)
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["🎥 Recommendations", "⭐ Watchlist", "💬 Reviews", "📝 Feedback"])

# 🎥 Recommendations
with tab1:
    st.subheader("Search & get movie recommendations")
    movie_list = movies["title"].values
    selected_movie = st.selectbox("Select a movie:", movie_list)

    if st.button("Show Recommendations"):
        names, posters = recommend(selected_movie)
        if not names:
            st.warning("No recommendations found.")
        else:
            st.markdown("### 🎬 Recommended Movies")
            cols = st.columns(5)
            for i in range(len(names)):
                with cols[i % 5]:
                    st.image(posters[i], width="stretch")
                    st.caption(names[i])

    row = movies[movies["title"] == selected_movie].iloc[0]
    movie_id = int(row["movie_id"])
    watchlist = load_json(WATCHLIST_FILE)
    is_in_watchlist = any(w["title"] == selected_movie for w in watchlist)

    if is_in_watchlist:
        if st.button("💔 Remove from Watchlist"):
            remove_from_watchlist(selected_movie)
            st.rerun()
    else:
        if st.button("🤍 Add to Watchlist"):
            add_to_watchlist(selected_movie, movie_id)
            st.success(f"Added '{selected_movie}' to watchlist")
            st.rerun()

# ⭐ Watchlist
with tab2:
    st.subheader("Your Watchlist")
    w = load_json(WATCHLIST_FILE)
    if not w:
        st.info("Your watchlist is empty.")
    else:
        cols = st.columns(3)
        for i, item in enumerate(w):
            with cols[i % 3]:
                st.image(fetch_poster(item["movie_id"], item["title"]), width="stretch")
                st.caption(item["title"])
                if st.button(f"Remove {item['title']}", key=f"rm_{i}"):
                    remove_from_watchlist(item["title"])
                    st.rerun()

# 💬 Reviews
with tab3:
    st.subheader("Write a review")
    movie_sel = st.selectbox("Select movie to review:", movies["title"].values)
    rating = st.slider("Rating (1–5)", 1, 5, 3)
    text = st.text_area("Your review")
    user = st.text_input("Your name", "Anonymous")
    if st.button("Submit Review"):
        if text.strip():
            add_review(movie_sel, rating, text.strip(), user)
            st.success("Review submitted!")
        else:
            st.warning("Write something before submitting!")

    st.markdown("### Reviews")
    revs = get_reviews(movie_sel)
    if revs:
        for r in sorted(revs, key=lambda x: x["timestamp"], reverse=True):
            st.markdown(f"**{r['user']}** rated ⭐ {r['rating']}/5")
            st.write(r["review"])
            st.caption(r["timestamp"])
            st.markdown("---")
    else:
        st.info("No reviews yet.")

# 📝 Feedback
with tab4:
    st.subheader("Your Feedback")
    name = st.text_input("Name")
    email = st.text_input("Email")
    msg = st.text_area("Message")

    if st.button("Send Feedback"):
        if not name.strip() or not email.strip() or not msg.strip():
            st.warning("Please fill all fields.")
        else:
            add_feedback(name.strip(), email.strip(), msg.strip())
            st.success("Thanks for your feedback ❤️")

    # Show all feedback below form
    st.markdown("### 💌 All Feedback")
    feedbacks = load_json(FEEDBACK_FILE)

    if feedbacks:
        for fb in sorted(feedbacks, key=lambda x: x["timestamp"], reverse=True):
            st.markdown(f"**{fb['name']}** ({fb['email']})")
            st.write(fb["message"])
            st.caption(fb["timestamp"])
            st.markdown("---")
    else:
        st.info("No feedback yet.")
