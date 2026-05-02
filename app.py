import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process
from model import MovieRecommender

# ─── PAGE CONFIG ───
st.set_page_config(
    page_title="CineMatch AI | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CUSTOM CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
    }
    
    .movie-card {
        background: linear-gradient(145deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .movie-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.2);
        border-color: #667eea;
    }
    
    .movie-rank {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    
    .similarity-score {
        background: rgba(102, 126, 234, 0.2);
        color: #a5b4fc;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    .search-container {
        background: rgba(30, 30, 46, 0.8);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 2rem;
    }
    
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.05) !important;
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 12px !important;
        color: white !important;
        font-size: 1.1rem !important;
        padding: 1rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
    }
    
    .recommendation-section {
        animation: fadeIn 0.6s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: rgba(255,255,255,0.5);
        margin-top: 3rem;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODEL (cached) ───
@st.cache_resource
def load_model():
    model = MovieRecommender()
    model.load()
    return model

# ─── FUZZY SEARCH ENGINE ───
def fuzzy_search(query, movies_df, limit=10):
    """Smart search: exact matches first, then fuzzy"""
    if not query or len(query) < 2:
        return []
    
    query_lower = query.lower()
    
    # Priority 1: Exact match
    exact = movies_df[movies_df['title'].str.lower() == query_lower]
    
    # Priority 2: Starts with query
    starts_with = movies_df[
        movies_df['title'].str.lower().str.startswith(query_lower)
    ]
    
    # Priority 3: Contains query
    contains = movies_df[
        movies_df['title'].str.contains(query, case=False, na=False)
    ]
    
    # Priority 4: Fuzzy fallback
    choices = movies_df['search_string'].tolist()
    fuzzy_results = process.extract(query, choices, scorer=fuzz.WRatio, limit=limit)
    
    fuzzy_matches = []
    for match_str, score, idx in fuzzy_results:
        if score > 60:
            row = movies_df.iloc[idx]
            fuzzy_matches.append({
                'movie_id': row['movieId'],
                'title': row['title'],
                'genres': row['genres'],
                'score': score
            })
    
    # Combine all, remove duplicates, prioritize exact matches
    all_results = []
    seen = set()
    
    for df_source in [exact, starts_with, contains]:
        for _, row in df_source.iterrows():
            if row['movieId'] not in seen:
                all_results.append({
                    'movie_id': row['movieId'],
                    'title': row['title'],
                    'genres': row['genres'],
                    'score': 100
                })
                seen.add(row['movieId'])
    
    # Add fuzzy results if not enough
    for match in fuzzy_matches:
        if match['movie_id'] not in seen and len(all_results) < limit:
            all_results.append(match)
            seen.add(match['movie_id'])
    
    return all_results[:limit]

# ─── UI RENDERING ───
def render_movie_card(movie, rank):
    genres = movie.get('genres', 'N/A').split('|')[:3]
    genre_tags = "".join([
        f'<span style="background:rgba(255,255,255,0.1);padding:2px 8px;border-radius:10px;font-size:0.75rem;margin-right:4px;">{g}</span>' 
        for g in genres
    ])
    
    st.markdown(f"""
    <div class="movie-card">
        <div class="movie-rank">{rank}</div>
        <h3 style="margin:0 0 0.5rem 0;color:#fff;font-size:1.1rem;">{movie['title']}</h3>
        <p style="margin:0;color:rgba(255,255,255,0.6);font-size:0.9rem;">{genre_tags}</p>
        <div class="similarity-score">⭐ {movie['predicted_rating']:.2f}/5.0</div>
    </div>
    """, unsafe_allow_html=True)

# ─── MAIN APP ───
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0;font-size:2.5rem;">🎬 CineMatch AI</h1>
        <p style="margin:0.5rem 0 0 0;font-size:1.2rem;opacity:0.9;">
            Discover your next favorite movie. Type any title, we'll find the magic.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    try:
        model = load_model()
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        st.info("Make sure you ran the artifact saving cell in your notebook first!")
        return
    
    # Search Section
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    with col1:
        search_query = st.text_input(
            "",
            placeholder="🔍 Type a movie name (e.g., 'Inception', 'Godfather', 'Matrix')...",
            key="movie_search",
            label_visibility="collapsed"
        )
    with col2:
        search_button = st.button("🚀 Find Movies", use_container_width=True, type="primary")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Live suggestions (appears as user types)
    if search_query and len(search_query) >= 2:
        suggestions = fuzzy_search(search_query, model.movies_df, limit=5)
        
        if suggestions:
            st.markdown("### 💡 Did you mean:")
            cols = st.columns(len(suggestions))
            selected_movie = None
            
            for idx, (col, movie) in enumerate(zip(cols, suggestions)):
                with col:
                    if st.button(
                        f"🎭 {movie['title'][:30]}...",
                        key=f"suggest_{idx}",
                        use_container_width=True
                    ):
                        selected_movie = movie
            
            if selected_movie:
                search_query = selected_movie['title']
                search_button = True
    
    # Process recommendation
    if search_button and search_query:
        with st.spinner("🤖 AI is analyzing movie patterns..."):
            # Get recommendations using YOUR model
            recommendations = model.get_similar_movies(search_query, n_recommendations=10)
            
            if recommendations is None:
                st.error("🎬 Movie not found or not in training data. Try another title!")
                return
            
            # Results Section
            st.markdown(f"""
            <div style="text-align:center;margin:2rem 0;">
                <h2 style="color:#fff;">Because you liked <span style="color:#667eea;">{search_query}</span></h2>
                <p style="color:rgba(255,255,255,0.6);">Top 10 movies picked by our Hybrid SVD AI model</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display recommendations in grid
            st.markdown('<div class="recommendation-section">', unsafe_allow_html=True)
            
            for i in range(0, len(recommendations), 5):
                cols = st.columns(5)
                for j, (col, rec) in enumerate(zip(cols, recommendations[i:i+5])):
                    with col:
                        render_movie_card(rec, rank=i+j+1)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Stats
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Movies in Database", f"{len(model.movies_df):,}")
            with col2:
                st.metric("Model RMSE", "0.8794")
            with col3:
                st.metric("Latent Factors", "50")
    
    # Empty state
    elif not search_query:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:rgba(255,255,255,0.5);">
            <h3>🎭 Start typing to discover movies</h3>
            <p>Our Hybrid SVD model has analyzed 10M+ ratings to find your perfect match</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show trending movies
        st.markdown("### 🔥 Trending Now")
        trending = model.movies_df.sample(5)
        cols = st.columns(5)
        for col, (_, movie) in zip(cols, trending.iterrows()):
            with col:
                st.markdown(f"""
                <div class="movie-card" style="text-align:center;">
                    <div style="font-size:2rem;">🎬</div>
                    <h4 style="margin:0.5rem 0;color:#fff;">{movie['title'][:25]}...</h4>
                    <p style="margin:0;color:rgba(255,255,255,0.6);">{movie['genres'].split('|')[0]}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🚀 Deployed on Streamlit Cloud • Built with ❤️ for Kaggle Portfolio</p>
        <p style="font-size:0.8rem;">Hybrid Model: Bias + SVD on Residuals | RMSE: 0.8794</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()