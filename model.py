# model.py — FINAL VERSION (uses compressed .npz)
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

class MovieRecommender:
    
    def __init__(self, artifacts_path='model_artifacts/'):
        self.path = artifacts_path
        self.movies_df = None
        self.item_factors = None
        self.item_factors_norm = None
        self.movie_map = None
        
    def load(self):
        """Load compressed artifacts"""
        # Load numpy arrays (compressed .npz)
        arrays = np.load(f"{self.path}model_arrays.npz")
        self.item_factors = arrays['item_factors']
        
        # Load mappings
        with open(f"{self.path}mappings.pkl", 'rb') as f:
            mappings = pickle.load(f)
        self.movie_map = mappings['movie_map']
        
        # Normalize for cosine similarity
        self.item_factors_norm = normalize(self.item_factors, norm='l2', axis=1)
        
        # Load movies
        self.movies_df = pd.read_csv(f"{self.path}movies_cleaned.csv")
        self.movies_df['search_string'] = (
            self.movies_df['title'] + " | " + 
            self.movies_df['genres'].str.replace('|', ', ')
        )
        
        print("✅ Model loaded!")
        return self
    
    def get_similar_movies(self, movie_title, n_recommendations=10):
        movie_row = self.movies_df[self.movies_df['title'] == movie_title]
        if len(movie_row) == 0:
            return None
        
        selected_id = movie_row['movieId'].values[0]
        selected_genres = set()
        if pd.notna(movie_row['genres'].iloc[0]):
            selected_genres = set(movie_row['genres'].iloc[0].split('|'))
        
        if selected_id not in self.movie_map:
            return None
        
        idx = self.movie_map[selected_id]
        selected_vec = self.item_factors_norm[idx].reshape(1, -1)
        
        results = []
        for mid, midx in self.movie_map.items():
            if mid == selected_id:
                continue
            
            other_vec = self.item_factors_norm[midx].reshape(1, -1)
            latent_sim = cosine_similarity(selected_vec, other_vec)[0][0]
            
            other_row = self.movies_df[self.movies_df['movieId'] == mid]
            genre_sim = 0
            title = "Unknown"
            genres = "Unknown"
            rating_count = 0
            
            if len(other_row) > 0:
                title = other_row['title'].values[0]
                genres = other_row['genres'].values[0] if pd.notna(other_row['genres'].iloc[0]) else "Unknown"
                
                if pd.notna(other_row['genres'].iloc[0]):
                    other_genres = set(other_row['genres'].iloc[0].split('|'))
                    if selected_genres and other_genres:
                        intersection = len(selected_genres & other_genres)
                        union = len(selected_genres | other_genres)
                        genre_sim = intersection / union if union > 0 else 0
                
                if 'movie_rating_count' in other_row.columns:
                    rating_count = other_row['movie_rating_count'].iloc[0]
                else:
                    rating_count = 100
            
            popularity_score = min(np.log1p(rating_count) / np.log1p(30000), 1.0)
            
            if genre_sim >= 0.5:
                combined = 0.3 * latent_sim + 0.4 * genre_sim + 0.3 * popularity_score
            elif genre_sim > 0:
                combined = 0.2 * latent_sim + 0.3 * genre_sim + 0.5 * popularity_score
            else:
                combined = 0.05 * latent_sim
            
            display_rating = 3.0 + combined * 1.7
            
            results.append({
                'movie_id': mid,
                'title': title,
                'genres': genres,
                'predicted_rating': min(display_rating, 5.0),
                'similarity_score': combined * 100
            })
        
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:n_recommendations]
    
    def search_movies(self, query, limit=10):
        if not query or len(query) < 2:
            return []
        matches = self.movies_df[
            self.movies_df['title'].str.contains(query, case=False, na=False)
        ]
        return matches.head(limit).to_dict('records')