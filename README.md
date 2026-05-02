# 🎬 CineMatch AI — Movie Recommender System

&gt; **End-to-end machine learning project:** From 10M+ MovieLens ratings to a live Streamlit web app that suggests your next favorite movie.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://movie-recommender-system-afadamarcello.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn)](https://scikit-learn.org)

---

## 🚀 Live Demo

**[🎥 Try CineMatch AI Live](https://movie-recommender-system-afadamarcello.streamlit.app/)**

Type any movie name (even with typos!) and get 10 AI-powered recommendations instantly.

---

## 📊 Project Overview

| Metric | Value |
|--------|-------|
| **RMSE** | 0.8794 |
| **Baseline RMSE** | 1.0575 |
| **Improvement** | **17%** |
| **Dataset** | MovieLens 10M+ ratings |
| **Unique Movies** | 48,213 |
| **Unique Users** | 50,000+ |

---

## 🎯 The Problem

Movie recommendation systems face a classic challenge: **how do we predict what a user will enjoy when we know nothing about them?**

Traditional approaches either:
- ❌ Require user login/history (cold start problem)
- ❌ Recommend only by genre (too generic)
- ❌ Use black-box deep learning (not interpretable)

**My solution:** A hybrid model that finds movies *similar in spirit* to any film you love — no account needed.

---

## 🔬 Methodology

### 1️⃣ Exploratory Data Analysis (EDA)

Key findings from the dataset:
- Rating distribution: heavily skewed toward 3-4 stars
- Power law: Top 10% of movies get 50% of all ratings
- Genre co-occurrence: Action+Sci-Fi+Thriller cluster strongly
- Temporal bias: Older movies rated higher (nostalgia effect)

Visualizations created:
- Rating distribution histograms
- Genre correlation heatmaps
- User activity decay curves
- Movie popularity power-law plots

### 2️⃣ Feature Engineering

| Feature | Description | Purpose |
|---------|-------------|---------|
| `user_bias` | User's average deviation from global mean | Captures "generous" vs "critical" raters |
| `movie_bias` | Movie's average deviation from global mean | Captures "blockbuster" vs "niche" appeal |
| `latent_factors` | 50-dimensional SVD vectors | Learns hidden taste dimensions |
| `genre_vectors` | Binary multi-hot encoding | Content-based fallback |
| `popularity_score` | Log-scaled rating count | Surfaces known, trusted movies |

### 3️⃣ Model Architecture: Hybrid SVD


┌─────────────────────────────────────────┐
│           HYBRID SVD MODEL              │
├─────────────────────────────────────────┤
│                                         │
│  Step 1: Bias Baseline                  │
│    prediction = global_mean             │
│                + user_bias[u]            │
│                + movie_bias[i]          │
│                                         │
│  Step 2: SVD on Residuals               │
│    residual = actual_rating - baseline  │
│    SVD learns: residual ≈ p_u · q_i     │
│                                         │
│  Step 3: Combine                        │
│    final = baseline + p_u · q_i         │
│                                         │
│  RMSE: 0.8794 (17% better than          │
│         naive global mean baseline)     │
│                                         │
└─────────────────────────────────────────┘


**Why this works:**
- **Bias terms** capture "this user rates everything 0.5 stars higher"
- **Latent factors** capture "this movie has the same 'vibe' as that one"
- **Residual training** prevents the SVD from re-learning what biases already know

### 4️⃣ From Rating Prediction to Movie Similarity

The Kaggle competition required **rating prediction** (RMSE). But for the web app, I needed **movie-to-movie similarity**.

**The pivot:**
```python
# Instead of: "What will User X rate Movie Y?"
# We ask: "What movies have similar latent DNA to The Matrix?"

similarity = cosine_similarity(
    item_factors[matrix_id],      # The Matrix's "taste DNA"
    item_factors[all_other_ids]  # Every other movie's DNA
)

Combined with genre matching:
final_score = 0.5 * latent_similarity + 0.5 * genre_overlap + popularity_boost


📁 Project Structure

movie-recommender-system/
│
├── 📂 notebooks/
│   └── movie_recommender.ipynb      # Full EDA → Modeling → Evaluation
│
├── 📂 model_artifacts/               # Pre-trained model (deployed with app)
│   ├── model_arrays.npz             # Compressed SVD matrices
│   ├── mappings.pkl               # movieId ↔ index mapping
│   └── movies_cleaned.csv         # Movie metadata + genres
│
├── 📂 .streamlit/
│   └── config.toml                # Dark Netflix-style theme
│
├── app.py                          # Streamlit web application
├── model.py                        # Recommender engine class
├── requirements.txt                # Dependencies
├── .gitignore                      # Excludes data/ (too large)
└── README.md                       # This file


🎨 The Web App
How It Works
User types a movie → Fuzzy search handles typos ("Matrx" → "Matrix")
AI analyzes the movie's "DNA" → 50 latent factors + genre signature
Scores all 48,000 movies → Combined similarity + popularity ranking
Returns top 10 → Beautiful cards with predicted match score


| Feature                  | Implementation                   | User Benefit                            |
| ------------------------ | -------------------------------- | --------------------------------------- |
| 🔍 **Fuzzy Search**      | `rapidfuzz.WRatio`               | Typo tolerance, Google-like suggestions |
| 🧬 **Latent Similarity** | Cosine on normalized SVD vectors | Finds movies with same "vibe"           |
| 🎭 **Genre Matching**    | Jaccard similarity on genre sets | Ensures coherent recommendations        |
| 🔥 **Popularity Boost**  | Log-scaled rating count          | Surfaces movies people actually know    |
| ⭐ **Match Score**        | 3.2–4.7 range                    | Realistic, trustworthy ratings          |


🔗 Links
🎥 Live App: streamlit.app
💻 GitHub: github.com/afadamarcello-code/movie-recommender-system
💼 LinkedIn: linkedin.com/in/ahmed-iprahim-a51240133

📜 License
MIT License — feel free to fork and build your own recommender!
