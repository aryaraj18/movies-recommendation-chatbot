"""
Recommendation Engine Module

Provides movie recommendations using TF-IDF and cosine similarity.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional
import pickle
import os


class MovieRecommender:
    """Content-based movie recommendation engine."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the recommender.
        
        Args:
            df: DataFrame with movie data and combined_features
        """
        self.df = df
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.vectorizer = None
        self._build_index()
    
    def _build_index(self):
        """Build TF-IDF index and similarity matrix."""
        print("Building recommendation index...")
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 2)
        )
        
        # Fit and transform
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df['combined_features'].fillna('')
        )
        
        # Calculate cosine similarity
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        print(f"Index built with {self.tfidf_matrix.shape[0]} movies and {self.tfidf_matrix.shape[1]} features")
    
    def get_similarity_scores(self, movie_idx: int) -> np.ndarray:
        """
        Get similarity scores for a movie.
        
        Args:
            movie_idx: Index of the movie
            
        Returns:
            Array of similarity scores
        """
        return self.cosine_sim[movie_idx]
    
    def get_title_to_idx(self) -> dict:
        """
        Create title to index mapping.
        
        Returns:
            Dictionary mapping title to index
        """
        return {title.lower(): idx for idx, title in enumerate(self.df['title'])}
    
    def get_idx_to_title(self) -> dict:
        """
        Create index to title mapping.
        
        Returns:
            Dictionary mapping index to title
        """
        return {idx: title for idx, title in enumerate(self.df['title'])}
    
    def recommend_by_title(self, title: str, n: int = 10) -> pd.DataFrame:
        """
        Recommend similar movies by title.
        
        Args:
            title: Movie title
            n: Number of recommendations
            
        Returns:
            DataFrame with recommended movies
        """
        # Find movie index
        title_lower = title.lower()
        title_to_idx = self.get_title_to_idx()
        
        # Try exact match first
        if title_lower in title_to_idx:
            movie_idx = title_to_idx[title_lower]
        else:
            # Try partial match
            matches = [k for k in title_to_idx.keys() if title_lower in k]
            if not matches:
                return pd.DataFrame()
            movie_idx = title_to_idx[matches[0]]
        
        return self.recommend_by_index(movie_idx, n)
    
    def recommend_by_index(self, movie_idx: int, n: int = 10) -> pd.DataFrame:
        """
        Recommend similar movies by index.
        
        Args:
            movie_idx: Index of the movie
            n: Number of recommendations
            
        Returns:
            DataFrame with recommended movies
        """
        # Get similarity scores
        sim_scores = self.get_similarity_scores(movie_idx)
        
        # Get top n+1 (excluding itself)
        top_indices = np.argsort(sim_scores)[::-1][1:n+1]
        
        # Get movie data
        recommendations = self.df.iloc[top_indices][[
            'title', 'vote_average', 'vote_count', 'genres', 
            'release_date', 'director'
        ]].copy()
        
        recommendations['similarity_score'] = sim_scores[top_indices]
        
        return recommendations
    
    def recommend_by_description(self, description: str, n: int = 10) -> pd.DataFrame:
        """
        Recommend movies based on a description.
        
        Args:
            description: Movie description or preference
            n: Number of recommendations
            
        Returns:
            DataFrame with recommended movies
        """
        # Transform description
        desc_vector = self.vectorizer.transform([description])
        
        # Calculate similarity
        sim_scores = cosine_similarity(desc_vector, self.tfidf_matrix).flatten()
        
        # Get top n
        top_indices = np.argsort(sim_scores)[::-1][:n]
        
        # Get movie data
        recommendations = self.df.iloc[top_indices][[
            'title', 'vote_average', 'vote_count', 'genres',
            'release_date', 'director'
        ]].copy()
        
        recommendations['similarity_score'] = sim_scores[top_indices]
        
        return recommendations
    
    def recommend_by_genre(self, genre: str, n: int = 20, 
                         min_rating: float = 7.0) -> pd.DataFrame:
        """
        Recommend movies by genre with minimum rating.
        
        Args:
            genre: Genre name
            n: Number of recommendations
            min_rating: Minimum rating
            
        Returns:
            DataFrame with recommended movies
        """
        # Filter by genre
        mask = self.df['genres'].str.lower().str.contains(genre.lower(), na=False)
        genre_df = self.df[mask].copy()
        
        # Filter by rating
        genre_df = genre_df[genre_df['vote_average'] >= min_rating]
        
        # Sort by rating
        genre_df = genre_df.sort_values('vote_average', ascending=False).head(n)
        
        return genre_df[['title', 'vote_average', 'vote_count', 'genres', 'release_date']]
    
    def get_similar_movies(self, movie_title: str, n: int = 5) -> List[Dict]:
        """
        Get similar movies as a list of dictionaries.
        
        Args:
            movie_title: Movie title
            n: Number of recommendations
            
        Returns:
            List of movie dictionaries
        """
        recommendations = self.recommend_by_title(movie_title, n)
        
        if recommendations.empty:
            return []
        
        results = []
        for _, row in recommendations.iterrows():
            results.append({
                'title': row['title'],
                'rating': row['vote_average'],
                'votes': row['vote_count'],
                'genre': row['genres'],
                'year': str(row['release_date'])[:4] if pd.notna(row['release_date']) else 'N/A',
                'director': row.get('director', 'N/A')
            })
        
        return results
    
    def save_model(self, filepath: str):
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'df': self.df,
            'tfidf_matrix': self.tfidf_matrix,
            'cosine_sim': self.cosine_sim,
            'vectorizer': self.vectorizer
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> 'MovieRecommender':
        """
        Load a saved model.
        
        Args:
            filepath: Path to the model file
            
        Returns:
            MovieRecommender instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        recommender = MovieRecommender(model_data['df'])
        recommender.tfidf_matrix = model_data['tfidf_matrix']
        recommender.cosine_sim = model_data['cosine_sim']
        recommender.vectorizer = model_data['vectorizer']
        
        print(f"Model loaded from {filepath}")
        return recommender


def create_recommender(df: pd.DataFrame) -> MovieRecommender:
    """
    Convenience function to create a recommender.
    
    Args:
        df: DataFrame with movie data
        
    Returns:
        MovieRecommender instance
    """
    return MovieRecommender(df)


if __name__ == "__main__":
    # Test the recommender
    from src.data_loader import MovieDataLoader
    from src.preprocess import preprocess_movies
    
    # Load and preprocess
    loader = MovieDataLoader('data/movies.csv')
    df = preprocess_movies(loader.df)
    
    # Create recommender
    recommender = MovieRecommender(df)
    
    # Get recommendations
    print("\nMovies similar to 'Inception':")
    recs = recommender.recommend_by_title('Inception', 5)
    print(recs)
