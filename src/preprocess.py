"""
Data Preprocessing Module

Cleans and preprocesses movie metadata for recommendation engine.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Tuple


class MoviePreprocessor:
    """Preprocess movie data for recommendations."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize preprocessor.
        
        Args:
            df: DataFrame with movie data
        """
        self.df = df.copy()
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the movie data.
        
        Returns:
            Cleaned DataFrame
        """
        # Make a copy
        df = self.df.copy()
        
        # Fill missing values
        df['title'] = df['title'].fillna('')
        df['overview'] = df['overview'].fillna('')
        df['genres'] = df['genres'].fillna('')
        df['director'] = df['director'].fillna('Unknown')
        df['cast'] = df['cast'].fillna('')
        df['tagline'] = df['tagline'].fillna('')
        
        # Convert vote_average to numeric
        df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0)
        df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce').fillna(0)
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)
        df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce').fillna(0)
        
        # Clean text fields
        df['title'] = df['title'].apply(self._clean_text)
        df['overview'] = df['overview'].apply(self._clean_text)
        df['genres'] = df['genres'].apply(self._clean_genres)
        df['director'] = df['director'].apply(self._clean_text)
        df['cast'] = df['cast'].apply(self._clean_cast)
        
        # Create combined features for recommendation
        df['combined_features'] = df.apply(self._create_combined_features, axis=1)
        
        self.df = df
        return df
    
    def _clean_text(self, text: str) -> str:
        """Clean text field."""
        if pd.isna(text):
            return ''
        text = str(text)
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip().lower()
    
    def _clean_genres(self, genres: str) -> str:
        """Clean genres field."""
        if pd.isna(genres):
            return ''
        # Handle both string and list formats
        genres = str(genres)
        # Remove brackets and quotes
        genres = genres.replace('[', '').replace(']', '')
        genres = genres.replace("'", '').replace('"', '')
        # Split by comma and clean each genre
        genre_list = [g.strip().lower() for g in genres.split(',')]
        return ' '.join(genre_list)
    
    def _clean_cast(self, cast: str) -> str:
        """Clean cast field."""
        if pd.isna(cast):
            return ''
        cast = str(cast)
        # Take first 3 actors
        actors = cast.split(',')[:3]
        return ' '.join([a.strip().lower() for a in actors])
    
    def _create_combined_features(self, row: pd.Series) -> str:
        """
        Create combined features for content-based filtering.
        
        Args:
            row: DataFrame row
            
        Returns:
            Combined features string
        """
        features = []
        
        # Add genres (most important)
        if row['genres']:
            features.append(row['genres'])
        
        # Add overview
        if row['overview']:
            features.append(row['overview'][:500])  # Limit length
        
        # Add director
        if row['director'] and row['director'] != 'unknown':
            features.append(row['director'])
        
        # Add cast
        if row['cast']:
            features.append(row['cast'])
        
        return ' '.join(features)
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        Get the processed DataFrame.
        
        Returns:
            Processed DataFrame
        """
        return self.df
    
    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate movies based on title.
        
        Returns:
            DataFrame without duplicates
        """
        self.df = self.df.drop_duplicates(subset=['title'], keep='first')
        return self.df
    
    def filter_by_votes(self, min_votes: int = 100) -> pd.DataFrame:
        """
        Filter movies by minimum vote count.
        
        Args:
            min_votes: Minimum number of votes
            
        Returns:
            Filtered DataFrame
        """
        self.df = self.df[self.df['vote_count'] >= min_votes]
        return self.df
    
    def get_feature_matrix(self) -> pd.DataFrame:
        """
        Get the combined features column.
        
        Returns:
            DataFrame with combined features
        """
        return self.df[['title', 'combined_features']].copy()


def preprocess_movies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to preprocess movie data.
    
    Args:
        df: Raw movie DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    preprocessor = MoviePreprocessor(df)
    df = preprocessor.clean_data()
    df = preprocessor.remove_duplicates()
    return df


if __name__ == "__main__":
    # Test preprocessing
    from src.data_loader import MovieDataLoader
    
    loader = MovieDataLoader('data/movies.csv')
    df = loader.df
    
    preprocessor = MoviePreprocessor(df)
    df_clean = preprocessor.clean_data()
    
    print(f"Original: {len(df)} movies")
    print(f"Cleaned: {len(df_clean)} movies")
    print(f"\nSample combined features:")
    print(df_clean['combined_features'].head(2).tolist())
