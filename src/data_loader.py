"""
Movie

Lo Data Loader Moduleads and manages movie metadata from Kaggle dataset.
"""

import pandas as pd
import os
from typing import Optional


class MovieDataLoader:
    """Load and manage movie data."""
    
    def __init__(self, data_path: str = 'data/movies.csv'):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the movies CSV file
        """
        self.data_path = data_path
        self.df = None
        self.load_data()
    
    def load_data(self) -> pd.DataFrame:
        """
        Load movie data from CSV.
        
        Returns:
            DataFrame with movie data
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Data file not found at {self.data_path}\n"
                f"Please download the Kaggle movies metadata dataset and place it in the data/ folder.\n"
                f"Download from: https://www.kaggle.com/datasets/rashikrahmanmr/movies-metadata"
            )
        
        # Read CSV with error handling
        try:
            self.df = pd.read_csv(self.data_path, low_memory=False)
            print(f"Loaded {len(self.df)} movies")
            return self.df
        except Exception as e:
            raise Exception(f"Error loading data: {e}")
    
    def get_info(self) -> dict:
        """
        Get dataset information.
        
        Returns:
            Dictionary with dataset info
        """
        return {
            'total_movies': len(self.df),
            'columns': list(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict()
        }
    
    def get_movie_by_title(self, title: str) -> pd.DataFrame:
        """
        Get movie(s) by title.
        
        Args:
            title: Movie title (partial match)
            
        Returns:
            DataFrame with matching movies
        """
        mask = self.df['title'].str.lower().str.contains(title.lower(), na=False)
        return self.df[mask]
    
    def get_movies_by_genre(self, genre: str, limit: int = 20) -> pd.DataFrame:
        """
        Get movies by genre.
        
        Args:
            genre: Genre name
            limit: Number of movies to return
            
        Returns:
            DataFrame with movies in the genre
        """
        mask = self.df['genres'].str.lower().str.contains(genre.lower(), na=False)
        return self.df[mask].head(limit)
    
    def get_movies_by_director(self, director: str, limit: int = 20) -> pd.DataFrame:
        """
        Get movies by director.
        
        Args:
            director: Director name
            limit: Number of movies to return
            
        Returns:
            DataFrame with movies by the director
        """
        mask = self.df['director'].str.lower().str.contains(director.lower(), na=False)
        return self.df[mask].head(limit)
    
    def get_top_rated(self, limit: int = 20, min_votes: int = 100) -> pd.DataFrame:
        """
        Get top rated movies.
        
        Args:
            limit: Number of movies to return
            min_votes: Minimum vote count
            
        Returns:
            DataFrame with top rated movies
        """
        df_filtered = self.df[self.df['vote_count'] >= min_votes]
        return df_filtered.nlargest(limit, 'vote_average')
    
    def get_popular_movies(self, limit: int = 20) -> pd.DataFrame:
        """
        Get popular movies by revenue.
        
        Args:
            limit: Number of movies to return
            
        Returns:
            DataFrame with popular movies
        """
        return self.df.nlargest(limit, 'revenue')
    
    def search_movies(self, query: str, limit: int = 10) -> pd.DataFrame:
        """
        Search movies by title or overview.
        
        Args:
            query: Search query
            limit: Number of results
            
        Returns:
            DataFrame with matching movies
        """
        title_match = self.df['title'].str.lower().str.contains(query.lower(), na=False)
        overview_match = self.df['overview'].str.lower().str.contains(query.lower(), na=False)
        return self.df[title_match | overview_match].head(limit)
    
    def get_random_movies(self, n: int = 10) -> pd.DataFrame:
        """
        Get random movies.
        
        Args:
            n: Number of movies
            
        Returns:
            DataFrame with random movies
        """
        return self.df.sample(n)
    
    def get_all_genres(self) -> list:
        """
        Get all unique genres.
        
        Returns:
            List of unique genres
        """
        genres = set()
        for genre_list in self.df['genres'].dropna():
            for genre in str(genre_list).split(','):
                genres.add(genre.strip())
        return sorted(list(genres))
    
    def get_all_directors(self) -> list:
        """
        Get all unique directors.
        
        Returns:
            List of unique directors
        """
        return self.df['director'].dropna().unique().tolist()


if __name__ == "__main__":
    # Test the data loader
    loader = MovieDataLoader()
    print(f"Total movies: {loader.get_info()['total_movies']}")
    
    # Test search
    results = loader.get_movie_by_title('Inception')
    print(f"\nFound {len(results)} movies with 'Inception'")
    print(results[['title', 'vote_average']].head())
