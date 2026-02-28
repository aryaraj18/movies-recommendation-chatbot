"""
Movie Search Module

Search and retrieve detailed information about movies.
"""

import pandas as pd
from typing import List, Dict, Optional


class MovieSearch:
    """Search and retrieve movie information."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the search engine.
        
        Args:
            df: DataFrame with movie data
        """
        self.df = df
    
    def get_movie_details(self, title: str) -> Optional[Dict]:
        """
        Get detailed information about a movie.
        
        Args:
            title: Movie title
            
        Returns:
            Dictionary with movie details
        """
        # Search for the movie
        mask = self.df['title'].str.lower() == title.lower()
        movie = self.df[mask]
        
        if movie.empty:
            # Try partial match
            mask = self.df['title'].str.lower().str.contains(title.lower(), na=False)
            movie = self.df[mask]
            
            if movie.empty:
                return None
        
        # Get the first match
        row = movie.iloc[0]
        
        # Build details dictionary
        details = {
            'title': row.get('title', 'N/A'),
            'original_title': row.get('original_title', row.get('title', 'N/A')),
            'overview': row.get('overview', 'N/A'),
            'tagline': row.get('tagline', 'N/A'),
            'genres': row.get('genres', 'N/A'),
            'rating': row.get('vote_average', 0),
            'vote_count': row.get('vote_count', 0),
            'release_date': row.get('release_date', 'N/A'),
            'runtime': row.get('runtime', 0),
            'language': row.get('original_language', 'N/A'),
            'status': row.get('status', 'N/A'),
            'director': row.get('director', 'N/A'),
            'cast': row.get('cast', 'N/A'),
            'production_companies': row.get('production_companies', 'N/A'),
            'budget': row.get('budget', 0),
            'revenue': row.get('revenue', 0),
            'homepage': row.get('homepage', 'N/A'),
            'imdb_id': row.get('imdb_id', 'N/A')
        }
        
        # Format runtime
        if details['runtime'] and details['runtime'] > 0:
            hours = int(details['runtime']) // 60
            minutes = int(details['runtime']) % 60
            details['runtime_formatted'] = f"{hours}h {minutes}m"
        else:
            details['runtime_formatted'] = "N/A"
        
        # Format year
        if details['release_date'] and str(details['release_date']) != 'N/A':
            details['year'] = str(details['release_date'])[:4]
        else:
            details['year'] = 'N/A'
        
        # Format budget and revenue
        details['budget_formatted'] = f"${details['budget']:,.0f}" if details['budget'] > 0 else "N/A"
        details['revenue_formatted'] = f"${details['revenue']:,.0f}" if details['revenue'] > 0 else "N/A"
        
        # Format rating
        details['rating_stars'] = '⭐' * int(details['rating'] // 2)
        
        # Generate IMDB link
        if details['imdb_id'] and details['imdb_id'] != 'N/A':
            details['imdb_link'] = f"https://www.imdb.com/title/{details['imdb_id']}/"
        else:
            details['imdb_link'] = 'N/A'
        
        return details
    
    def search_by_title(self, query: str, limit: int = 10) -> pd.DataFrame:
        """
        Search movies by title.
        
        Args:
            query: Search query
            limit: Number of results
            
        Returns:
            DataFrame with matching movies
        """
        mask = self.df['title'].str.lower().str.contains(query.lower(), na=False)
        results = self.df[mask][[
            'title', 'vote_average', 'vote_count', 'genres', 'release_date'
        ]].head(limit)
        
        return results
    
    def search_by_genre(self, genre: str, limit: int = 20) -> pd.DataFrame:
        """
        Search movies by genre.
        
        Args:
            genre: Genre name
            limit: Number of results
            
        Returns:
            DataFrame with matching movies
        """
        mask = self.df['genres'].str.lower().str.contains(genre.lower(), na=False)
        results = self.df[mask][[
            'title', 'vote_average', 'vote_count', 'genres', 'release_date'
        ]].sort_values('vote_average', ascending=False).head(limit)
        
        return results
    
    def search_by_director(self, director: str, limit: int = 20) -> pd.DataFrame:
        """
        Search movies by director.
        
        Args:
            director: Director name
            limit: Number of results
            
        Returns:
            DataFrame with matching movies
        """
        mask = self.df['director'].str.lower().str.contains(director.lower(), na=False)
        results = self.df[mask][[
            'title', 'vote_average', 'vote_count', 'genres', 'release_date', 'director'
        ]].sort_values('vote_average', ascending=False).head(limit)
        
        return results
    
    def search_by_actor(self, actor: str, limit: int = 20) -> pd.DataFrame:
        """
        Search movies by actor.
        
        Args:
            actor: Actor name
            limit: Number of results
            
        Returns:
            DataFrame with matching movies
        """
        mask = self.df['cast'].str.lower().str.contains(actor.lower(), na=False)
        results = self.df[mask][[
            'title', 'vote_average', 'vote_count', 'genres', 'release_date', 'cast'
        ]].sort_values('vote_average', ascending=False).head(limit)
        
        return results
    
    def search_by_year(self, year: int, limit: int = 20) -> pd.DataFrame:
        """
        Search movies by release year.
        
        Args:
            year: Release year
            limit: Number of results
            
        Returns:
            DataFrame with matching movies
        """
        mask = self.df['release_date'].str.startswith(str(year), na=False)
        results = self.df[mask][[
            'title', 'vote_average', 'vote_count', 'genres', 'release_date'
        ]].sort_values('vote_average', ascending=False).head(limit)
        
        return results
    
    def get_top_rated(self, limit: int = 20, min_votes: int = 100) -> pd.DataFrame:
        """
        Get top rated movies.
        
        Args:
            limit: Number of results
            min_votes: Minimum vote count
            
        Returns:
            DataFrame with top rated movies
        """
        df_filtered = self.df[self.df['vote_count'] >= min_votes]
        results = df_filtered.nlargest(limit, 'vote_average')[[
            'title', 'vote_average', 'vote_count', 'genres', 'release_date'
        ]]
        
        return results
    
    def get_popular(self, limit: int = 20) -> pd.DataFrame:
        """
        Get popular movies by revenue.
        
        Args:
            limit: Number of results
            
        Returns:
            DataFrame with popular movies
        """
        results = self.df.nlargest(limit, 'revenue')[[
            'title', 'vote_average', 'vote_count', 'revenue', 'genres', 'release_date'
        ]]
        
        return results
    
    def get_recent_releases(self, limit: int = 20) -> pd.DataFrame:
        """
        Get recent movie releases.
        
        Args:
            limit: Number of results
            
        Returns:
            DataFrame with recent releases
        """
        # Sort by release date
        df_sorted = self.df.dropna(subset=['release_date'])
        df_sorted = df_sorted.sort_values('release_date', ascending=False)
        
        results = df_sorted.head(limit)[[
            'title', 'vote_average', 'vote_count', 'genres', 'release_date'
        ]]
        
        return results
    
    def get_all_genres(self) -> List[str]:
        """
        Get all unique genres.
        
        Returns:
            List of unique genres
        """
        genres = set()
        for genre_list in self.df['genres'].dropna():
            genre_str = str(genre_list).replace(',', ' ')
            for genre in genre_str.split():
                if genre and len(genre) > 2:
                    genres.add(genre.strip().title())
        return sorted(list(genres))
    
    def get_all_directors(self) -> List[str]:
        """
        Get all unique directors.
        
        Returns:
            List of unique directors
        """
        directors = self.df['director'].dropna().unique().tolist()
        return sorted([d for d in directors if d and d != 'Unknown'])
    
    def format_movie_card(self, title: str) -> str:
        """
        Format movie information as a nice card.
        
        Args:
            title: Movie title
            
        Returns:
            Formatted string with movie info
        """
        details = self.get_movie_details(title)
        
        if not details:
            return f"Movie '{title}' not found. 😕"
        
        card = f"""
🎬 **{details['title']}** ({details['year']})
{details['rating_stars']} **{details['rating']}/10** ({details['vote_count']:,} votes)

📅 **Release Date:** {details['release_date']}
⏱️ **Runtime:** {details['runtime_formatted']}
🎭 **Genre:** {details['genres']}
🎬 **Director:** {details['director']}

📖 **Overview:**
{details['overview']}
"""
        
        if details['tagline']:
            card += f"\n> *{details['tagline']}*"
        
        if details['imdb_link'] != 'N/A':
            card += f"\n🔗 [View on IMDB]({details['imdb_link']})"
        
        return card


def create_search(df: pd.DataFrame) -> MovieSearch:
    """
    Convenience function to create a search engine.
    
    Args:
        df: DataFrame with movie data
        
    Returns:
        MovieSearch instance
    """
    return MovieSearch(df)


if __name__ == "__main__":
    # Test the search
    from src.data_loader import MovieDataLoader
    
    loader = MovieDataLoader('data/movies.csv')
    search = MovieSearch(loader.df)
    
    # Get details
    details = search.get_movie_details('Inception')
    if details:
        print(f"Title: {details['title']}")
        print(f"Rating: {details['rating']}")
        print(f"Genres: {details['genres']}")
