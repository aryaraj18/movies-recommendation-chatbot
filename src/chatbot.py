"""
Chatbot Module

Natural language chatbot for movie recommendations and information.
"""

import re
from typing import List, Dict, Tuple, Optional


class MovieChatbot:
    """Chatbot for movie recommendations and information."""
    
    def __init__(self, recommender, search_engine):
        """
        Initialize the chatbot.
        
        Args:
            recommender: MovieRecommender instance
            search_engine: MovieSearch instance
        """
        self.recommender = recommender
        self.search = search_engine
        
        # Define patterns for intent recognition
        self.patterns = {
            'recommend': [
                r'recommend',
                r'similar to',
                r'movies like',
                r'what should i watch',
                r'suggestions',
                r'like (.*)',
            ],
            'info': [
                r'tell me about',
                r'information about',
                r'details about',
                r'what is (.*)',
                r'about (.*)',
            ],
            'top_rated': [
                r'top rated',
                r'best movies',
                r'highest rated',
                r'best rated',
            ],
            'genre': [
                r'movies in (.*)',
                r'(.*) movies',
                r'genre (.*)',
                r'show me (.*)',
            ],
            'director': [
                r'movies by (.*)',
                r'director (.*)',
                r'films by (.*)',
            ],
            'actor': [
                r'movies with (.*)',
                r'starring (.*)',
                r'cast (.*)',
            ],
            'random': [
                r'surprise me',
                r'random movie',
                r'pick a movie',
            ],
            'help': [
                r'help',
                r'what can you do',
                r'commands',
            ]
        }
    
    def parse_message(self, message: str) -> Tuple[str, Optional[str]]:
        """
        Parse user message to determine intent and extract entities.
        
        Args:
            message: User message
            
        Returns:
            (intent, entity) tuple
        """
        message_lower = message.lower()
        
        # Check recommend patterns
        for pattern in self.patterns['recommend']:
            match = re.search(pattern, message_lower)
            if match:
                entity = match.group(1) if match.groups() else None
                return ('recommend', entity)
        
        # Check info patterns
        for pattern in self.patterns['info']:
            match = re.search(pattern, message_lower)
            if match:
                entity = match.group(1) if match.groups() else None
                return ('info', entity)
        
        # Check top rated patterns
        for pattern in self.patterns['top_rated']:
            if pattern in message_lower:
                return ('top_rated', None)
        
        # Check genre patterns
        for pattern in self.patterns['genre']:
            match = re.search(pattern, message_lower)
            if match:
                entity = match.group(1) if match.groups() else None
                return ('genre', entity)
        
        # Check director patterns
        for pattern in self.patterns['director']:
            match = re.search(pattern, message_lower)
            if match:
                entity = match.group(1) if match.groups() else None
                return ('director', entity)
        
        # Check actor patterns
        for pattern in self.patterns['actor']:
            match = re.search(pattern, message_lower)
            if match:
                entity = match.group(1) if match.groups() else None
                return ('actor', entity)
        
        # Check random patterns
        for pattern in self.patterns['random']:
            if pattern in message_lower:
                return ('random', None)
        
        # Check help patterns
        for pattern in self.patterns['help']:
            if pattern in message_lower:
                return ('help', None)
        
        # Default to search
        return ('search', message)
    
    def handle_recommend(self, movie: str) -> str:
        """
        Handle movie recommendation request.
        
        Args:
            movie: Movie name
            
        Returns:
            Response string
        """
        if not movie:
            return "Sure! Tell me a movie you liked, and I'll recommend similar ones. 🎬"
        
        # Get recommendations
        results = self.recommender.get_similar_movies(movie, n=10)
        
        if not results:
            return f"I couldn't find any movies similar to '{movie}'. Try a different movie! 😕"
        
        # Format response
        response = f"Here are some movies similar to **{movie}**:\n\n"
        
        for i, m in enumerate(results, 1):
            rating = m.get('rating', 0)
            genre = m.get('genre', 'N/A')
            year = m.get('year', 'N/A')
            response += f"{i}. **{m['title']}** ({year}) - ⭐ {rating:.1f}\n"
            response += f"   🎭 {genre}\n\n"
        
        return response
    
    def handle_info(self, movie: str) -> str:
        """
        Handle movie information request.
        
        Args:
            movie: Movie name
            
        Returns:
            Response string
        """
        if not movie:
            return "Sure! Tell me which movie you want to know about. 🎬"
        
        # Get movie details
        details = self.search.get_movie_details(movie)
        
        if not details:
            return f"I couldn't find information about '{movie}'. Try a different movie! 😕"
        

        response = f"""
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
            response += f"\n> *{details['tagline']}*\n"
        
        if details['imdb_link'] != 'N/A':
            response += f"\n🔗 [View on IMDB]({details['imdb_link']})"
        
        return response
    
    def handle_top_rated(self, genre: str = None) -> str:
        """
        Handle top rated movies request.
        
        Args:
            genre: Optional genre filter
            
        Returns:
            Response string
        """
        if genre:
            # Filter by genre
            results = self.search.search_by_genre(genre, limit=10)
        else:
            # Get all top rated
            results = self.search.get_top_rated(limit=10)
        
        if results.empty:
            return "I couldn't find any top rated movies. 😕"
        
        # Format response
        if genre:
            response = f"Here are the top rated **{genre}** movies:\n\n"
        else:
            response = "Here are the top rated movies:\n\n"
        
        for i, (_, row) in enumerate(results.iterrows(), 1):
            response += f"{i}. **{row['title']}** - ⭐ {row['vote_average']:.1f}\n"
        
        return response
    
    def handle_genre(self, genre: str) -> str:
        """
        Handle genre search request.
        
        Args:
            genre: Genre name
            
        Returns:
            Response string
        """
        if not genre:
            genres = self.search.get_all_genres()
            return f"Available genres: {', '.join(genres[:20])}..."
        
        results = self.search.search_by_genre(genre, limit=10)
        
        if results.empty:
            return f"I couldn't find any movies in the '{genre}' genre. 😕"
        
        response = f"Here are some **{genre}** movies:\n\n"
        
        for i, (_, row) in enumerate(results.iterrows(), 1):
            response += f"{i}. **{row['title']}** - ⭐ {row['vote_average']:.1f}\n"
        
        return response
    
    def handle_director(self, director: str) -> str:
        """
        Handle director search request.
        
        Args:
            director: Director name
            
        Returns:
            Response string
        """
        if not director:
            return "Which director's movies would you like to see?"
        
        results = self.search.search_by_director(director, limit=10)
        
        if results.empty:
            return f"I couldn't find any movies by '{director}'. 😕"
        
        response = f"Here are some movies by **{director}**:\n\n"
        
        for i, (_, row) in enumerate(results.iterrows(), 1):
            response += f"{i}. **{row['title']}** ({row['release_date'][:4]}) - ⭐ {row['vote_average']:.1f}\n"
        
        return response
    
    def handle_actor(self, actor: str) -> str:
        """
        Handle actor search request.
        
        Args:
            actor: Actor name
            
        Returns:
            Response string
        """
        if not actor:
            return "Which actor's movies would you like to see?"
        
        results = self.search.search_by_actor(actor, limit=10)
        
        if results.empty:
            return f"I couldn't find any movies starring '{actor}'. 😕"
        
        response = f"Here are some movies starring **{actor}**:\n\n"
        
        for i, (_, row) in enumerate(results.iterrows(), 1):
            response += f"{i}. **{row['title']}** ({row['release_date'][:4]}) - ⭐ {row['vote_average']:.1f}\n"
        
        return response
    
    def handle_random(self) -> str:
        """
        Handle random movie request.
        
        Returns:
            Response string
        """
        import random
        
        # Get random movie
        movie = self.search.get_random_movies(1)
        
        if movie.empty:
            return "I couldn't find any movies. 😕"
        
        row = movie.iloc[0]
        title = row['title']
        
        # Get details for the random movie
        details = self.search.get_movie_details(title)
        
        if details:
            return f"""
🎲 **Random Movie Pick:**

🎬 **{details['title']}** ({details['year']})
{details['rating_stars']} **{details['rating']}/10**

🎭 **Genre:** {details['genres']}
🎬 **Director:** {details['director']}

📖 {details['overview'][:300]}...
"""
        else:
            return f"How about watching **{title}**? 🎬"
    
    def handle_help(self) -> str:
        """
        Handle help request.
        
        Returns:
            Response string
        """
        return """
🤖 **Movie Chatbot Commands:**

Here are some things you can ask me:

🎬 **Recommendations:**
- "Recommend movies like Inception"
- "Movies similar to The Matrix"
- "Suggest something like Avatar"

ℹ️ **Movie Info:**
- "Tell me about Titanic"
- "What's Inception about?"

⭐ **Top Rated:**
- "Show me top rated movies"
- "Best horror movies"
- "Top action films"

🎭 **By Genre:**
- "Movies in comedy"
- "Sci-fi movies"
- "Horror films"

🎬 **By Director:**
- "Movies by Christopher Nolan"
- "Films by Quentin Tarantino"

🎭 **By Actor:**
- "Movies with Tom Hanks"
- "Starring Leonardo DiCaprio"

🎲 **Random:**
- "Surprise me"
- "Pick a random movie"

Just type naturally and I'll understand! 😊
"""
    
    def handle_search(self, query: str) -> str:
        """
        Handle general search request.
        
        Args:
            query: Search query
            
        Returns:
            Response string
        """
        results = self.search.search_by_title(query, limit=5)
        
        if results.empty:
            return f"I couldn't find anything matching '{query}'. Try a different search! 😕"
        
        response = f"Here are some movies matching '{query}':\n\n"
        
        for i, (_, row) in enumerate(results.iterrows(), 1):
            response += f"{i}. **{row['title']}** - ⭐ {row['vote_average']:.1f}\n"
        
        response += "\nTell me which one you want to know more about!"
        
        return response
    
    def get_response(self, message: str) -> str:
        """
        Get chatbot response for user message.
        
        Args:
            message: User message
            
        Returns:
            Response string
        """
        # Parse message
        intent, entity = self.parse_message(message)
        
        # Handle based on intent
        if intent == 'recommend':
            return self.handle_recommend(entity)
        elif intent == 'info':
            return self.handle_info(entity)
        elif intent == 'top_rated':
            return self.handle_top_rated()
        elif intent == 'genre':
            return self.handle_genre(entity)
        elif intent == 'director':
            return self.handle_director(entity)
        elif intent == 'actor':
            return self.handle_actor(entity)
        elif intent == 'random':
            return self.handle_random()
        elif intent == 'help':
            return self.handle_help()
        elif intent == 'search':
            return self.handle_search(message)
        else:
            return "I'm not sure I understood that. Type 'help' for a list of commands! 😊"


def create_chatbot(recommender, search_engine) -> MovieChatbot:
    """
    Convenience function to create a chatbot.
    
    Args:
        recommender: MovieRecommender instance
        search_engine: MovieSearch instance
        
    Returns:
        MovieChatbot instance
    """
    return MovieChatbot(recommender, search_engine)


if __name__ == "__main__":
    # Test the chatbot
    from src.data_loader import MovieDataLoader
    from src.preprocess import preprocess_movies
    from src.recommend import create_recommender
    from src.search import create_search
    
    # Load data
    loader = MovieDataLoader('data/movies.csv')
    df = preprocess_movies(loader.df)
    
    # Create components
    recommender = create_recommender(df)
    search = create_search(df)
    
    # Create chatbot
    chatbot = create_chatbot(recommender, search)
    
    # Test
    print(chatbot.get_response("Recommend movies like Inception"))
