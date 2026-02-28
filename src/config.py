"""
Configuration file for Movie Recommendation Chatbot
"""

# Data configuration
DATA_PATH = 'data/movies.csv'
MIN_VOTE_COUNT = 100
MIN_RATING = 5.0

# Recommendation configuration
TOP_N_RECOMMENDATIONS = 10
SIMILARITY_THRESHOLD = 0.1

# Chat configuration
MAX_OVERVIEW_LENGTH = 500
MAX_CAST_DISPLAY = 5

# App configuration
STREAMLIT_PORT = 8501
STREAMLIT_HOST = 'localhost'

# Model configuration
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)
