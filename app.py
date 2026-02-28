"""
Flask API for Movie Recommendation Chatbot
Powered by OpenRouter AI (GPT-OSS-20B)
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import ast
import os
import re
import requests as http_requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-oss-20b:free"

# Global variables
movies_df = None
credits_df = None

DATA_PATH = "data/movies_metadata.csv"
CREDITS_PATH = "data/credits.csv"

SYSTEM_PROMPT = """You are MovieBot, a friendly AI assistant that ONLY provides movie recommendations and movie-related information.

STRICT RULES:
1. You can ONLY discuss movies, movie recommendations, and movie-related topics.
2. If a user asks about ANYTHING not related to movies (coding, math, science, personal advice, weather, politics, etc.), you MUST politely decline and say something like: "I'm MovieBot \u2014 I only help with movie recommendations! \ud83c\udfac Ask me about movies, genres, or recommendations instead."
3. Use ONLY the movie data provided in the [MOVIE DATA] section to make recommendations. Do NOT invent or hallucinate movie details not present in the data.
4. Be conversational, friendly, and concise.
5. When recommending movies, format them clearly with title, year, rating, and a brief description when available.
6. Use markdown formatting: **bold** for titles, numbered lists for recommendations.
7. Use emojis sparingly for a fun touch.
8. If the movie data context is empty or doesn't match, say you couldn't find matching movies and suggest the user try different keywords or genres.
9. Keep responses focused and scannable \u2014 avoid walls of text."""


def load_data():
    global movies_df, credits_df

    print("Loading movie data...")
    movies_df = pd.read_csv(DATA_PATH, low_memory=False)

    # Clean data
    movies_df = movies_df.dropna(subset=["title"])
    movies_df = movies_df[movies_df["vote_average"].notna()]
    movies_df = movies_df[movies_df["vote_average"] > 0]

    def process_genres(genres):
        if pd.isna(genres):
            return ""
        try:
            if isinstance(genres, str):
                genres_list = ast.literal_eval(genres)
                return " ".join([g["name"] for g in genres_list])
        except Exception:
            return str(genres).lower()
        return ""

    movies_df["genres_processed"] = movies_df["genres"].apply(process_genres)
    movies_df["overview"] = movies_df["overview"].fillna("")
    movies_df = movies_df.reset_index(drop=True)

    print("Loading credits...")
    credits_df = pd.read_csv(CREDITS_PATH, low_memory=False)

    print(f"Loaded {len(movies_df)} movies!")
    return movies_df


def get_cast_for_movie(row):
    """Get cast for a movie row."""
    cast = []
    try:
        movie_id = row.get("id")
        if movie_id and credits_df is not None:
            movie_credits = credits_df[credits_df["id"] == movie_id]
            if not movie_credits.empty:
                cast_str = movie_credits.iloc[0].get("cast", "[]")
                if pd.notna(cast_str):
                    cast_list = ast.literal_eval(cast_str)[:5]
                    cast = [c["name"] for c in cast_list]
    except Exception:
        pass
    return cast


def search_movies(query, n=15):
    """Search movies by title, genre, or keywords."""
    query_lower = query.lower().strip()
    if not query_lower:
        return []

    movies_df["score"] = 0

    movies_df.loc[movies_df["title"].str.lower() == query_lower, "score"] = 100
    movies_df.loc[
        movies_df["title"].str.lower().str.startswith(query_lower), "score"
    ] += 50
    movies_df.loc[
        movies_df["title"].str.lower().str.contains(query_lower, na=False, regex=False),
        "score",
    ] += 10
    movies_df.loc[
        movies_df["genres_processed"].str.lower().str.contains(
            query_lower, na=False, regex=False
        ),
        "score",
    ] += 5
    movies_df.loc[
        movies_df["overview"].str.lower().str.contains(query_lower, na=False, regex=False),
        "score",
    ] += 3

    results_df = movies_df[movies_df["score"] > 0].nlargest(n, "score")

    results = []
    for _, row in results_df.iterrows():
        cast = get_cast_for_movie(row)
        results.append(
            {
                "title": row["title"],
                "year": str(row.get("release_date", ""))[:4]
                if pd.notna(row.get("release_date"))
                else "N/A",
                "rating": float(row["vote_average"])
                if pd.notna(row["vote_average"])
                else 0,
                "vote_count": int(row["vote_count"])
                if pd.notna(row["vote_count"])
                else 0,
                "overview": str(row["overview"])[:300]
                if pd.notna(row["overview"])
                else "",
                "genres": row["genres_processed"],
                "cast": cast,
                "runtime": int(row.get("runtime", 0))
                if pd.notna(row.get("runtime"))
                else 0,
            }
        )

    movies_df["score"] = 0
    return results


def _extract_keywords(query):
    """Extract meaningful keywords from a natural-language query."""
    q = query.lower().strip()
    if not q:
        return []

    # Split on whitespace + punctuation, keep hyphenated words.
    parts = re.split(r"[^a-z0-9\-']+", q)

    stop_words = {
        "i",
        "me",
        "my",
        "we",
        "us",
        "you",
        "your",
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "about",
        "that",
        "this",
        "it",
        "its",
        "and",
        "or",
        "but",
        "not",
        "no",
        "so",
        "if",
        "than",
        "too",
        "very",
        "just",
        "want",
        "wanna",
        "watch",
        "movie",
        "movies",
        "film",
        "films",
        "show",
        "find",
        "give",
        "get",
        "looking",
        "look",
        "need",
        "like",
        "some",
        "something",
        "recommend",
        "recommendation",
        "recommendations",
        "suggest",
        "suggestion",
        "suggestions",
        "any",
        "good",
        "great",
        "really",
        "also",
        "where",
        "what",
        "when",
        "how",
        "who",
        "which",
        "set",
        "setting",
        "featuring",
        "features",
        "based",
        "kind",
        "type",
        "genre",
        "story",
        "plot",
    }

    keywords = []
    for p in parts:
        if not p or p in stop_words:
            continue
        if len(p) <= 2:
            continue
        keywords.append(p)

    # Add a couple of common compound theme phrases if present
    compounds = [
        "post apocalyptic",
        "post-apocalyptic",
        "science fiction",
        "sci-fi",
        "time travel",
        "time-travel",
        "coming of age",
        "coming-of-age",
    ]
    for c in compounds:
        if c in q:
            keywords.append(c)

    # De-dup while preserving order
    seen = set()
    out = []
    for k in keywords:
        if k in seen:
            continue
        seen.add(k)
        out.append(k)

    return out


def search_movies_keywords(query, n=15):
    """Keyword-based search for natural language queries (e.g., themes, settings)."""
    keywords = _extract_keywords(query)
    if not keywords:
        return []

    # Expand a few theme synonyms to increase recall.
    synonyms = {
        "post apocalyptic": ["post apocalyptic", "post-apocalyptic", "apocalyptic", "apocalypse"],
        "post-apocalyptic": ["post apocalyptic", "post-apocalyptic", "apocalyptic", "apocalypse"],
        "apocalyptic": ["post-apocalyptic", "apocalyptic", "apocalypse", "wasteland"],
        "dystopian": ["dystopian", "dystopia", "totalitarian", "oppressive", "regime"],
        "dystopia": ["dystopian", "dystopia", "totalitarian", "oppressive"],
        "teen": ["teen", "teenager", "teenage", "young", "adolescent", "youth"],
        "teenager": ["teen", "teenager", "teenage", "young"],
        "zombie": ["zombie", "undead", "infected"],
        "alien": ["alien", "extraterrestrial", "invasion"],
        "space": ["space", "galaxy", "interstellar", "astronaut"],
    }

    expanded = []
    for k in keywords:
        expanded.append(k)
        expanded.extend(synonyms.get(k, []))

    # De-dup while preserving order
    seen = set()
    expanded_unique = []
    for k in expanded:
        if k in seen:
            continue
        seen.add(k)
        expanded_unique.append(k)

    movies_df["kw_score"] = 0

    for kw in expanded_unique:
        # Overview: strongest signal for descriptive queries
        movies_df.loc[
            movies_df["overview"].str.lower().str.contains(kw, na=False, regex=False),
            "kw_score",
        ] += 3
        # Genres: helpful when users mention genre-adjacent keywords
        movies_df.loc[
            movies_df["genres_processed"].str.lower().str.contains(
                kw, na=False, regex=False
            ),
            "kw_score",
        ] += 5
        # Title: small boost
        movies_df.loc[
            movies_df["title"].str.lower().str.contains(kw, na=False, regex=False),
            "kw_score",
        ] += 2

    results_df = movies_df[movies_df["kw_score"] > 0].copy()

    # Prefer movies with some minimum popularity to reduce obscure noise.
    if "vote_count" in results_df.columns:
        results_df = results_df[results_df["vote_count"].fillna(0) >= 20]

    # Rank primarily by keyword score, tie-break by rating.
    if "vote_average" in results_df.columns:
        results_df["final_score"] = results_df["kw_score"] + results_df["vote_average"].fillna(0) * 0.3
        results_df = results_df.nlargest(n, "final_score")
    else:
        results_df = results_df.nlargest(n, "kw_score")

    results = []
    for _, row in results_df.iterrows():
        cast = get_cast_for_movie(row)
        results.append(
            {
                "title": row["title"],
                "year": str(row.get("release_date", ""))[:4]
                if pd.notna(row.get("release_date"))
                else "N/A",
                "rating": float(row["vote_average"])
                if pd.notna(row["vote_average"])
                else 0,
                "vote_count": int(row["vote_count"])
                if pd.notna(row["vote_count"])
                else 0,
                "overview": str(row["overview"])[:300]
                if pd.notna(row["overview"])
                else "",
                "genres": row["genres_processed"],
                "cast": cast,
                "runtime": int(row.get("runtime", 0))
                if pd.notna(row.get("runtime"))
                else 0,
            }
        )

    movies_df["kw_score"] = 0
    return results


def get_top_rated(genre=None, n=15):
    """Get top rated movies, optionally filtered by genre."""
    df = movies_df.copy()

    if genre:
        df = df[
            df["genres_processed"].str.lower().str.contains(genre.lower(), na=False)
        ]

    df = df[df["vote_count"] >= 100]
    df = df.nlargest(n, "vote_average")

    results = []
    for _, row in df.iterrows():
        results.append(
            {
                "title": row["title"],
                "year": str(row.get("release_date", ""))[:4]
                if pd.notna(row.get("release_date"))
                else "N/A",
                "rating": float(row["vote_average"])
                if pd.notna(row["vote_average"])
                else 0,
                "vote_count": int(row["vote_count"])
                if pd.notna(row["vote_count"])
                else 0,
                "overview": str(row["overview"])[:300]
                if pd.notna(row["overview"])
                else "",
                "genres": row["genres_processed"],
            }
        )

    return results


def get_similar_movies(title, n=15):
    """Find movies similar to given title based on genre match."""
    title_lower = title.lower().strip()
    matches = movies_df[movies_df["title"].str.lower() == title_lower]

    if matches.empty:
        matches = movies_df[
            movies_df["title"].str.lower().str.contains(title_lower, na=False)
        ]

    if matches.empty:
        return []

    row = matches.iloc[0]
    genres = row["genres_processed"].lower().split() if row["genres_processed"] else []

    if not genres:
        return []

    movies_df["genre_match"] = 0
    for genre in genres:
        movies_df.loc[
            movies_df["genres_processed"].str.lower().str.contains(genre, na=False),
            "genre_match",
        ] += 1

    movies_df.loc[
        movies_df["title"].str.lower() == title_lower, "genre_match"
    ] = 0

    results_df = movies_df[movies_df["genre_match"] > 0].copy()
    results_df = results_df[results_df["vote_count"] >= 50]
    results_df["combined"] = (
        results_df["genre_match"] * 10 + results_df["vote_average"]
    )
    results_df = results_df.nlargest(n, "combined")

    results = []
    for _, r in results_df.iterrows():
        results.append(
            {
                "title": r["title"],
                "year": str(r.get("release_date", ""))[:4]
                if pd.notna(r.get("release_date"))
                else "N/A",
                "rating": float(r["vote_average"])
                if pd.notna(r["vote_average"])
                else 0,
                "overview": str(r["overview"])[:300]
                if pd.notna(r["overview"])
                else "",
                "genres": r["genres_processed"],
            }
        )

    movies_df["genre_match"] = 0
    return results


def build_context(message):
    """Build movie data context for the AI based on the user's message."""
    message_lower = message.lower().strip()
    context_movies = []

    genres = [
        "horror", "comedy", "action", "drama", "sci-fi", "science fiction",
        "romance", "thriller", "animation", "adventure", "fantasy", "mystery",
        "crime", "documentary", "war", "western", "musical", "family", "history",
    ]

    detected_genre = None
    for g in genres:
        if g in message_lower:
            detected_genre = g
            break

    similar_keywords = ["similar to", "like ", "movies like", "films like", "recommend"]
    is_similar = any(kw in message_lower for kw in similar_keywords)
    is_top = any(
        kw in message_lower for kw in ["top", "best", "highest rated", "top rated"]
    )
    is_info = any(
        kw in message_lower
        for kw in ["tell me about", "about ", "info", "details", "what is", "plot"]
    )
    is_random = any(
        kw in message_lower for kw in ["random", "surprise", "anything", "something"]
    )
    is_greeting = any(
        kw in message_lower
        for kw in ["hello", "hi", "hey", "help", "start", "what can"]
    )

    if is_greeting:
        context_movies = get_top_rated(None, 5)
    elif is_similar:
        title = message_lower
        for kw in [
            "similar to", "movies like", "films like",
            "recommend me", "recommend", "like",
        ]:
            title = title.replace(kw, "")
        title = title.strip().strip("\"'")
        if title:
            context_movies = get_similar_movies(title, 15)
            if not context_movies:
                context_movies = search_movies(title, 15)
    elif is_top and detected_genre:
        context_movies = get_top_rated(detected_genre, 15)
    elif is_top:
        context_movies = get_top_rated(None, 15)
    elif detected_genre:
        context_movies = get_top_rated(detected_genre, 15)
    elif is_info:
        title = message_lower
        for kw in [
            "tell me about", "about", "info about",
            "details about", "what is", "plot of", "plot",
        ]:
            title = title.replace(kw, "")
        title = title.strip().strip("\"'")
        if title:
            context_movies = search_movies(title, 5)
    elif is_random:
        import random

        sample = movies_df[movies_df["vote_count"] >= 100].sample(5)
        for _, row in sample.iterrows():
            context_movies.append(
                {
                    "title": row["title"],
                    "year": str(row.get("release_date", ""))[:4]
                    if pd.notna(row.get("release_date"))
                    else "N/A",
                    "rating": float(row["vote_average"])
                    if pd.notna(row["vote_average"])
                    else 0,
                    "overview": str(row["overview"])[:300]
                    if pd.notna(row["overview"])
                    else "",
                    "genres": row["genres_processed"],
                }
            )
    else:
        context_movies = search_movies(message, 15)

    # Fallback: handle natural language queries by keyword scoring
    if not context_movies:
        context_movies = search_movies_keywords(message, 15)

    if not context_movies:
        return "[MOVIE DATA]\nNo matching movies found in the database for this query.\n[/MOVIE DATA]"

    context = "[MOVIE DATA]\n"
    for m in context_movies:
        context += f"- {m['title']} ({m.get('year', 'N/A')}) | Rating: {m.get('rating', 0):.1f}/10"
        if m.get("genres"):
            context += f" | Genres: {m['genres']}"
        if m.get("overview"):
            context += f" | Overview: {m['overview']}"
        if m.get("cast"):
            context += f" | Cast: {', '.join(m['cast'])}"
        if m.get("runtime"):
            context += f" | Runtime: {m['runtime']} min"
        context += "\n"
    context += "[/MOVIE DATA]"

    return context


# ---- AI Provider Functions ----


def call_openrouter(system_content, messages, api_key, model):
    """Call OpenRouter API (OpenAI-compatible)."""
    api_key = api_key or OPENROUTER_API_KEY
    model = model or "openai/gpt-oss-20b:free"

    if not api_key:
        return "OpenRouter API key not configured. Please add your key in Settings."

    all_messages = [{"role": "system", "content": system_content}] + messages

    response = http_requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "MovieBot",
        },
        json={
            "model": model,
            "messages": all_messages,
            "max_tokens": 1024,
            "temperature": 0.7,
        },
        timeout=60,
    )

    if response.status_code != 200:
        error = ""
        try:
            error = response.json().get("error", {}).get("message", "")
        except Exception:
            pass
        raise Exception(f"OpenRouter error ({response.status_code}): {error}")

    return response.json()["choices"][0]["message"]["content"]


def call_ollama(system_content, messages, server_url, model):
    """Call Ollama API (OpenAI-compatible endpoint)."""
    server_url = (server_url or "http://localhost:11434").rstrip("/")

    if not model:
        return (
            "Ollama model not configured. "
            "Please set a model name in Settings (e.g. llama3, mistral)."
        )

    all_messages = [{"role": "system", "content": system_content}] + messages

    response = http_requests.post(
        f"{server_url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": model,
            "messages": all_messages,
            "temperature": 0.7,
        },
        timeout=120,
    )

    if response.status_code != 200:
        raise Exception(
            f"Ollama error ({response.status_code}): {response.text[:200]}"
        )

    return response.json()["choices"][0]["message"]["content"]


def call_anthropic(system_content, messages, api_key, model):
    """Call Anthropic Messages API."""
    model = model or "claude-sonnet-4-20250514"

    if not api_key:
        return "Anthropic API key not configured. Please add your key in Settings."

    anthropic_messages = []
    for m in messages:
        if m["role"] in ("user", "assistant"):
            anthropic_messages.append(
                {"role": m["role"], "content": m["content"]}
            )

    response = http_requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 1024,
            "system": system_content,
            "messages": anthropic_messages,
        },
        timeout=60,
    )

    if response.status_code != 200:
        error = ""
        try:
            error = response.json().get("error", {}).get("message", "")
        except Exception:
            pass
        raise Exception(f"Anthropic error ({response.status_code}): {error}")

    return response.json()["content"][0]["text"]


def call_gemini(system_content, messages, api_key, model):
    """Call Google Gemini API."""
    model = model or "gemini-2.0-flash"

    if not api_key:
        return "Gemini API key not configured. Please add your key in Settings."

    contents = []
    for m in messages:
        role = m["role"]
        if role == "assistant":
            role = "model"
        if role in ("user", "model"):
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

    response = http_requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
        headers={"Content-Type": "application/json"},
        json={
            "system_instruction": {"parts": [{"text": system_content}]},
            "contents": contents,
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1024,
            },
        },
        timeout=60,
    )

    if response.status_code != 200:
        error = ""
        try:
            error = response.json().get("error", {}).get("message", "")
        except Exception:
            pass
        raise Exception(f"Gemini error ({response.status_code}): {error}")

    return response.json()["candidates"][0]["content"]["parts"][0]["text"]


def call_ai(system_content, messages, provider_config):
    """Route to the correct AI provider."""
    provider = provider_config.get("name", "openrouter")
    api_key = provider_config.get("apiKey", "")
    model = provider_config.get("model", "")
    server_url = provider_config.get("serverUrl", "")

    if provider == "openrouter":
        return call_openrouter(system_content, messages, api_key, model)
    elif provider == "ollama":
        return call_ollama(system_content, messages, server_url, model)
    elif provider == "anthropic":
        return call_anthropic(system_content, messages, api_key, model)
    elif provider == "gemini":
        return call_gemini(system_content, messages, api_key, model)
    else:
        return "Unknown AI provider. Please configure a provider in Settings."


# ---- Routes ----


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "")
    history = data.get("history", [])
    provider_config = data.get("provider", {"name": "openrouter"})

    if not message:
        return jsonify({"reply": "Please type a message!"})

    context = build_context(message)
    system_content = SYSTEM_PROMPT + "\n\n" + context

    messages = []
    for msg in history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    try:
        reply = call_ai(system_content, messages, provider_config)
        return jsonify({"reply": reply})
    except http_requests.exceptions.Timeout:
        return jsonify(
            {"reply": "The request timed out. Please try again in a moment."}
        )
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})


@app.route("/api/test-connection", methods=["POST"])
def test_connection():
    data = request.json
    provider_config = data.get("provider", {})

    try:
        reply = call_ai(
            "Reply with exactly: Connected successfully!",
            [{"role": "user", "content": "Test connection"}],
            provider_config,
        )
        return jsonify({"success": True, "message": reply[:100]})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)[:200]})


@app.route("/api/models", methods=["POST"])
def get_models():
    data = request.json
    provider = data.get("provider", "")

    if provider == "ollama":
        server_url = data.get("serverUrl", "http://localhost:11434").rstrip("/")
        try:
            res = http_requests.get(f"{server_url}/api/tags", timeout=10)
            if res.status_code == 200:
                models = [m["name"] for m in res.json().get("models", [])]
                return jsonify({"success": True, "models": models})
            else:
                return jsonify(
                    {"success": False, "message": f"Ollama returned status {res.status_code}"}
                )
        except Exception as e:
            return jsonify({"success": False, "message": str(e)[:200]})

    elif provider == "openrouter":
        try:
            res = http_requests.get(
                "https://openrouter.ai/api/v1/models", timeout=15
            )
            if res.status_code == 200:
                models = []
                for m in res.json().get("data", []):
                    models.append(
                        {"id": m["id"], "name": m.get("name", m["id"])}
                    )
                return jsonify({"success": True, "models": models})
            else:
                return jsonify(
                    {"success": False, "message": f"OpenRouter returned status {res.status_code}"}
                )
        except Exception as e:
            return jsonify({"success": False, "message": str(e)[:200]})

    elif provider == "anthropic":
        models = [
            {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4"},
            {"id": "claude-haiku-4-20250414", "name": "Claude Haiku 4"},
            {"id": "claude-3-7-sonnet-20250219", "name": "Claude 3.7 Sonnet"},
            {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet"},
            {"id": "claude-3-5-haiku-20241022", "name": "Claude 3.5 Haiku"},
            {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus"},
            {"id": "claude-3-sonnet-20240229", "name": "Claude 3 Sonnet"},
            {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku"},
        ]
        return jsonify({"success": True, "models": models})

    elif provider == "gemini":
        models = [
            {"id": "gemini-2.5-pro-preview-05-06", "name": "Gemini 2.5 Pro Preview"},
            {"id": "gemini-2.5-flash-preview-04-17", "name": "Gemini 2.5 Flash Preview"},
            {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash"},
            {"id": "gemini-2.0-flash-lite", "name": "Gemini 2.0 Flash Lite"},
            {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro"},
            {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash"},
            {"id": "gemini-1.5-flash-8b", "name": "Gemini 1.5 Flash 8B"},
        ]
        return jsonify({"success": True, "models": models})

    return jsonify({"success": False, "message": "Unknown provider"})


@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify(
        {
            "total_movies": len(movies_df) if movies_df is not None else 0,
        }
    )


if __name__ == "__main__":
    load_data()
    print("=" * 50)
    print("  MovieBot - AI Movie Recommendation Chatbot")
    print("=" * 50)
    print(f"  Database  : {len(movies_df)} movies")
    print(f"  Providers : OpenRouter, Ollama, Anthropic, Gemini")
    print(f"  Server    : http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
