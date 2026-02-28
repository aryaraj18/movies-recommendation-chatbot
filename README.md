# 🎬 MovieBot — AI-Powered Movie Recommendation Chatbot

An intelligent movie recommendation chatbot that uses AI to understand natural language queries and suggest movies from a database of **42,000+ titles**. Ask for movies by genre, theme, mood, or even complex descriptions like *"post-apocalyptic dystopian world with a teen protagonist"* — and get relevant recommendations instantly.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-2.3+-green?logo=flask)

## ✨ Features

- **Natural Language Understanding** — Ask for movies the way you'd ask a friend. The chatbot parses complex queries, extracts keywords, and finds matching movies by theme, setting, genre, cast, and more.
- **42,000+ Movie Database** — Powered by a comprehensive movie metadata dataset with titles, genres, overviews, ratings, cast, and runtime.
- **Multi-Provider AI Support** — Choose your preferred AI backend:
  - **OpenRouter** (GPT, Mistral, LLaMA, and many more — includes free models)
  - **Ollama** (run local models — fully offline/private)
  - **Anthropic** (Claude)
  - **Google Gemini**
- **Smart Search** — Keyword scoring with synonym expansion, genre detection, and multi-term matching to surface the most relevant results.
- **Modern Chat UI** — Dark glassmorphism design with markdown rendering, typing indicators, suggestion chips, and responsive layout.
- **In-App Settings** — Configure your AI provider, API keys, and model selection (with searchable dropdowns) — all from the UI.

## 🖼️ Screenshots
<img width="1145" height="957" alt="Image" src="https://github.com/user-attachments/assets/14dbc78d-3fbc-4929-9fa9-62bf34a9c807" />
<img width="665" height="747" alt="Image" src="https://github.com/user-attachments/assets/4232ba1e-46ad-4d32-8082-3fc4345e7bbb" />

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- An API key from one of the supported providers, **or** [Ollama](https://ollama.com/) installed locally for offline use

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/aryaraj18/movies-recommendation-chatbot
   cd movies-recommendation
   ```

   > The movie dataset (42,000+ movies) is already included in the `data/` folder — no extra downloads needed.

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   Create a `.env` file in the project root:

   ```
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

   > You can get a free API key from [OpenRouter](https://openrouter.ai/). This is optional if you plan to use Ollama or another provider.

5. **Run the app**

   ```bash
   python app.py
   ```

6. **Open in browser**

   Navigate to [http://localhost:5000](http://localhost:5000)

## 💬 Usage Examples

| Query | What It Does |
|---|---|
| *"recommend sci-fi movies"* | Returns top-rated science fiction movies |
| *"movies like Interstellar"* | Finds movies with similar genres |
| *"post-apocalyptic dystopian teen protagonist"* | Searches by keywords across overviews & genres |
| *"top rated horror"* | Returns highest-rated horror movies |
| *"tell me about Inception"* | Shows details about a specific movie |
| *"surprise me"* | Returns random popular movies |

## 🛠️ Tech Stack

- **Backend:** Python, Flask, Pandas
- **Frontend:** HTML, CSS, JavaScript (vanilla — no frameworks)
- **AI Integration:** OpenRouter, Ollama, Anthropic, Google Gemini
- **Data:** The Movies Dataset (Kaggle) — 42,000+ movies

## 📁 Project Structure

```
Movies Recommendation/
├── app.py                # Flask backend — routes, search logic, AI provider calls
├── index.html            # Frontend — chat UI, settings modal, CSS & JS
├── requirements.txt      # Python dependencies
├── .env                  # API keys (not tracked in git)
└── data/
    ├── movies_metadata.csv   # Movie titles, genres, overviews, ratings
    └── credits.csv           # Cast and crew data
```

## ⚙️ How It Works

1. **User sends a message** → The frontend posts it to `/api/chat`.
2. **Query analysis** → `build_context()` detects the intent — genre request, similarity search, info lookup, or natural language description.
3. **Movie search** → Relevant movies are found via genre filtering, title matching, or keyword scoring with synonym expansion.
4. **AI generation** → The matched movies are sent as context to the selected AI model, which is constrained to movie-only topics via a system prompt.
5. **Response** → The AI returns a conversational, markdown-formatted reply with recommendations.

## 👤 Author

- **Name**: Arya Raj
- **Role**: Data Science Student (Semester 4 rn xd)
- **Goal**: AI Engineer / Data Scientist

---

⭐ If you found this helpful, please star the repository!
