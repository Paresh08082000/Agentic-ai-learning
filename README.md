# Agentic AI Learning — LangGraph Chatbot

A multi-utility chatbot built with **LangGraph**, **Groq LLM**, and **Streamlit**.  
Features: web search, RAG over uploaded PDFs, stock price lookup, and calculator — all with persistent conversation history.

---

## Project Structure

```
Agentic-ai-learning/
├── chatbot/
│   ├── streamlit_frontend_rag.py       # Main app (PDF RAG + tools)
│   ├── langgraph_rag_backend.py        # LangGraph backend with RAG
│   ├── langgraph_tool_backend.py       # LangGraph backend with tools only
│   ├── streamlit_frontend_tool.py      # Tool-only frontend
│   └── requirements.txt
├── langgraph-tutorials/
├── .gitignore
└── README.md
```

---

## Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/Paresh08082000/Agentic-ai-learning.git
cd Agentic-ai-learning
```

### 2. Create and activate a virtual environment
```bash
python -m venv myenv
source myenv/bin/activate        # Mac/Linux
myenv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r chatbot/requirements.txt
```

### 4. Set up environment variables
Create a `.env` file inside the `chatbot/` folder:
```
GROQ_API_KEY=your_groq_api_key_here
```
> Get a free API key at https://console.groq.com

### 5. Run the app
```bash
streamlit run chatbot/streamlit_frontend_rag.py
```

---

## Deploy on Streamlit Community Cloud (Free)

### 1. Push this repo to GitHub (already done)

### 2. Go to [share.streamlit.io](https://share.streamlit.io)
Sign in with your GitHub account.

### 3. Click "New app" and fill in:
| Field | Value |
|---|---|
| Repository | `Paresh08082000/Agentic-ai-learning` |
| Branch | `main` |
| Main file path | `chatbot/streamlit_frontend_rag.py` |

### 4. Add your secret key
Click **Advanced settings → Secrets** and paste:
```toml
GROQ_API_KEY = "your_actual_groq_api_key_here"
```

### 5. Click Deploy
The app will be live at a `*.streamlit.app` URL in ~2-3 minutes.

---

## .gitignore explained

The following are excluded from version control:

| Pattern | Reason |
|---|---|
| `.env` | Contains secret API keys — never commit this |
| `myenv/`, `venv/` | Virtual environment — install locally via `requirements.txt` |
| `__pycache__/`, `*.pyc` | Python bytecode — auto-generated |
| `*.db`, `*.db-shm`, `*.db-wal` | SQLite chat history — resets on each deploy anyway |
| `.DS_Store` | macOS system file |

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Agent framework | LangGraph |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector store | FAISS (in-memory, per thread) |
| Web search | DuckDuckGo |
| Frontend | Streamlit |
| Conversation memory | LangGraph SQLite checkpointer |

## Deployed app link

https://pnb-chatbot.streamlit.app/
