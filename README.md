# rag-mini-app
RAG app: chat with your documents using citations.

## What this is
A simple local RAG demo:
- **FastAPI backend** for ingest + search + answer
- **Streamlit frontend** for upload + chat UI
- Uses a local LLM via **Ollama**

## Project structure
- `backend/` — FastAPI + RAG logic
- `frontend/` — Streamlit UI
- `requirements.txt` — Python dependencies

## Run locally (Mac/Linux/Windows)
### 1) Install dependencies
```bash
python -m venv .venv
# Mac/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
