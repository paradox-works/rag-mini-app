import os
import re
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from pydantic import BaseModel, Field

from .rag import LocalVectorStore, build_prompt, ollama_generate

app = FastAPI(title="RAG Mini App (Local)")

# Storage
DATA_DIR = os.getenv("DATA_DIR", "data")
UPLOAD_DIR = Path(DATA_DIR) / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

STORE = LocalVectorStore(data_dir=DATA_DIR)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Safety limits (tune as needed)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))  # 10 MB
ALLOWED_EXTS = {".pdf", ".txt"}

# Optional API key (recommended if you ever run on LAN)
# If ADMIN_KEY is empty, endpoints are open (local demo).
ADMIN_KEY = os.getenv("ADMIN_KEY", "").strip() or None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=12)


def _require_key(x_api_key: Optional[str]) -> None:
    if ADMIN_KEY is None:
        return
    if x_api_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _safe_display_name(name: str) -> str:
    # For UI only — NOT for filesystem paths
    name = (name or "upload").strip()
    name = re.sub(r"[^a-zA-Z0-9._ -]+", "_", name)
    return name[:200] or "upload"


@app.get("/health")
def health():
    return {"ok": True, "chunks": len(STORE.chunks), "model": OLLAMA_MODEL}


@app.post("/reset")
def reset(x_api_key: Optional[str] = Header(default=None)):
    _require_key(x_api_key)
    STORE.reset()

    # Also remove uploaded files
    if UPLOAD_DIR.exists():
        for p in UPLOAD_DIR.glob("*"):
            try:
                if p.is_file():
                    p.unlink()
            except Exception:
                pass

    return {"ok": True}


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(default=None),
):
    _require_key(x_api_key)

    # Validate extension
    display_name = _safe_display_name(file.filename)
    ext = Path(display_name).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail="Only .pdf or .txt files are allowed")

    # Server-side filename (prevents path traversal & collisions)
    server_name = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / server_name

    # Stream to disk with size limit
    size = 0
    try:
        with save_path.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="File too large")
                f.write(chunk)
    finally:
        await file.close()

    # Add to vector store (store uses display_name as source label)
    result = STORE.add_document(str(save_path), source_name=display_name)
    return {"ok": True, **result}


@app.post("/query")
def query(req: QueryRequest, x_api_key: Optional[str] = Header(default=None)):
    # Optional: protect query too by uncommenting next line
    # _require_key(x_api_key)

    retrieved = STORE.search(req.question, top_k=req.top_k)
    prompt, sources = build_prompt(req.question, retrieved)

    answer = ollama_generate(prompt, model=OLLAMA_MODEL, temperature=0.2)
    return {"ok": True, "answer": answer, "sources": sources}
