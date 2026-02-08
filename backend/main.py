import os

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from .rag import LocalVectorStore, build_prompt, ollama_generate

app = FastAPI(title="RAG Mini App (Local)")

STORE = LocalVectorStore(data_dir="data")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 4


@app.get("/health")
def health():
    return {"ok": True, "chunks": len(STORE.chunks), "model": OLLAMA_MODEL}


@app.post("/reset")
def reset():
    STORE.reset()
    return {"ok": True}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    os.makedirs("data/uploads", exist_ok=True)
    save_path = os.path.join("data/uploads", file.filename)

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    result = STORE.add_document(save_path, source_name=file.filename)
    return {"ok": True, **result}


@app.post("/query")
def query(req: QueryRequest):
    retrieved = STORE.search(req.question, top_k=req.top_k)
    prompt, sources = build_prompt(req.question, retrieved)

    answer = ollama_generate(prompt, model=OLLAMA_MODEL, temperature=0.2)
    return {"ok": True, "answer": answer, "sources": sources}
