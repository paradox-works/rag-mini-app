from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import faiss
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


def _clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return _clean_text(f.read())


def read_pdf_file(path: str) -> str:
    reader = PdfReader(path)
    parts: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)
    return _clean_text("\n\n".join(parts))


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """
    Simple chunker that tries to keep chunks readable on CPU-only machines.
    chunk_size/overlap are in characters (approx).
    """
    text = _clean_text(text)
    if not text:
        return []

    # Split into sentences-ish
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    cur = ""

    for sent in sentences:
        if not sent:
            continue
        if len(cur) + len(sent) + 1 <= chunk_size:
            cur = (cur + " " + sent).strip()
        else:
            if cur:
                chunks.append(cur)
            # start next chunk with overlap from the end of previous chunk
            if overlap > 0 and chunks:
                tail = chunks[-1][-overlap:]
                cur = (tail + " " + sent).strip()
            else:
                cur = sent.strip()

    if cur:
        chunks.append(cur)

    # Deduplicate tiny chunks
    chunks = [c for c in chunks if len(c) >= 50]
    return chunks


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norm


@dataclass
class DocChunk:
    text: str
    source: str  # filename
    chunk_id: int


class LocalVectorStore:
    """
    Minimal FAISS cosine-sim vector store, single-user demo.
    Persists index + metadata to ./data by default.
    """

    def __init__(self, data_dir: str = "data", embed_model_name: str = "all-MiniLM-L6-v2"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        self.embedder = SentenceTransformer(embed_model_name)
        self.index: Optional[faiss.Index] = None
        self.dim: Optional[int] = None
        self.chunks: List[DocChunk] = []

        self._index_path = os.path.join(self.data_dir, "index.faiss")
        self._meta_path = os.path.join(self.data_dir, "meta.json")

        self._try_load()

    def _try_load(self) -> None:
        if os.path.exists(self._index_path) and os.path.exists(self._meta_path):
            try:
                self.index = faiss.read_index(self._index_path)
                with open(self._meta_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self.chunks = [DocChunk(**c) for c in raw.get("chunks", [])]
                self.dim = int(raw.get("dim", 0)) or None
            except Exception:
                # If load fails, start fresh
                self.index = None
                self.chunks = []
                self.dim = None

    def _save(self) -> None:
        if self.index is None or self.dim is None:
            return
        faiss.write_index(self.index, self._index_path)
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {"dim": self.dim, "chunks": [c.__dict__ for c in self.chunks]},
                f,
                ensure_ascii=False,
                indent=2,
            )

    def reset(self) -> None:
        self.index = None
        self.dim = None
        self.chunks = []
        for p in [self._index_path, self._meta_path]:
            if os.path.exists(p):
                os.remove(p)

    def _embed(self, texts: List[str]) -> np.ndarray:
        emb = self.embedder.encode(texts, normalize_embeddings=False, show_progress_bar=False)
        emb = np.array(emb, dtype=np.float32)
        emb = _l2_normalize(emb)  # cosine similarity via inner product
        return emb

    def add_document(self, file_path: str, source_name: Optional[str] = None) -> Dict:
        source = source_name or os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            text = read_pdf_file(file_path)
        else:
            text = read_text_file(file_path)

        chunks = chunk_text(text)
        if not chunks:
            return {"added_chunks": 0, "source": source}

        new_chunks: List[DocChunk] = []
        start_id = len(self.chunks)
        for i, c in enumerate(chunks):
            new_chunks.append(DocChunk(text=c, source=source, chunk_id=start_id + i))

        vectors = self._embed([c.text for c in new_chunks])
        dim = vectors.shape[1]

        if self.index is None:
            # Cosine similarity = inner product on normalized vectors
            self.index = faiss.IndexFlatIP(dim)
            self.dim = dim
        else:
            if self.dim != dim:
                raise ValueError(f"Embedding dim mismatch: expected {self.dim}, got {dim}")

        self.index.add(vectors)
        self.chunks.extend(new_chunks)
        self._save()

        return {"added_chunks": len(new_chunks), "source": source}

    def search(self, query: str, top_k: int = 4) -> List[Tuple[DocChunk, float]]:
        if self.index is None or not self.chunks:
            return []

        qv = self._embed([query])
        scores, idxs = self.index.search(qv, top_k)
        results: List[Tuple[DocChunk, float]] = []
        for i, score in zip(idxs[0].tolist(), scores[0].tolist()):
            if i < 0 or i >= len(self.chunks):
                continue
            results.append((self.chunks[i], float(score)))
        return results


def ollama_generate(prompt: str, model: str = "llama3.2", temperature: float = 0.2) -> str:
    """
    Requires Ollama running locally: http://localhost:11434
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def build_prompt(question: str, retrieved: List[Tuple[DocChunk, float]]) -> Tuple[str, List[Dict]]:
    """
    Returns (prompt, sources_list)
    sources_list includes source + preview text for UI.
    """
    sources: List[Dict] = []
    context_lines: List[str] = []

    for n, (chunk, score) in enumerate(retrieved, start=1):
        preview = chunk.text[:240] + ("..." if len(chunk.text) > 240 else "")
        sources.append(
            {
                "n": n,
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                "score": round(score, 4),
                "preview": preview,
            }
        )
        context_lines.append(f"[{n}] Source: {chunk.source} (chunk {chunk.chunk_id})\n{chunk.text}")

    context = "\n\n".join(context_lines) if context_lines else "None"

    prompt = f"""
You are a helpful assistant. Answer the question ONLY using the context below.
If the context does not contain the answer, say: "I don't know based on the provided documents."
When you use information from the context, cite it like [1], [2], etc.

Question:
{question}

Context:
{context}

Answer:
""".strip()

    return prompt, sources
