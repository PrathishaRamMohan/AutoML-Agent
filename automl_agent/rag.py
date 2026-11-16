"""Agentic RAG over the ML best-practice knowledge base (knowledge_base/*.md).

* Documents are chunked by '## ' section.
* Default embedder: TF-IDF (no downloads, works offline).
  Set EMBEDDINGS=minilm in .env to use sentence-transformers (all-MiniLM-L6-v2).
* agentic_retrieve(): the agent issues several targeted queries; if the best
  hit is weak, it reformulates the query (via LLM when available) and retries.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from automl_agent import config


@dataclass
class Chunk:
    source: str
    section: str
    text: str


def load_chunks() -> list[Chunk]:
    chunks = []
    for path in sorted(config.KB_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        title = raw.splitlines()[0].lstrip("# ").strip() if raw else path.stem
        for part in re.split(r"\n(?=## )", raw):
            lines = part.strip().splitlines()
            if not lines:
                continue
            section = lines[0].lstrip("# ").strip()
            body = "\n".join(lines[1:]).strip()
            if body:
                chunks.append(Chunk(source=path.name, section=f"{title} > {section}", text=body))
    return chunks


class VectorStore:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        docs = [f"{c.section}. {c.text}" for c in chunks]
        self.kind = "tfidf"
        if config.EMBEDDINGS == "minilm":
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                self.matrix = self.model.encode(docs, normalize_embeddings=True)
                self.kind = "minilm"
            except Exception:  # fall back silently if not installed
                pass
        if self.kind == "tfidf":
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
            m = self.vectorizer.fit_transform(docs).toarray()
            self.matrix = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)

    def _embed(self, query: str) -> np.ndarray:
        if self.kind == "minilm":
            return self.model.encode([query], normalize_embeddings=True)[0]
        v = self.vectorizer.transform([query]).toarray()[0]
        return v / (np.linalg.norm(v) + 1e-12)

    def search(self, query: str, k: int = 3) -> list[dict]:
        sims = self.matrix @ self._embed(query)
        top = np.argsort(-sims)[:k]
        return [
            {"source": self.chunks[i].source, "section": self.chunks[i].section,
             "text": self.chunks[i].text, "score": round(float(sims[i]), 3)}
            for i in top
        ]


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    return VectorStore(load_chunks())


def _reformulate(query: str) -> str:
    """Ask the LLM to rewrite a query that retrieved weak results; fallback = keyword broadening."""
    from automl_agent.llm import ask_llm

    rewritten = ask_llm(
        "Rewrite this search query for a machine-learning best-practices knowledge base. "
        "Return only the improved query, max 12 words.",
        query,
        fallback="",
    ).strip().strip('"')
    if rewritten:
        return rewritten
    return " ".join(w for w in re.findall(r"[a-zA-Z]+", query) if len(w) > 3) + " machine learning best practice"


def agentic_retrieve(queries: list[str], k: int = 2, min_score: float = 0.12) -> tuple[list[dict], list[str]]:
    """Run several queries, self-correct weak ones, de-duplicate. Returns (hits, retrieval_log)."""
    store = get_store()
    seen, hits, log = set(), [], []
    for q in queries:
        results = store.search(q, k)
        best = results[0]["score"] if results else 0
        log.append(f"query='{q}' best_score={best}")
        if best < min_score:
            q2 = _reformulate(q)
            results = store.search(q2, k)
            log.append(f"  weak result -> reformulated to '{q2}' best_score={results[0]['score']}")
        for r in results:
            key = (r["source"], r["section"])
            if key not in seen and r["score"] >= min_score / 2:
                seen.add(key)
                hits.append(r)
    return hits, log


def format_context(hits: list[dict], max_chars: int = 3500) -> str:
    out, total = [], 0
    for i, h in enumerate(hits, 1):
        block = f"[{i}] {h['section']} ({h['source']})\n{h['text']}"
        if total + len(block) > max_chars:
            break
        out.append(block)
        total += len(block)
    return "\n\n".join(out)
