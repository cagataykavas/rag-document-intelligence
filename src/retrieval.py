from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    source: str = "memory"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str


def chunk_documents(documents: Iterable[Document], words_per_chunk: int = 90, overlap: int = 20) -> list[Chunk]:
    if overlap >= words_per_chunk:
        raise ValueError("overlap must be smaller than words_per_chunk")
    chunks: list[Chunk] = []
    step = words_per_chunk - overlap
    for document in documents:
        words = document.text.split()
        for index, start in enumerate(range(0, len(words), step)):
            text = " ".join(words[start:start + words_per_chunk]).strip()
            if text:
                chunks.append(Chunk(f"{document.doc_id}:{index}", document.doc_id, document.source, text))
    return chunks


class SparseRetriever:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.chunks: list[Chunk] = []
        self.matrix = None

    def fit(self, chunks: list[Chunk]) -> "SparseRetriever":
        if not chunks:
            raise ValueError("at least one chunk is required")
        self.chunks = chunks
        self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in chunks])
        return self

    def search(self, query: str, k: int = 5) -> list[dict]:
        if self.matrix is None:
            raise RuntimeError("fit must be called before search")
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranking = scores.argsort()[::-1][:k]
        return [{"score": float(scores[i]), "chunk_id": self.chunks[i].chunk_id, "doc_id": self.chunks[i].doc_id, "source": self.chunks[i].source, "text": self.chunks[i].text} for i in ranking]


def reciprocal_rank(results: list[dict], relevant_doc_id: str) -> float:
    for rank, result in enumerate(results, start=1):
        if result["doc_id"] == relevant_doc_id:
            return 1.0 / rank
    return 0.0
