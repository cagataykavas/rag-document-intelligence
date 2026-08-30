from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval import Chunk


@dataclass(frozen=True)
class HybridWeights:
    word: float = 0.65
    character: float = 0.35

    def normalized(self) -> tuple[float, float]:
        if self.word < 0 or self.character < 0:
            raise ValueError("retrieval weights cannot be negative")
        total = self.word + self.character
        if total <= 0:
            raise ValueError("at least one retrieval weight must be positive")
        return self.word / total, self.character / total


class HybridSparseRetriever:
    """Fuse word and character TF-IDF for robust local retrieval.

    Word n-grams favor semantic lexical matches while character n-grams help
    with identifiers, spelling variants, compound tokens, and requirement IDs.
    This is deliberately described as hybrid *sparse* retrieval, not as a dense
    embedding system.
    """

    def __init__(self, weights: HybridWeights | None = None) -> None:
        self.weights = weights or HybridWeights()
        self.word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        self.chunks: list[Chunk] = []
        self.word_matrix = None
        self.char_matrix = None

    def fit(self, chunks: list[Chunk]) -> "HybridSparseRetriever":
        if not chunks:
            raise ValueError("at least one chunk is required")
        self.chunks = list(chunks)
        corpus = [chunk.text for chunk in chunks]
        self.word_matrix = self.word_vectorizer.fit_transform(corpus)
        self.char_matrix = self.char_vectorizer.fit_transform(corpus)
        return self

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        diversity_penalty: float = 0.08,
    ) -> list[dict]:
        if self.word_matrix is None or self.char_matrix is None:
            raise RuntimeError("fit must be called before search")
        if k < 1:
            raise ValueError("k must be positive")
        if diversity_penalty < 0:
            raise ValueError("diversity_penalty cannot be negative")

        word_weight, char_weight = self.weights.normalized()
        word_query = self.word_vectorizer.transform([query])
        char_query = self.char_vectorizer.transform([query])
        word_scores = cosine_similarity(word_query, self.word_matrix).ravel()
        char_scores = cosine_similarity(char_query, self.char_matrix).ravel()
        fused = word_weight * word_scores + char_weight * char_scores

        candidates = list(np.argsort(fused)[::-1])
        selected: list[int] = []
        doc_counts: dict[str, int] = {}
        while candidates and len(selected) < min(k, len(candidates)):
            best_position = max(
                range(len(candidates)),
                key=lambda position: (
                    float(fused[candidates[position]])
                    - diversity_penalty * doc_counts.get(self.chunks[candidates[position]].doc_id, 0)
                ),
            )
            index = candidates.pop(best_position)
            selected.append(index)
            doc_id = self.chunks[index].doc_id
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

        return [
            {
                "rank": rank,
                "score": float(fused[index]),
                "word_score": float(word_scores[index]),
                "character_score": float(char_scores[index]),
                "chunk_id": self.chunks[index].chunk_id,
                "doc_id": self.chunks[index].doc_id,
                "source": self.chunks[index].source,
                "text": self.chunks[index].text,
            }
            for rank, index in enumerate(selected, start=1)
        ]
