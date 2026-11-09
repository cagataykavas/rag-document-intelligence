from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

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


@dataclass(frozen=True)
class RetrievalAdmissionPolicy:
    """Bound and filter evidence before it can enter the generation path."""

    min_fused_score: float = 1e-12
    min_relative_score: float = 0.0
    max_query_chars: int = 4_096
    max_results: int = 100

    def __post_init__(self) -> None:
        if (
            type(self.min_fused_score) not in {int, float}
            or not isfinite(self.min_fused_score)
            or self.min_fused_score < 0
        ):
            raise ValueError("min_fused_score must be finite and non-negative")
        if (
            type(self.min_relative_score) not in {int, float}
            or not isfinite(self.min_relative_score)
            or not 0 <= self.min_relative_score <= 1
        ):
            raise ValueError("min_relative_score must be finite and in [0, 1]")
        if type(self.max_query_chars) is not int or not 1 <= self.max_query_chars <= 100_000:
            raise ValueError("max_query_chars must be an integer in [1, 100000]")
        if type(self.max_results) is not int or not 1 <= self.max_results <= 10_000:
            raise ValueError("max_results must be an integer in [1, 10000]")


class HybridSparseRetriever:
    """Fuse word and character TF-IDF for robust local retrieval.

    Word n-grams favor semantic lexical matches while character n-grams help
    with identifiers, spelling variants, compound tokens, and requirement IDs.
    This is deliberately described as hybrid *sparse* retrieval, not as a dense
    embedding system.
    """

    def __init__(
        self,
        weights: HybridWeights | None = None,
        admission_policy: RetrievalAdmissionPolicy | None = None,
    ) -> None:
        self.weights = weights or HybridWeights()
        self.admission_policy = admission_policy or RetrievalAdmissionPolicy()
        self.word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        self.chunks: list[Chunk] = []
        self.word_matrix = None
        self.char_matrix = None

    def fit(self, chunks: list[Chunk]) -> "HybridSparseRetriever":
        if not chunks:
            raise ValueError("at least one chunk is required")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("chunk IDs must be unique")
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
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if len(query) > self.admission_policy.max_query_chars:
            raise ValueError("query exceeds the configured character budget")
        if any(ord(character) < 32 and character not in "\t\n\r" for character in query):
            raise ValueError("query contains unsupported control characters")
        if type(k) is not int or not 1 <= k <= self.admission_policy.max_results:
            raise ValueError("k must be a positive integer within the result budget")
        if (
            type(diversity_penalty) not in {int, float}
            or not isfinite(diversity_penalty)
            or diversity_penalty < 0
        ):
            raise ValueError("diversity_penalty must be finite and non-negative")

        word_weight, char_weight = self.weights.normalized()
        word_query = self.word_vectorizer.transform([query])
        char_query = self.char_vectorizer.transform([query])
        word_scores = cosine_similarity(word_query, self.word_matrix).ravel()
        char_scores = cosine_similarity(char_query, self.char_matrix).ravel()
        fused = word_weight * word_scores + char_weight * char_scores
        if not np.all(np.isfinite(fused)):
            raise ValueError("retrieval produced non-finite scores")

        top_score = float(np.max(fused))
        admission_threshold = max(
            self.admission_policy.min_fused_score,
            top_score * self.admission_policy.min_relative_score,
        )
        candidates = [
            index
            for index in range(len(self.chunks))
            if float(fused[index]) >= admission_threshold
            and (float(word_scores[index]) > 0 or float(char_scores[index]) > 0)
        ]
        selected: list[int] = []
        doc_counts: dict[str, int] = {}
        target_count = min(k, len(candidates))
        while candidates and len(selected) < target_count:
            index = min(
                candidates,
                key=lambda candidate: (
                    -(
                        float(fused[candidate])
                        - diversity_penalty * doc_counts.get(self.chunks[candidate].doc_id, 0)
                    ),
                    -float(fused[candidate]),
                    -float(word_scores[candidate]),
                    -float(char_scores[candidate]),
                    self.chunks[candidate].chunk_id,
                ),
            )
            candidates.remove(index)
            selected.append(index)
            doc_id = self.chunks[index].doc_id
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

        return [
            {
                "rank": rank,
                "score": float(fused[index]),
                "word_score": float(word_scores[index]),
                "character_score": float(char_scores[index]),
                "admission_threshold": admission_threshold,
                "chunk_id": self.chunks[index].chunk_id,
                "doc_id": self.chunks[index].doc_id,
                "source": self.chunks[index].source,
                "text": self.chunks[index].text,
            }
            for rank, index in enumerate(selected, start=1)
        ]
