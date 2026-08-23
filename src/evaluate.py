from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from src.engine import RAGEngine
from src.prompting import PromptMode
from src.store import DocumentStore


@dataclass(frozen=True)
class EvalCase:
    question: str
    relevant_doc_id: str
    expected_phrase: str | None = None


@dataclass(frozen=True)
class EvalMetrics:
    cases: int
    recall_at_k: float
    mean_reciprocal_rank: float
    citation_doc_accuracy: float
    answer_phrase_accuracy: float
    average_latency_ms: float


def reciprocal_rank(results: list[dict], relevant_doc_id: str) -> float:
    for rank, result in enumerate(results, start=1):
        if result["doc_id"] == relevant_doc_id:
            return 1.0 / rank
    return 0.0


def evaluate(
    store: DocumentStore,
    cases: list[EvalCase],
    *,
    k: int = 5,
    prompt_mode: PromptMode = PromptMode.ZERO_SHOT,
) -> EvalMetrics:
    engine = RAGEngine(store)
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    citation_matches: list[float] = []
    phrase_matches: list[float] = []
    latencies: list[float] = []

    chunk_to_doc = {chunk.chunk_id: chunk.doc_id for chunk in store.chunks()}
    for case in cases:
        started = time.perf_counter()
        retrieval = engine.retrieve(case.question, k=k)
        answer = engine.answer(case.question, k=k, mode=prompt_mode, use_llm=False)
        latencies.append((time.perf_counter() - started) * 1000)

        recalls.append(float(any(item["doc_id"] == case.relevant_doc_id for item in retrieval)))
        reciprocal_ranks.append(reciprocal_rank(retrieval, case.relevant_doc_id))
        cited_docs = {chunk_to_doc.get(chunk_id) for chunk_id in answer.citations}
        citation_matches.append(float(case.relevant_doc_id in cited_docs))
        if case.expected_phrase:
            phrase_matches.append(float(case.expected_phrase.lower() in answer.answer.lower()))
        else:
            phrase_matches.append(1.0)

    if not cases:
        return EvalMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return EvalMetrics(
        cases=len(cases),
        recall_at_k=mean(recalls),
        mean_reciprocal_rank=mean(reciprocal_ranks),
        citation_doc_accuracy=mean(citation_matches),
        answer_phrase_accuracy=mean(phrase_matches),
        average_latency_ms=mean(latencies),
    )


def load_cases(path: Path) -> list[EvalCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [EvalCase(**item) for item in payload]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval/citation behavior over an indexed SQLite corpus.")
    parser.add_argument("--database", default="data/rag.db")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--prompt-mode", choices=[item.value for item in PromptMode], default=PromptMode.ZERO_SHOT.value)
    args = parser.parse_args()
    metrics = evaluate(
        DocumentStore(args.database),
        load_cases(args.cases),
        k=args.k,
        prompt_mode=PromptMode(args.prompt_mode),
    )
    print(json.dumps(asdict(metrics), indent=2))


if __name__ == "__main__":
    main()
