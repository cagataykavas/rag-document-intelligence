from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from src.engine import RAGEngine
from src.prompting import PromptMode
from src.retrieval_eval import RetrievalCase, evaluate_rankings
from src.store import DocumentStore


@dataclass(frozen=True)
class EvalCase:
    question: str
    relevant_doc_id: str
    expected_phrase: str | None = None


@dataclass(frozen=True)
class EvalMetrics:
    cases: int
    hit_rate_at_k: float
    recall_at_k: float
    precision_at_k: float
    mean_reciprocal_rank: float
    ndcg_at_k: float
    citation_doc_accuracy: float
    citation_validity_rate: float
    lexical_grounding_rate: float
    answer_phrase_accuracy: float
    average_latency_ms: float


def evaluate(
    store: DocumentStore,
    cases: list[EvalCase],
    *,
    k: int = 5,
    prompt_mode: PromptMode = PromptMode.ZERO_SHOT,
) -> EvalMetrics:
    engine = RAGEngine(store)
    rankings: list[list[dict]] = []
    retrieval_cases: list[RetrievalCase] = []
    citation_matches: list[float] = []
    citation_validity: list[float] = []
    grounding_rates: list[float] = []
    phrase_matches: list[float] = []
    latencies: list[float] = []

    chunk_to_doc = {chunk.chunk_id: chunk.doc_id for chunk in store.chunks()}
    for case in cases:
        started = time.perf_counter()
        retrieval = engine.retrieve(case.question, k=k)
        answer = engine.answer(case.question, k=k, mode=prompt_mode, use_llm=False)
        latencies.append((time.perf_counter() - started) * 1000)

        rankings.append(retrieval)
        retrieval_cases.append(
            RetrievalCase(case.question, frozenset({case.relevant_doc_id}))
        )
        cited_docs = {chunk_to_doc.get(chunk_id) for chunk_id in answer.citations}
        citation_matches.append(float(case.relevant_doc_id in cited_docs))
        citation_validity.append(float(answer.citation_audit["citation_validity_rate"]))
        grounding_rates.append(float(answer.citation_audit["lexical_grounding_rate"]))
        if case.expected_phrase:
            phrase_matches.append(float(case.expected_phrase.lower() in answer.answer.lower()))
        else:
            phrase_matches.append(1.0)

    if not cases:
        return EvalMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    retrieval_metrics = evaluate_rankings(rankings, retrieval_cases, k=k)
    return EvalMetrics(
        cases=len(cases),
        hit_rate_at_k=retrieval_metrics.hit_rate_at_k,
        recall_at_k=retrieval_metrics.recall_at_k,
        precision_at_k=retrieval_metrics.precision_at_k,
        mean_reciprocal_rank=retrieval_metrics.mean_reciprocal_rank,
        ndcg_at_k=retrieval_metrics.ndcg_at_k,
        citation_doc_accuracy=mean(citation_matches),
        citation_validity_rate=mean(citation_validity),
        lexical_grounding_rate=mean(grounding_rates),
        answer_phrase_accuracy=mean(phrase_matches),
        average_latency_ms=mean(latencies),
    )


def load_cases(path: Path) -> list[EvalCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [EvalCase(**item) for item in payload]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval, citation and grounding behavior over an indexed corpus."
    )
    parser.add_argument("--database", default="data/rag.db")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--prompt-mode",
        choices=[item.value for item in PromptMode],
        default=PromptMode.ZERO_SHOT.value,
    )
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
