from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.engine import OpenAICompatibleGenerator, RAGEngine
from src.evidence_policy import apply_evidence_policy
from src.parsers import parse
from src.prompt_eval import PromptEvalCase, evaluate_prompt_modes
from src.prompting import EvidenceSnippet, PromptMode
from src.store import DocumentStore


def ingest_command(args) -> None:
    store = DocumentStore(args.database)
    results = []
    for raw in args.paths:
        path = Path(raw)
        candidates = sorted(path.rglob("*")) if path.is_dir() else [path]
        for candidate in candidates:
            if (
                not candidate.is_file()
                or candidate.suffix.lower() not in {".txt", ".md", ".json", ".csv", ".pdf"}
            ):
                continue
            parsed = parse(candidate)
            results.append(asdict(store.ingest(parsed, args.words_per_chunk, args.overlap)))
    print(json.dumps(results, indent=2))


def query_command(args) -> None:
    engine = RAGEngine(DocumentStore(args.database))
    result = engine.answer(
        args.question,
        k=args.k,
        mode=PromptMode(args.prompt_mode),
        use_llm=args.llm,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


def list_command(args) -> None:
    print(json.dumps(DocumentStore(args.database).documents(), ensure_ascii=False, indent=2))


def prompt_benchmark_command(args) -> None:
    payload = json.loads(args.cases.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit("prompt benchmark cases must be a JSON array")

    engine = RAGEngine(DocumentStore(args.database))
    cases: list[PromptEvalCase] = []
    quarantined_total = 0
    for row in payload:
        question = str(row["question"])
        retrieval = engine.retrieve(question, k=args.k)
        evidence = [
            EvidenceSnippet(item["chunk_id"], item["source"], item["text"])
            for item in retrieval
            if item["score"] > 0
        ]
        safe_evidence, findings = apply_evidence_policy(evidence)
        quarantined_total += sum(item.quarantined for item in findings)
        cases.append(
            PromptEvalCase(
                question=question,
                evidence=tuple(safe_evidence),
                expected_insufficient_evidence=bool(
                    row.get("expected_insufficient_evidence", False)
                ),
            )
        )

    generator = OpenAICompatibleGenerator()
    result = {
        "cases": len(cases),
        "retrieval_k": args.k,
        "quarantined_chunks": quarantined_total,
        "generator": generator.model,
        "modes": evaluate_prompt_modes(cases, generator.generate),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rag-docs",
        description="Persistent evidence-grounded document intelligence CLI.",
    )
    parser.add_argument("--database", default="data/rag.db")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest")
    ingest.add_argument("paths", nargs="+")
    ingest.add_argument("--words-per-chunk", type=int, default=110)
    ingest.add_argument("--overlap", type=int, default=25)
    ingest.set_defaults(func=ingest_command)

    query = sub.add_parser("query")
    query.add_argument("question")
    query.add_argument("--k", type=int, default=5)
    query.add_argument(
        "--prompt-mode",
        choices=[item.value for item in PromptMode],
        default=PromptMode.ZERO_SHOT.value,
    )
    query.add_argument("--llm", action="store_true")
    query.set_defaults(func=query_command)

    benchmark = sub.add_parser(
        "prompt-benchmark",
        help="compare zero/one/few-shot structured grounding behavior using a configured LLM",
    )
    benchmark.add_argument("--cases", type=Path, required=True)
    benchmark.add_argument("--k", type=int, default=5)
    benchmark.set_defaults(func=prompt_benchmark_command)

    listing = sub.add_parser("list")
    listing.set_defaults(func=list_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
