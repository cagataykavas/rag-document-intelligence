from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.engine import RAGEngine
from src.parsers import parse
from src.prompting import PromptMode
from src.store import DocumentStore


def ingest_command(args) -> None:
    store = DocumentStore(args.database)
    results = []
    for raw in args.paths:
        path = Path(raw)
        candidates = sorted(path.rglob("*")) if path.is_dir() else [path]
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix.lower() not in {".txt", ".md", ".json", ".csv", ".pdf"}:
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


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag-docs", description="Persistent evidence-grounded document intelligence CLI.")
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
    query.add_argument("--prompt-mode", choices=[item.value for item in PromptMode], default=PromptMode.ZERO_SHOT.value)
    query.add_argument("--llm", action="store_true")
    query.set_defaults(func=query_command)

    listing = sub.add_parser("list")
    listing.set_defaults(func=list_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
