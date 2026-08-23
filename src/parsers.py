"""Parsing boundary for text, Markdown, JSON, CSV and PDF documents."""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader


@dataclass(frozen=True)
class ParsedDocument:
    source: str
    media_type: str
    text: str
    metadata: dict[str, Any]


def parse_text(path: Path) -> ParsedDocument:
    return ParsedDocument(str(path), "text/plain", path.read_text(encoding="utf-8"), {})


def parse_markdown(path: Path) -> ParsedDocument:
    return ParsedDocument(str(path), "text/markdown", path.read_text(encoding="utf-8"), {"format": "markdown"})


def parse_json(path: Path) -> ParsedDocument:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pretty = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    return ParsedDocument(str(path), "application/json", pretty, {"root_type": type(payload).__name__})


def parse_csv(path: Path) -> ParsedDocument:
    raw = path.read_text(encoding="utf-8")
    rows = list(csv.DictReader(io.StringIO(raw)))
    lines = [" | ".join(f"{key}: {value}" for key, value in row.items()) for row in rows]
    return ParsedDocument(str(path), "text/csv", "\n".join(lines), {"rows": len(rows)})


def parse_pdf(path: Path) -> ParsedDocument:
    reader = PdfReader(str(path))
    pages: list[str] = []
    non_empty_pages = 0
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            non_empty_pages += 1
        pages.append(f"[Page {page_number}]\n{text}")
    return ParsedDocument(
        str(path),
        "application/pdf",
        "\n\n".join(pages),
        {"pages": len(reader.pages), "pages_with_text": non_empty_pages},
    )


def parse(path: str | Path) -> ParsedDocument:
    path = Path(path)
    parsers = {
        ".txt": parse_text,
        ".md": parse_markdown,
        ".json": parse_json,
        ".csv": parse_csv,
        ".pdf": parse_pdf,
    }
    try:
        return parsers[path.suffix.lower()](path)
    except KeyError as exc:
        raise ValueError(f"unsupported document type: {path.suffix}") from exc
