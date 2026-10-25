from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from src.parsers import ParsedDocument
from src.retrieval import Chunk, Document, SourceLocator, chunk_documents

DDL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    media_type TEXT NOT NULL,
    content_sha256 TEXT NOT NULL UNIQUE,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    text TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    page_number INTEGER,
    page_start_utf8 INTEGER,
    page_end_utf8 INTEGER,
    page_text_sha256 TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, ordinal);
"""


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    inserted: bool
    chunks: int
    content_sha256: str


class DocumentStore:
    def __init__(self, path: str | Path = "data/rag.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(DDL)
            self._migrate_chunk_locators(conn)

    @staticmethod
    def _migrate_chunk_locators(conn: sqlite3.Connection) -> None:
        existing = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(chunks)").fetchall()
        }
        columns = {
            "page_number": "INTEGER",
            "page_start_utf8": "INTEGER",
            "page_end_utf8": "INTEGER",
            "page_text_sha256": "TEXT",
        }
        for name, column_type in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE chunks ADD COLUMN {name} {column_type}")

    @staticmethod
    def _chunk_pdf_pages(
        parsed: ParsedDocument,
        doc_id: str,
        words_per_chunk: int,
        overlap: int,
    ) -> list[Chunk]:
        if overlap >= words_per_chunk:
            raise ValueError("overlap must be smaller than words_per_chunk")
        if not parsed.pages:
            raise ValueError("PDF provenance requires at least one parsed page")
        expected_numbers = tuple(range(1, len(parsed.pages) + 1))
        if tuple(page.number for page in parsed.pages) != expected_numbers:
            raise ValueError("PDF page numbers must be contiguous and one-based")
        rendered = "\n\n".join(
            f"[Page {page.number}]\n{page.text}" for page in parsed.pages
        )
        if parsed.text != rendered:
            raise ValueError("PDF page inventory does not match normalized text")
        if parsed.metadata.get("pages") != len(parsed.pages) or parsed.metadata.get(
            "pages_with_text"
        ) != sum(bool(page.text) for page in parsed.pages):
            raise ValueError("PDF page metadata does not match page inventory")
        step = words_per_chunk - overlap
        chunks: list[Chunk] = []
        for page in parsed.pages:
            matches = tuple(re.finditer(r"\S+", page.text))
            page_bytes = page.text.encode("utf-8")
            page_sha256 = hashlib.sha256(page_bytes).hexdigest()
            for index, word_start in enumerate(range(0, len(matches), step)):
                selected = matches[word_start : word_start + words_per_chunk]
                if not selected:
                    continue
                start_char = selected[0].start()
                end_char = selected[-1].end()
                text = page.text[start_char:end_char]
                start_utf8 = len(page.text[:start_char].encode("utf-8"))
                end_utf8 = len(page.text[:end_char].encode("utf-8"))
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}:p{page.number}:{index}",
                        doc_id=doc_id,
                        source=parsed.source,
                        text=text,
                        source_locator=SourceLocator(
                            page_number=page.number,
                            page_start_utf8=start_utf8,
                            page_end_utf8=end_utf8,
                            page_text_sha256=page_sha256,
                        ),
                    )
                )
        return chunks

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def ingest(self, parsed: ParsedDocument, words_per_chunk: int = 110, overlap: int = 25) -> IngestResult:
        digest = hashlib.sha256(parsed.text.encode("utf-8")).hexdigest()
        doc_id = f"doc-{digest[:16]}"
        chunks = (
            self._chunk_pdf_pages(parsed, doc_id, words_per_chunk, overlap)
            if parsed.media_type == "application/pdf"
            else chunk_documents(
                [Document(doc_id=doc_id, text=parsed.text, source=parsed.source)],
                words_per_chunk=words_per_chunk,
                overlap=overlap,
            )
        )
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT doc_id FROM documents WHERE content_sha256 = ?",
                (digest,),
            ).fetchone()
            if existing:
                count = conn.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (existing["doc_id"],)).fetchone()[0]
                return IngestResult(existing["doc_id"], False, int(count), digest)

            conn.execute(
                """
                INSERT INTO documents (doc_id, source, media_type, content_sha256, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (doc_id, parsed.source, parsed.media_type, digest, json.dumps(parsed.metadata, ensure_ascii=False)),
            )
            conn.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, doc_id, source, text, ordinal,
                    page_number, page_start_utf8, page_end_utf8, page_text_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.source,
                        chunk.text,
                        index,
                        chunk.source_locator.page_number
                        if chunk.source_locator is not None
                        else None,
                        chunk.source_locator.page_start_utf8
                        if chunk.source_locator is not None
                        else None,
                        chunk.source_locator.page_end_utf8
                        if chunk.source_locator is not None
                        else None,
                        chunk.source_locator.page_text_sha256
                        if chunk.source_locator is not None
                        else None,
                    )
                    for index, chunk in enumerate(chunks)
                ],
            )
            conn.commit()
        return IngestResult(doc_id, True, len(chunks), digest)

    def chunks(self) -> list[Chunk]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, doc_id, source, text,
                       page_number, page_start_utf8, page_end_utf8, page_text_sha256
                FROM chunks ORDER BY doc_id, ordinal
                """
            ).fetchall()
        chunks: list[Chunk] = []
        for row in rows:
            locator_values = (
                row["page_number"],
                row["page_start_utf8"],
                row["page_end_utf8"],
                row["page_text_sha256"],
            )
            populated = tuple(value is not None for value in locator_values)
            if any(populated) and not all(populated):
                raise ValueError("stored chunk source locator is incomplete")
            locator = (
                SourceLocator(*locator_values) if all(populated) else None
            )
            if locator is not None:
                text_bytes = row["text"].encode("utf-8")
                if locator.page_end_utf8 - locator.page_start_utf8 != len(text_bytes):
                    raise ValueError("stored chunk source locator length mismatch")
            chunks.append(
                Chunk(
                    row["chunk_id"],
                    row["doc_id"],
                    row["source"],
                    row["text"],
                    locator,
                )
            )
        return chunks

    def documents(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT doc_id, source, media_type, content_sha256, metadata_json, created_at FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [
            {
                "doc_id": row["doc_id"],
                "source": row["source"],
                "media_type": row["media_type"],
                "content_sha256": row["content_sha256"],
                "metadata": json.loads(row["metadata_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
