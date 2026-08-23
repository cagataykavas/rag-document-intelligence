from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from src.parsers import ParsedDocument
from src.retrieval import Chunk, Document, chunk_documents

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
    ordinal INTEGER NOT NULL
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

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def ingest(self, parsed: ParsedDocument, words_per_chunk: int = 110, overlap: int = 25) -> IngestResult:
        digest = hashlib.sha256(parsed.text.encode("utf-8")).hexdigest()
        doc_id = f"doc-{digest[:16]}"
        chunks = chunk_documents(
            [Document(doc_id=doc_id, text=parsed.text, source=parsed.source)],
            words_per_chunk=words_per_chunk,
            overlap=overlap,
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
                "INSERT INTO chunks (chunk_id, doc_id, source, text, ordinal) VALUES (?, ?, ?, ?, ?)",
                [(chunk.chunk_id, chunk.doc_id, chunk.source, chunk.text, index) for index, chunk in enumerate(chunks)],
            )
            conn.commit()
        return IngestResult(doc_id, True, len(chunks), digest)

    def chunks(self) -> list[Chunk]:
        with self.connect() as conn:
            rows = conn.execute("SELECT chunk_id, doc_id, source, text FROM chunks ORDER BY doc_id, ordinal").fetchall()
        return [Chunk(row["chunk_id"], row["doc_id"], row["source"], row["text"]) for row in rows]

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
