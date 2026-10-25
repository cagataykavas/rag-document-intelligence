from __future__ import annotations

import sqlite3

import pytest

from src.engine import RAGEngine
from src.evidence_snapshot import EvidenceSnapshotError, build_evidence_snapshot
from src.parsers import ParsedDocument, ParsedPage, parse_pdf
from src.prompting import EvidenceSnippet
from src.store import DocumentStore


class FakePdfPage:
    def __init__(self, text: str | None) -> None:
        self.text = text

    def extract_text(self) -> str | None:
        return self.text


class FakePdfReader:
    def __init__(self, _path: str) -> None:
        self.pages = [
            FakePdfPage("  Birinci sayfa kanıtı.  "),
            FakePdfPage(None),
            FakePdfPage("Üçüncü sayfa kanıtı."),
        ]


def pdf_document() -> ParsedDocument:
    pages = (
        ParsedPage(1, "alpha birinci sayfa kanıtı delta epsilon"),
        ParsedPage(2, "zeta ikinci sayfa kanıtı eta theta"),
    )
    return ParsedDocument(
        source="policy.pdf",
        media_type="application/pdf",
        text="\n\n".join(f"[Page {page.number}]\n{page.text}" for page in pages),
        metadata={"pages": 2, "pages_with_text": 2},
        pages=pages,
    )


def test_pdf_parser_retains_normalized_page_boundaries(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.parsers.PdfReader", FakePdfReader)
    path = tmp_path / "policy.pdf"
    path.write_bytes(b"fake-pdf")

    parsed = parse_pdf(path)

    assert parsed.pages == (
        ParsedPage(1, "Birinci sayfa kanıtı."),
        ParsedPage(2, ""),
        ParsedPage(3, "Üçüncü sayfa kanıtı."),
    )
    assert parsed.metadata == {"pages": 3, "pages_with_text": 2}


def test_pdf_chunks_never_cross_pages_and_offsets_are_exact(tmp_path) -> None:
    store = DocumentStore(tmp_path / "rag.db")
    store.ingest(pdf_document(), words_per_chunk=4, overlap=1)

    chunks = store.chunks()

    assert chunks
    assert {chunk.source_locator.page_number for chunk in chunks} == {1, 2}
    pages = {page.number: page.text for page in pdf_document().pages}
    for chunk in chunks:
        locator = chunk.source_locator
        assert locator is not None
        page_bytes = pages[locator.page_number].encode("utf-8")
        assert (
            page_bytes[locator.page_start_utf8 : locator.page_end_utf8].decode("utf-8")
            == chunk.text
        )

    reopened = DocumentStore(tmp_path / "rag.db")
    assert reopened.chunks() == chunks


def test_pdf_inventory_mismatch_fails_closed(tmp_path) -> None:
    parsed = pdf_document()
    tampered = ParsedDocument(
        parsed.source,
        parsed.media_type,
        parsed.text + " altered",
        parsed.metadata,
        parsed.pages,
    )

    with pytest.raises(ValueError, match="inventory does not match"):
        DocumentStore(tmp_path / "rag.db").ingest(tampered)


def test_retrieval_exposes_page_locator(tmp_path) -> None:
    store = DocumentStore(tmp_path / "rag.db")
    store.ingest(pdf_document(), words_per_chunk=4, overlap=1)

    rows = RAGEngine(store).retrieve("ikinci sayfa kanıtı", k=2)

    assert rows[0]["source_locator"]["page_number"] == 2
    assert "page_text_sha256" in rows[0]["source_locator"]


def test_evidence_snapshot_binds_page_locator() -> None:
    base = {
        "page_number": 3,
        "page_start_utf8": 10,
        "page_end_utf8": 16,
        "page_text_sha256": "a" * 64,
    }
    original = build_evidence_snapshot(
        [EvidenceSnippet("policy:1", "policy.pdf", "kanıt", base)]
    )
    changed = build_evidence_snapshot(
        [EvidenceSnippet("policy:1", "policy.pdf", "kanıt", {**base, "page_number": 4})]
    )

    assert original.snapshot_sha256 != changed.snapshot_sha256
    assert original.commitments[0].source_locator_sha256 is not None


def test_evidence_snapshot_rejects_inconsistent_page_range() -> None:
    locator = {
        "page_number": 1,
        "page_start_utf8": 0,
        "page_end_utf8": 99,
        "page_text_sha256": "b" * 64,
    }

    with pytest.raises(EvidenceSnapshotError, match="source locator values"):
        build_evidence_snapshot(
            [EvidenceSnippet("policy:1", "policy.pdf", "short", locator)]
        )


def test_existing_chunk_table_is_migrated_without_reindexing(tmp_path) -> None:
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                source TEXT NOT NULL,
                text TEXT NOT NULL,
                ordinal INTEGER NOT NULL
            )
            """
        )

    store = DocumentStore(path)
    with store.connect() as connection:
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(chunks)")
        }

    assert {
        "page_number",
        "page_start_utf8",
        "page_end_utf8",
        "page_text_sha256",
    } <= columns
