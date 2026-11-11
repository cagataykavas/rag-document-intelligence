from pathlib import Path

from src.parsers import ParsedDocument
from src.store import DocumentStore


def document(source: str, text: str) -> ParsedDocument:
    return ParsedDocument(source, "text/markdown", text, {"classification": "public"})


def ingest_corpus(store: DocumentStore, documents: list[ParsedDocument]) -> None:
    for item in documents:
        store.ingest(item, words_per_chunk=8, overlap=2)


def test_snapshot_is_order_independent_and_stable_across_restart(
    tmp_path: Path,
) -> None:
    documents = [
        document(
            "security.md", "Administrative access requires strong authentication."
        ),
        document("retention.md", "Audit evidence is retained for seven years."),
    ]
    first = DocumentStore(tmp_path / "first.db")
    second = DocumentStore(tmp_path / "second.db")
    ingest_corpus(first, documents)
    ingest_corpus(second, list(reversed(documents)))

    first_snapshot = first.corpus_snapshot()
    second_snapshot = second.corpus_snapshot()
    restarted_snapshot = DocumentStore(first.path).corpus_snapshot()

    assert first_snapshot.to_dict() == second_snapshot.to_dict()
    assert restarted_snapshot.to_dict() == first_snapshot.to_dict()
    assert first_snapshot.document_count == 2
    assert first_snapshot.chunk_count == len(first_snapshot.chunks)


def test_snapshot_changes_when_retrieval_material_changes(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path / "rag.db")
    empty = store.corpus_snapshot()
    store.ingest(
        document(
            "security.md", "Administrative access requires strong authentication."
        ),
        words_per_chunk=8,
        overlap=2,
    )
    populated = store.corpus_snapshot()

    with store.connect() as conn:
        conn.execute("UPDATE chunks SET text = text || ' Audit logging is mandatory.'")
        conn.commit()
    changed = store.corpus_snapshot()

    assert empty.document_count == 0
    assert empty.chunk_count == 0
    assert empty.sha256 != populated.sha256
    assert changed.document_count == populated.document_count
    assert changed.chunk_count == populated.chunk_count
    assert changed.sha256 != populated.sha256


def test_public_snapshot_contains_no_document_content_or_source(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path / "rag.db")
    store.ingest(
        document(
            "private-customer-name.md",
            "A sensitive sentence that must not enter metadata.",
        ),
        words_per_chunk=20,
        overlap=2,
    )

    public_snapshot = store.corpus_snapshot().to_dict()
    serialized = str(public_snapshot)

    assert set(public_snapshot) == {"version", "sha256", "documents", "chunks"}
    assert "private-customer-name" not in serialized
    assert "sensitive sentence" not in serialized
