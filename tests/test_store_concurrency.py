from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from src.parsers import ParsedDocument
from src.store import DocumentStore


def test_concurrent_duplicate_ingestion_commits_one_document(tmp_path: Path) -> None:
    database = tmp_path / "rag.db"
    workers = 12
    stores = [DocumentStore(database) for _ in range(workers)]
    start = Barrier(workers)
    document = ParsedDocument(
        source="policy.md",
        media_type="text/markdown",
        text=" ".join(
            [
                "The service retains audit evidence for seven years and protects it from change."
            ]
            * 20
        ),
        metadata={"classification": "public-demo"},
    )

    def ingest(store: DocumentStore):
        start.wait()
        return store.ingest(document, words_per_chunk=30, overlap=5)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(ingest, stores))

    assert sum(result.inserted for result in results) == 1
    assert len({result.doc_id for result in results}) == 1
    assert len({result.content_sha256 for result in results}) == 1
    assert len({result.chunks for result in results}) == 1

    persisted = DocumentStore(database)
    assert len(persisted.documents()) == 1
    assert len(persisted.chunks()) == results[0].chunks
