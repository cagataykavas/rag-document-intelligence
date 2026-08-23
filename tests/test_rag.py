from pathlib import Path

from src.conflicts import ConflictKind, Requirement, compare
from src.engine import RAGEngine
from src.parsers import ParsedDocument
from src.prompting import PromptMode, build_prompt, EvidenceSnippet
from src.store import DocumentStore


def build_store(tmp_path: Path) -> DocumentStore:
    store = DocumentStore(tmp_path / "rag.db")
    store.ingest(
        ParsedDocument(
            source="security.md",
            media_type="text/markdown",
            text=(
                "REQ-1 The service shall require authenticated access for all administrative endpoints. "
                "REQ-2 Audit logs must be retained for seven years and protected from modification."
            ),
            metadata={},
        ),
        words_per_chunk=18,
        overlap=4,
    )
    store.ingest(
        ParsedDocument(
            source="performance.md",
            media_type="text/markdown",
            text="REQ-3 The API shall respond within 250 ms at the ninety-fifth percentile under the reference workload.",
            metadata={},
        ),
        words_per_chunk=18,
        overlap=4,
    )
    return store


def test_content_deduplication(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path / "rag.db")
    document = ParsedDocument("a.txt", "text/plain", "hello retrieval world", {})
    first = store.ingest(document, words_per_chunk=10, overlap=2)
    second = store.ingest(document, words_per_chunk=10, overlap=2)
    assert first.inserted is True
    assert second.inserted is False
    assert first.doc_id == second.doc_id


def test_grounded_extractive_query_has_citation(tmp_path: Path) -> None:
    store = build_store(tmp_path)
    answer = RAGEngine(store).answer("How long are audit logs retained?", k=3)
    assert answer.insufficient_evidence is False
    assert answer.citations
    assert any("seven years" in item["text"].lower() for item in answer.retrieval)


def test_zero_one_few_shot_modes_change_prompt() -> None:
    evidence = [EvidenceSnippet("chunk-1", "demo", "The policy requires authentication.")]
    zero = build_prompt("Is auth required?", evidence, PromptMode.ZERO_SHOT)
    one = build_prompt("Is auth required?", evidence, PromptMode.ONE_SHOT)
    few = build_prompt("Is auth required?", evidence, PromptMode.FEW_SHOT)
    assert "Example question" not in zero
    assert one.count("Example question") == 1
    assert few.count("Example question") == 2


def test_numeric_requirement_conflict() -> None:
    left = Requirement("REQ-A", "The API shall respond within 250 ms under peak load.", "a")
    right = Requirement("REQ-B", "The API shall respond within 500 ms under peak load.", "b")
    result = compare(left, right, minimum_overlap=0.1)
    assert result.kind is ConflictKind.NUMERIC


def test_negation_conflict() -> None:
    left = Requirement("REQ-A", "Administrative access shall require authentication.", "a")
    right = Requirement("REQ-B", "Administrative access shall not require authentication.", "b")
    result = compare(left, right, minimum_overlap=0.1)
    assert result.kind is ConflictKind.NEGATION
