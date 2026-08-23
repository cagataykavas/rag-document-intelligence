from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.conflicts import Requirement, find_conflicts
from src.engine import RAGEngine
from src.parsers import parse
from src.prompting import PromptMode
from src.store import DocumentStore

STORE = DocumentStore()
ENGINE = RAGEngine(STORE)

app = FastAPI(
    title="RAG Document Intelligence",
    version="1.0.0",
    description=(
        "Persistent document ingestion, evidence retrieval, zero/one/few-shot grounded "
        "answering and deterministic requirement-conflict analysis."
    ),
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    k: int = Field(default=5, ge=1, le=20)
    prompt_mode: PromptMode = PromptMode.ZERO_SHOT
    use_llm: bool = False


class InlineDocumentRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)


class RequirementInput(BaseModel):
    requirement_id: str = Field(min_length=1, max_length=100)
    source: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=3, max_length=10000)


class ConflictRequest(BaseModel):
    requirements: list[RequirementInput] = Field(min_length=2, max_length=500)
    minimum_overlap: float = Field(default=0.18, ge=0.0, le=1.0)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "documents": len(STORE.documents()), "chunks": len(STORE.chunks())}


@app.get("/documents")
def documents() -> list[dict]:
    return STORE.documents()


@app.post("/documents/inline", status_code=201)
def ingest_inline(request: InlineDocumentRequest) -> dict:
    suffix = Path(request.name).suffix.lower() or ".txt"
    if suffix not in {".txt", ".md"}:
        raise HTTPException(400, "inline ingestion supports .txt and .md names")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / Path(request.name).name
        path.write_text(request.text, encoding="utf-8")
        parsed = parse(path)
        # Keep the logical caller-provided source instead of a temporary filesystem path.
        parsed = type(parsed)(request.name, parsed.media_type, parsed.text, parsed.metadata)
        return asdict(STORE.ingest(parsed))


@app.post("/documents/upload", status_code=201)
async def ingest_upload(
    file: UploadFile = File(...),
    words_per_chunk: int = Form(default=110, ge=30, le=1000),
    overlap: int = Form(default=25, ge=0, le=300),
) -> dict:
    suffix = Path(file.filename or "document.txt").suffix.lower()
    if suffix not in {".txt", ".md", ".json", ".csv", ".pdf"}:
        raise HTTPException(415, f"unsupported file type: {suffix or 'unknown'}")
    if overlap >= words_per_chunk:
        raise HTTPException(400, "overlap must be smaller than words_per_chunk")
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(413, "demo upload limit is 20 MB")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / (Path(file.filename or "document.txt").name)
        path.write_bytes(content)
        try:
            parsed = parse(path)
        except Exception as exc:
            raise HTTPException(422, f"document parsing failed: {exc}") from exc
        parsed = type(parsed)(file.filename or path.name, parsed.media_type, parsed.text, parsed.metadata)
        return asdict(STORE.ingest(parsed, words_per_chunk=words_per_chunk, overlap=overlap))


@app.post("/query")
def query(request: QueryRequest) -> dict:
    try:
        result = ENGINE.answer(
            request.question,
            k=request.k,
            mode=request.prompt_mode,
            use_llm=request.use_llm,
        )
    except Exception as exc:
        if request.use_llm:
            raise HTTPException(502, f"generator request failed: {exc}") from exc
        raise
    return asdict(result)


@app.post("/requirements/conflicts")
def conflicts(request: ConflictRequest) -> dict[str, object]:
    rows = [
        Requirement(item.requirement_id, item.text, item.source)
        for item in request.requirements
    ]
    detected = find_conflicts(rows, minimum_overlap=request.minimum_overlap)
    return {
        "requirements": len(rows),
        "conflicts": [asdict(item) for item in detected],
        "conflict_count": len(detected),
    }
