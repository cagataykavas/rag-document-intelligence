from __future__ import annotations

import json
import os
from dataclasses import dataclass

import httpx

from src.grounding import audit_grounding
from src.hybrid import HybridSparseRetriever
from src.prompting import EvidenceSnippet, PromptMode, build_prompt
from src.store import DocumentStore


@dataclass(frozen=True)
class Answer:
    answer: str
    citations: tuple[str, ...]
    rejected_citations: tuple[str, ...]
    confidence: float
    insufficient_evidence: bool
    retrieval: tuple[dict, ...]
    citation_audit: dict
    prompt_mode: str
    generator: str
    retriever: str


class OpenAICompatibleGenerator:
    """Minimal adapter for OpenAI-compatible chat-completion servers.

    It works with local gateways and hosted providers exposing the conventional
    `/chat/completions` JSON contract; no provider-specific SDK is required.
    """

    def __init__(self) -> None:
        self.base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
        self.api_key = os.getenv("LLM_API_KEY", "local")
        self.model = os.getenv("LLM_MODEL", "local-model")

    def generate(self, prompt: str) -> dict:
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError("generator response must decode to a JSON object")
        return payload


class RAGEngine:
    def __init__(self, store: DocumentStore) -> None:
        self.store = store

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        chunks = self.store.chunks()
        if not chunks:
            return []
        return HybridSparseRetriever().fit(chunks).search(query, k=k)

    def answer(
        self,
        question: str,
        *,
        k: int = 5,
        mode: PromptMode = PromptMode.ZERO_SHOT,
        use_llm: bool = False,
    ) -> Answer:
        retrieval = self.retrieve(question, k=k)
        evidence = [
            EvidenceSnippet(item["chunk_id"], item["source"], item["text"])
            for item in retrieval
            if item["score"] > 0
        ]
        if not evidence:
            text = "The indexed documents do not provide sufficient evidence for this question."
            audit = audit_grounding(text, (), retrieval)
            return Answer(
                answer=text,
                citations=(),
                rejected_citations=(),
                confidence=0.0,
                insufficient_evidence=True,
                retrieval=tuple(retrieval),
                citation_audit=audit.to_dict(),
                prompt_mode=mode.value,
                generator="extractive",
                retriever="hybrid-word-char-tfidf",
            )

        prompt = build_prompt(question, evidence, mode)
        if use_llm:
            payload = OpenAICompatibleGenerator().generate(prompt)
            allowed = {item.chunk_id for item in evidence}
            requested_citations = tuple(str(item) for item in payload.get("citations", []))
            citations = tuple(item for item in requested_citations if item in allowed)
            rejected = tuple(item for item in requested_citations if item not in allowed)
            answer_text = str(payload.get("answer", ""))
            audit = audit_grounding(answer_text, requested_citations, retrieval)
            return Answer(
                answer=answer_text,
                citations=citations,
                rejected_citations=rejected,
                confidence=max(0.0, min(1.0, float(payload.get("confidence", 0.0)))),
                insufficient_evidence=bool(payload.get("insufficient_evidence", False)),
                retrieval=tuple(retrieval),
                citation_audit=audit.to_dict(),
                prompt_mode=mode.value,
                generator="openai-compatible",
                retriever="hybrid-word-char-tfidf",
            )

        top = evidence[0]
        score = float(retrieval[0]["score"])
        answer_text = top.text.strip()
        if len(answer_text) > 700:
            answer_text = answer_text[:697].rstrip() + "..."
        citations = (top.chunk_id,)
        audit = audit_grounding(answer_text, citations, retrieval)
        return Answer(
            answer=answer_text,
            citations=citations,
            rejected_citations=(),
            confidence=max(0.0, min(1.0, score)),
            insufficient_evidence=False,
            retrieval=tuple(retrieval),
            citation_audit=audit.to_dict(),
            prompt_mode=mode.value,
            generator="extractive",
            retriever="hybrid-word-char-tfidf",
        )
