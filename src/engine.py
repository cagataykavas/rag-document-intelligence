from __future__ import annotations

import json
import os
from dataclasses import dataclass

import httpx

from src.prompting import EvidenceSnippet, PromptMode, build_prompt
from src.retrieval import SparseRetriever
from src.store import DocumentStore


@dataclass(frozen=True)
class Answer:
    answer: str
    citations: tuple[str, ...]
    confidence: float
    insufficient_evidence: bool
    retrieval: tuple[dict, ...]
    prompt_mode: str
    generator: str


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
        return json.loads(content)


class RAGEngine:
    def __init__(self, store: DocumentStore) -> None:
        self.store = store

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        chunks = self.store.chunks()
        if not chunks:
            return []
        return SparseRetriever().fit(chunks).search(query, k=k)

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
            return Answer(
                answer="The indexed documents do not provide sufficient evidence for this question.",
                citations=(),
                confidence=0.0,
                insufficient_evidence=True,
                retrieval=tuple(retrieval),
                prompt_mode=mode.value,
                generator="extractive",
            )

        prompt = build_prompt(question, evidence, mode)
        if use_llm:
            payload = OpenAICompatibleGenerator().generate(prompt)
            allowed = {item.chunk_id for item in evidence}
            citations = tuple(item for item in payload.get("citations", []) if item in allowed)
            return Answer(
                answer=str(payload.get("answer", "")),
                citations=citations,
                confidence=max(0.0, min(1.0, float(payload.get("confidence", 0.0)))),
                insufficient_evidence=bool(payload.get("insufficient_evidence", False)),
                retrieval=tuple(retrieval),
                prompt_mode=mode.value,
                generator="openai-compatible",
            )

        top = evidence[0]
        score = float(retrieval[0]["score"])
        answer_text = top.text.strip()
        if len(answer_text) > 700:
            answer_text = answer_text[:697].rstrip() + "..."
        return Answer(
            answer=answer_text,
            citations=(top.chunk_id,),
            confidence=max(0.0, min(1.0, score)),
            insufficient_evidence=False,
            retrieval=tuple(retrieval),
            prompt_mode=mode.value,
            generator="extractive",
        )
