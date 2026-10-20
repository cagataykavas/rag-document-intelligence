from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from typing import Any

import httpx

from src.claims import trace_claim_support
from src.evidence_policy import apply_evidence_policy
from src.generation_policy import admit_generation
from src.grounding import audit_grounding
from src.hybrid import HybridSparseRetriever
from src.prompting import EvidenceSnippet, PromptMode, build_prompt
from src.store import DocumentStore


DEFAULT_MAX_RESPONSE_BYTES = 128 * 1024
HARD_MAX_RESPONSE_BYTES = 1024 * 1024
RESPONSE_CHUNK_BYTES = 16 * 1024


class GeneratorTransportError(RuntimeError):
    """Stable, non-sensitive failure at the model transport boundary."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class Answer:
    answer: str
    citations: tuple[str, ...]
    rejected_citations: tuple[str, ...]
    confidence: float
    insufficient_evidence: bool
    retrieval: tuple[dict, ...]
    citation_audit: dict
    claim_support: tuple[dict, ...]
    evidence_policy: tuple[dict, ...]
    quarantined_chunk_ids: tuple[str, ...]
    generation_admission: dict
    prompt_mode: str
    generator: str
    retriever: str


class OpenAICompatibleGenerator:
    """Minimal adapter for OpenAI-compatible chat-completion servers.

    It works with local gateways and hosted providers exposing the conventional
    `/chat/completions` JSON contract; no provider-specific SDK is required.
    """

    def __init__(
        self,
        *,
        max_response_bytes: int | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1").rstrip(
            "/"
        )
        self.api_key = os.getenv("LLM_API_KEY", "local")
        self.model = os.getenv("LLM_MODEL", "local-model")
        configured_limit: object = (
            max_response_bytes
            if max_response_bytes is not None
            else os.getenv("LLM_MAX_RESPONSE_BYTES", str(DEFAULT_MAX_RESPONSE_BYTES))
        )
        self.max_response_bytes = _validated_response_limit(configured_limit)
        self._transport = transport

    def generate(self, prompt: str) -> Any:
        try:
            with httpx.Client(
                timeout=60,
                transport=self._transport,
            ) as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Accept-Encoding": "identity",
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "n": 1,
                        "response_format": {"type": "json_object"},
                    },
                ) as response:
                    response.raise_for_status()
                    body = self._read_bounded_body(response)
        except GeneratorTransportError:
            raise
        except httpx.HTTPError as exc:
            raise GeneratorTransportError("GENERATOR_HTTP_ERROR") from exc

        try:
            envelope = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GeneratorTransportError("GENERATOR_RESPONSE_NOT_JSON") from exc
        content = _message_content(envelope)
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise GeneratorTransportError("GENERATOR_CONTENT_NOT_JSON") from exc

    def _read_bounded_body(self, response: httpx.Response) -> bytes:
        content_encoding = response.headers.get("content-encoding", "identity")
        if content_encoding.strip().lower() not in {"", "identity"}:
            raise GeneratorTransportError("UNSUPPORTED_RESPONSE_ENCODING")

        lengths = response.headers.get_list("content-length")
        if lengths:
            declared = lengths[0].strip() if len(lengths) == 1 else ""
            if not declared.isascii() or not declared.isdigit():
                raise GeneratorTransportError("INVALID_RESPONSE_LENGTH")
            if int(declared) > self.max_response_bytes:
                raise GeneratorTransportError("GENERATOR_RESPONSE_TOO_LARGE")

        chunks: list[bytes] = []
        received = 0
        for chunk in response.iter_raw(chunk_size=RESPONSE_CHUNK_BYTES):
            received += len(chunk)
            if received > self.max_response_bytes:
                raise GeneratorTransportError("GENERATOR_RESPONSE_TOO_LARGE")
            chunks.append(chunk)
        return b"".join(chunks)


def _validated_response_limit(value: object) -> int:
    if isinstance(value, bool):
        raise GeneratorTransportError("INVALID_RESPONSE_BYTE_LIMIT")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeneratorTransportError("INVALID_RESPONSE_BYTE_LIMIT") from exc
    if str(converted) != str(value).strip():
        raise GeneratorTransportError("INVALID_RESPONSE_BYTE_LIMIT")
    if not 1 <= converted <= HARD_MAX_RESPONSE_BYTES:
        raise GeneratorTransportError("INVALID_RESPONSE_BYTE_LIMIT")
    return converted


def _message_content(envelope: object) -> str:
    if not isinstance(envelope, dict) or "choices" not in envelope:
        raise GeneratorTransportError("INVALID_GENERATOR_ENVELOPE")
    choices = envelope["choices"]
    if not isinstance(choices, list) or len(choices) != 1:
        raise GeneratorTransportError("INVALID_GENERATOR_ENVELOPE")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise GeneratorTransportError("INVALID_GENERATOR_ENVELOPE")
    content = choice["message"].get("content")
    if not isinstance(content, str):
        raise GeneratorTransportError("INVALID_GENERATOR_ENVELOPE")
    return content


class RAGEngine:
    def __init__(self, store: DocumentStore) -> None:
        self.store = store

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        chunks = self.store.chunks()
        if not chunks:
            return []
        return HybridSparseRetriever().fit(chunks).search(query, k=k)

    def _insufficient_answer(
        self,
        retrieval: list[dict],
        mode: PromptMode,
        policy_findings: tuple[dict, ...] = (),
        quarantined: tuple[str, ...] = (),
        generation_admission: dict | None = None,
        generator: str = "extractive",
    ) -> Answer:
        if generation_admission and not generation_admission.get("accepted", True):
            text = (
                "The generated response failed the admission policy, so the system "
                "will not return it as a factual answer."
            )
        elif quarantined:
            text = (
                "Potentially relevant retrieved evidence was quarantined by the untrusted-evidence "
                "policy, so the system will not use it to generate a factual answer."
            )
        else:
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
            claim_support=(),
            evidence_policy=policy_findings,
            quarantined_chunk_ids=quarantined,
            generation_admission=generation_admission or {"applied": False},
            prompt_mode=mode.value,
            generator=generator,
            retriever="hybrid-word-char-tfidf",
        )

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
            return self._insufficient_answer(retrieval, mode)

        safe_evidence, findings = apply_evidence_policy(evidence)
        finding_rows = tuple(item.to_dict() for item in findings)
        quarantined = tuple(item.chunk_id for item in findings if item.quarantined)
        if not safe_evidence:
            return self._insufficient_answer(retrieval, mode, finding_rows, quarantined)

        prompt = build_prompt(question, safe_evidence, mode)
        if use_llm:
            payload = OpenAICompatibleGenerator().generate(prompt)
            allowed = {item.chunk_id for item in safe_evidence}
            admission = admit_generation(payload, allowed)
            admission_audit = admission.to_audit_dict()
            if not admission.accepted:
                rejected = tuple(
                    item for item in admission.citations if item not in allowed
                )
                result = self._insufficient_answer(
                    retrieval,
                    mode,
                    finding_rows,
                    quarantined,
                    admission_audit,
                    "openai-compatible",
                )
                return replace(result, rejected_citations=rejected)

            citations = admission.citations
            answer_text = admission.answer
            audit = audit_grounding(answer_text, citations, retrieval)
            claims = tuple(
                item.to_dict()
                for item in trace_claim_support(answer_text, citations, retrieval)
            )
            return Answer(
                answer=answer_text,
                citations=citations,
                rejected_citations=(),
                confidence=admission.confidence,
                insufficient_evidence=admission.insufficient_evidence,
                retrieval=tuple(retrieval),
                citation_audit=audit.to_dict(),
                claim_support=claims,
                evidence_policy=finding_rows,
                quarantined_chunk_ids=quarantined,
                generation_admission=admission_audit,
                prompt_mode=mode.value,
                generator="openai-compatible",
                retriever="hybrid-word-char-tfidf",
            )

        top = safe_evidence[0]
        retrieval_by_id = {item["chunk_id"]: item for item in retrieval}
        score = float(retrieval_by_id[top.chunk_id]["score"])
        answer_text = top.text.strip()
        if len(answer_text) > 700:
            answer_text = answer_text[:697].rstrip() + "..."
        citations = (top.chunk_id,)
        audit = audit_grounding(answer_text, citations, retrieval)
        claims = tuple(
            item.to_dict()
            for item in trace_claim_support(answer_text, citations, retrieval)
        )
        return Answer(
            answer=answer_text,
            citations=citations,
            rejected_citations=(),
            confidence=max(0.0, min(1.0, score)),
            insufficient_evidence=False,
            retrieval=tuple(retrieval),
            citation_audit=audit.to_dict(),
            claim_support=claims,
            evidence_policy=finding_rows,
            quarantined_chunk_ids=quarantined,
            generation_admission={"applied": False},
            prompt_mode=mode.value,
            generator="extractive",
            retriever="hybrid-word-char-tfidf",
        )
