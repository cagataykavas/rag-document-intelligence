from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

import httpx

from src.admission_authentication import AdmissionAuthenticator
from src.claims import claim_support_sha256, trace_claim_support
from src.evidence_policy import apply_evidence_policy
from src.evidence_snapshot import build_evidence_snapshot
from src.generation_policy import GenerationPolicy, admit_generation, bind_claim_support
from src.grounding import audit_grounding
from src.hybrid import HybridSparseRetriever
from src.prompting import EvidenceSnippet, PromptMode, build_prompt
from src.store import DocumentStore


DEFAULT_MAX_RESPONSE_BYTES = 128 * 1024
HARD_MAX_RESPONSE_BYTES = 1024 * 1024
RESPONSE_CHUNK_BYTES = 16 * 1024
DEFAULT_REQUEST_DEADLINE_SECONDS = 60.0
DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0
DEFAULT_READ_TIMEOUT_SECONDS = 30.0
CONTENT_BOUND_GENERATION_POLICY = GenerationPolicy(require_content_bound_evidence=True)


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
    generation_admission_authentication: dict
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
        request_deadline_seconds: float | None = None,
        connect_timeout_seconds: float | None = None,
        read_timeout_seconds: float | None = None,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
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
        self.request_deadline_seconds = _validated_positive_seconds(
            request_deadline_seconds
            if request_deadline_seconds is not None
            else os.getenv(
                "LLM_REQUEST_DEADLINE_SECONDS",
                str(DEFAULT_REQUEST_DEADLINE_SECONDS),
            ),
            "INVALID_REQUEST_DEADLINE",
        )
        self.connect_timeout_seconds = _validated_positive_seconds(
            connect_timeout_seconds
            if connect_timeout_seconds is not None
            else os.getenv(
                "LLM_CONNECT_TIMEOUT_SECONDS",
                str(DEFAULT_CONNECT_TIMEOUT_SECONDS),
            ),
            "INVALID_CONNECT_TIMEOUT",
        )
        self.read_timeout_seconds = _validated_positive_seconds(
            read_timeout_seconds
            if read_timeout_seconds is not None
            else os.getenv(
                "LLM_READ_TIMEOUT_SECONDS",
                str(DEFAULT_READ_TIMEOUT_SECONDS),
            ),
            "INVALID_READ_TIMEOUT",
        )
        if self.connect_timeout_seconds > self.request_deadline_seconds:
            raise GeneratorTransportError("CONNECT_TIMEOUT_EXCEEDS_DEADLINE")
        if self.read_timeout_seconds > self.request_deadline_seconds:
            raise GeneratorTransportError("READ_TIMEOUT_EXCEEDS_DEADLINE")
        self._transport = transport
        self._clock = clock
        self._client: httpx.Client | None = None

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> OpenAICompatibleGenerator:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _pooled_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(transport=self._transport)
        return self._client

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise GeneratorTransportError("GENERATOR_DEADLINE_EXCEEDED")
        return remaining

    def generate(self, prompt: str) -> Any:
        deadline = self._clock() + self.request_deadline_seconds
        try:
            remaining = self._remaining(deadline)
            timeout = httpx.Timeout(
                timeout=remaining,
                connect=min(self.connect_timeout_seconds, remaining),
                read=min(self.read_timeout_seconds, remaining),
                write=min(self.connect_timeout_seconds, remaining),
                pool=min(self.connect_timeout_seconds, remaining),
            )
            with self._pooled_client().stream(
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
                timeout=timeout,
            ) as response:
                response.raise_for_status()
                body = self._read_bounded_body(response, deadline)
        except GeneratorTransportError:
            raise
        except httpx.TimeoutException as exc:
            code = (
                "GENERATOR_DEADLINE_EXCEEDED"
                if self._clock() >= deadline
                else "GENERATOR_PHASE_TIMEOUT"
            )
            raise GeneratorTransportError(code) from exc
        except httpx.HTTPError as exc:
            raise GeneratorTransportError("GENERATOR_HTTP_ERROR") from exc

        self._remaining(deadline)
        try:
            envelope = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GeneratorTransportError("GENERATOR_RESPONSE_NOT_JSON") from exc
        content = _message_content(envelope)
        self._remaining(deadline)
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise GeneratorTransportError("GENERATOR_CONTENT_NOT_JSON") from exc
        self._remaining(deadline)
        return payload

    def _read_bounded_body(self, response: httpx.Response, deadline: float) -> bytes:
        self._remaining(deadline)
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
            self._remaining(deadline)
            received += len(chunk)
            if received > self.max_response_bytes:
                raise GeneratorTransportError("GENERATOR_RESPONSE_TOO_LARGE")
            chunks.append(chunk)
        self._remaining(deadline)
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


def _validated_positive_seconds(value: object, code: str) -> float:
    if isinstance(value, bool):
        raise GeneratorTransportError(code)
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeneratorTransportError(code) from exc
    if not math.isfinite(converted) or converted <= 0:
        raise GeneratorTransportError(code)
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
    def __init__(
        self,
        store: DocumentStore,
        generator: OpenAICompatibleGenerator | None = None,
        admission_authenticator: AdmissionAuthenticator | None = None,
    ) -> None:
        self.store = store
        self.generator = generator
        self.admission_authenticator = admission_authenticator

    def close(self) -> None:
        if self.generator is not None:
            self.generator.close()

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
        generation_admission_authentication: dict | None = None,
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
            generation_admission_authentication=(
                generation_admission_authentication or {"applied": False}
            ),
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

        evidence_snapshot = build_evidence_snapshot(safe_evidence) if use_llm else None
        prompt = build_prompt(question, safe_evidence, mode)
        if use_llm:
            if self.generator is None:
                self.generator = OpenAICompatibleGenerator()
            payload = self.generator.generate(prompt)
            allowed = {item.chunk_id for item in safe_evidence}
            assert evidence_snapshot is not None
            admission = admit_generation(
                payload,
                allowed,
                CONTENT_BOUND_GENERATION_POLICY,
                evidence_snapshot_sha256=evidence_snapshot.snapshot_sha256,
            )
            claim_rows = ()
            if admission.accepted:
                claim_rows = tuple(
                    trace_claim_support(
                        admission.answer,
                        admission.citations,
                        retrieval,
                        admission.claim_citations,
                    )
                )
                admission = bind_claim_support(
                    admission,
                    claim_support_sha256(claim_rows),
                )
            admission_audit = admission.to_audit_dict()
            admission_authentication = (
                self.admission_authenticator.authenticate(admission_audit)
                if self.admission_authenticator is not None
                else {"applied": False}
            )
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
                    admission_authentication,
                    "openai-compatible",
                )
                return replace(result, rejected_citations=rejected)

            citations = admission.citations
            answer_text = admission.answer
            audit = audit_grounding(answer_text, citations, retrieval)
            claims = tuple(item.to_dict() for item in claim_rows)
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
                generation_admission_authentication=admission_authentication,
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
            generation_admission_authentication={"applied": False},
            prompt_mode=mode.value,
            generator="extractive",
            retriever="hybrid-word-char-tfidf",
        )
