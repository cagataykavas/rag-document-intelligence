import json

import httpx
import pytest

from src.engine import GeneratorTransportError, OpenAICompatibleGenerator


class ChunkStream(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.iterated = False
        self.yielded_chunks = 0

    def __iter__(self):
        self.iterated = True
        for chunk in self.chunks:
            self.yielded_chunks += 1
            yield chunk


def response_body(answer: str = "Bounded response") -> bytes:
    content = json.dumps(
        {
            "answer": answer,
            "citations": ["doc-1:0"],
            "confidence": 0.9,
            "insufficient_evidence": False,
        }
    )
    return json.dumps({"choices": [{"message": {"content": content}}]}).encode()


def generator_for(
    handler,
    *,
    max_response_bytes: int,
) -> OpenAICompatibleGenerator:
    return OpenAICompatibleGenerator(
        max_response_bytes=max_response_bytes,
        transport=httpx.MockTransport(handler),
    )


def assert_error(code: str, generator: OpenAICompatibleGenerator) -> None:
    with pytest.raises(GeneratorTransportError) as error:
        generator.generate("question")
    assert error.value.code == code
    assert str(error.value) == code


def test_streams_exact_boundary_and_requests_identity_encoding() -> None:
    body = response_body()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept-encoding"] == "identity"
        request_payload = json.loads(request.content)
        assert request_payload["n"] == 1
        return httpx.Response(
            200,
            stream=ChunkStream([body[:7], body[7:31], body[31:]]),
        )

    payload = generator_for(handler, max_response_bytes=len(body)).generate("question")

    assert payload["answer"] == "Bounded response"
    assert payload["citations"] == ["doc-1:0"]


def test_rejects_oversized_declared_length_before_reading_body() -> None:
    stream = ChunkStream([response_body()])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Length": "1000"},
            stream=stream,
        )

    assert_error(
        "GENERATOR_RESPONSE_TOO_LARGE",
        generator_for(handler, max_response_bytes=999),
    )
    assert stream.iterated is False


def test_rejects_oversized_chunked_body_without_content_length() -> None:
    stream = ChunkStream([b"x" * (16 * 1024)] * 100)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    assert_error(
        "GENERATOR_RESPONSE_TOO_LARGE",
        generator_for(handler, max_response_bytes=128 * 1024),
    )
    assert stream.yielded_chunks == 9


@pytest.mark.parametrize(
    ("headers", "code"),
    [
        ({"Content-Length": "not-a-number"}, "INVALID_RESPONSE_LENGTH"),
        ({"Content-Encoding": "gzip"}, "UNSUPPORTED_RESPONSE_ENCODING"),
    ],
)
def test_rejects_ambiguous_transport_metadata(
    headers: dict[str, str], code: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers=headers, stream=ChunkStream([response_body()])
        )

    assert_error(code, generator_for(handler, max_response_bytes=1024))


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (b"not-json", "GENERATOR_RESPONSE_NOT_JSON"),
        (json.dumps({"choices": []}).encode(), "INVALID_GENERATOR_ENVELOPE"),
        (
            json.dumps({"choices": [{"message": {"content": "not-json"}}]}).encode(),
            "GENERATOR_CONTENT_NOT_JSON",
        ),
    ],
)
def test_rejects_malformed_provider_responses(body: bytes, code: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=ChunkStream([body]))

    assert_error(code, generator_for(handler, max_response_bytes=1024))


@pytest.mark.parametrize("value", [0, 1024 * 1024 + 1, True, "1.5", "unbounded"])
def test_rejects_invalid_response_byte_limits(value: object) -> None:
    with pytest.raises(GeneratorTransportError) as error:
        OpenAICompatibleGenerator(max_response_bytes=value)  # type: ignore[arg-type]
    assert error.value.code == "INVALID_RESPONSE_BYTE_LIMIT"


def test_wraps_http_errors_without_exposing_provider_details() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="provider secret diagnostic")

    assert_error(
        "GENERATOR_HTTP_ERROR",
        generator_for(handler, max_response_bytes=1024),
    )
