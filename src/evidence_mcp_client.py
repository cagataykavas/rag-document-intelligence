"""Read-only synchronous agent adapter using actual official MCP stdio sessions.

Each attempt owns a fresh server subprocess/session. This deliberately trades
startup latency for simple bounded failure isolation; no shared background loop.
Only transport failures retry, at most twice, and only read-only operations.
"""
import asyncio
import math
import sys
from datetime import timedelta
from pathlib import Path
from typing import Annotated

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.evidence_mcp import MAX_OUTPUT_BYTES
from src.evidence_telemetry import span


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Match(StrictModel):
    doc_id: Annotated[str, Field(min_length=1, max_length=100)]
    title: str


class Matches(StrictModel):
    matches: Annotated[list[Match], Field(max_length=3)]


class Sentence(StrictModel):
    sentence: Annotated[int, Field(ge=0)]
    text: str


class Evidence(StrictModel):
    doc_id: Annotated[str, Field(min_length=1, max_length=100)]
    sentences: Annotated[list[Sentence], Field(min_length=1)]


class Envelope(StrictModel):
    corpus_fingerprint: str
    result: dict


class MCPContractError(ValueError):
    """Nonretryable remote tool error or response-contract violation."""


class MCPCorpusTools:
    def __init__(self, corpus_path, fingerprint, *, timeout=15.0, attempts=2,
                 telemetry=None, parameters=None):
        if not math.isfinite(timeout) or not 0 < timeout <= 120:
            raise ValueError('timeout must be finite and in (0,120]')
        if type(attempts) is not int or not 1 <= attempts <= 2:
            raise ValueError('attempts must be 1 or 2')
        self.fingerprint = fingerprint
        self.timeout, self.attempts, self.telemetry = timeout, attempts, telemetry
        self.parameters = parameters or StdioServerParameters(command=sys.executable,
            args=['-m', 'src.evidence_mcp', '--corpus', str(Path(corpus_path).resolve())],
            cwd=str(Path(__file__).resolve().parents[1]))
        self.binding = f'mcp-stdio-v1;schema=1;timeout={timeout};attempts={attempts}'

    def validate(self, response, name, arguments):
        if response.isError:
            raise MCPContractError('remote tool rejected request')
        if len(response.content) != 1 or response.content[0].type != 'text':
            raise MCPContractError('expected one text response')
        encoded = response.content[0].text
        if len(encoded.encode('utf-8')) > MAX_OUTPUT_BYTES:
            raise MCPContractError('response exceeds byte budget')
        try:
            envelope = Envelope.model_validate_json(encoded)
            if envelope.corpus_fingerprint != self.fingerprint:
                raise MCPContractError('corpus identity mismatch')
            if name == 'search_corpus':
                matches = Matches.model_validate(envelope.result).model_dump()['matches']
                if len({m['doc_id'] for m in matches}) != len(matches):
                    raise MCPContractError('duplicate search IDs')
                return matches
            result = Evidence.model_validate(envelope.result).model_dump()
            if result['doc_id'] != arguments['doc_id']:
                raise MCPContractError('evidence document mismatch')
            if [s['sentence'] for s in result['sentences']] != list(range(len(result['sentences']))):
                raise MCPContractError('noncontiguous sentence numbering')
            return result
        except ValidationError as exc:
            raise MCPContractError('invalid response schema') from exc

    async def _attempt(self, name, arguments):
        # Cancellation covers initialization, discovery and call. SDK closes/kills
        # its subprocess on context exit; teardown can add SDK cleanup grace.
        with anyio.fail_after(self.timeout):
            async with stdio_client(self.parameters) as (reader, writer):
                async with ClientSession(reader, writer,
                        read_timeout_seconds=timedelta(seconds=self.timeout)) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    if {t.name for t in listed.tools} != {'search_corpus', 'get_evidence'}:
                        raise MCPContractError('tool capability mismatch')
                    response = await session.call_tool(name, arguments)
                    return self.validate(response, name, arguments)

    def _invoke(self, name, arguments):
        return asyncio.run(self._attempt(name, arguments))

    def close(self):
        """Fresh sessions already close after every attempt."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _call(self, name, arguments):
        for attempt in range(1, self.attempts + 1):
            with span(self.telemetry, 'mcp.attempt', **{'tool.name': name, 'retry.attempt': attempt}):
                try:
                    return self._invoke(name, arguments)
                except Exception as exc:
                    # anyio task groups can wrap contract errors. Never retry them.
                    def contract(error):
                        return isinstance(error, MCPContractError) or any(
                            contract(e) for e in getattr(error, 'exceptions', ()))
                    if contract(exc):
                        raise MCPContractError('MCP response contract rejected') from exc
                    if attempt == self.attempts:
                        raise OSError('MCP transport unavailable after bounded attempts') from exc
                    # Raising inside the span marks failed attempts without retaining
                    # potentially private remote error strings.
                    if self.telemetry:
                        from opentelemetry.trace import get_current_span, Status, StatusCode
                        active = get_current_span()
                        active.set_attribute('error.type', type(exc).__name__)
                        active.set_status(Status(StatusCode.ERROR))
        raise AssertionError('unreachable')

    def search(self, query):
        return self._call('search_corpus', {'query': query})

    def read(self, doc_id):
        return self._call('get_evidence', {'doc_id': doc_id})
