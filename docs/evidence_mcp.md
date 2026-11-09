# Local MCP corpus retrieval

`src.evidence_mcp` exposes the bounded evidence agent's `CorpusTools` through the
[official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk),
pinned to **1.26.0**. This is a real MCP stdio server, not a custom JSON-RPC imitation.
It shares the agent's retriever implementation and corpus fingerprint. The agent
itself currently calls its in-process tools; it does **not** route its model loop
through MCP. The adapter is a separately tested integration surface.

## Run

From the repository root, using Python 3.11+ on a POSIX system:

```sh
python -m venv .venv-mcp
.venv-mcp/bin/pip install -r requirements.txt -r requirements-mcp.txt
.venv-mcp/bin/python -m src.evidence_mcp --corpus /absolute/path/to/corpus.jsonl
```

The process speaks MCP on stdin/stdout. Start it through an MCP stdio client,
not an interactive chat shell. It does not listen on a port. The operator selects
the corpus once at startup; tool arguments cannot change that file or its path.
Use only a corpus you intend to expose to the local client.

Corpus JSONL uses one object per line:

```json
{"doc_id":"study-1","title":"Sleep study","abstract":["Participants slept eight hours.","Memory scores improved."]}
```

SciFact's `id` field is also accepted in place of `doc_id`. Document identifiers
must be unique, and the corpus must be nonempty.

## Tools and boundaries

- `search_corpus({"query": "sleep memory"})`: returns up to three IDs and titles
  using the existing sparse retriever. Query is a strict string of 1–500
  characters; whitespace-only queries are rejected.
- `get_evidence({"doc_id": "study-1"})`: returns numbered abstract sentences.
  The strict string ID must exist in the configured corpus and be 1–100 characters.
- Both tools return JSON text with `corpus_fingerprint` and `result`, and advertise
  read-only, non-destructive, idempotent, closed-world hints.
- Serialized tool output is capped at 65,536 UTF-8 bytes; larger responses fail
  with a tool error rather than silently truncating citation evidence.
- Unexpected properties, missing required arguments, wrong types, unknown tools,
  and unknown document IDs produce tool errors. There is no write, shell,
  arbitrary filesystem-read, or network-fetch tool.
- The MCP API permits direct reads of any configured corpus ID. The bounded
  agent's separate search-before-read and citation-membership controls are not
  authorization controls for this API.
- Titles and abstracts are untrusted data. Returning a source sentence does not
  prove entailment or truth, and consumers must not execute its instructions.
- There is no remote deployment, authentication, per-user corpus filtering,
  distributed orchestration, or production-security claim. An authorized local
  client can read the entire selected corpus. Tool annotations are descriptive
  hints, not an access-control mechanism.

## Reproduce the integration test

```sh
.venv-mcp/bin/pytest -q tests/test_evidence_mcp.py
.venv-mcp/bin/ruff check src/evidence_mcp.py tests/test_evidence_mcp.py
```

The test launches a real subprocess and connects with `mcp.ClientSession` over
`mcp.client.stdio.stdio_client`. It performs protocol initialization, capability
and tool discovery, successful search/read calls, matching corpus fingerprints,
ten invalid-input/unknown-tool calls, and a successful read after rejection.
It confirms that source instruction text remains data and that the original
fixture corpus bytes are unchanged. Set `EVIDENCE_MCP_REPORT` to an output JSON
path when running pytest to record the exact invalid calls, responses, protocol
version, SDK version, and corpus hashes (see `evidence/mcp-integration.json`). It also rejects an oversized response. It uses three synthetic documents, no credentials,
no private data, and no model download or hosted model API. The test is skipped
when the optional MCP dependency is absent, so base-suite success alone does not
prove this integration passed.

## Integrated agent mode

The original standalone adapter tests above remain historical transport evidence.
The agent now offers explicit `--transport mcp` with strict client response
validation, bounded read-only retries and SDK span export. See
[integrated transport and observability](integrated-mcp-observability.md) for
architecture, reproduction, measured parity and limitations.
