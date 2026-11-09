"""Test-only real MCP server: transport/contract faults, never production flags."""
import asyncio
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from src.evidence_agent import CorpusTools

mode, corpus_path, marker = sys.argv[1:]
with Path(marker + '.pids').open('a') as stream:
    stream.write(str(os.getpid()) + '\n')
corpus = CorpusTools(list(map(json.loads, Path(corpus_path).read_text().splitlines())))
server = Server('fault-fixture')

@server.list_tools()
async def listing():
    return [Tool(name=name, inputSchema={'type': 'object'})
            for name in ['search_corpus', 'get_evidence']]

@server.call_tool()
async def call(name, arguments):
    count = int(Path(marker).read_text()) if Path(marker).exists() else 0
    Path(marker).write_text(str(count + 1))
    if mode == 'disconnect' or (mode == 'disconnect_once' and count == 0):
        os._exit(19)
    if mode == 'timeout' or (mode == 'timeout_once' and count == 0):
        await asyncio.sleep(60)
    if mode == 'remote_error':
        raise ValueError('PRIVATE SECRET remote failure')
    result = {'matches': corpus.search(arguments['query'])} if name == 'search_corpus' else corpus.read(arguments['doc_id'])
    fingerprint = corpus.fingerprint
    if mode == 'wrong_corpus':
        fingerprint = 'wrong'
    if mode == 'bad_schema':
        result['unexpected'] = 'PRIVATE SECRET'
    if mode == 'disconnect_after_success':
        asyncio.get_running_loop().call_later(0.1, os._exit, 19)
    return [TextContent(type='text', text=json.dumps({'corpus_fingerprint': fingerprint, 'result': result}))]

async def main():
    if mode == 'startup_timeout':
        await asyncio.sleep(60)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

asyncio.run(main())
