"""Real sessions, owned task groups, timeout/disconnect recovery and cleanup."""
import json
import os
import sys
import time
from pathlib import Path

import pytest
pytest.importorskip('mcp')
pytest.importorskip('opentelemetry.sdk')
from mcp import StdioServerParameters
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from src.evidence_agent import CorpusTools
from src.evidence_mcp_client import MCPContractError
from src.evidence_mcp_persistent import PersistentMCPCorpusTools
from src.evidence_telemetry import Telemetry


@pytest.fixture
def setup(tmp_path, request):
    docs = [{'doc_id': 'one', 'title': 'Sleep', 'abstract': ['Sleep improves memory.']}]
    path = tmp_path / 'corpus.jsonl'
    path.write_text('\n'.join(map(json.dumps, docs)))
    exporter = InMemorySpanExporter()
    telemetry = Telemetry(exporter)
    clients = []
    def client(mode=None, **kwargs):
        parameters = None
        if mode:
            parameters = StdioServerParameters(command=sys.executable, args=[
                str(Path(__file__).parent / 'fixtures' / 'mcp_fault_server.py'),
                mode, str(path), str(tmp_path / 'count')])
        tools = PersistentMCPCorpusTools(path, CorpusTools(docs).fingerprint,
            telemetry=telemetry, parameters=parameters, **kwargs)
        clients.append(tools)
        return tools
    yield client, exporter, telemetry
    for tools in clients:
        tools.close()
        assert tools._thread is None or not tools._thread.is_alive()
    pids = tmp_path / 'count.pids'
    if pids.exists():
        for pid in map(int, pids.read_text().splitlines()):
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)  # verify actual test-server processes were reaped
    telemetry.shutdown()
    if os.environ.get('EVIDENCE_PERSISTENT_SPANS'):
        destination = Path(os.environ['EVIDENCE_PERSISTENT_SPANS'])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / (request.node.name + '.jsonl')).write_text('\n'.join(
            json.dumps(json.loads(s.to_json())) for s in exporter.get_finished_spans()) + '\n')


def test_reuses_one_process_and_cleans_up(setup):
    factory, exporter, telemetry = setup
    client = factory('normal')
    with telemetry.span('fixture.task'):
        with client:
            assert client.search('sleep')[0]['doc_id'] == 'one'
            assert client.read('one')['sentences'][0]['sentence'] == 0
            assert client.search('memory')[0]['doc_id'] == 'one'
            assert client.connections_started == 1
        assert not client._thread.is_alive()
        client.close()  # idempotent
    spans = exporter.get_finished_spans()
    assert len([s for s in spans if s.name == 'mcp.connect']) == 1
    assert len([s for s in spans if s.name == 'mcp.attempt']) == 3
    assert len({s.context.trace_id for s in spans}) == 1
    ids = {s.context.span_id for s in spans}
    assert all(not s.parent or s.parent.span_id in ids for s in spans)
    with pytest.raises(RuntimeError, match='closed'):
        client.read('one')


@pytest.mark.parametrize('mode', ['wrong_corpus', 'bad_schema', 'remote_error'])
def test_contract_faults_not_retried(setup, tmp_path, mode):
    client = setup[0](mode)
    with pytest.raises(MCPContractError):
        client.search('sleep')
    assert (tmp_path / 'count').read_text() == '1'
    assert client.connections_started == 1


def test_disconnect_once_reconnects_then_reuses(setup, tmp_path):
    client = setup[0]('disconnect_once')
    assert client.search('sleep')[0]['doc_id'] == 'one'
    assert client.read('one')['doc_id'] == 'one'
    assert client.connections_started == 2
    assert (tmp_path / 'count').read_text() == '3'
    attempts = [s for s in setup[1].get_finished_spans() if s.name == 'mcp.attempt']
    assert [s.status.status_code for s in attempts] == [StatusCode.ERROR, StatusCode.UNSET, StatusCode.UNSET]


@pytest.mark.parametrize('mode', ['disconnect', 'timeout'])
def test_bounded_failure_and_owner_cleanup(setup, tmp_path, mode):
    client = setup[0](mode, timeout=2)
    started = time.monotonic()
    with pytest.raises(OSError):
        client.search('sleep')
    client.close()
    assert time.monotonic() - started < 12
    assert (tmp_path / 'count').read_text() == '2'
    assert client.connections_started == 2
    assert not client._thread.is_alive()


def test_close_before_start_is_safe(setup):
    client = setup[0]()
    client.close()
    client.close()
    assert client._thread is None


def test_idle_disconnect_reconnects_on_next_call(setup, tmp_path):
    client = setup[0]('disconnect_after_success')
    assert client.search('sleep')[0]['doc_id'] == 'one'
    time.sleep(0.3)
    assert client.read('one')['doc_id'] == 'one'
    assert client.connections_started == 2
    assert (tmp_path / 'count').read_text() == '2'


def test_timeout_once_reconnects_and_then_reuses(setup, tmp_path):
    client = setup[0]('timeout_once', timeout=2)
    assert client.search('sleep')[0]['doc_id'] == 'one'
    assert client.read('one')['doc_id'] == 'one'
    assert client.connections_started == 2
    assert (tmp_path / 'count').read_text() == '3'


def test_handshake_timeout_closes_children(setup, tmp_path):
    client = setup[0]('startup_timeout', timeout=2)
    with pytest.raises(OSError):
        client.search('sleep')
    client.close()
    assert not (tmp_path / 'count').exists()  # no tool dispatch before initialize
    assert len((tmp_path / 'count.pids').read_text().splitlines()) == 2


def test_concurrent_call_and_close_are_rejected(setup, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    client = setup[0]('timeout_once', timeout=2)
    with ThreadPoolExecutor(max_workers=1) as pool:
        work = pool.submit(client.search, 'sleep')
        deadline = time.monotonic() + 4
        while not (tmp_path / 'count').exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (tmp_path / 'count').exists()
        with pytest.raises(RuntimeError, match='serial caller'):
            client.read('one')
        with pytest.raises(RuntimeError, match='during a call'):
            client.close()
        assert work.result(timeout=8)[0]['doc_id'] == 'one'
    assert client.read('one')['doc_id'] == 'one'


def test_snapshot_identity_survives_reuse_and_rejects_changed_reconnect(setup, tmp_path):
    client = setup[0]('normal')
    assert client.read('one')['sentences'][0]['text'] == 'Sleep improves memory.'
    (tmp_path / 'corpus.jsonl').write_text(json.dumps({
        'doc_id': 'one', 'title': 'Sleep', 'abstract': ['A different corpus snapshot.']}))
    assert client.read('one')['sentences'][0]['text'] == 'Sleep improves memory.'
    client.close()
    new_client = setup[0]('normal')  # expects the original fingerprint
    with pytest.raises(MCPContractError):
        new_client.read('one')


def test_lifecycle_switch_cannot_resume_existing_checkpoint(setup, tmp_path):
    from src.evidence_agent import EvidenceAgent
    from src.evidence_mcp_client import MCPCorpusTools
    class Policy:
        identity = 'scripted-config-check'
        def generate(self, messages):
            return json.dumps({'tool': 'finish', 'verdict': 'NOT_ENOUGH_INFO', 'citations': []})
    reused = setup[0]()
    fresh = MCPCorpusTools(tmp_path / 'corpus.jsonl', reused.fingerprint)
    db = tmp_path / 'config.db'
    EvidenceAgent(db, fresh, Policy()).run('id', 'claim')
    with pytest.raises(ValueError, match='configuration'):
        EvidenceAgent(db, reused, Policy()).run('id', 'claim')


def test_closed_bridge_fence_does_not_dispatch(setup):
    client = setup[0]()
    client._closed = True  # simulate the bridge's uncertain-outcome fence
    with pytest.raises(OSError, match='uncertain transport outcome'):
        client._invoke('search_corpus', {'query': 'sleep'})
    assert client._thread is None
