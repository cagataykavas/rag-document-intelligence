"""Actual stdio subprocess faults and SDK span objects, no mock JSON traces."""
import json
import os
import sys
import time
from pathlib import Path
import pytest
pytest.importorskip('mcp')
pytest.importorskip('opentelemetry.sdk')
from mcp import StdioServerParameters
from mcp.types import CallToolResult, TextContent
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from src.evidence_agent import CorpusTools, EvidenceAgent
from src.evidence_mcp_client import MCPCorpusTools, MCPContractError
from src.evidence_telemetry import Telemetry

@pytest.fixture
def setup(tmp_path, request):
    docs = [{'doc_id': 'one', 'title': 'PRIVATE SECRET sleep',
             'abstract': ['PRIVATE SECRET sleep improves memory.']},
            {'doc_id': 'two', 'title': 'Plants', 'abstract': ['Plants grow.']}]
    path = tmp_path / 'corpus.jsonl'
    path.write_text('\n'.join(map(json.dumps, docs)))
    exporter = InMemorySpanExporter()
    telemetry = Telemetry(exporter)
    yield path, CorpusTools(docs), exporter, telemetry
    telemetry.shutdown()
    if os.environ.get('EVIDENCE_INTEGRATED_SPANS'):
        destination = Path(os.environ['EVIDENCE_INTEGRATED_SPANS'])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / (request.node.name + '.jsonl')).write_text(
            '\n'.join(json.dumps(json.loads(s.to_json())) for s in exporter.get_finished_spans()) + '\n')

class Policy:
    identity = 'scripted-test-only'
    def generate(self, messages):
        state = json.loads(messages[-1]['content'])
        history = state['history']
        if not history:
            return json.dumps({'tool': 'search', 'query': 'PRIVATE SECRET sleep'})
        if not any(e.get('action', {}).get('tool') == 'read' for e in history):
            return json.dumps({'tool': 'read', 'doc_id': 'one'})
        return json.dumps({'tool': 'finish', 'verdict': 'SUPPORT',
                           'citations': [{'doc_id': 'one', 'sentence': 0}]})

def fault_tools(setup, tmp_path, mode, **kwargs):
    path, direct, _, telemetry = setup
    parameters = StdioServerParameters(command=sys.executable, args=[
        str(Path(__file__).parent / 'fixtures' / 'mcp_fault_server.py'),
        mode, str(path), str(tmp_path / 'count')])
    return MCPCorpusTools(path, direct.fingerprint, telemetry=telemetry,
                          parameters=parameters, **kwargs)

def test_real_agent_transport_parity_and_sdk_linkage(setup, tmp_path):
    path, direct, exporter, telemetry = setup
    mcp = MCPCorpusTools(path, direct.fingerprint, telemetry=telemetry)
    states = [EvidenceAgent(tmp_path / f'{name}.db', tools, Policy(), telemetry=telemetry)
              .run('id', 'PRIVATE SECRET claim') for name, tools in [('direct', direct), ('mcp', mcp)]]
    assert states[0]['answer'] == states[1]['answer']
    assert [e['result'] for e in states[0]['trace']] == [e['result'] for e in states[1]['trace']]
    spans = exporter.get_finished_spans()
    tasks = [s for s in spans if s.name == 'agent.task']
    assert len(tasks) == 2
    assert len([s for s in spans if s.name == 'mcp.attempt']) == 2
    for task in tasks:
        children = [s for s in spans if s.context.trace_id == task.context.trace_id and s != task]
        assert children and all(s.parent for s in children)
        ids = {s.context.span_id for s in spans}
        assert all(s.parent.span_id in ids for s in children)
    assert 'PRIVATE SECRET' not in '\n'.join(s.to_json() for s in spans)
    assert all(s.status.status_code != StatusCode.ERROR for s in spans)

@pytest.mark.parametrize('mode', ['wrong_corpus', 'bad_schema', 'remote_error'])
def test_contract_faults_do_not_retry(setup, tmp_path, mode):
    tools = fault_tools(setup, tmp_path, mode)
    with pytest.raises(MCPContractError):
        tools.search('sleep')
    assert (tmp_path / 'count').read_text() == '1'
    spans = setup[2].get_finished_spans()
    assert spans[-1].status.status_code == StatusCode.ERROR
    assert 'PRIVATE SECRET' not in spans[-1].to_json()

def test_disconnect_retry_recovers(setup, tmp_path):
    tools = fault_tools(setup, tmp_path, 'disconnect_once')
    assert tools.search('sleep')[0]['doc_id'] == 'one'
    spans = setup[2].get_finished_spans()
    assert [s.status.status_code for s in spans] == [StatusCode.ERROR, StatusCode.UNSET]
    assert (tmp_path / 'count').read_text() == '2'

@pytest.mark.parametrize('mode', ['disconnect', 'timeout'])
def test_transport_fault_is_bounded(setup, tmp_path, mode):
    tools = fault_tools(setup, tmp_path, mode, timeout=2.0)
    started = time.monotonic()
    with pytest.raises(OSError):
        tools.search('sleep')
    assert time.monotonic() - started < 12  # includes SDK teardown; not realtime guarantee
    assert len(setup[2].get_finished_spans()) == 2
    assert (tmp_path / 'count').read_text() == '2'  # both attempts reached the server call
    assert all(s.status.status_code == StatusCode.ERROR for s in setup[2].get_finished_spans())

@pytest.mark.parametrize('result', [
    {'matches': [{'doc_id': 'one', 'title': 'x'}] * 2},
    {'matches': [{'doc_id': 1, 'title': 'x'}]},
    {'matches': [], 'extra': 1},
])
def test_strict_result_shape(setup, result):
    path, direct, _, _ = setup
    client = MCPCorpusTools(path, direct.fingerprint)
    response = CallToolResult(content=[TextContent(type='text', text=json.dumps(
        {'corpus_fingerprint': direct.fingerprint, 'result': result}))])
    with pytest.raises(MCPContractError):
        client.validate(response, 'search_corpus', {'query': 'sleep'})

def test_transport_config_change_rejected(setup, tmp_path):
    path, direct, _, _ = setup
    EvidenceAgent(tmp_path / 'state.db', direct, Policy()).run('id', 'claim')
    with pytest.raises(ValueError, match='configuration'):
        EvidenceAgent(tmp_path / 'state.db', MCPCorpusTools(path, direct.fingerprint), Policy()).run('id', 'claim')


@pytest.mark.parametrize('result', [
    {'doc_id': 'other', 'sentences': [{'sentence': 0, 'text': 'x'}]},
    {'doc_id': 'one', 'sentences': [{'sentence': 1, 'text': 'x'}]},
    {'doc_id': 'one', 'sentences': [{'sentence': True, 'text': 'x'}]},
    {'doc_id': 'one', 'sentences': []},
    {'doc_id': 'one', 'sentences': [{'sentence': 0, 'text': 'x' * 66000}]},
])
def test_read_result_binding_and_limits(setup, result):
    path, direct, _, _ = setup
    client = MCPCorpusTools(path, direct.fingerprint)
    response = CallToolResult(content=[TextContent(type='text', text=json.dumps(
        {'corpus_fingerprint': direct.fingerprint, 'result': result}))])
    with pytest.raises(MCPContractError):
        client.validate(response, 'get_evidence', {'doc_id': 'one'})


def test_model_and_exhaustion_error_spans_are_redacted(setup, tmp_path):
    class FailingPolicy:
        identity = 'failure-fixture'
        def generate(self, messages):
            raise RuntimeError('PRIVATE SECRET model fault')
    _, tools, exporter, telemetry = setup
    state = EvidenceAgent(tmp_path / 'failed.db', tools, FailingPolicy(),
                          max_steps=1, telemetry=telemetry).run('id', 'PRIVATE SECRET')
    assert state['status'] == 'exhausted'
    spans = exporter.get_finished_spans()
    for name in ['agent.task', 'agent.policy']:
        assert next(s for s in spans if s.name == name).status.status_code == StatusCode.ERROR
    assert 'PRIVATE SECRET' not in '\n'.join(s.to_json() for s in spans)


def test_checkpoint_error_is_exported_without_exception_text(setup, tmp_path, monkeypatch):
    _, tools, exporter, telemetry = setup
    agent = EvidenceAgent(tmp_path / 'fail.db', tools, Policy(), telemetry=telemetry)
    def fail(state):
        raise OSError('PRIVATE SECRET disk path')
    monkeypatch.setattr(agent, '_save_state', fail)
    with pytest.raises(OSError):
        agent.run('id', 'claim')
    spans = exporter.get_finished_spans()
    assert {s.name for s in spans} == {'checkpoint.save', 'agent.task'}
    assert all(s.status.status_code == StatusCode.ERROR for s in spans)
    assert 'PRIVATE SECRET' not in '\n'.join(s.to_json() for s in spans)


def test_canonical_corpus_identity():
    a = CorpusTools([{'doc_id': 42, 'title': 'Paper', 'abstract': ['A sentence.'], 'unused': 10}])
    b = CorpusTools([{'doc_id': '42', 'title': 'Paper', 'abstract': ['A sentence.']}])
    c = CorpusTools([{'doc_id': '42', 'title': 'Paper', 'abstract': ['Changed sentence.']}])
    assert a.fingerprint == b.fingerprint
    assert a.fingerprint != c.fingerprint
