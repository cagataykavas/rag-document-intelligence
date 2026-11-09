from typing import ClassVar

import numpy as np
import pytest

from src.dense import DenseRetriever, normalize, reciprocal_rank_fusion
from src.retrieval import Chunk


class FakeEncoder:
    signature: ClassVar[dict] = {'model': 'test-only'}
    calls = 0

    def encode(self, texts):
        self.calls += 1
        return np.array([[len(t), t.count('x') + 1] for t in texts], dtype=float)


def chunks():
    return [Chunk('a', 'a', 'test', 'xxx'), Chunk('b', 'b', 'test', 'word')]


def test_normalization_and_validation():
    assert np.allclose(normalize([[3, 4]]), [[.6, .8]])
    for bad in ([[0, 0]], [[float('nan'), 1]], [1, 2]):
        with pytest.raises(ValueError):
            normalize(bad)


def test_cache_and_invalidation(tmp_path):
    encoder = FakeEncoder()
    first = DenseRetriever(encoder, tmp_path).fit(chunks())
    second = DenseRetriever(encoder, tmp_path).fit(chunks())
    assert second.cache_hit and encoder.calls == 1
    assert first.vector_sha256 == second.vector_sha256
    assert first.search('xxx')[0]['doc_id'] == 'a'
    third = DenseRetriever(encoder, tmp_path).fit(list(reversed(chunks())))
    assert third.cache_key != first.cache_key and not third.cache_hit
    encoder.signature = {'model': 'changed'}
    assert DenseRetriever(encoder, tmp_path).fit(chunks()).cache_key != first.cache_key


def test_corrupt_cache_rejected(tmp_path):
    retriever = DenseRetriever(FakeEncoder(), tmp_path).fit(chunks())
    (tmp_path / (retriever.cache_key + '.npy')).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        DenseRetriever(FakeEncoder(), tmp_path).fit(chunks())


def test_dense_contracts():
    retriever = DenseRetriever(FakeEncoder())
    with pytest.raises(RuntimeError):
        retriever.search('x')
    with pytest.raises(ValueError):
        retriever.fit([])
    retriever.fit(chunks())
    with pytest.raises(ValueError):
        retriever.search('x', k=0)
    assert len(retriever.search('x', k=99)) == 2


def test_rrf_duplicate_credit_and_stable_ties():
    a, b = {'doc_id': 'a'}, {'doc_id': 'b'}
    results = reciprocal_rank_fusion([a, a], [b], k=2)
    assert [r['doc_id'] for r in results] == ['a', 'b']
    assert results[0]['score'] == results[1]['score'] == 1 / 61
    assert reciprocal_rank_fusion([a, b], [b], k=1)[0]['doc_id'] == 'b'


def test_dense_ties_use_document_id():
    retriever = DenseRetriever(FakeEncoder()).fit([
        Chunk('z', 'z', 'test', 'same'), Chunk('a', 'a', 'test', 'same')])
    assert [r['doc_id'] for r in retriever.search('same')] == ['a', 'z']


def test_duplicate_chunks_and_wrong_encoder_rows():
    with pytest.raises(ValueError, match='unique'):
        DenseRetriever(FakeEncoder()).fit([chunks()[0], chunks()[0]])
    encoder = FakeEncoder()
    encoder.encode = lambda texts: np.array([[1, 2]])
    with pytest.raises(ValueError, match='row mismatch'):
        DenseRetriever(encoder).fit(chunks())


def test_cache_valid_checksum_invalid_shape(tmp_path):
    import hashlib
    import json
    retriever = DenseRetriever(FakeEncoder(), tmp_path).fit(chunks())
    path = tmp_path / (retriever.cache_key + '.npy')
    np.save(path, np.array([[1., 0.]]))
    path.with_suffix('.json').write_text(json.dumps({'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'signature': FakeEncoder.signature}))
    with pytest.raises(ValueError, match='row mismatch'):
        DenseRetriever(FakeEncoder(), tmp_path).fit(chunks())


def test_cache_valid_checksum_unnormalized(tmp_path):
    import hashlib
    import json
    retriever = DenseRetriever(FakeEncoder(), tmp_path).fit(chunks())
    path = tmp_path / (retriever.cache_key + '.npy')
    np.save(path, np.array([[3., 4.], [3., 4.]]))
    path.with_suffix('.json').write_text(json.dumps({'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'signature': FakeEncoder.signature}))
    with pytest.raises(ValueError, match='not normalized'):
        DenseRetriever(FakeEncoder(), tmp_path).fit(chunks())


@pytest.mark.parametrize('kwargs', [{'k': 0}, {'rank_constant': -1}])
def test_rrf_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], **kwargs)


def test_truncation_counts_special_tokens():
    from src.dense_benchmark import truncation

    def tokenizer(texts, **kwargs):
        assert kwargs == {'truncation': False, 'padding': False}
        return {'input_ids': [list(range(n)) for n in (256, 257, 100)]}

    result = truncation(tokenizer, ['a', 'b', 'c'])
    assert result['over_limit'] == 1
    assert result['fraction_over_limit'] == 1 / 3


def test_benchmark_artifact_contract(tmp_path, monkeypatch):
    import json

    from src import dense_benchmark

    class TinyEncoder(FakeEncoder):
        def __init__(self, **kwargs):
            self.tokenizer = lambda texts, **kw: {'input_ids': [[1, 2] for _ in texts]}

    monkeypatch.setattr(dense_benchmark, 'MiniLMEncoder', TinyEncoder)
    monkeypatch.setattr(dense_benchmark.importlib.metadata, 'version', lambda p: 'test-only')
    monkeypatch.setattr(dense_benchmark.subprocess, 'check_output', lambda *a, **kw: '')
    (tmp_path / 'corpus.jsonl').write_text('\n'.join(json.dumps(d) for d in [
        {'doc_id': 1, 'title': 'alpha', 'abstract': ['alpha treatment']},
        {'doc_id': 2, 'title': 'beta', 'abstract': ['beta treatment']}]))
    (tmp_path / 'claims_dev.jsonl').write_text(json.dumps(
        {'id': 1, 'claim': 'alpha treatment', 'evidence': {'1': []}}))
    output = tmp_path / 'result.json'
    dense_benchmark.run(tmp_path, output, tmp_path / 'cache')
    result = json.loads(output.read_text())
    assert set(result['models']) == {'word', 'hybrid', 'dense', 'rrf'}
    assert result['documents'] == 2 and result['queries'] == 1
    assert result['models']['word']['metrics']['1']['recall_at_k'] == 1
    assert result['models']['dense']['cache_hit'] is False
    assert len(result['models']['dense']['vector_sha256']) == 64
    assert result['models']['rrf']['query_latency_ms']['p50'] > 0


def test_normalization_rejects_finite_overflow():
    with pytest.raises(ValueError, match='nonfinite'):
        normalize([[3e38, 3e38]])


def test_cache_rejects_wrong_signature(tmp_path):
    import json
    retriever = DenseRetriever(FakeEncoder(), tmp_path).fit(chunks())
    path = tmp_path / (retriever.cache_key + '.json')
    manifest = json.loads(path.read_text())
    manifest['signature'] = {'model': 'wrong'}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='signature'):
        DenseRetriever(FakeEncoder(), tmp_path).fit(chunks())


def test_encoder_and_cache_reject_wrong_dimension(tmp_path):
    encoder = FakeEncoder()
    DenseRetriever(encoder, tmp_path).fit(chunks())
    encoder.dimension = 384
    with pytest.raises(ValueError, match='dimension'):
        DenseRetriever(encoder, tmp_path).fit(chunks())
    with pytest.raises(ValueError, match='dimension'):
        DenseRetriever(encoder).fit(chunks())
