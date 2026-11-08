import pytest

from src.hybrid import HybridSparseRetriever
from src.retrieval import Chunk
from src.scifact_benchmark import load_scifact, paired_bootstrap


@pytest.mark.parametrize('size,k,expected', [(2, 2, 2), (4, 4, 4), (5, 10, 5), (5, 3, 3)])
def test_hybrid_returns_requested_number(size, k, expected):
    chunks = [Chunk(str(i), str(i), 'fixture', f'policy evidence token{i}') for i in range(size)]
    results = HybridSparseRetriever().fit(chunks).search('policy', k=k)
    assert len(results) == expected
    assert len({row['chunk_id'] for row in results}) == expected


def test_paired_bootstrap_constant_difference():
    result = paired_bootstrap([0.1] * 10, seed=2026, samples=100)
    assert result['mean'] == pytest.approx(0.1)
    assert result['lower_95'] == pytest.approx(0.1)
    assert result['upper_95'] == pytest.approx(0.1)


def test_scifact_rejects_unknown_relevance(tmp_path):
    (tmp_path / 'corpus.jsonl').write_text('{"doc_id":1,"title":"Test","abstract":["Text"]}\n')
    (tmp_path / 'claims_dev.jsonl').write_text('{"id":1,"claim":"Query","evidence":{"2":[]}}\n')
    with pytest.raises(ValueError, match='unknown'):
        load_scifact(tmp_path)
