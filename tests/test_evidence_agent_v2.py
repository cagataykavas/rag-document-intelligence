from src.evidence_agent_constrained import candidate_actions


def test_initial_search_is_explicitly_forced_scaffolding():
    result = candidate_actions({'claim': 'water', 'history': [], 'remaining_actions': 6})
    assert result == [({'tool': 'search', 'query': 'water'}, 'Search for the claim.')]


def test_finish_only_has_no_retrieval_actions_and_cites_read_sentences_only():
    state = {'claim': 'Water is wet', 'remaining_actions': 1, 'history': [
        {'action': {'tool': 'search'}, 'result': [{'doc_id': '1', 'title': 'Paper'}, {'doc_id': '2', 'title': 'Unread'}]},
        {'action': {'tool': 'read'}, 'result': {'doc_id': '1', 'sentences': [
            {'sentence': 0, 'text': 'Water is wet.'}, {'sentence': 1, 'text': 'Another sentence.'}]}}]}
    options = candidate_actions(state, finish_only=True)
    assert len(options) == 5
    assert all(a['tool'] == 'finish' for a, _ in options)
    assert all(c['doc_id'] == '1' for a, _ in options for c in a['citations'])
    assert options[-1][0]['verdict'] == 'NOT_ENOUGH_INFO'


def test_no_repeated_read_and_lexical_top_three_sentence_budget():
    state = {'claim': 'water', 'remaining_actions': 3, 'history': [
        {'action': {'tool': 'search'}, 'result': [{'doc_id': '1', 'title': 'Read'}, {'doc_id': '2', 'title': 'Unread'}]},
        {'action': {'tool': 'read'}, 'result': {'doc_id': '1', 'sentences': [
            {'sentence': i, 'text': 'water' if i == 4 else 'other'} for i in range(5)]}}]}
    options = candidate_actions(state)
    reads = [a for a, _ in options if a['tool'] == 'read']
    assert reads == [{'tool': 'read', 'doc_id': '2'}]
    cited = {c['sentence'] for a, _ in options if a['tool'] == 'finish' for c in a['citations']}
    assert cited == {0, 1, 4}


def test_requires_one_read_before_abstention():
    state = {'claim': 'water', 'remaining_actions': 5, 'history': [
        {'action': {'tool': 'search'}, 'result': [{'doc_id': '1', 'title': 'Paper'}]}]}
    actions = [a for a, _ in candidate_actions(state)]
    assert actions == [{'tool': 'read', 'doc_id': '1'}]
