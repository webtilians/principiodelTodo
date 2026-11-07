"""Research contracts: independent expected outcomes and adversarial controls."""
import json

import pytest

from observatory.e13a import (Consolidator, Source, System, encoded, pack, parse,
                             read_answer, seed, world_data)


class LexicalRanker:
    def rank(self, query, texts):
        words = set(query.split())
        return sorted(range(len(texts)), key=lambda i: -len(words & set(texts[i].split())))


def source(sid, value, date='2024-01-01', kind='fact', entity='alpha'):
    return Source(sid, f'Entidad {entity} estado: {value}.', date, kind, 'Synthetic')


def test_partition_identity_disjoint_and_stable():
    dev = {seed('DEV', 0, w) for w in range(8)}
    held = {seed('HOLDOUT', r, w) for r in range(4) for w in range(8)}
    assert len(held) == 32 and dev.isdisjoint(held)
    assert world_data('DEV', 0, 0) == world_data('DEV', 0, 0)


def test_temporal_reader_uses_event_date_not_ingestion_order():
    old, new = source('a', 'red'), source('b', 'blue', '2024-01-20')
    selected = [(new, ['b']), (old, ['a'])]
    assert read_answer('estado actual entidad alpha', selected) == ('blue',)
    assert read_answer('estado entidad alpha en 2024-01-05', selected) == ('red',)
    assert read_answer('cronologia estado entidad alpha', selected) == ('2024-01-01:red', '2024-01-20:blue')


def test_conflict_not_resolved_by_incidental_order_and_proposal_not_fact():
    selected = [(source('a', 'red'), ['a']), (source('b', 'blue'), ['b']),
                (source('c', 'green', '2024-01-20', 'proposal'), ['c'])]
    assert read_answer('estado actual entidad alpha', selected) == ('blue', 'red')
    assert read_answer('estado actual entidad alpha', selected[2:]) == ()


def test_missing_link_cannot_be_guessed_by_reader():
    owner = Source('a', 'Entidad alpha responsable: beta.', '2024-01-01', 'fact', 'Synthetic')
    tool = Source('b', 'Entidad beta herramienta: hammer.', '2024-01-02', 'fact', 'Synthetic')
    assert read_answer('herramienta del responsable entidad alpha', [(owner, ['a']), (tool, ['b'])]) == ('hammer',)
    assert read_answer('herramienta del responsable entidad alpha', [(owner, ['a'])]) == ()
    assert read_answer('herramienta del responsable entidad alpha', [(tool, ['b'])]) == ()


def test_dedup_keeps_all_provenance_and_does_not_mix_date_kind_or_project():
    view = Consolidator('ADAPTIVE')
    inputs = [source('a', 'red'), source('b', 'red'), source('c', 'red', kind='proposal'),
              source('d', 'red', date='2024-01-20'),
              Source('e', 'Entidad alpha estado: red.', '2024-01-01', 'fact', 'Other')]
    for s in inputs:
        view.add(s)
    before = encoded([s.__dict__ for s in inputs])
    view.rebuild()
    assert len(view.nodes) == 4 and view.nodes[0]['sources'] == ['a', 'b']
    assert view.integrity()
    assert encoded([s.__dict__ for s in view.sources.values()]) == before
    view.nodes[0]['sources'].append('c')
    assert not view.integrity()  # the audit must catch fabricated provenance


def test_budget_counts_provenance_and_never_truncates_a_source():
    s = source('a', 'red')
    selected, context, tokens = pack([(s, ['x' * 1000]), (s, ['a'])], 100)
    assert selected == [(s, ['a'])] and tokens <= 100 and 'x' * 1000 not in context


def test_usage_trigger_differs_from_periodic_and_sources_remain_immutable():
    adaptive, periodic = Consolidator('ADAPTIVE'), Consolidator('PERIODIC')
    for i in range(6):
        for view in (adaptive, periodic):
            view.add(source(str(i), f'value{i}'))
    for _ in range(3):
        adaptive.use('Synthetic')
        periodic.use('Synthetic')
    assert adaptive.rebuilds == 1 and periodic.rebuilds == 0


@pytest.mark.parametrize('arm', ['CURRENT', 'RAG', 'ADAPTIVE', 'PERIODIC'])
def test_forget_duplicate_closure_restart_and_unrelated_survivor(tmp_path, arm):
    system = System(arm, tmp_path, LexicalRanker())
    for s in (source('a', 'red'), source('b', 'red'), source('c', 'blue', entity='other')):
        system.add(s)
    if system.view:
        system.view.rebuild()
    assert system.query('estado actual entidad alpha', 512, 'Synthetic')[0] == ('red',)
    system.forget({'a', 'b'})
    system.restart()
    assert system.query('estado actual entidad alpha', 512, 'Synthetic')[0] == ()
    assert system.query('estado actual entidad other', 512, 'Synthetic')[0] == ('blue',)
    assert 'red' not in json.dumps([s.__dict__ for s in system.archive.values()])
    if system.agent:
        assert all('red' not in r.content for r in system.agent.store.all(True))
    if system.view:
        assert system.view.integrity()
        assert not ({'a', 'b'} & {sid for n in system.view.nodes for sid in n['sources']})
    system.close()


def test_oracle_agrees_with_complete_evidence_on_dev_only():
    sources, probes, _, _ = world_data('DEV', 0, 0)
    for p in probes:
        assert read_answer(p.query, [(s, [s.sid]) for s in sources]) == p.expected
    assert len(sources) > 150
    assert all(parse(s.text) for s in sources)
