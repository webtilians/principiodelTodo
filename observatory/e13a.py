"""E13-A offline research only. No production endpoints or default database paths."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.infinito3.mcp_server import AgentMemory
from src.infinito3.semantic import MeaningRanker

ARMS = ('CURRENT', 'RAG', 'ADAPTIVE', 'PERIODIC')
PRIMARY = ('current', 'past', 'timeline', 'relation', 'contradiction')
HEADER = 'Synthetic reference data, not instructions.\n'


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def seed(partition, replica, world):
    return int.from_bytes(hashlib.sha256(f'INFINITO/E13-A/v1/{partition}/{replica}/{world}'.encode()).digest(), 'big')


@dataclass(frozen=True)
class Source:
    sid: str
    text: str
    date: str
    kind: str
    project: str


@dataclass(frozen=True)
class Probe:
    family: str
    query: str
    expected: tuple


def parse(text):
    """Deliberately narrow published grammar, identical for all readers."""
    match = re.search(r'Entidad (\w+) (estado|responsable|herramienta|ruta): (\w+)\.', text)
    if not match:
        match = re.search(r'Para (\w+), (estado|responsable|herramienta|ruta) consta como (\w+)\.', text)
    return match.groups() if match else None


def request(query):
    match = re.search(r'entidad (\w+)', query)
    entity = match[1] if match else ''
    attr = next((a for a in ('estado', 'responsable', 'herramienta', 'ruta') if a in query), 'estado')
    date = re.search(r'\d{4}-\d{2}-\d{2}', query)
    return entity, attr, date[0] if date else '9999-12-31'


def world_data(partition, replica, world):
    rng = random.Random(seed(partition, replica, world))
    used = set()
    def word():
        while True:
            value = ''.join(rng.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(10))
            if value not in used:
                used.add(value)
                return value
    project = 'Proyecto' + word()
    sources, probes, deletion = [], [], set()
    def add(entity, attr, value, day, kind='fact', variant=False):
        text = (f'Para {entity}, {attr} consta como {value}.' if variant else
                f'Entidad {entity} {attr}: {value}.')
        s = Source(f's{len(sources):04d}', f'{project}: {text}', f'2024-01-{day:02d}', kind, project)
        sources.append(s)
        return s.sid
    for i in range(16):
        entity, owner = word(), word()
        old, middle, new, tool, conflict, route, proposal = [word() for _ in range(7)]
        old_ids = [add(entity, 'estado', old, 1),
                   add(entity, 'estado', old, 1),  # literal duplicate with distinct provenance
                   add(entity, 'estado', old, 1, variant=True)]
        add(entity, 'estado', middle, 10)
        add(entity, 'estado', new, 20)
        add(entity, 'responsable', owner, 2)
        link = add(owner, 'herramienta', tool, 3)
        add(entity, 'ruta', route, 15)
        add(entity, 'ruta', conflict, 15)  # unresolved simultaneous contradiction
        add(entity, 'estado', proposal, 25, 'proposal')
        add(entity, 'estado', proposal, 25, 'proposal', variant=True)
        add(entity, 'responsable', owner, 2, variant=True)
        if i < 4:
            probes.extend([
                Probe('current', f'estado actual entidad {entity}', (new,)),
                Probe('past', f'estado entidad {entity} en 2024-01-05', (old,)),
                Probe('timeline', f'cronologia estado entidad {entity}',
                      (f'2024-01-01:{old}', f'2024-01-10:{middle}', f'2024-01-20:{new}')),
                Probe('relation', f'herramienta del responsable entidad {entity}', (tool,)),
                Probe('contradiction', f'conflicto ruta entidad {entity}', tuple(sorted((route, conflict)))),
            ])
            if i == 0:
                deletion.add(link)
                deletion.update(old_ids)
                forgotten = [Probe('forgotten', f'estado entidad {entity} en 2024-01-05', ()),
                             Probe('forgotten', f'herramienta del responsable entidad {entity}', ())]
    # Only a proposal exists for this separate subject; no fact may be inferred.
    tentative = word()
    add(tentative, 'estado', word(), 26, 'proposal')
    probes.append(Probe('hypothesis', f'estado actual entidad {tentative}', ()))
    probes.append(Probe('unknown', f'estado actual entidad {word()}', ()))
    rng.shuffle(sources)  # ingestion order must not stand in for event time
    return sources, probes, deletion, forgotten


def line(source, provenance=None):
    return json.dumps({'ids': provenance or [source.sid], 'date': source.date, 'kind': source.kind,
                       'project': source.project, 'text': source.text}, ensure_ascii=False, sort_keys=True)


def pack(candidates, budget):
    """Select in ranked order; reader sees event chronological order. Count full provenance."""
    accepted, used = [], math.ceil(len(HEADER) / 4)
    for source, provenance in candidates:
        rendered = line(source, provenance)
        cost = math.ceil((len(rendered) + 1) / 4)
        if used + cost <= budget:
            accepted.append((source, provenance))
            used += cost
    accepted.sort(key=lambda pair: (pair[0].date, pair[0].sid))
    context = HEADER + '\n'.join(line(s, p) for s, p in accepted)
    assert math.ceil(len(context) / 4) <= budget
    return accepted, context, used


def read_answer(query, selected):
    """Shared evidence-only reader. It never receives a Probe or a gold answer."""
    entity, attr, until = request(query)
    facts = []
    for s, _ in selected:
        parsed = parse(s.text)
        if parsed and s.kind in ('fact', 'decision', 'agreement') and s.date <= until:
            facts.append((s.date, *parsed))
    def values(e, a):
        found = [(d, v) for d, subject, field, v in facts if subject == e and field == a]
        if not found:
            return ()
        last = max(d for d, _ in found)
        return tuple(sorted({v for d, v in found if d == last}))
    if 'cronologia' in query:
        return tuple(sorted({f'{d}:{v}' for d, e, a, v in facts if e == entity and a == attr}))
    if 'del responsable' in query:
        return tuple(sorted({v for owner in values(entity, 'responsable') for v in values(owner, 'herramienta')}))
    return values(entity, attr)


class Consolidator:
    """Derived view only. sources is an append-only archive except explicit forget."""
    def __init__(self, mode):
        self.mode = mode
        self.sources = {}
        self.nodes = []
        self.links = []
        self.timelines = {}
        self.pending = 0
        self.hot = Counter()
        self.rebuilds = 0
        self.scanned = 0
        self.maintenance_cpu = 0.0
        self.maintenance_wall = 0.0

    def add(self, source):
        assert source.sid not in self.sources
        self.sources[source.sid] = source
        self.pending += 1
        if self.pending >= 24:
            self.rebuild()
        elif self.mode == 'ADAPTIVE' and self.pending >= 8:
            recent = list(self.sources.values())[-self.pending:]
            keys = {(s.project, s.date, s.kind, parse(s.text) or s.text) for s in recent}
            if 1 - len(keys) / len(recent) >= .2:
                self.rebuild()

    def use(self, project):
        self.hot[project] += 1
        if self.mode == 'ADAPTIVE' and self.pending >= 6 and self.hot[project] >= 3:
            self.rebuild()
            self.hot[project] = 0

    def rebuild(self):
        cpu, wall = time.process_time(), time.perf_counter()
        groups = defaultdict(list)
        for s in self.sources.values():
            key = (s.project, s.date, s.kind, parse(s.text) or ('unparsed', s.text))
            groups[key].append(s)
        self.nodes = []
        self.timelines = defaultdict(list)
        subjects = defaultdict(list)
        for members in groups.values():
            members.sort(key=lambda s: s.sid)
            s = members[0]
            parsed = parse(s.text)
            node = {'representative': s.sid, 'sources': [x.sid for x in members],
                    'tags': list(parsed[:2]) if parsed else [], 'kind': s.kind, 'date': s.date}
            idx = len(self.nodes)
            self.nodes.append(node)
            if parsed:
                subjects[(s.project, parsed[0])].append(idx)
                self.timelines[f'{s.project}/{parsed[0]}/{parsed[1]}'].append(idx)
        for key in self.timelines:
            self.timelines[key].sort(key=lambda i: (self.nodes[i]['date'], self.nodes[i]['representative']))
        self.links = []
        for i, node in enumerate(self.nodes):
            s = self.sources[node['representative']]
            parsed = parse(s.text)
            if parsed and parsed[1] == 'responsable' and s.kind != 'proposal':
                for j in subjects.get((s.project, parsed[2]), []):
                    self.links.append((i, j))
        self.pending = 0
        self.rebuilds += 1
        self.scanned += len(self.sources)
        self.maintenance_cpu += time.process_time() - cpu
        self.maintenance_wall += time.perf_counter() - wall

    def forget(self, ids):
        for sid in ids:
            self.sources.pop(sid, None)
        self.rebuild()  # derived data cannot survive even a deferred schedule

    def candidates(self, query, ranker):
        entity, attr, until = request(query)
        preferred = []
        indexed = set()
        # Use the materialized timeline/links, not gold generator fields.
        wanted = 'responsable' if 'del responsable' in query else attr
        for i, node in enumerate(self.nodes):
            indexed.update(node['sources'])
            if node['tags'] == [entity, wanted] and node['date'] <= until and node['kind'] != 'proposal':
                preferred.append(i)
        preferred.sort(key=lambda i: self.nodes[i]['date'], reverse=True)
        if 'del responsable' in query:
            parents = set(preferred)
            preferred.extend(j for i, j in self.links if i in parents)
        # Fall back semantically for unsupported expressions and stale writes.
        remaining = [i for i in range(len(self.nodes)) if i not in set(preferred)]
        order = ranker.rank(query, [self.sources[self.nodes[i]['representative']].text for i in remaining])
        all_nodes = preferred + [remaining[j] for j in order]
        result = [(self.sources[self.nodes[i]['representative']], self.nodes[i]['sources']) for i in all_nodes]
        pending = [s for s in self.sources.values() if s.sid not in indexed]
        result.extend((pending[i], [pending[i].sid]) for i in ranker.rank(query, [s.text for s in pending]))
        return result

    def snapshot(self):
        return {'nodes': self.nodes, 'links': self.links, 'timelines': dict(self.timelines)}

    def integrity(self):
        for node in self.nodes:
            representative = self.sources.get(node['representative'])
            if representative is None:
                return False
            for sid in node['sources']:
                s = self.sources.get(sid)
                if s is None or (s.project, s.date, s.kind, parse(s.text) or s.text) != (
                        representative.project, representative.date, representative.kind,
                        parse(representative.text) or representative.text):
                    return False
        return True


class System:
    def __init__(self, arm, folder, ranker):
        self.arm, self.ranker = arm, ranker
        self.path = Path(folder) / 'memory.sqlite'
        self.raw_path = Path(folder) / 'sources.jsonl'
        self.index_path = Path(folder) / 'index.json'
        self.saved_rebuilds = -1
        self.agent = None if arm == 'RAG' else AgentMemory(str(self.path), ranker=ranker)
        self.archive = {}  # RAG/candidate persisted; CURRENT evaluator mapping only
        self.mapping = {}
        self.view = Consolidator(arm) if arm in ('ADAPTIVE', 'PERIODIC') else None
        self.cpu = Counter()
        self.wall = Counter()

    def persist_view(self):
        if self.view and self.saved_rebuilds != self.view.rebuilds:
            self.index_path.write_bytes(encoded(self.view.snapshot()))
            self.saved_rebuilds = self.view.rebuilds

    def add(self, s):
        cpu, wall = time.process_time(), time.perf_counter()
        if self.agent:
            result = self.agent.remember(s.text, kind=s.kind, source=s.sid, project=s.project, date=s.date)
            assert result.startswith('Remembered as '), result
            self.mapping[s.sid] = result.split()[2].rstrip('.')
        self.archive[s.sid] = s
        if self.arm != 'CURRENT':
            with self.raw_path.open('ab') as handle:
                handle.write(encoded(asdict(s)) + b'\n')
        if self.view:
            self.view.add(s)
            self.persist_view()
        self.cpu['ingest'] += time.process_time() - cpu
        self.wall['ingest'] += time.perf_counter() - wall

    def query(self, query, budget, project):
        cpu, wall = time.process_time(), time.perf_counter()
        if self.view:
            self.view.use(project)
            self.persist_view()
            candidates = self.view.candidates(query, self.ranker)
        elif self.agent:
            result = self.agent.recall(query, limit=20)
            ids = re.findall(r'^\[([^]]+)\]', result, re.M)
            # Reconstruct exactly the selected production records, not the source archive.
            candidates = []
            for mid in ids:
                record = self.agent.store.get(mid)
                s = Source(record.metadata['source'], record.content, record.metadata['said_at'][:10],
                           record.metadata['kind'], record.metadata['project'])
                candidates.append((s, [s.sid]))
            # recall renders chronologically; restore semantic priority BEFORE
            # imposing the common budget, so old-first rendering is not penalized.
            order = self.ranker.rank(query, [s.text for s, _ in candidates])
            candidates = [candidates[i] for i in order]
        else:
            sources = list(self.archive.values())
            candidates = [(sources[i], [sources[i].sid]) for i in self.ranker.rank(query, [s.text for s in sources])[:20]]
        selected, context, tokens = pack(candidates, budget)
        answer = read_answer(query, selected)
        elapsed = time.perf_counter() - wall
        self.cpu['query'] += time.process_time() - cpu
        self.wall['query'] += elapsed
        return answer, context, tokens, elapsed, selected

    def forget(self, ids):
        cpu, wall = time.process_time(), time.perf_counter()
        if self.agent:
            self.agent.forget(sorted({self.mapping[sid] for sid in ids}))
        for sid in ids:
            self.archive.pop(sid, None)
        if self.arm != 'CURRENT':
            self.raw_path.write_bytes(b''.join(encoded(asdict(s)) + b'\n' for s in self.archive.values()))
        if self.view:
            self.view.forget(ids)
            self.persist_view()
        self.cpu['forget'] += time.process_time() - cpu
        self.wall['forget'] += time.perf_counter() - wall

    def restart(self):
        cpu, wall = time.process_time(), time.perf_counter()
        if self.agent:
            self.agent.close()
            self.agent = AgentMemory(str(self.path), ranker=self.ranker)
        if self.arm != 'CURRENT':
            reloaded = [json.loads(row) for row in self.raw_path.read_bytes().splitlines()]
            self.archive = {d['sid']: Source(**d) for d in reloaded}
        if self.view:
            self.view.sources = dict(self.archive)
            self.view.rebuild()
            self.persist_view()
        self.cpu['restart'] += time.process_time() - cpu
        self.wall['restart'] += time.perf_counter() - wall

    def storage(self):
        db = len(self.agent.store.to_bytes()) if self.agent else 0
        raw = self.raw_path.stat().st_size if self.raw_path.exists() else 0
        index = self.index_path.stat().st_size if self.index_path.exists() else 0
        return {'sqlite_bytes': db, 'source_bytes': raw, 'index_bytes': index, 'total_bytes': db + raw + index,
                'rows': len(self.agent.store.all(True)) if self.agent else len(self.archive),
                'nodes': len(self.view.nodes) if self.view else 0}

    def close(self):
        if self.agent:
            self.agent.close()


def percentile(values, fraction):
    return sorted(values)[min(len(values) - 1, math.ceil(len(values) * fraction) - 1)] if values else 0


def run_world(partition, replica, world, ranker):
    sources, probes, deletion, forgotten = world_data(partition, replica, world)
    project = sources[0].project
    # Equal, shared embedding preparation: charge same measured cost to every arm.
    ranker._cache.clear()
    cpu, wall = time.process_time(), time.perf_counter()
    for text in dict.fromkeys([s.text for s in sources] + [p.query for p in probes + forgotten]):
        ranker.vector(text)
    embed_cpu, embed_wall = time.process_time() - cpu, time.perf_counter() - wall
    result = {'partition': partition, 'replica': replica, 'world': world, 'seed': str(seed(partition, replica, world)),
              'data_sha256': digest([asdict(s) for s in sources]), 'sources': [asdict(s) for s in sources],
              'delete_ids': sorted(deletion), 'arms': {}}
    # Rotate arm order across worlds to reduce drift in timing comparisons.
    arms = list(ARMS[world % 4:] + ARMS[:world % 4])
    for arm in arms:
        with tempfile.TemporaryDirectory(prefix='infinito-e13a-synthetic-') as folder:
            system = System(arm, folder, ranker)
            rows, latencies = [], []
            for i, s in enumerate(sources):
                system.add(s)
                if i % 12 == 11:
                    # Usage only; no oracle labels or scores go back to the view.
                    system.query(probes[(i // 12) % len(probes)].query, 512, project)
            intact = not system.view or (
                system.view.integrity() and digest([asdict(s) for s in system.view.sources.values()]) ==
                digest([asdict(s) for s in sources]))
            storage = system.storage()
            before_index = system.view.snapshot() if system.view else None
            for budget in (512, 256):
                for p in probes:
                    answer, context, tokens, latency, selected = system.query(p.query, budget, project)
                    rows.append({'family': p.family, 'query': p.query, 'budget': budget,
                                 'expected': p.expected, 'answer': answer, 'correct': answer == p.expected,
                                 'context': context, 'tokens': tokens,
                                 'provenance_valid': all(sid in system.archive for _, ids in selected for sid in ids)})
                    latencies.append(latency)
            system.forget(deletion)
            # Both immediate and restart/rebuild outcomes are independently recorded.
            survivors = [Probe('survivor', p.query, p.expected) for p in probes[5:10]]
            for phase in ('immediate', 'restart'):
                if phase == 'restart':
                    system.restart()
                for p in forgotten + survivors:
                    answer, context, tokens, latency, selected = system.query(p.query, 512, project)
                    rows.append({'family': p.family, 'phase': phase, 'query': p.query, 'budget': 512,
                                 'expected': p.expected, 'answer': answer, 'correct': answer == p.expected,
                                 'context': context, 'tokens': tokens,
                                 'provenance_valid': all(sid in system.archive for _, ids in selected for sid in ids)})
                    latencies.append(latency)
                intact = intact and (not system.view or system.view.integrity())
            # Stronger than answer absence: inspect all surviving derivative state for deleted source ids.
            no_deleted = not system.view or not (set(deletion) & {
                sid for n in system.view.nodes for sid in n['sources']})
            result['arms'][arm] = {'queries': rows, 'storage': storage, 'after_forget_storage': system.storage(),
                                  'index_before': before_index,
                                  'index_after': system.view.snapshot() if system.view else None,
                                  'archive_after': [asdict(s) for s in system.archive.values()] if arm != 'CURRENT' else None,
                                  'database_after': [{'text': r.content, 'metadata': r.metadata} for r in system.agent.store.all(True)] if system.agent else None,
                                  'integrity': intact and no_deleted,
                                  'cpu_s': dict(system.cpu), 'wall_s': dict(system.wall),
                                  'embedding_cpu_s': embed_cpu, 'embedding_wall_s': embed_wall,
                                  'query_p50_s': percentile(latencies, .5), 'query_p95_s': percentile(latencies, .95),
                                  'query_latencies_s': latencies,
                                  'rebuilds': system.view.rebuilds if system.view else 0,
                                  'scanned': system.view.scanned if system.view else 0,
                                  'maintenance_cpu_s': system.view.maintenance_cpu if system.view else 0,
                                  'maintenance_wall_s': system.view.maintenance_wall if system.view else 0}
            system.close()
    return result


def summarize(worlds):
    summary = {}
    for arm in ARMS:
        rows = [q for w in worlds for q in w['arms'][arm]['queries']]
        strata = {}
        for budget in (512, 256):
            for family in PRIMARY + ('hypothesis', 'unknown', 'forgotten', 'survivor'):
                selected = [q for q in rows if q['budget'] == budget and q['family'] == family]
                if selected:
                    strata[f'{budget}/{family}'] = {'correct': sum(q['correct'] for q in selected), 'n': len(selected)}
        primary = [q for q in rows if q['budget'] == 512 and q['family'] in PRIMARY]
        summary[arm] = {'accuracy': sum(q['correct'] for q in primary) / len(primary), 'strata': strata,
                        'mean_storage_bytes': statistics.mean(w['arms'][arm]['storage']['total_bytes'] for w in worlds),
                        'total_cpu_s': sum(sum(w['arms'][arm]['cpu_s'].values()) + w['arms'][arm]['embedding_cpu_s'] for w in worlds),
                        'query_p95_s': percentile([t for w in worlds for t in w['arms'][arm]['query_latencies_s']], .95),
                        'maintenance_cpu_s': sum(w['arms'][arm]['maintenance_cpu_s'] for w in worlds),
                        'rebuilds': sum(w['arms'][arm]['rebuilds'] for w in worlds)}
    comparisons = {}
    for baseline in ('CURRENT', 'RAG', 'PERIODIC'):
        deltas = []
        for w in worlds:
            scores = {}
            for arm in ('ADAPTIVE', baseline):
                qs = [q for q in w['arms'][arm]['queries'] if q['budget'] == 512 and q['family'] in PRIMARY]
                scores[arm] = sum(q['correct'] for q in qs) / len(qs)
            deltas.append(scores['ADAPTIVE'] - scores[baseline])
        rng = random.Random(13)
        boot = [statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(5000)]
        comparisons[baseline] = {'delta': statistics.mean(deltas), 'ci95': [percentile(boot, .025), percentile(boot, .975)],
                                 'world_deltas': deltas}
    g0 = all(a['integrity'] and all(q['tokens'] <= q['budget'] and q['provenance_valid'] for q in a['queries'])
             for w in worlds for a in w['arms'].values()) and len({w['seed'] for w in worlds}) == len(worlds)
    g1 = all(comparisons[a]['delta'] >= .05 and comparisons[a]['ci95'][0] > 0 for a in ('CURRENT', 'RAG'))
    for family in PRIMARY:
        rates = {a: summary[a]['strata'][f'512/{family}']['correct'] / summary[a]['strata'][f'512/{family}']['n'] for a in ARMS}
        g1 = g1 and all(rates['ADAPTIVE'] - rates[a] >= -.02 for a in ('CURRENT', 'RAG'))
    g2 = all(q['correct'] and q['provenance_valid'] for w in worlds
             for q in w['arms']['ADAPTIVE']['queries'] if q['family'] in ('hypothesis', 'forgotten', 'survivor')) and g0
    a, c, p = (summary[k] for k in ('ADAPTIVE', 'CURRENT', 'PERIODIC'))
    g3 = a['mean_storage_bytes'] <= 1.5*c['mean_storage_bytes'] and a['total_cpu_s'] <= 2*c['total_cpu_s'] and a['query_p95_s'] <= 2*c['query_p95_s']
    g4 = g2 and (a['accuracy'] - p['accuracy'] >= .02 or
                (p['maintenance_cpu_s'] > 0 and a['maintenance_cpu_s'] <= .8*p['maintenance_cpu_s'] and a['accuracy'] - p['accuracy'] >= -.01))
    gates = dict(G0=bool(g0), G1=bool(g1), G2=bool(g2), G3=bool(g3), G4=bool(g4))
    return {'systems': summary, 'comparisons': comparisons, 'gates': gates,
            'verdict': 'E13A_INCONCLUSIVE' if not g0 else 'E13A_PASS' if all(gates.values()) else 'E13A_FAIL'}


def fingerprint(paths):
    return {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in paths}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--partition', choices=['DEV', 'HOLDOUT'], required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--freeze', type=Path)
    args = ap.parse_args()
    files = [ROOT / p for p in ('observatory/e13a.py', 'tests/test_e13a.py', 'scripts/audit_e13a.py',
                                'docs/INFINITO_E13A_PREREG_2026-11-05.md', 'observatory/requirements_e13a.txt')]
    hashes = fingerprint(files)
    model_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.model.iterdir() if p.is_file()}
    if args.partition == 'HOLDOUT':
        if not args.freeze:
            ap.error('HOLDOUT requires a committed freeze manifest')
        frozen = json.loads(args.freeze.read_text())
        assert frozen['files'] == hashes, 'code/protocol differs from frozen DEV selection'
        assert frozen['model_files'] == model_hashes, 'model differs from frozen DEV model'
        relative = args.freeze.resolve().relative_to(ROOT).as_posix()
        committed = subprocess.check_output(['git', 'show', f'HEAD:{relative}'], cwd=ROOT)
        assert json.loads(committed) == frozen, 'freeze manifest must be committed before HOLDOUT'
    if args.output.exists():
        ap.error('output already exists: never overwrite a consumed run')
    args.output.mkdir(parents=True)
    ranker = MeaningRanker(str(args.model))
    ranker.warm()
    worlds = []
    for replica in range(1 if args.partition == 'DEV' else 4):
        for world in range(8):
            result = run_world(args.partition, replica, world, ranker)
            worlds.append(result)
            (args.output / f'world-{replica}-{world}.json').write_bytes(encoded(result))
            print(f'{args.partition} {replica}/{world} complete', flush=True)
    report = {'partition': args.partition, 'code': hashes,
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'environment': {'python': sys.version, 'platform': platform.platform(), 'processor': platform.processor()},
              'model_files': model_hashes,
              'model_bytes': sum(p.stat().st_size for p in args.model.iterdir() if p.is_file()),
              'api_calls': 0, 'summary': summarize(worlds),
              'artifacts': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob('world-*.json')}}
    (args.output / 'summary.json').write_bytes(encoded(report))
    print(json.dumps(report['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
