"""Independent artifact audit: no imports from learner, generator or product."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def extract(text):
    patterns = (r'Entidad (\w+) (\w+): (\w+)\.', r'Para (\w+), (\w+) consta como (\w+)\.')
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.groups()
    raise AssertionError(f'Unsupported evidence: {text}')


def interpret(query, lines):
    subject = re.search(r'entidad (\w+)', query)[1]
    attr = next(a for a in ('estado', 'responsable', 'herramienta', 'ruta') if a in query)
    dates = re.findall(r'\d{4}-\d{2}-\d{2}', query)
    cutoff = dates[0] if dates else '9999-12-31'
    facts = [(x['date'], *extract(x['text'])) for x in lines if x['kind'] != 'proposal' and x['date'] <= cutoff]
    def latest(entity, field):
        matches = [x for x in facts if x[1:3] == (entity, field)]
        return sorted({x[3] for x in matches if x[0] == max(y[0] for y in matches)}) if matches else []
    if query.startswith('cronologia'):
        return sorted({f'{d}:{v}' for d, e, a, v in facts if e == subject and a == attr})
    if 'del responsable' in query:
        return sorted({v for owner in latest(subject, 'responsable') for v in latest(owner, 'herramienta')})
    return latest(subject, attr)


def audit(folder):
    report = json.loads((folder / 'summary.json').read_bytes())
    seeds, case_count, discrete = set(), 0, []
    scores = {arm: [0, 0] for arm in report['summary']['systems']}
    for name, expected_hash in report['artifacts'].items():
        payload = (folder / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected_hash
        w = json.loads(payload)
        assert w['seed'] not in seeds
        namespace = f"INFINITO/E13-A/v1/{w['partition']}/{w['replica']}/{w['world']}"
        assert w['seed'] == str(int.from_bytes(hashlib.sha256(namespace.encode()).digest(), 'big'))
        seeds.add(w['seed'])
        sources = {s['sid']: s for s in w['sources']}
        deleted = set(w['delete_ids'])
        for arm, a in w['arms'].items():
            for stage in ('before', 'after'):
                index = a.get('index_' + stage)
                if index is not None:
                    for n in index['nodes']:
                        rep = sources[n['representative']]
                        assert n['kind'] == rep['kind'] and n['date'] == rep['date']
                        for sid in n['sources']:
                            s = sources[sid]
                            assert extract(s['text']) == extract(rep['text'])
                            assert (s['kind'], s['date'], s['project']) == (rep['kind'], rep['date'], rep['project'])
                            assert stage != 'after' or sid not in deleted
                    for left, right in index['links']:
                        l = extract(sources[index['nodes'][left]['representative']]['text'])
                        r = extract(sources[index['nodes'][right]['representative']]['text'])
                        assert l[1] == 'responsable' and l[2] == r[0]
            if a.get('archive_after') is not None:
                assert {s['sid'] for s in a['archive_after']} == set(sources) - deleted
                assert all(s == sources[s['sid']] for s in a['archive_after'])
            if a.get('database_after') is not None:
                forbidden = {extract(sources[sid]['text']) for sid in deleted}
                assert all(extract(r['text']) not in forbidden for r in a['database_after'])
            for q in a['queries']:
                lines = [json.loads(x) for x in q['context'].splitlines()[1:]]
                assert math.ceil(len(q['context']) / 4) <= q['tokens'] <= q['budget']
                for x in lines:
                    for sid in x['ids']:
                        s = sources[sid]
                        assert (extract(x['text']), x['date'], x['kind'], x['project']) == (
                            extract(s['text']), s['date'], s['kind'], s['project'])
                        assert 'phase' not in q or sid not in deleted
                assert interpret(q['query'], lines) == q['answer']
                assert q['correct'] == (q['answer'] == q['expected'])
                if q['budget'] == 512 and q['family'] in ('current', 'past', 'timeline', 'relation', 'contradiction'):
                    scores[arm][0] += q['correct']
                    scores[arm][1] += 1
                case_count += 1
                discrete.append((name, arm, q['family'], q.get('phase'), q['budget'], q['query'], q['expected'], q['answer'], q['context']))
    for arm, (correct, count) in scores.items():
        assert correct / count == report['summary']['systems'][arm]['accuracy']
    return {'partition': report['partition'], 'worlds': len(seeds), 'queries': case_count,
            'scores': scores, 'seeds': sorted(seeds),
            'discrete_sha256': hashlib.sha256(json.dumps(sorted(discrete), sort_keys=True).encode()).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--disjoint', type=Path)
    args = parser.parse_args()
    result = audit(args.folder)
    if args.compare:
        assert result['discrete_sha256'] == audit(args.compare)['discrete_sha256']
        result['reproduced'] = True
    if args.disjoint:
        assert set(result['seeds']).isdisjoint(audit(args.disjoint)['seeds'])
        result['disjoint'] = True
    print(json.dumps(result, indent=2))
