"""Audit saved rows and fixed gates without model inference or training."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ARMS = ('relational_joint', 'relational_aux', 'unary_joint', 'unary_aux', 'gru')
VARIANTS = ('U_before', 'U_after', 'S_before', 'S_after')


def accuracy(rows, mode='read'):
    return sum(r['predictions'][mode] == r['truth'] for r in rows)/len(rows)


def contrast(a, b, mode_a='read', mode_b='read', seed=0):
    aa = {(r['seed'], r['variant']): int(r['predictions'][mode_a] == r['truth']) for r in a}
    bb = {(r['seed'], r['variant']): int(r['predictions'][mode_b] == r['truth']) for r in b}
    assert aa.keys() == bb.keys()
    worlds = sorted({k[0] for k in aa})
    d = np.array([np.mean([aa[(s, v)]-bb[(s, v)] for v in VARIANTS]) for s in worlds])
    rng = np.random.default_rng(seed)
    samples = d[rng.integers(len(d), size=(2000, len(d)))].mean(1)
    return dict(delta=float(d.mean()), ci95=np.quantile(samples, [.025, .975]).tolist(), worlds=len(d))


def summarize(rows):
    result = {mode: accuracy(rows, mode) for mode in rows[0]['predictions']}
    result['selection'] = sum(r['selected'] == r['winner'] for r in rows)/len(rows)
    result['winner_weight'] = float(np.mean([r['winner_weight'] for r in rows]))
    seeds = {r['seed'] for r in rows}
    result['all_four'] = sum(all(r['predictions']['read'] == r['truth'] for r in rows if r['seed'] == s) for s in seeds)/len(seeds)
    result['by_variant'] = {v: accuracy([r for r in rows if r['variant'] == v]) for v in VARIANTS}
    result['by_k'] = {k: accuracy([r for r in rows if r['k'] == k]) for k in sorted({r['k'] for r in rows})}
    result['selection_decision_table'] = {f'selection_{s}_decision_{d}': sum((r['selected'] == r['winner']) == s and (r['predictions']['read'] == r['truth']) == d for r in rows) for s in (False, True) for d in (False, True)}
    result['baselines'] = {name: sum(r['baselines'][name] == r['truth'] for r in rows)/len(rows) for name in rows[0]['baselines']}
    return result


def audit(root):
    report = {'replicas': {}, 'technical': 'F10G_TECHNICAL_COMPLETE', 'confirmations_opened': False}
    all_hashes, pooled = set(), []
    for replica in range(4):
        directory = root/str(replica)
        train = json.loads((directory/'training.json').read_text())
        rows = json.loads((directory/'rows.json').read_text())
        assert train['replica'] == replica and set(train['arms']) == set(ARMS)
        assert len(rows) == 5*2*128*4
        assert len({(r['split'], r['arm'], r['seed'], r['variant']) for r in rows}) == len(rows)
        reference = {(r['split'], r['seed'], r['variant']): (r['hash'], r['truth'], r['winner'], r['k'], r['baselines']) for r in rows if r['arm'] == ARMS[0]}
        for arm in ARMS:
            observed = {(r['split'], r['seed'], r['variant']): (r['hash'], r['truth'], r['winner'], r['k'], r['baselines']) for r in rows if r['arm'] == arm}
            assert observed == reference, 'Arms must evaluate exactly the same worlds and labels'
        local_hashes = set(train['train_hashes'])
        assert len(local_hashes) == 2048
        assert not local_hashes & all_hashes
        all_hashes |= local_hashes
        for split, base in [('dev', 18200000), ('ood', 18400000)]:
            subset = [r for r in rows if r['split'] == split and r['arm'] == ARMS[0]]
            assert {r['seed'] for r in subset} == set(range(base+10000*replica, base+10000*replica+128))
            hashes = {r['hash'] for r in subset}
            assert len(hashes) == 512 and not hashes & all_hashes
            all_hashes |= hashes
        for arm, meta in train['arms'].items():
            assert meta['steps'] == 640 and meta['examples'] == 40960
            assert hashlib.sha256((directory/(arm+'.pt')).read_bytes()).hexdigest() == meta['sha256']
        target = train['arms'][ARMS[0]]['parameters']
        assert all(train['arms'][a]['parameters'] == target for a in ARMS[:-1])
        assert abs(train['arms']['gru']['parameters']-target)/target <= .05
        for r in rows:
            assert r['replica'] == replica and r['baselines']['symbolic'] == r['truth']
        metrics = {split: {arm: summarize([r for r in rows if r['split'] == split and r['arm'] == arm]) for arm in ARMS} for split in ('dev', 'ood')}
        comparisons = {}
        for split in ('dev', 'ood'):
            arms = {a: [r for r in rows if r['split'] == split and r['arm'] == a] for a in ARMS}
            comparisons[split] = {a: {mode: contrast(arms[a], arms[a], mode_b=mode, seed=187000+replica) for mode in ('skip', 'shuffle')} for a in ARMS}
            comparisons[split]['decoupling'] = contrast(arms['relational_aux'], arms['relational_joint'], seed=187000+replica)
            comparisons[split]['architecture'] = contrast(arms['relational_aux'], arms['unary_aux'], seed=187000+replica)
        m, c = metrics['ood']['relational_aux'], comparisons['ood']
        gates = dict(selection=m['selection'] >= .90, decision=m['read'] >= .90,
                     variants=min(m['by_variant'].values()) >= .85, interpretation=m['oracle_patch'] >= .95,
                     all_four=m['all_four'] >= .80, read_skip=c['relational_aux']['skip']['ci95'][0] > .20,
                     read_shuffle=c['relational_aux']['shuffle']['ci95'][0] > .10,
                     improvement=c['decoupling']['delta'] >= .05 and c['decoupling']['ci95'][0] > 0)
        architecture = c['architecture']['delta'] >= .03 and c['architecture']['ci95'][0] > 0
        report['replicas'][replica] = dict(metrics=metrics, contrasts=comparisons, gates=gates,
                                          decoupling_pass=all(gates.values()), architecture_pass=architecture,
                                          budget=train['arms'])
        pooled.extend(rows)
    report['decoupling'] = 'F10G_DECOUPLING_PASS' if all(r['decoupling_pass'] for r in report['replicas'].values()) else 'F10G_DECOUPLING_FAIL'
    report['architecture'] = 'F10G_ARCHITECTURE_PASS' if all(r['architecture_pass'] for r in report['replicas'].values()) else 'F10G_ARCHITECTURE_FAIL'
    # Descriptive aggregate only, no pooled independence claim.
    report['aggregate'] = {split: {arm: summarize([r for r in pooled if r['split'] == split and r['arm'] == arm]) for arm in ARMS} for split in ('dev', 'ood')}
    report['unique_case_hashes'] = len(all_hashes)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.root)
    args.output.write_text(json.dumps(result, indent=2))
    print(result['technical'], result['decoupling'], result['architecture'])
