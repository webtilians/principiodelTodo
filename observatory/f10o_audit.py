"""F10-O raw-log and paired-world audit; no training and no model selection."""
import argparse, collections, gzip, hashlib, json, pathlib
import numpy as np

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); args=ap.parse_args()
    root=pathlib.Path(args.root); report={'replicas':{},'gate_counts':collections.Counter()}
    all_seeds=set()
    for rep in range(4):
        d=root/str(rep); frozen=json.loads((d/'frozen.json').read_text()); complete=json.loads((d/'complete.json').read_text())
        assert sha(d/'frozen.json')==complete['frozen_sha'] and sha(d/'eval.json')==complete['eval_sha']
        assert all(sha(d/k)==v for k,v in frozen['files'].items())
        counts=collections.Counter(); observed_seeds=set(); cases=set(); n=0
        with gzip.open(d/'train.jsonl.gz','rt',encoding='utf8') as f:
            for line in f:
                r=json.loads(line); o=r['observed']; n+=1; counts[r['pid']]+=1
                assert set(o)=={'context','actions','answer','success'}
                assert not (o['context'][0]==1 and o['context'][1]==1)
                assert o['actions']!=[1,2,3] # SRX
                assert 27100000+rep*10000<=r['seed']<27100000+rep*10000+3000
                observed_seeds.add(r['seed']); cases.add(r['case_sha'])
        assert n==98304 and len(cases)==3072 and len(observed_seeds)==768
        assert [counts[i] for i in range(85)]==frozen['train_counts']
        assert not all_seeds&observed_seeds; all_seeds|=observed_seeds
        rows=json.loads((d/'eval.json').read_text()); assert len(rows)==6*(768+512)
        assert not cases&{r['case_sha'] for r in rows}
        eval_seeds={r['seed'] for r in rows}; assert not eval_seeds&all_seeds; all_seeds|=eval_seeds
        for r in rows:
            base=27200000 if r['split']=='dev' else 27400000
            assert base+rep*10000<=r['seed']<base+rep*10000+(3000 if r['split']=='dev' else 128)
            if r['split']=='ood': assert (r['task'],r['flip'])==('MEMORY',1)
        by={arm:sorted([r for r in rows if r['split']=='ood' and r['arm']==arm],key=lambda r:(r['base_world'],r['variant'])) for arm in ('T','R','S','B','I','T_action_permuted')}
        for rr in by.values():
            assert len(rr)==512 and len({r['case_sha'] for r in rr})==512
            assert all(sum(x['base_world']==j for x in rr)==4 for j in range(128))
        for arm in by: assert [r['case_sha'] for r in by[arm]]==[r['case_sha'] for r in by['T']]
        rng=np.random.default_rng(316000+rep); boot=rng.integers(0,128,size=(2000,128))
        def vals(arm,mode='normal'): return np.array([r['results'][mode]['success'] for r in by[arm]],float).reshape(128,4).mean(1)
        def metric(v):
            lo,hi=np.quantile(v[boot].mean(1),[.025,.975]); return {'mean':float(v.mean()),'ci95':[float(lo),float(hi)]}
        metrics={a:metric(vals(a)) for a in by}
        t=vals('T'); contrasts={a:metric(t-vals(a)) for a in ('R','S','B','I','T_action_permuted')}
        causal={m:metric(t-vals('T',m)) for m in ('shuffle','x_disabled','public_randomized')}
        dev={str((task,flip)):float(np.mean([r['results']['normal']['success'] for r in rows if r['split']=='dev' and r['arm']=='T' and (r['task'],r['flip'])==(task,flip)])) for task,flip in (('PUBLIC',0),('PUBLIC',1),('MEMORY',0))}
        strata={f'{field}={value}':float(np.mean([r['results']['normal']['success'] for r in by['T'] if r[field]==value])) for field in ('variant','n','source_mode') for value in sorted(set(r[field] for r in by['T']))}
        plans={a:dict(collections.Counter(r['program'] or 'EMPTY' for r in rr)) for a,rr in by.items()}
        gates={'G0':all(x>=.8 for x in dev.values()),
            'G1':metrics['T']['mean']>=.8 and metrics['T']['ci95'][0]>.7 and min(strata.values())>=.7 and plans['T'].get('SRX',0)>=.9*512,
            'G2':all(causal[m]['mean']>=.2 and causal[m]['ci95'][0]>0 for m in ('shuffle','x_disabled')),
            'G3':all(contrasts[a]['mean']>=.1 and contrasts[a]['ci95'][0]>0 for a in ('R','S','B')),
            'G4_diagnostic':metrics['I']['mean']>=.8}
        report['gate_counts'].update({k:int(v) for k,v in gates.items()})
        report['replicas'][str(rep)]={'metrics':metrics,'contrasts':contrasts,'causal':causal,'dev':dev,'strata':strata,'plans':plans,'gates':gates,'cost':complete,'fit':frozen['fit']}
    report['technical']='F10O_TECHNICAL_COMPLETE'
    report['status']='F10O_TERMINAL_TRANSFER_PASS' if all(report['gate_counts'][k]>=3 for k in ('G0','G1','G2','G3')) else 'F10O_TERMINAL_TRANSFER_FAIL'
    report['all_sha_and_budget_checks']=True
    (root/'science.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(report['status'],dict(report['gate_counts']))
    for r,v in report['replicas'].items(): print(r,{a:round(m['mean'],4) for a,m in v['metrics'].items()},v['plans']['T'])
if __name__=='__main__': main()
