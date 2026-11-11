"""Post-run replay of every trial and evaluation against immutable environment."""
import argparse, gzip, hashlib, importlib.util, json, pathlib, sys
import numpy as np
import torch

def main():
    p=argparse.ArgumentParser(); p.add_argument('--frozen',required=True); p.add_argument('--original',required=True); p.add_argument('--root',required=True); a=p.parse_args()
    frozen=pathlib.Path(a.frozen).resolve(); sys.path.insert(0,str(frozen))
    from observatory.f10m_experiment import frozen_original_models,selection_cache
    from observatory.f10n_memory import EffectMemory,RewardMemory,execute
    spec=importlib.util.spec_from_file_location('worlds',frozen/'observatory/f10n_worlds.py'); w=importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
    w.TRAIN_BASE=27100000; w.DEV_BASE=27200000; w.OOD_BASE=27400000
    g0=json.loads((pathlib.Path(a.original)/'f10l-independent-g0-original/f10l-g0.json').read_text())
    torch.set_num_threads(1); root=pathlib.Path(a.root); report={}
    runner=pathlib.Path(__file__).with_name('f10o_terminal.py')
    spec=importlib.util.spec_from_file_location('terminal_runner',runner); terminal=importlib.util.module_from_spec(spec); spec.loader.exec_module(terminal)
    codes=torch.tensor([[w.OPS.index(o) for o in s]+[4]*(3-len(s)) for s in w.PROGRAMS])
    lengths=torch.tensor([len(s) for s in w.PROGRAMS])
    for rep in range(4):
        d=root/str(rep); orig=pathlib.Path(a.original)/f'f10l-stage1-{rep}'
        selector,xor,lut,original=frozen_original_models(rep,orig,g0)
        train=w.dataset('train',rep); slots=selection_cache(train,selector)
        lookup={(r['seed'],r['variant']):(r,s) for r,s in zip(train,slots)}
        effects=EffectMemory(); reward=RewardMemory(); count=0
        with gzip.open(d/'train.jsonl.gz','rt',encoding='utf8') as f:
            for line in f:
                x=json.loads(line); row,slot=lookup[(x['seed'],x['variant'])]; seq=w.PROGRAMS[x['pid']]
                result=execute(row,seq,slot,lut); obs=x['observed']
                assert row['case_sha']==x['case_sha'] and obs['context']==w.context(row).tolist()
                assert obs['actions']==[w.OPS.index(o) for o in seq]+[4]*(3-len(seq))
                assert (obs['answer'],obs['success'])==(result['answer'],result['success'])
                effects.observe(result,row['public_bit']); reward.observe(row,x['pid'],result); count+=1
        assert effects.snapshot()==json.loads((d/'I.json').read_text())
        assert reward.snapshot()==json.loads((d/'B.json').read_text())
        cases=w.dataset('dev',rep)+w.dataset('ood',rep); slots=selection_cache(cases,selector)
        lookup={(r['seed'],r['variant']):(r,s) for r,s in zip(cases,slots)}
        evals=json.loads((d/'eval.json').read_text()); modes=0
        models={}
        for arm in ('T','R','S'):
            m=terminal.TerminalModel(); m.load_state_dict(torch.load(d/(arm+'.pt'),weights_only=True)); m.eval(); models[arm]=m
        planned={}
        for x in evals:
            row,slot=lookup[(x['seed'],x['variant'])]
            assert x['truth']==row['truth'] and x['case_sha']==row['case_sha']
            key=(x['arm'],row['task'],row['flip'],row['public_bit'])
            if key not in planned:
                arm=x['arm']
                if arm=='I': seq,_=effects.plan(row['task'],row['flip'],row['public_bit'])
                elif arm=='B': seq,_=reward.rank(row['task'],row['flip'],row['public_bit'],'additive_reward_table')
                else:
                    model=models['T' if arm=='T_action_permuted' else arm]
                    ac=codes if arm!='T_action_permuted' else torch.where(codes==4,4,(codes+1)%4)
                    with torch.no_grad(): scores=torch.sigmoid(model(torch.tensor(w.context(row)).repeat(85,1),ac)[:,0])-.01*lengths
                    seq=w.PROGRAMS[int(scores.argmax())]
                planned[key]=''.join(seq)
            assert planned[key]==x['program']
            for mode,record in x['results'].items():
                result=execute(row,tuple(x['program']),slot,lut,mode)
                assert all(result[k]==v for k,v in record.items()); modes+=1
        meta=json.loads((d/'frozen.json').read_text())
        assert meta['original']==json.loads(json.dumps(original))
        assert hashlib.sha256(runner.read_bytes()).hexdigest()==meta['code_sha256']
        report[str(rep)]={'training_trials_replayed':count,'evaluation_rows_replayed':len(evals),'interventions_replayed':modes,'I_B_reconstructed_exactly':True,'original_SHA_verified':True,'all_plans_recomputed_from_frozen_models':True}
        print('REPLAY_PASS',rep,count,modes,flush=True)
    (root/'replay.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__': main()
