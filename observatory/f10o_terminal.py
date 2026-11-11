"""Prospective F10-O. Run with an immutable F10-N checkout, never import product."""
import argparse, collections, gzip, hashlib, importlib.util, json, pathlib, sys, time
import numpy as np
import torch
from torch import nn

def digest(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def save(p, obj):
    pathlib.Path(p).write_text(json.dumps(obj, sort_keys=True, separators=(',', ':')), encoding='utf8')

class TerminalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(5,8)
        self.initial=nn.Linear(3,32)
        self.cell=nn.GRUCell(8,32)
        self.head=nn.Linear(32,2)
    def forward(self, context, actions):
        h=torch.tanh(self.initial(context))
        for t in range(3):
            new=self.cell(self.embedding(actions[:,t]),h)
            h=torch.where((actions[:,t]!=4).unsqueeze(1),new,h)
        return self.head(h)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--frozen', required=True)
    ap.add_argument('--original',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    frozen=pathlib.Path(args.frozen).resolve()
    sys.path.insert(0,str(frozen))
    from observatory.f10n_memory import EffectMemory,RewardMemory,execute
    from observatory.f10m_experiment import frozen_original_models,selection_cache
    spec=importlib.util.spec_from_file_location('f10o_worlds',frozen/'observatory/f10n_worlds.py')
    worlds=importlib.util.module_from_spec(spec); spec.loader.exec_module(worlds)
    worlds.TRAIN_BASE=27100000; worlds.DEV_BASE=27200000; worlds.OOD_BASE=27400000
    programs=worlds.PROGRAMS
    trainids=np.array([i for i,s in enumerate(programs) if s!=worlds.HELDOUT_PROGRAM])
    codes=torch.tensor([[worlds.OPS.index(o) for o in s]+[4]*(3-len(s)) for s in programs])
    lens=torch.tensor([len(s) for s in programs])
    g0=json.loads((pathlib.Path(args.original)/'f10l-independent-g0-original/f10l-g0.json').read_text())
    out=pathlib.Path(args.out); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    for rep in range(4):
        start=time.perf_counter(); dest=out/str(rep); dest.mkdir(exist_ok=False)
        orig=pathlib.Path(args.original)/f'f10l-stage1-{rep}'
        selector,xor,lut,original=frozen_original_models(rep,orig,g0)
        rows=worlds.dataset('train',rep)
        assert len(rows)==3072 and all((r['task'],r['flip']) in worlds.TASKS for r in rows)
        selected=selection_cache(rows,selector)
        contexts=np.array([worlds.context(r) for r in rows])
        effects=EffectMemory(); rewards=RewardMemory()
        observations=[]; counts=np.zeros(85,int)
        with gzip.open(dest/'train.jsonl.gz','wt',encoding='utf8') as stream:
            for epoch in range(16):
                order=np.random.default_rng(276000+100*rep+epoch).permutation(len(rows))
                rng=np.random.default_rng(286000+100*rep+epoch)
                for i in order:
                    row=rows[int(i)]
                    for _ in range(2):
                        pid=int(rng.choice(trainids)); counts[pid]+=1
                        result=execute(row,programs[pid],selected[int(i)],lut)
                        # Privileged control is the only learner receiving this object.
                        effects.observe(result,row['public_bit']); rewards.observe(row,pid,result)
                        # Strict public boundary: no state, truth, trace, slot or read payload.
                        public={'context':contexts[i].tolist(),'actions':codes[pid].tolist(),
                                'answer':result['answer'],'success':result['success']}
                        observations.append(public)
                        stream.write(json.dumps({'seed':row['seed'],'variant':row['variant'],
                            'case_sha':row['case_sha'],'pid':pid,'observed':public})+'\n')
        assert len(observations)==98304 and counts[programs.index(worlds.HELDOUT_PROGRAM)]==0
        x=torch.tensor([r['context'] for r in observations],dtype=torch.float32)
        a=torch.tensor([r['actions'] for r in observations])
        y=torch.tensor([[r['success'],r['answer']] for r in observations],dtype=torch.float32)
        models={}; fit={}
        for arm in ('T','R','S'):
            ts=time.perf_counter(); torch.manual_seed(296000+rep)
            model=TerminalModel(); optim=torch.optim.Adam(model.parameters(),lr=.003)
            rng=np.random.default_rng(306000+rep)
            for k in range(512):
                sl=slice(k*192,(k+1)*192); target=y[sl].clone()
                if arm=='S': target[:,1]=target[torch.tensor(rng.permutation(192)),1]
                pred=model(x[sl],a[sl]); loss=nn.functional.binary_cross_entropy_with_logits(pred,target,reduction='none')
                loss=(loss[:,0]+(0 if arm=='R' else 1)*loss[:,1]).mean()
                optim.zero_grad(); loss.backward(); optim.step()
            model.eval(); models[arm]=model
            torch.save(model.state_dict(),dest/f'{arm}.pt')
            fit[arm]={'updates':512,'examples':len(observations),'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.perf_counter()-ts}
        save(dest/'I.json',effects.snapshot()); save(dest/'B.json',rewards.snapshot())
        manifest={p.name:digest(p) for p in dest.iterdir() if p.is_file()}
        meta={'status':'FROZEN_BEFORE_DEV_OOD','replica':rep,'files':manifest,'fit':fit,
              'original':original,'train_counts':counts.tolist(),'train_cases':[r['case_sha'] for r in rows],
              'train_seeds':sorted(set(r['seed'] for r in rows)),
              'code_sha256':digest(__file__),'prereg_sha256':digest(pathlib.Path(__file__).parents[1]/'docs/INFINITO_F10O_PREREG_2026-11-03.md'),
              'environment':{'torch':torch.__version__,'numpy':np.__version__,'python':sys.version}}
        save(dest/'frozen.json',meta)
        print('FROZEN_BEFORE_EVAL',rep,flush=True)
        records=[]; decisions={}; plan_seconds=collections.defaultdict(float)
        for split in ('dev','ood'):
            cases=worlds.dataset(split,rep); choices=selection_cache(cases,selector)
            assert not set(r['case_sha'] for r in cases)&set(meta['train_cases'])
            for arm in ('T','R','S','B','I','T_action_permuted'):
                for case,slot in zip(cases,choices):
                    key=(arm,case['task'],case['flip'],case['public_bit'])
                    if key not in decisions:
                        ts=time.perf_counter()
                        if arm=='I': seq,_=effects.plan(case['task'],case['flip'],case['public_bit'])
                        elif arm=='B': seq,_=rewards.rank(case['task'],case['flip'],case['public_bit'],'additive_reward_table')
                        else:
                            model=models['T' if arm=='T_action_permuted' else arm]
                            ac=codes if arm!='T_action_permuted' else torch.where(codes==4,4,(codes+1)%4)
                            with torch.no_grad():
                                pred=model(torch.tensor(worlds.context(case)).repeat(85,1),ac)
                                pid=int(torch.argmax(torch.sigmoid(pred[:,0])-.01*lens))
                            seq=programs[pid]
                        decisions[key]=seq; plan_seconds[arm]+=time.perf_counter()-ts
                    seq=decisions[key]
                    results={mode:execute(case,seq,slot,lut,mode) for mode in ('normal','shuffle','x_disabled','public_randomized')}
                    records.append({k:case[k] for k in ('seed','base_world','variant','task','flip','n','source_mode','case_sha')} | {
                        'split':split,'arm':arm,'program':''.join(seq),'truth':case['truth'],
                        'results':{m:{k:r[k] for k in ('answer','success','reward','operations')} for m,r in results.items()}})
        save(dest/'eval.json',records)
        assert all(digest(dest/name)==h for name,h in manifest.items())
        for name,h in original['all_5_original_checkpoint_sha'].items(): assert digest(orig/(name+'.pt'))==h
        save(dest/'complete.json',{'frozen_sha':digest(dest/'frozen.json'),'eval_sha':digest(dest/'eval.json'),
            'seconds':time.perf_counter()-start,'plan_seconds_per_8_contexts':dict(plan_seconds),
            'all_frozen_sha_verified':True,'original_sha_verified':True})
        print('COMPLETE',rep,round(time.perf_counter()-start,2),flush=True)

if __name__=='__main__': main()
