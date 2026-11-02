"""Fixed-budget F10-G training followed by frozen new-world evaluation."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from observatory.f10g_worlds import dataset, KEYS, VARIANTS
from observatory.f10g_models import Actor, count, gru_width

ARMS = ('relational_joint', 'relational_aux', 'unary_joint', 'unary_aux', 'gru')
MODES = ('read', 'skip', 'shuffle', 'erase_sign', 'erase_time', 'oracle_patch')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensors(cases):
    return (torch.tensor(np.stack([c['history'] for c in cases])),
            torch.tensor(np.stack([c['memory'] for c in cases])),
            torch.tensor([c['truth'] for c in cases], dtype=torch.float32),
            torch.tensor([c['winner'] for c in cases]))


def model_for(arm):
    return Actor('gru', gru_width()) if arm == 'gru' else Actor(arm.split('_')[0])


def train(index, out):
    cases = dataset('train', index)
    x, m, y, winner = tensors(cases)
    report = dict(replica=index, torch=torch.__version__, numpy=np.__version__,
                  python=platform.python_version(), train_hashes=[c['hash'] for c in cases], arms={})
    for arm in ARMS:
        torch.manual_seed(185000+index)
        model = model_for(arm)
        optimizer = torch.optim.Adam(model.parameters(), lr=.002)
        losses, steps = [], 0
        start = time.perf_counter()
        for epoch in range(20):
            order = np.random.default_rng(186000+100*index+epoch).permutation(len(y))
            total = 0
            for offset in range(0, len(y), 64):
                ids = order[offset:offset+64]
                logits, scores, _, hidden = model(x[ids], m[ids])
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y[ids])
                if arm.endswith('aux'):
                    value = m[ids, winner[ids], KEYS]
                    direct = model.interpret(hidden, value, torch.ones_like(value), x[ids, -1, -1])
                    loss = loss + nn.functional.cross_entropy(scores, winner[ids]) + nn.functional.binary_cross_entropy_with_logits(direct, y[ids])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                total += float(loss.detach())*len(ids)
                steps += 1
            losses.append(total/len(y))
            if (epoch+1) % 5 == 0:
                print(index, arm, epoch+1, round(losses[-1], 4), flush=True)
        seconds = time.perf_counter()-start
        path = out/(arm+'.pt')
        torch.save(model.state_dict(), path)
        report['arms'][arm] = dict(parameters=count(model), steps=steps, examples=20*len(y),
                                  seconds=seconds, losses=losses, sha256=sha(path))
        (out/'training.json').write_text(json.dumps(report, indent=2))
    return report


def evaluate(index, out, training):
    rows = []
    for split in ('dev', 'ood'):
        cases = dataset(split, index)
        x, m, y, winner = tensors(cases)
        for arm in ARMS:
            path = out/(arm+'.pt')
            assert sha(path) == training['arms'][arm]['sha256']
            model = model_for(arm)
            model.load_state_dict(torch.load(path, weights_only=True))
            model.eval()
            with torch.no_grad():
                for start in range(0, len(cases), 64):
                    stop = start+64
                    xx, mm, ww = x[start:stop], m[start:stop], winner[start:stop]
                    result = {}
                    for mode in MODES:
                        patch = mm[torch.arange(len(mm)), ww, KEYS] if mode == 'oracle_patch' else None
                        logit, scores, weights, _ = model(xx, mm, mode, patch)
                        result[mode] = (logit >= 0).int().tolist()
                        if mode == 'read':
                            selected = scores.argmax(-1).tolist()
                            weight = weights[torch.arange(len(mm)), ww].tolist()
                    for b, c in enumerate(cases[start:stop]):
                        mem = c['memory']
                        query = int(c['history'][-1, :KEYS].argmax())
                        valid = np.flatnonzero(mem[:, query])
                        latest = valid[np.argmax(mem[valid, KEYS+2])]
                        inv = int(c['history'][-1, -1])
                        baselines = {'last_position': int(mem[valid[-1], KEYS] > 0)^inv,
                                     'latest_unsigned': int(mem[latest, KEYS] > 0)^inv,
                                     'symbolic': int(mem[c['winner'], KEYS] > 0)^inv,
                                     'coin': int(np.random.default_rng(c['seed']+99).integers(2))}
                        rows.append(dict(replica=index, split=split, arm=arm, seed=c['seed'], variant=c['variant'],
                                         k=c['k'], hash=c['hash'], truth=c['truth'], winner=c['winner'],
                                         selected=selected[b], winner_weight=weight[b],
                                         predictions={mode: result[mode][b] for mode in MODES}, baselines=baselines))
            assert sha(path) == training['arms'][arm]['sha256']
    (out/'rows.json').write_text(json.dumps(rows, separators=(',', ':')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, choices=range(4), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    training = train(args.index, args.output)
    # All five final checkpoints saved before any DEV/OOD is generated.
    evaluate(args.index, args.output, training)
    print('F10G_REPLICA_COMPLETE', args.index, flush=True)


if __name__ == '__main__':
    main()
