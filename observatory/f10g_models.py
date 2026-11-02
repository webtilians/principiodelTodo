"""Capacity-controlled learned selectors; no evaluator labels in forward."""
import torch
from torch import nn
from observatory.f10g_worlds import KEYS, FEATURES


class Actor(nn.Module):
    def __init__(self, kind='relational', width=32):
        super().__init__()
        self.kind = kind
        self.gru = nn.GRU(FEATURES+1, width, batch_first=True)
        if kind != 'gru':
            self.selector = nn.Sequential(nn.Linear(6, 16), nn.Tanh(), nn.Linear(16, 1))
        self.head = nn.Sequential(nn.Linear(width+(1 if kind == 'gru' else 3), 32), nn.Tanh(), nn.Linear(32, 1))

    def interpret(self, hidden, value, available, inversion):
        if self.kind == 'gru':
            return self.head(torch.cat((hidden, inversion[:, None]), 1)).flatten()
        return self.head(torch.cat((hidden, value[:, None], available[:, None], inversion[:, None]), 1)).flatten()

    def forward(self, history, memory, intervention='read', patch_value=None):
        _, h = self.gru(history)
        h = h[-1]
        inv = history[:, -1, -1]
        valid = (memory[:, :, :KEYS] * history[:, -1, None, :KEYS]).sum(-1) > 0
        sign, time = memory[:, :, KEYS+1], memory[:, :, KEYS+2]
        src = memory[:, :, KEYS+3:KEYS+5]
        if intervention == 'erase_sign':
            sign = torch.zeros_like(sign)
        if intervention == 'erase_time':
            time = torch.zeros_like(time)
        if self.kind == 'gru':
            scores = torch.zeros_like(sign)
        elif self.kind == 'unary':
            features = torch.stack((sign, time, src[:, :, 0], src[:, :, 1], time**2, sign*time), -1)
            scores = self.selector(features).squeeze(-1)
        else:
            n = memory.shape[1]
            si, sj = sign[:, :, None].expand(-1, -1, n), sign[:, None, :].expand(-1, n, -1)
            ti, tj = time[:, :, None].expand(-1, -1, n), time[:, None, :].expand(-1, n, -1)
            features = torch.stack((si, sj, ti, tj, (src[:, :, None]*src[:, None, :]).sum(-1), ti-tj), -1)
            pair = self.selector(features).squeeze(-1)
            scores = (pair*valid[:, None, :]).sum(-1)/valid.sum(-1, keepdim=True).clamp(min=1)
        logits = (scores*4).masked_fill(~valid, -1e9)
        weights = logits.softmax(-1)*valid
        values = memory[:, :, KEYS].clone()
        if intervention == 'shuffle':
            # Fixed cyclic permutation: metadata and attention unchanged.
            for b in range(len(values)):
                ids = torch.where(valid[b])[0]
                values[b, ids] = values[b, ids.roll(1)]
        read = (weights*values).sum(-1)
        available = valid.any(-1).float()
        if intervention == 'skip' or self.kind == 'gru':
            read, available = torch.zeros_like(read), torch.zeros_like(available)
        if patch_value is not None:
            read, available = patch_value, torch.ones_like(available)
        return self.interpret(h, read, available, inv), logits, weights, h


def count(model):
    return sum(p.numel() for p in model.parameters())


def gru_width():
    target = count(Actor())
    return min(range(16, 65), key=lambda w: (abs(count(Actor('gru', w))-target), w))
