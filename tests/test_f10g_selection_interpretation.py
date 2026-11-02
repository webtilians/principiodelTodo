import unittest
import numpy as np
import torch
from observatory.f10g_worlds import world, seed_for, KEYS, VARIANTS
from observatory.f10g_models import Actor, count, gru_width
from observatory.f10g_experiment import tensors


class F10GTests(unittest.TestCase):
    def test_generator_semantics_and_pairing(self):
        for split in ('train', 'dev', 'ood'):
            for j in range(8):
                cases = world(split, 0, j)
                self.assertEqual([c['variant'] for c in cases], list(VARIANTS))
                self.assertEqual(cases[0]['truth'], cases[1]['truth'])
                self.assertNotEqual(cases[2]['truth'], cases[3]['truth'])
                for c in cases:
                    m, h = c['memory'], c['history']
                    q = h[-1, :KEYS].argmax()
                    signed = [i for i in range(32) if m[i, q] and m[i, KEYS+1]]
                    winner = max(signed, key=lambda i: m[i, KEYS+2])
                    self.assertEqual(winner, c['winner'])
                    self.assertEqual(c['truth'], int(m[winner, KEYS] > 0)^int(h[-1, -1]))
                    self.assertEqual(int(m[:, q].sum()), c['k'])
                    np.testing.assert_array_equal(m, h[:32, :29])
                np.testing.assert_array_equal(cases[0]['memory'][:, :25], cases[3]['memory'][:, :25])
                np.testing.assert_array_equal(cases[0]['memory'][:, 27:], cases[3]['memory'][:, 27:])

    def test_no_confirmation_or_out_of_range(self):
        for args in [('confirm', 0, 0), ('train', 4, 0), ('dev', 0, 128)]:
            with self.assertRaises(ValueError):
                seed_for(*args)

    def test_capacity(self):
        target = count(Actor())
        self.assertEqual(target, count(Actor('unary')))
        self.assertLessEqual(abs(count(Actor('gru', gru_width()))-target)/target, .05)

    def test_selector_no_value_leak_and_interventions(self):
        torch.set_num_threads(1)
        x, m, y, w = tensors(world('train', 0, 0))
        for kind in ('unary', 'relational'):
            model = Actor(kind).eval()
            with torch.no_grad():
                normal, scores, weights, hidden = model(x, m)
                changed = m.clone(); changed[:, :, KEYS] *= -1
                self.assertTrue(torch.equal(scores, model(x, changed)[1]))
                self.assertTrue(torch.equal(weights, model(x, m, 'shuffle')[2]))
                self.assertTrue(torch.equal(hidden, model(x, m, 'skip')[3]))
                self.assertTrue(torch.equal(model(x, m, 'skip')[0], model(x, changed, 'skip')[0]))
                value = m[torch.arange(4), w, KEYS]
                direct = model.interpret(hidden, value, torch.ones(4), x[:, -1, -1])
                self.assertTrue(torch.equal(direct, model(x, m, patch_value=value)[0]))
                perm = torch.randperm(32)
                self.assertTrue(torch.allclose(normal, model(x, m[:, perm])[0], atol=1e-6))


if __name__ == '__main__':
    unittest.main()
