import unittest
import torch
from interventions import Intervention, trim_generation


class InterventionTests(unittest.TestCase):
    def setUp(self):
        self.down = torch.nn.Linear(5, 3, bias=False)
        with torch.no_grad():
            self.down.weight.copy_(torch.arange(15).reshape(3, 5) / 10)

    def test_row_isolation_prefill_and_cached_decode(self):
        x = torch.ones(2, 4, 5)
        specs = [{'kind': 'neuron', 'neuron': 2, 'amplitude': 2.0}, {'kind': 'sham'}]
        base = self.down(x)
        with Intervention(self.down, specs):
            y = self.down(x)
            z = self.down(x[:, :1])
        self.assertTrue(torch.equal(y[:, 0], base[:, 0]))
        self.assertTrue(torch.equal(y[1], base[1]))
        expected = self.down.weight[:, 2] * 2
        torch.testing.assert_close(y[0, 1:] - base[0, 1:], expected.expand(3, -1))
        torch.testing.assert_close(z[0, 0] - base[0, 0], expected)

    def test_prefill_only_releases_cached_decode(self):
        x = torch.ones(1, 3, 5)
        spec = [{'kind': 'neuron', 'neuron': 1, 'amplitude': 3.0}]
        with Intervention(self.down, spec, 'prefill_only'):
            y = self.down(x)
            z = self.down(x[:, :1])
        torch.testing.assert_close(z, self.down(x[:, :1]))
        self.assertFalse(torch.equal(y[:, 1:], self.down(x)[:, 1:]))

    def test_decode_only_does_not_touch_prefill(self):
        x = torch.ones(1, 3, 5)
        with Intervention(self.down, [{'kind': 'neuron', 'neuron': 1, 'amplitude': 3.0}], 'decode_only'):
            y = self.down(x)
            z = self.down(x[:, :1])
        torch.testing.assert_close(y, self.down(x))
        self.assertFalse(torch.equal(z, self.down(x[:, :1])))

    def test_random_control_matches_norm_and_sham_is_inert(self):
        x = torch.ones(2, 3, 5)
        specs = [{'kind': 'random_residual', 'injected_l2_norm': 2.5, 'direction_seed': 8}, {'kind': 'sham'}]
        base = self.down(x)
        with Intervention(self.down, specs):
            y = self.down(x)
        torch.testing.assert_close(torch.linalg.vector_norm(y[0, 1:] - base[0, 1:], dim=-1), torch.full((2,), 2.5))
        torch.testing.assert_close(y[1], base[1])
        torch.testing.assert_close(y[:, 0], base[:, 0])

    def test_eos_padding_is_not_counted_as_generation(self):
        self.assertEqual(trim_generation([7, 4, 3, 3], {3, 4}, 4), ([7, 4], 'eos'))
        self.assertEqual(trim_generation([7, 8], {3, 4}, 2), ([7, 8], 'cap'))


if __name__ == '__main__':
    unittest.main()
