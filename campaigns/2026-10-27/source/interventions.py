"""Position-safe, per-row neuron and norm-matched residual interventions.

Torch is imported only on a GPU host or by small synthetic-tensor tests.
"""
import torch


def active_positions(length, first_call, timing, decoded_so_far=0):
    """The first invocation is prefill; later length-one calls are decoding."""
    if first_call:
        return slice(1, length) if timing in {'continuous', 'prefill_only', 'pulse_8'} else slice(0, 0)
    # Prefill predicts generated token 1; seven cached calls predict tokens 2..8.
    enabled = timing in {'continuous', 'decode_only'} or (timing == 'pulse_8' and decoded_so_far < 7)
    return slice(0, length) if enabled else slice(0, 0)


class Intervention:
    """One spec per batch row, same prompt for all rows so there is no padding.

    Neuron specs add to the down-projection *input*. Residual-control specs add
    at its output, with identical token timing. These are comparisons at the
    same linear sub-update, not claims of exchangeable statistical nulls.
    """

    def __init__(self, down, specs, timing='continuous'):
        self.down, self.specs, self.timing = down, specs, timing
        self.calls = 0
        self.decoded_so_far = 0
        self.handles = []
        self.first = True

    def __enter__(self):
        device = self.down.weight.device
        self.rows = torch.tensor([i for i, s in enumerate(self.specs) if s['kind'] == 'neuron'], device=device)
        self.channels = torch.tensor([s['neuron'] for s in self.specs if s['kind'] == 'neuron'], device=device)
        self.amps = torch.tensor([s['amplitude'] for s in self.specs if s['kind'] == 'neuron'], device=device)
        residual_rows, residual_vectors = [], []
        for i, s in enumerate(self.specs):
            if s['kind'] == 'random_residual':
                g = torch.Generator(device='cpu').manual_seed(s['direction_seed'])
                v = torch.randn(self.down.weight.shape[0], generator=g, dtype=torch.float32)
                v = v / torch.linalg.vector_norm(v) * s['injected_l2_norm']
                residual_rows.append(i)
                residual_vectors.append(v)
        self.residual_rows = torch.tensor(residual_rows, device=device)
        self.residual_vectors = torch.stack(residual_vectors).to(device) if residual_vectors else None
        self.handles = [self.down.register_forward_pre_hook(self._pre), self.down.register_forward_hook(self._post)]
        return self

    def _pre(self, _module, inputs):
        x = inputs[0]
        if x.shape[0] != len(self.specs):
            raise RuntimeError('batch/spec mismatch; beam expansion is unsupported')
        self.positions = active_positions(x.shape[1], self.calls == 0, self.timing, self.decoded_so_far)
        self.calls += 1
        if self.calls > 1:
            self.decoded_so_far += x.shape[1]
        if not len(self.rows) or self.positions.stop == self.positions.start:
            return inputs
        y = x.clone()
        # Broadcasting index arrays preserves row/channel pairing at all positions.
        pos = torch.arange(x.shape[1], device=x.device)[self.positions]
        y[self.rows[:, None], pos[None, :], self.channels[:, None]] += self.amps[:, None].to(x.dtype)
        return (y,) + inputs[1:]

    def _post(self, _module, _inputs, output):
        if self.residual_vectors is None or self.positions.stop == self.positions.start:
            return output
        y = output.clone()
        y[self.residual_rows, self.positions, :] += self.residual_vectors[:, None, :].to(y.dtype)
        return y

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()


def trim_generation(ids, eos_ids, maximum):
    """Remove batch padding after the first configured EOS, retaining that EOS."""
    ids = list(ids)
    for i, value in enumerate(ids):
        if value in eos_ids:
            return ids[:i + 1], 'eos'
    return ids, 'cap' if len(ids) >= maximum else 'other'
