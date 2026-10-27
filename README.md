# Qwen3.8-27B-Neuron Dictionary

**Exploratory intervention measurements for 704 sampled MLP channels.**

A measurement archive for **Qwen/Qwen3.8-27B**, containing recorded token-level effects, bounded text excerpts, calibration arrays, and an audit. It is not yet a dictionary of substantive behavioral discoveries. The verified examples mostly change formatting, preamble, and response length; they do not demonstrate a coherent persona change or a newly controllable capability.

## What is here

| Layer (zero-based) | Entries | Different paired excerpts |
|---|---:|---:|
| 12 | 256 | 43 |
| 20 | 192 | 80 |
| 32 | 128 | 30 |
| 44 | 128 | 34 |
| **Total** | **704** | **187 / 704 (26.6%)** |

“Different” means the stored text excerpts differ. It does **not** mean the factual answer changed. Each example uses its entry's largest measured probe, selected on the same probe battery; it is not held-out evidence.

## Start here

- [Behavioral discovery register](discoveries/README.md): two concrete archived candidates, complete available outputs, failure cases, and the tests still needed. **Zero candidates have passed behavioral confirmation.**
- [Dictionary](data/dictionary.csv): one row per sampled channel, with measured effect summaries.
- [All probe effects](data/probe_effects.csv): 11,264 archived channel–probe measurements.
- [Paired examples](data/examples.csv): 704 bounded before/after excerpts.
- [Probe definitions](data/probes.csv): exact prompts, target strings, and readout definitions.
- [Audited findings](docs/FINDINGS.md), [method](docs/METHODS.md), [data card](docs/DATA_CARD.md), and [verification](docs/VERIFICATION.md).

Use `python tools/browse.py --layer 20 --limit 10` to browse the data without loading a model.

## What a “press” means

At the input to an MLP's down projection, add a constant to one channel at every prompt position except the first, and continue adding it during cached generation. Historical runs used **4 × that channel's observed calibration maximum**. This is an additive offset, not a fourfold multiplication of the current activation.

The recorded effect is `log P(target first token | prompt, baseline) − log P(target first token | prompt, intervention)`. **Negative values mean the intervention increased that particular token's probability.** This is a first-position readout, not full-answer probability or factual accuracy.

## Scope and evidence

These are sampled entries from four of 64 text layers, not a complete map of the model. The “largest measured probe” is an incidental argmax, not a semantic neuron name. The model remains one checkpoint; results do not establish general laws of neural networks.

Raw profiles and calibration arrays were recovered and checked against all 704 entries. All 18 checkpoint weight shards and the configuration metadata match the pinned official Qwen3.8-27B revision. Six selected example pairs were freshly replayed on a B200: all 12 archived excerpts matched. These selected examples are demonstrations, not a representative behavioral benchmark. See [verification](docs/VERIFICATION.md) for the whole-dictionary replay status and remaining provenance limits. The repository does not distribute model weights.

**Numerical replay caveat:** the full 11,264-cell B200 replay correlated with the archive at r=0.996, but 20.44% of cells exceeded the predeclared 0.15-nat tolerance. Both measurement versions are included. Strong effects retained direction; exact numerical reproduction is not claimed.

## Verify locally

```bash
python -m pip install numpy
python tools/verify_release.py
python -m unittest discover -s tests -v
```

Verification checks hashes, identities, cross-file joins, all 11,264 effect cells, summary arithmetic, calibration amplitudes, example flags, and generated summary counts. It does not silently treat archived observations as a fresh GPU replication.

## Reproduce an intervention

The new [probe runner](tools/reproduce_probe.py) records complete text, generated IDs, all configured EOS IDs, checkpoint fingerprints, and both sign conventions. Run it on a GPU host with the matching checkpoint. See [reproduction instructions](docs/REPRODUCE.md).

## Credits

Inspired by [Lyra's neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer) and [Goodfire's “Neural Networks Are Already Clean”](https://already-clean.goodfire.com/). The original explorer and its authors deserve credit for the starting tools and methodology. This repository packages the subsequent 27B experiment artifacts with newly written audit and browsing tools; it does not redistribute the original explorer code or UI.

For related causal-intervention work, see [ROME](https://arxiv.org/abs/2202.05262) and [instruction-following activation steering](https://arxiv.org/abs/2410.12877). This release makes no claim to have invented neuron steering.
