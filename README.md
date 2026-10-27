# Qwen3.8-27B-Neuron Dictionary

A reproducible archive of neuron interventions in **Qwen/Qwen3.8-27B**: paired generated outputs, token IDs, calibration, controls, exact experimental source, and verification tools.

**This release contains exploratory measurements, not a validated dictionary of neuron meanings. No new behavioral discovery is confirmed.**

## Stopped behavioral campaign

The two-B200 campaign was stopped for review after **5.29 hours**. It completed the same eight prompts in both directions for **30,592 neurons**, producing **489,472 committed intervention continuations**. Partial chunks are preserved separately. The target was 1,114,112 MLP channels; the released run is a partial census.

| Result | What the evidence supports |
|---|---|
| First-person snake narration at L20 N3432 | A fresh same-prompt replay changes a factual description to first-person narration. Generalization to other subjects is untested. |
| Riddle/wordplay framing under several interventions | Ordinary requests sometimes receive fabricated riddle interpretations. Other-channel controls and broader output damage prevent a selective-neuron claim. |
| Timing and dose effects | Some prompt-time or initial-pulse effects differ from decoding-only pushes. Large doses can produce empty or degraded outputs. |
| Automatic candidate ranking | Surface phrase changes often receive high scores without a substantive behavioral change. Four early candidates received followups; later channels did not receive automatic held-out followups. |

The run was stopped because continued exhaustive generation under that selection scheme was not justified for the goal of finding compelling behavioral changes. The full outputs are retained so others can inspect, re-score, or challenge that assessment.

- **[Campaign findings and limitations](docs/CAMPAIGN_FINDINGS.md)**
- **[Complete campaign data download](https://github.com/Volkopat/Qwen3.8-27B-Neuron-Dictionary/releases/tag/campaign-2026-10-27)** ? archive, SHA-256 receipt, source and per-file manifest.
- **[Campaign data card and reproduction](campaigns/2026-10-27/README.md)**
- **[Browsable campaign neuron index](data/campaign_neurons.csv)** ? coverage and descriptive output-change counts, without invented semantic labels.
- **[Selected full-output evidence](data/campaign_examples.json)** ? paired examples and their controls, including failure cases.

## Earlier probe archive

The earlier archive contains **704 sampled channels** from layers 12, 20, 32 and 44, **11,264 first-target-token effect measurements**, and 704 bounded paired excerpts. These are separate experiments with a different dose convention, not additional confirmed behavioral discoveries.

- [Earlier channel index](data/dictionary.csv), [all probe effects](data/probe_effects.csv), [paired excerpts](data/examples.csv), and [probe definitions](data/probes.csv).
- [Earlier findings](docs/FINDINGS.md), [methods](docs/METHODS.md), [data card](docs/DATA_CARD.md), and [verification record](docs/VERIFICATION.md).
- [Behavioral candidate register](discoveries/README.md).

All 18 checkpoint weight shards matched the pinned official revision. A fresh replay covered every earlier effect cell, but **20.44% exceeded the declared 0.15-nat tolerance**. Six selected example pairs reproduced all 12 short excerpts. Exact numerical reproduction of the earlier archive is not claimed.

## Intervention and interpretation

The campaign adds a signed constant to one post-gating MLP channel at the input to its down projection. Each row uses the same prompt as a sham row in its batch; every batch has equal prompt lengths. Both signs use a nominal injected contribution norm of 0.3 times the layer's small frozen calibration residual norm. Full generation is capped at 96 tokens for screening and 192 for followups; capped continuations are not completed answers.

Followups compare the selected channel with a sham, opposite sign, another channel and two seeded random residual directions. These match **nominal** linear contribution norms, not exactly realized BF16 perturbations. Such controls do not establish a calibrated statistical null. Large additive offsets test overdrive; they do not establish a channel's natural function.

The earlier archive instead uses a positive offset of four times each channel's observed calibration maximum and primarily measures the first target token at the first assistant position. Do not merge the two protocols or interpret text differences as answer accuracy.

## Verify and inspect

```bash
python -m pip install numpy
python tools/verify_release.py
python tools/verify_behavioral_candidates.py
python tools/verify_campaign_bundle.py
python -m unittest discover -s tests -v
python tools/browse.py --layer 20 --limit 10
```

To independently audit the downloaded and extracted full campaign:

```bash
python tools/audit_campaign.py --root /path/to/extracted/campaign --output campaign-audit.json --entries campaign-neurons.csv
```

An archive-integrity PASS establishes consistency of saved artifacts. It does not certify behavioral interpretation, mathematical novelty, or statistical significance. No model weights are distributed, and none of these verification commands loads a language model.

## Credits

The research workspace began from [Lyra's neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer); its authors deserve credit for the starting tools and methodology. [Goodfire's neuron research](https://already-clean.goodfire.com/) was an additional inspiration. This package does not redistribute the upstream explorer UI or tracked source. See [NOTICE.md](NOTICE.md) for provenance and licensing boundaries.

Related prior work includes [ROME](https://arxiv.org/abs/2202.05262) and [instruction-following activation steering](https://arxiv.org/abs/2410.12877). This release does not claim to have invented neuron steering.
