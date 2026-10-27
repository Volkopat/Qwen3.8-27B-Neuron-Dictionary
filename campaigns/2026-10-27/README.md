# Stopped campaign data card

Campaign `atlas-20261027-02` ran on two NVIDIA B200s and was stopped at the user's request after **5.29 wall hours** (approximately 10.58 allocated GPU-hours). Both worker processes exited normally. All campaign and export workloads were removed; the archived data remains available.

| Data | Released coverage |
|---|---:|
| Model | Qwen/Qwen3.8-27B, pinned revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Eligible post-gating MLP channels | 1,114,112 across 64 layers |
| Fully screened channels | 30,592 (2.75%) |
| Discovery prompts per complete channel | 8 |
| Signed residual-unit doses | −0.3 and +0.3 |
| Committed intervention continuations | 489,472 |
| Additional partial continuations | 384, excluded from committed coverage |
| Early candidates with followups | 4, each with 16 prompts and six arms |
| Confirmed behavioral discoveries | 0 |

These are counts of saved experiments, not independent behavioral replications or success rates. Each continuation is generated greedily with a cap: 96 tokens for the screen and 192 for followups. An EOS is distinguished from a token cap, and complete generated token IDs are retained.

## Download the complete archive

[Release and download](https://github.com/Volkopat/Qwen3.8-27B-Neuron-Dictionary/releases/tag/campaign-2026-10-27)

- Filename: `qwen38-27b-behavior-campaign-20261027.tar.gz`
- Size: **76,476,621 bytes**
- SHA-256: `eee2181ffbf8374392f3d7847fe213fbd1a7cd412aeace7e7826c318509dc143`
- Files: **997**, including the archive's manifest; 996 payload-file hashes were independently checked after download.

[archive.json](archive.json) records the download URL, size and checksum. [verification.json](verification.json) contains the streaming audit of committed and partial records. The [neuron index](../../data/campaign_neurons.csv) and [selected full-output examples](../../data/campaign_examples.json) are also directly available in the repository.

Extract the archive into a dedicated directory, then run:

```bash
python tools/audit_campaign.py --root /path/to/extracted/campaign --output campaign-audit.json --entries campaign-neurons.csv
```

This command reads saved artifacts; it does not load weights or execute archived campaign code. It checks frozen source hashes, compressed chunk fingerprints, row uniqueness and complete combinations, baseline joins, generation metadata, and the four candidate followups. Partial artifacts are retained and reported separately. It does not assess semantic coherence or statistical significance.

## Archive layout

| Path | Contents |
|---|---|
| `ARCHIVE_MANIFEST.json` | SHA-256 and byte size of every other archived file |
| `code/` | Exact source and configuration used for the run |
| `preflight.json` | All 18 checkpoint weight-shard digest comparisons |
| `results/pilot.json` | Health checks, throughput, memory, and full pilot seed replays |
| `results/calibration.npz` and `.json` | Frozen activation maxima, projection-column norms and residual calibration |
| `results/rank-*/layer-L*/chunk_*.jsonl.gz` | Committed screen rows and matching same-batch baselines |
| `results/rank-*/layer-L*/*.partial` | Incomplete screen work, excluded from committed coverage |
| `results/rank-*/selection_*.json` | Candidate selections frozen before followups |
| `results/rank-*/candidate_*.json` | All followup arms and discovery-only timing/dose diagnostics |
| `results/rank-*/runtime.json` and `progress.json` | Runtime identity and stopped-run counters |
| `results/worker-*.log` | Execution logs |

## Repository layout

The [source directory](source/) is a byte-identical copy of the archived `code/` directory. [evidence/](evidence/) provides small source receipts, calibration, pilot and candidate artifacts for convenient review; full screen chunks are in the release asset. [stop.json](stop.json) records the final counters and user-requested stop.

The CSV index contains one row for each fully screened channel. Its changed-text counts and degradation flags are descriptive surface measures, not behavior names or success labels. The original weak novelty score is retained in the archive so its failures can be inspected, rather than rewritten retrospectively.

## Reproducing or extending experiments

The archived runner is the exact laboratory instrument, not a universal model adapter. It expects the model path and Run:ai/PVC layout in its frozen configuration and a preflight receipt for those exact weight files. The verified runtime was BF16 PyTorch 2.8.0+cu128 with Transformers 5.21.0 on B200. Optional optimized linear-attention kernels were absent; reference PyTorch paths were used.

For a new run, use a fresh campaign directory and identity, provide an authenticated Qwen3.8-27B checkpoint and its own tokenizer, create a current preflight receipt, and update the paths before freezing the new source hashes. Reusing the old checkpoint file timestamps as a new authentication receipt is invalid. Do not overwrite the stopped results or silently reuse these held-out prompts for further tuning. No weights are distributed in either the repository or data asset.

## Scientific limits

This is a partial census on a small fixed prompt battery. Additive pushes can greatly exceed observed calibration activations. Comparators match nominal linear norms, not realized BF16 deltas. Automatic ranking counted surface phrase changes, and only four early candidates received followups. The full archive enables reanalysis; it does not establish that any channel has a single semantic function. See [the synthesis](../../docs/CAMPAIGN_FINDINGS.md) for observations, failures and rejected interpretations.
