# Verification record

Audit date: **2026-10-27**. This document separates archive validation from fresh execution.

| Check | Status |
|---|---|
| Unique dictionary keys and complete example joins | PASS: 704 entries, four layers |
| Complete signed effect vectors | PASS: 11,264 cells recovered from four original JSON profiles |
| Magnitudes, concentrations, selected probes | PASS within recorded rounding; four rounded argmax ties retained |
| Intervention offsets vs recovered calibration maxima | PASS for every entry at 4 × observed maximum |
| Example excerpts and change flags | PASS: 187 differing excerpt pairs; source flags agree |
| Current checkpoint identity | Config, generation config, and tokenizer config exactly match pinned official revision |
| Whole-weight integrity | PASS: all 18 shards SHA-256 match official pinned-revision LFS digests; 55,563,006,776 bytes checked |
| Historical immutable per-run identity | NOT RECORDED: current checkpoint recovery cannot retroactively supply it |
| Fresh whole-dictionary GPU replay | COMPLETED WITH DIFFERENCES: all 704 entries / 11,264 cells replayed; 8,962 cells within the predeclared 0.15-nat tolerance, 2,302 outside |
| Fresh reproduction of example generations | PASS for six preselected entries: all 12 archived excerpts matched; full 12-token and 96-token continuations retained. Remaining 698 example pairs not freshly replayed |
| Broad semantic/capability findings | NOT ESTABLISHED by this release |

## Independently checked arithmetic and provenance

Two independent source/claim reviews were performed in addition to the programmatic release audit. The original remote artifacts were retrieved through CPU-only read jobs; those helper jobs were deleted after collection. The exported archive contained 253 artifact files; only the 27B dictionary/calibration subset is distributed here.

The recovered checkpoint config, generation config, and tokenizer config have the same SHA-256 hashes as the official files at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The HF cache reference and available download receipt identify that revision. A separate CPU-only job hashed every weight shard and matched all 18 to the official pinned-revision LFS SHA-256 digests; see `evidence/checkpoint_weights.json`. This authenticates the current checkpoint, while historical per-run revision receipts remain unavailable.

The primary dictionary omits layer-wide “verified” flags and unsupported semantic labels. The historical standardized contrast, empirical z-score counts, and layer-level correlations are separately reconstructed with explicit definitions. Original source CSVs are preserved unchanged under `evidence/source_csv/` with their limitations documented in the data card.

## Whole-dictionary replay result

A fresh BF16 replay on an NVIDIA B200 covered every archived entry and all 16 probes. The agreement was **not exact**:

| Comparison | Result |
|---|---:|
| Replayed entries / cells | 704 / 11,264 |
| Cells within declared 0.15-nat tolerance | 8,962 / 11,264 (79.56%) |
| Cells outside that tolerance | 2,302 / 11,264 (20.44%) |
| Median absolute delta difference | 0.074591 nats |
| Maximum absolute delta difference | 0.753625 nats |
| Pearson correlation across recorded cells | 0.996108 |
| Sign agreement across all cells | 91.59% |
| Sign agreement where archived absolute effect is at least 1 nat | 100% |

These are descriptive comparisons of dependent cells, not confidence intervals or independent replications across tasks. The original software/kernel environment was not fully pinned. Different numerical execution paths are a plausible explanation for some differences, but this audit does not identify their cause. Do not describe the entire dictionary as reproduced within tolerance or bit-identical.

`evidence/dictionary_replay.json` contains every archived and replayed vector; `evidence/replay_comparison.json` contains derived comparisons. The primary CSV preserves the historical measurement version rather than silently replacing it. The six selected text examples all matched at the historical short cap, a narrower result than numerical reproduction of the entire archive. That example replay used the CSV-rounded offsets; the dictionary replay used unrounded calibration maxima as in the original measurement loop.

## Re-run the checks

```bash
python tools/verify_release.py
python -m unittest discover -s tests -v
```

`evidence/verification_report.json` records the machine-readable archive audit and a separate GPU-replay status. **An archive-integrity PASS is not a claim that GPU measurements all matched.** The offline checker validates identity fields against the preserved official reference; the source comparison itself was performed online against the pinned official files. `evidence/SHA256SUMS.json` fingerprints the release files, excluding itself and the regenerable verification report. Tests deliberately corrupt a sign, duplicate an entry, flip a change flag, substitute another model identity, and change a revision to verify that the audit detects them.

The earlier n² law, general stopping rate, per-neuron out-of-sample claims, and Qwen2.5 reporting interpretation are excluded from release findings. No figure from the older model's paper outline is reused.
