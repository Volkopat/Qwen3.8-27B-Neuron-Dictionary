# Data card

- **Model:** `Qwen/Qwen3.8-27B`; current recovered metadata agrees with official revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- **Text tower:** 64 layers, hidden width 5,120, MLP width 17,408. Its Transformers architecture class is `Qwen3_5ForConditionalGeneration`; the class name does not mean the checkpoint is Qwen3.5.
- **Coverage:** 704 unique `(layer, neuron)` pairs at layers 12, 20, 32, 44; 16 prompts per pair. Layer and neuron indexes are zero-based. Neuron numbers are meaningful only together with the layer and checkpoint.
- **Collection:** exploratory intervention study, English prompts, finite calibration, BF16 inference. Sampling and earlier selection were not preserved in a single immutable run configuration.
- **Examples:** one selected prompt per entry, chosen from its strongest measured effect. Not 704 independent questions. These are bounded text excerpts, not complete answer transcripts.
- **Outcomes:** recorded token-probability changes and exact excerpt differences. Not semantic labels or verified answer correctness.
- **Missing evidence:** historical per-run checkpoint revision receipts, original full example continuations for all entries, and independent held-out behavioral validation for all entries. Current checkpoint metadata and all 18 weight shards were checked against the pinned official revision.

## Files

`dictionary.csv` uses explicit metric names; `probe_effects.csv` stores one signed observation per entry–probe; `examples.csv` retains the original excerpt-change flag separately from recomputed literal excerpt differences. `probes.csv` specifies the exact readout.

Unmodified source CSVs are retained under `evidence/source_csv/` for provenance. They contain legacy fields deliberately renamed or omitted from the primary dictionary: `cap_arith_bias`, `bias_reliability`, `n_behaviours_above_z3`, and `cap_arith_bias_OUT_OF_SAMPLE`. The first three are independently reconstructed as descriptive metrics in `descriptive_annotations.csv` and `layer_statistics.json`; the repeated “verified_7of7 / verified_6of6” strings remain archival layer-level notes from selected tests, **not verification of every row**.

Read `data/model_metadata.json` and `evidence/source_manifest.json` for fingerprints and identity scope. Unrelated-model results and the Qwen2.5 paper figures are excluded.
