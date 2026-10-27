# Tables

## dictionary.csv

| Column | Meaning |
|---|---|
| `entry_id` | Stable layer–channel identifier within this checkpoint, e.g. `L20_N2302` |
| `layer`, `neuron` | Zero-based text layer and MLP channel index |
| `sign` | Sign of the constant additive offset; all archived entries use +1 |
| `additive_offset` | Signed activation-unit offset, rounded in the historical artifact |
| `calibration_max_multiplier` | 4 × the finite-corpus observed maximum absolute channel activation |
| `mean_abs_first_token_delta_nats` | Mean absolute baseline-minus-intervention log-probability change across 16 prompts |
| `max_abs_first_token_delta_nats` | Largest absolute change across those prompts |
| `largest_measured_probe` | Selected argmax probe; incidental annotation, not a neuron meaning |
| `effect_concentration` | Largest absolute change divided by the sum of absolute changes |

## probe_effects.csv

One row per `(entry_id, probe_id)`: 11,264 cells. `baseline_minus_intervention_logp_nats` follows the historical sign convention: **negative means higher intervention probability** for the independently tokenized first target token at the first assistant position.

## examples.csv

One selected probe per entry. `baseline_excerpt` and `intervention_excerpt` are bounded stored excerpts; `excerpt_differs` is recomputed from those strings. `source_changed_flag` preserves the original generation comparison. Both flags agree for all 704 entries. The signed delta is the selected probe's first-position readout, not a score of the excerpt's correctness.

## probes.csv and model_metadata.json

The probe table contains exact prompts, target strings, and the readout definition. Metadata identifies the pinned model revision, architecture, verified fingerprints, and provenance limitations. See the [data card](../docs/DATA_CARD.md).

## Descriptive annotations and layer statistics

`descriptive_annotations.csv` contains a dimensionless contrast: mean standardized arithmetic delta minus mean standardized capital delta. Each probe is standardized across all sampled entries in its layer, including the scored entry, using population standard deviation. `count_abs_empirical_z_gt_3` counts readouts exceeding an empirical absolute z-score of 3. Neither field is a significance test or an intrinsic neuron meaning.

`layer_statistics.json` publishes every reference entry ID, probe order, means, population standard deviations, and the layer-level split-half Pearson correlation. The layer correlation is not a confidence score for individual neurons. Standardization can change the sign of the raw arithmetic–capital contrast; do not interpret this annotation as a difference in nats.
