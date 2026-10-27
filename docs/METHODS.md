# Method and metric definitions

## Intervention

For an MLP down projection `y = W_down h + b`, change one channel to `h_n + a`. The immediate residual write changes by `a W_down[:, n]`. This identity follows from linearity of the down projection; later layers and generation can respond nonlinearly.

Historical offset: `a = sign × 4 × observed_abs_max[n]`. The archive contains positive signs only. `observed_abs_max` is a finite-corpus maximum, not a population bound or typical firing rate. The recovered calibration summary records 200,704 tokens. A positive offset can make a negative activation less negative; it is not necessarily “amplifying” that activation.

The historical hook changes prompt positions 1 onward and every single-token cached decoding step. Therefore examples combine altered prompt processing and altered generation. They do not isolate a decoding-only mechanism.

## Readout

Sixteen English prompts cover capitals, multiplication, simple facts, and self-identification. Each target is independently tokenized. Only its first token is scored at the first assistant output position. The complete target text is retained so this limitation is visible; in particular the water probe uses `H`, not the complete formula `H2O`.

Distinct target strings can share their first token (for example `144` and `169` in the replay). Sixteen probe rows therefore do not mean sixteen independent skills or semantic dimensions. Case, token boundaries, and whether the model would normally begin with an introductory phrase all affect this readout.

For entry i and probe j:

`d_ij = log P_baseline(target_first_token_j | prompt_j) − log P_intervention_i(target_first_token_j | prompt_j)`.

- `d < 0`: measured target token becomes more likely.
- `d > 0`: measured target token becomes less likely.
- Mean absolute effect: `mean_j |d_ij|`.
- Maximum absolute effect: `max_j |d_ij|`.
- Concentration: `max_j |d_ij| / sum_j |d_ij|`.

Concentration measures distribution across this small probe battery. Neither high nor low concentration establishes semantic selectivity. The argmax probe was selected using the same measurements, so its effect is subject to winner selection.

Historical values are rounded to three decimal places. Verification tolerances account for rounding, rather than insisting that summaries recomputed from rounded vectors be bit-identical. Four entries have rounded ties at the maximum; the archived selected probe is retained when its magnitude is within rounding tolerance of that maximum.

## Examples

The recorded example instrument defaults to greedy generation with a 12-new-token cap, flattens newlines, and stores at most 70 characters. The included source excerpts are no longer than 62 characters. Full original generations and token IDs were not preserved in these CSVs. An absent answer may occur later; formatting differences do not establish changed knowledge.

No held-out semantic accuracy, causal circuit localization, universal scale law, or broad capability improvement is inferred from these examples.

## Layer-standardized descriptive contrast

Within each sampled layer let `D` be its neuron-by-probe delta matrix, and `Z = (D − column_mean(D)) / (column_population_std(D) + 1e-9)`. The stored descriptive contrast is `mean(Z_arithmetic) − mean(Z_capitals)`. The count field is the number of the 16 entries with `abs(Z) > 3`. These formulas reproduce all 704 historical bias and count values within rounding. They use the full sampled layer, including each scored entry; they are not leave-one-out nulls.

The split-half Pearson correlation compares the contrast on capital/arithmetic pairs 1–2 with the contrast on pairs 3–4, across the sampled neurons. It is a layer-level internal-consistency statistic, not per-neuron validation. The z-score threshold has no calibrated p-value or multiple-testing correction.
