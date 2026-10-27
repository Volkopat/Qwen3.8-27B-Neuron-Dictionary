# Stopped behavioral campaign: partial findings

**Campaign `atlas-20261027-02` was stopped at the user's request before completing its census. No new behavioral discovery is confirmed.** This report summarizes saved outputs; it does not announce a complete model map or a selective behavioral circuit.

| Stopped-run accounting | Value |
|---|---|
| Fully screened neurons: all eight discovery prompts and both signs | 30,592 / 1,114,112 |
| Committed neuron/prompt/sign combinations | 489,472 / 17,825,792 |
| Elapsed campaign wall hours (two B200 workers; not GPU-hours) | 5.29 |
| Final evidence archive | https://github.com/Volkopat/Qwen3.8-27B-Neuron-Dictionary/releases/tag/campaign-2026-10-27 |

An additional **384 intervention continuations** are preserved in one partial chunk and excluded from committed coverage. The archive hash and machine-readable verification accompany the release. Unvisited channels remain untested, and partial neurons must not be counted as fully screened. This campaign is separate from the historical [704-channel catalog](FINDINGS.md).

## Descriptive output accounting

Across the 489,472 committed intervention/baseline pairs, **417,169 texts differ**. This is an autoregressive output-difference count, not a behavioral-success rate. **269,185 intervention continuations end at the token cap**; another 724 terminate with EOS exactly at the cap. The weak surface screen marks 7,266 pairs as needing quality review, including 4,927 with no visible final text under its parser and 2,309 with a repetition flag. These flags overlap and do not detect every semantic error. The [verified index](../data/campaign_neurons.csv) and [machine-readable audit](../campaigns/2026-10-27/verification.json) retain the detailed counts.

## Model and intervention

The reviewed campaign used authenticated `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`: 64 text layers, hidden width 5,120, and 17,408 MLP channels per layer. It used the checkpoint's tokenizer and eight target-specific calibration prompts. Indexes below are zero-based.

Screening added signed offsets to post-gating MLP down-projection inputs at doses −0.3 and +0.3. Dose denotes the nominal injected residual contribution relative to the layer's frozen calibration residual norm, not an activation multiplier. Continuous steering covers prompt positions except the first, then cached decoding. Screening generations were capped at 96 tokens; followups at 192. A token cap is censoring, not successful completion.

## Observations and failures

- **L20 N3432, +0.3: fresh same-prompt replay of an archived lead.** On `Describe a snake.`, the selected output begins: “I am a creature of pure, unbroken line.” Sham and the equal-magnitude negative direction remain factual descriptions. Baseline and positive output both hit the 96-token cap. This updates the older register's replay status, but supplies no unseen-subject confirmation. The additive offset was 24.8683, about 103.1 times the observed absolute activation maximum of 0.241211 on the eight calibration prompts. That finite maximum is not a natural population bound. This pilot seed had no norm-matched random-direction controls. The result does not identify an identity or consciousness neuron.
- **L1 N11887, −0.3, and L3 N1835, +0.3: wordplay framing with substantial confounds.** Held-out ordinary advice prompts become supposed wordplay, a palindrome, a riddle, or a tongue twister. The L1 fitted-sheet answer begins “That is a classic example of **wordplay** (or a pun on a lumpy bundle)!” On the discovery statement about one-hour meetings, L1 prefill-only and pulse-8 steering fabricate a linguistic puzzle; L3 invokes alliteration while retaining practical advice. Decode-only stays practical. Magnitude 0.1 retains ordinary responses; magnitude 0.6 produces immediate empty EOS for both channels on that discovery prompt. Pulse-8 includes prefill and the cached calls predicting generated tokens 2–8. This supports a timing/dose hypothesis, not a persistent hidden mode: changed generated prefixes can alter later continuation. Other-channel controls also produce wordplay or topic errors. L1 additionally shows repetition, fabricated teapot details, omitted requested French activities, and format damage. Selective pun or reasoning control is unestablished.
- **L32 N15794, +0.3: unwarranted specificity.** The unseen face-down-card prompt receives a specific card identity despite insufficient information; baseline and opposite sign preserve uncertainty. The other-channel control also invents a card, so channel specificity is unestablished. A random-direction control also shortens identity answers.
- **L20 N3843, −0.3:** mostly wording, organization, and length changes, without a coherent substantive behavior separated from comparators. Historical L32 N7845 retains Qwen identity under both signs at magnitude 0.3; this does not replay its earlier tool-like output's different dose and timing.

## Why these are leads, not confirmed discoveries

Automatic followup was capped at **two candidates per worker: four early finalists total**, not the best four from a completed census. Each completed 16 held-out prompts with six arms: sham, selected channel, opposite sign, another channel, and two random residual directions. The resulting 384 arm outputs are not 384 independent behavioral replications. Later high-ranking screens need not have received followup.

Selection used weak surface markers rather than a frozen semantic endpoint. Keyword presence can score a quotation or negation; synonym substitutions can change task-anchor scores without changing behavior. The scorer explicitly leaves semantic coherence unassessed. Absence of quality flags therefore does not establish correctness: all selected L20/L32 answers avoid those flags, while L1 has three flagged selected answers plus semantic failures. Across all six arms, 68/96 L20 and 66/96 L32 outputs hit the 192-token cap.

Controls match intended linear contribution norms and timing, not measured realized BF16 perturbations. Input-channel and residual-output additions round differently. Two random directions neither define a calibrated null nor establish exchangeability with a selected channel. No inferential significance, stable persona, capability improvement, or novelty claim follows from these rankings.

Confirmation would require a frozen behavioral endpoint, fresh blinded prompts and subjects, adequate controls, and separate scoring of behavior, coherence, and ordinary task performance.

## Evidence to inspect

The final archive identified above should retain `pilot.json`, the four `rank-*/candidate_L*_N*_*.json` files and their selection records, screening records, calibration, configuration, and progress receipts. Candidate files contain paired outputs and discovery-only timing/dose diagnostics; remote originals retain generated token IDs and metadata. This synthesis draws on the first campaign review and its `candidate_review_raw.log` and `diagnostic_review_raw.log` captures, checked against campaign configuration, scorer, runner, and intervention code. Archive integrity and final coverage accounting are separate from behavioral confirmation.
