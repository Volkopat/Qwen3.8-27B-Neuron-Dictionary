# Math search receipts and all-neuron campaign protocol

Prepared 2026-10-27; local searches timestamped 17:56:09–17:56:27 UTC; web retrieval completed before this document was written (after 17:56:58 UTC). This is a pre-experiment literature/protocol receipt, not experimental evidence or a claim that an independently running experiment has not started. No jobs, model loads, GPU operations, or writes outside this file were performed by this research task.

## Required target and scope

New surgery experiments must use **Qwen3.8-27B exclusively**, with its own tokenizer and target-specific calibration. Before execution, the runner must record the actual repository/checkpoint revision, configuration and architecture, tokenizer revision, numerical precision/quantization, model-code revision, module inventory and intervention hook location. Identity was not verified by this read-only literature task. A model-name string is insufficient. If the checkpoint cannot be verified, report the blocker rather than substituting another model. Historical pilots do not confirm target-model findings.

“All neurons” needs an explicit enumeration: for this protocol, every intervenable post-nonlinearity/post-gating MLP channel of every applicable layer. Record exact count from the verified configuration and modules, plus attempted/completed/failed/skipped counts. An MLP-channel census is not a census of attention heads, residual directions, or all possible mechanisms. Proxy-only evaluation is a proxy screen, not exhaustive behavioral confirmation. Do not infer architectural dimensions from the advertised 27B size.

## Actual local interfaces and read-only query receipts

Discovery began with `rg --files` at `D:\Open-Source-Lab`, followed by targeted source and usage reads.

### Math KB

Root: `D:\Open-Source-Lab\Math KB`.

Usage read: `run/search.py`, `tools/kb.py`, `tools/kb_batch.py`, `tools/kb_facts.py`, and `docs/SEARCH_ENGINE.md`.

The main interface is `python run/search.py find "QUERY" --k 5`; `run/search.py` forwards to `tools/kb.py`. Other operations include `technique`, `analogy`, `context`, and JSONL `serve`. The documented batch interface is `python tools/kb_batch.py requests.json responses.json [--agent NAME] [--cpu | --gpu]`. Its `find` operations start a dense server and use lock/ledger machinery. **Neither was executed**, because this assignment forbids model loads and external-path changes.

The actual model-free interface used was `tools/kb_facts.py`'s `method(query)` function. Its CLI equivalent is:

```powershell
python -B 'D:\Open-Source-Lab\Math KB\tools\kb_facts.py' method 'false discovery rate'
```

Executed as one in-memory Python batch importing `kb_facts` from its tools directory, with `PYTHONDONTWRITEBYTECODE=1` and `python -B`; no request/output files or server. `method` reads the pre-existing `data/facts/method_index.parquet`. The inspected implementation performs no writes for this operation. The index was 198,463,760 bytes, last modified 2026-10-03 14:45:06 UTC. These are model-extracted abstract metadata, not verified theorem statements or authoritative method-origin attributions.

| Query, exact spelling | Observed response |
|---|---|
| `false discovery rate` | Exact method key; 141 indexed uses |
| `confidence sequence` | Exact key; 2 uses |
| `sequential testing` | Exact key; 32 uses |
| `causal inference` | Exact key; 619 uses |
| `Benjamini` | Five returned keys; leading key `benjamini hochberg procedure`, 102 uses |
| `dose response` | Exact key; 2 uses |

Follow-up read-only `pyarrow.parquet.read_table` filtered the same method index for the following exact keys and inspected referenced `facts.parquet` rows. Selected IDs below are catalogue leads, **not web-verified publications or theorem evidence**. Follow-up selection sorted ID strings descending, so it is not a reliable chronology for old-style arXiv IDs.

| Exact key | Indexed uses | Selected catalogue leads |
|---|---:|---|
| `confidence sequence` | 2 | 2607.17409; 2407.13977 |
| `benjamini yekutieli procedure` | 8 | 2512.10467; 2504.11759; 2401.16651 |
| `sequential testing` | 32 | 2608.27996; 2607.08522; 2606.23962 |
| `randomization test` | 17 | physics/0404016; 2608.20438; 2608.05527 |
| `winner s curse` | 3 | 2512.24521; 2408.06323; 2311.09083 |
| `dose response` | 2 | 2609.07559; 2601.14094 |
| `norm matched controls` | — | No exact-key row |

All 16 selected paper-facts rows had `new_result: "none"`. Nothing in that field supplies an applicable theorem. A failed/empty metadata field is not evidence that the underlying paper lacks results. Dense semantic/theorem-chunk retrieval was not performed under the no-model-load restriction.

### OpenAI mathematics collection

Actual local source: `D:\Mathematics\OpenAI\math-main`. Read its `README.md` and `CONTENTS.md`; no dedicated search executable was found in the targeted filename search. The documented navigation interface is the manuscript map `CONTENTS.md`, overview PDF, manuscript directories and Lean formalization catalogue. This is an on-disk manuscript collection, not an OpenAI API math-search endpoint.

README describes 721 manuscripts in 373 families, with mixed verification status and approximately 42% of top-line results formalized. These are catalogue self-reports, not independently audited mathematical validity. `CONTENTS.md` SHA256: `080BADCB677FB80BE0E695C21A8EC88BCB3A46DD3E3078CA10061E6B289BE175`.

Executed `rg -n -i -- PATTERN 'D:\Mathematics\OpenAI\math-main\CONTENTS.md'` separately for each pattern:

| Pattern | Matching lines | Assessment |
|---|---:|---|
| `false discovery` | 0 | No direct catalogue hit |
| `multiple testing` | 0 | No direct catalogue hit |
| `confidence sequence` | 0 | No direct catalogue hit |
| `dose.response` | 0 | No direct catalogue hit |
| `selection bias` | 0 | No direct catalogue hit |
| `sequential` | 4 | Complexity/geometry, not sequential inference |
| `causal` | 1 | QFT causal commutation, not causal inference |
| `winner` | 5 | Substrings in Swinnerton-Dyer, not winner's curse |
| `randomization` | 3 | Derandomization in complexity theory |
| `concentration` | 4 | Chemical concentration and a random-SAT threshold argument; no applicable testing theorem retrieved |
| `spherical` | 44 | Broad mathematical matches, not a matched-control theorem |
| `isotrop` | 16 | Broad matches including anisotropic algebraic groups |

Counts refer to matching lines, not manuscripts. This bounded catalogue search supplied **no directly applicable statistical theorem**; it does not establish absence from all manuscript full texts. No Lean proof was imported or checked.

## Primary published sources retrieved on the web

Search interface: `web.run` via the tool's `search_query`, `open`, and `click` operations. Exact search strings included:

- `Benjamini Yekutieli 2001 false discovery rate dependency Annals Statistics`
- `Howard Ramdas McAuliffe Sekhon 2021 time uniform nonparametric nonasymptotic confidence sequences`
- `Geva 2021 transformer feed forward layers key value memories Dai 2022 knowledge neurons`
- `site.aclanthology.org knowledge neurons pretrained transformers 2022`
- `site.jmlr.org Phipson Smyth permutation p values should never be zero`
- `"Benjamini" "Yekutieli" "2001" "pdf" site:tau.ac.il`
- `"The control of the false discovery rate" "dependency" site:projecteuclid.org/journalArticle/Download`

Sources and what was actually available:

1. **Geva et al., EMNLP 2021**, [Transformer Feed-Forward Layers Are Key-Value Memories](https://aclanthology.org/2021.emnlp-main.446/). Publisher abstract/metadata and searchable PDF text retrieved. Supports the established weighted-value-vector view of FFNs; does not establish target-model channel semantics.
2. **Geva et al., EMNLP 2022**, [Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space](https://aclanthology.org/2022.emnlp-main.3/). Publisher abstract/metadata retrieved. Prior empirical work on FFN sub-updates and prediction control; not a statistical guarantee for this campaign.
3. **Dai et al., ACL 2022**, [Knowledge Neurons in Pretrained Transformers](https://aclanthology.org/2022.acl-long.581/). Publisher abstract/metadata retrieved. BERT cloze-fact attribution/editing is prior art, not a Qwen3.8-27B replication.
4. **Wang et al., EMNLP 2022**, [Finding Skill Neurons in Pre-trained Transformer-based Language Models](https://aclanthology.org/2022.emnlp-main.765/). Publisher page opened. Documents predictive/task-specific neurons and perturbations after prompt tuning; establishes prior art, not universal single-neuron behavior.
5. **Howard et al., Annals of Statistics 2021**, [Time-uniform, nonparametric, nonasymptotic confidence sequences](https://arxiv.org/abs/1810.08240), DOI [10.1214/20-AOS1991](https://doi.org/10.1214/20-AOS1991). Author arXiv page opened; publisher PDF search excerpt retrieved. Relevant retrieved result: confidence sequences can provide coverage uniformly over time under their stated assumptions. Specific boundary implementation was not imported or verified here.
6. **Phipson & Smyth, 2010**, [Permutation P-values Should Never Be Zero](https://arxiv.org/html/1603.05766v1), DOI [10.2202/1544-6115.1585](https://doi.org/10.2202/1544-6115.1585). Author-posted corrected full HTML opened. Explains discrete randomization nulls and the danger of understated Monte Carlo p-values in multiple testing.
7. **Benjamini & Yekutieli, 2001**, [The control of the false discovery rate in multiple testing under dependency](https://doi.org/10.1214/aos/1013699998). Bibliographic identity found, but direct publisher HTML/PDF returned an iframe/access block and author-page access returned 502. **Full primary theorem text was not retrieved**. BH/BY conditions below are explicit protocol guidance using standard statements; do not describe this session as a full primary-proof check. No secondary source is used to claim a newly verified theorem.

Single-neuron attribution and intervention are established research directions. This bounded search does not justify novelty of an all-neuron campaign; any novelty claim needs a dedicated systematic review and comparison of coverage, target, interventions, controls and endpoints.

## Concrete protocol: own derivations and design recommendations

Everything below is a proposed design or elementary derivation, not a theorem retrieved from either local collection.

### Intervention and matched residual controls

At a fixed layer/token hook, write the MLP contribution as `y = W_down a + b`, with post-gating channel vector `a` and column `w_n = W_down[:, n]`. An additive channel change `a_n <- a_n + delta` gives exactly:

```text
Delta y_neuron = delta * w_n
rho = ||Delta y_neuron||_2 = |delta| * ||w_n||_2
```

This identity is conditional on that linear projection and hook. Validate actual tensor orientation and any subsequent normalization/scaling. Changing a pre-gate quantity, scaling/ablating the channel, or editing a whole block has different semantics. For ablation `delta = -a_n(x,t)`; norm matching must therefore be per item/token, not just at the mean activation.

For each matched control draw `g ~ N(0,I_d)`, set `v = g / ||g||_2` and inject `Delta y_random = rho * v` **at exactly the same residual contribution, layer, token mask and timing**. Use seeded multiple directions and preferably antipodal pairs. If delta or rho changes by item/token, match that realized norm. Do not renormalize the full residual after only one arm. If neuron surgery also triggers persistent hook behavior, the residual arm must match it.

This compares direction-specific impact at equal L2 perturbation size. Equal L2 norm does not match covariance, alignment with the residual, downstream Jacobian sensitivity, or natural activation likelihood. A neuron selected for maximal impact is not a uniform random direction. Add complementary controls: randomly selected real channels calibrated by their own target-model activation scales, sign reversal, sham/no-op, and optionally layer-covariance-matched residual directions. Keep these distinct scientific comparators.

Use two dose conventions and report both: (i) native channel dose `delta = c * s_n`, where `s_n` is a frozen target-calibration scale; (ii) fixed residual dose `delta = sign * rho / ||w_n||_2` for nonzero columns. The first reflects channel-scale manipulation, the second removes projection-column norm as a magnitude confound. Report zero/tiny-column handling and actual injected norm. Neither convention proves biological or mechanistic uniqueness.

### Estimands, paired measurements and sampling units

Fix behavior scores, direction of improvement and a minimum meaningful effect before confirmation. For each independent prompt family/item define paired differences:

```text
D_i = score(neuron intervention, i) - score(baseline, i)
C_i = score(neuron intervention, i) - mean_j score(random residual direction j, i)
```

Keep baseline-effect and direction-specificity estimands separate. Pair identical prompt, tokenization, decoding policy, context and stopping budget. For stochastic generation use multiple independent seeds, matched seed schedules where useful, and preserve the pairing; nominally equal seeds need not force the same random-number consumption after divergent trajectories. Cluster related paraphrases by source/template/fact. Repeated seeds and tokens do not become independent prompt replications. State the prompt population to which uncertainty applies; on a fixed deterministic benchmark, the measured finite-set mean is descriptive unless a sampling/randomization model is specified.

Track syntax validity, semantic score, refusal/format changes, output length, truncation, fluency and broad capability degradation. A shorter output or corrupted answer format can move a behavioral scorer without achieving the proposed behavior. Use blinded human or frozen independent scoring for confirmation; labels invented while reading discovery outputs require new held-out items.

### Exhaustive discovery, selection bias and held-out confirmation

1. Split prompt families into target calibration, discovery and untouched confirmation sets before scores are inspected. Freeze split IDs, scorer version, seeds and hashes. Prevent template/fact leakage across splits.
2. Enumerate every eligible channel. Give each the same preregistered minimum screen, including a sham and signed, calibrated intervention(s). Cache/batch only when intervention isolation is preserved. If using gradients/logits as a first pass, record which channels later received actual behavioral tests; retain proxy-only status for the rest.
3. Rank discovery effects or allocate extra discovery budget adaptively. Log every channel, sign, dose, behavior metric, token window and attempted tuning choice. Screening losers and failed trials remain part of coverage accounting.
4. Before opening confirmation, freeze candidate set, primary endpoint, sign, dose, token policy, comparator ensemble, test, sample size and multiplicity family. Ranking and dose tuning use discovery only. Reusing confirmation to choose a better sign, dose or label turns it into discovery.
5. Estimate fresh effects on all frozen candidates, including failures to replicate. Correct the entire frozen confirmation family. Discovery-only selection can reduce confirmation multiplicity to the selected hypotheses when the independent confirmation p-values remain valid conditional on discovery. It does not confirm every untested channel or remove multiplicity over endpoints actually searched on confirmation.

Own selection-bias derivation: if `hat(theta)_j = theta_j + epsilon_j` and all true effects are zero, the selected maximum is `max_j epsilon_j`, usually positive. For independent Gaussian errors the scale grows approximately as `sigma * sqrt(2 log M)` for large M; correlated channels change effective search size. This is an illustrative asymptotic, not a calibrated threshold. The winner's discovery effect and its same-data confidence interval cannot be reported as unbiased confirmation. Resampling selected data without repeating selection does not fix the problem.

### Valid nulls before FDR

Every inferential p-value needs a stated null and a valid construction: under that null `P(p <= u) <= u`. An empirical percentile among random directions, a normalized effect score, a bootstrap fraction, or a softmax over neuron effects is not automatically such a p-value. In particular, a learned/selected neuron and random sphere directions are not exchangeable merely because their norms agree. Report their contrast descriptively unless a defensible test addresses the specified null.

For a simple fixed-sample confirmation option, rescale each independent cluster score to [0,1], so its paired difference is in [-1,1]. Under independent identically sampled clusters, for the one-sided null `E[D] <= tau`, a conservative p-value from Hoeffding's inequality is:

```text
p = min(1, exp(-N * max(mean(D)-tau, 0)^2 / 2))
```

This is our elementary application of the bounded-mean inequality, not a theorem retrieved in the local search. Derivation: a width-2 variable gives `P(mean(D)-E[D] >= t) <= exp(-N*t^2/2)`. The same construction applies to C when independent cluster-level comparator averaging leaves C in [-1,1]. Fix tau, N, all averaging rules and the score bounds in advance. This can be conservative; do not apply it to unbounded logits or treat dependent paraphrases as independent. Explicitly randomize comparator directions independently across clusters or formulate inference conditional on a frozen direction ensemble. Alternative paired permutation/sign-flip tests need their own randomization/exchangeability/symmetry assumptions; a zero-mean null alone does not license arbitrary sign flips.

If a legitimate Monte Carlo randomization test is chosen, a common conservative construction is `(1 + exceedances)/(B + 1)` under an appropriate exchangeable null scheme. Adding one does not repair an invalid randomization. Its smallest attainable p-value is `1/(B+1)`, which must be checked against the multiplicity threshold before spending the budget.

For a frozen family of m valid p-values, BH uses ordered thresholds `k*q/m` under independence or suitable positive-dependence conditions. Shared prompts/correlated channels do not automatically satisfy those conditions. BY uses `k*q/(m*H_m)`, where `H_m = sum_{j=1}^m 1/j`, to accommodate arbitrary dependence of otherwise valid p-values. Neither repairs invalid null tests or confirmation leakage. A conservative fallback is Bonferroni at `q/m`, which controls family-wise error by a union bound with valid marginal p-values. At rank one, Monte Carlo resolution for BY requires roughly `B+1 >= m*H_m/q`; low-budget direction ensembles cannot support tiny adjusted p-values. Report adjusted decisions, effect sizes and uncertainty, not a claim that every discovered neuron has probability q of being false.

### Sequential screening and dose response

Adaptive discovery allocation is allowed as an explicitly exploratory efficiency device; preserve a minimum pass across every channel before calling the screen exhaustive. Repeated fixed-N p-value peeking during confirmation is not fixed-N inference. Prefer fixed-sample confirmation here. If early confirmation stopping is needed, preregister a confidence-sequence/e-process construction with checked boundedness/martingale assumptions and handle multiplicity across hypotheses as well as time. Howard et al. provides relevant theory, not permission to reuse arbitrary intervals after optional stopping.

Freeze a modest signed dose grid from target calibration, for example multipliers `{0, -0.5, +0.5, -1, +1, -2, +2}` of `s_n`; these are candidate design values, not validated doses. Examine all grid points for finalists on discovery, with realized residual norms and matched controls. Choose one primary dose for independent confirmation or include all confirmed doses in the testing family. Do not assume monotonicity: saturation, sign asymmetry, normalization and downstream nonlinearities can create nonmonotone responses. Freeze generation budget and intervention token mask; count censored outputs separately. A largest-dose failure does not establish a selective mechanism, and a sharp threshold does not by itself imply a phase transition.

## Execution handoff

This receipt is ready for the campaign coordinator before it authorizes GPU experiments. It supplies searched sources and a concrete proposed statistical protocol, not checkpoint verification or executed measurements. Before running, the coordinator/runner must finalize the target manifest, complete neuron enumeration, intervention semantics, calibrated dose convention, split/scorer hashes, fixed confirmation sample size, valid-null test and multiplicity family. Preserve this file's retrieval provenance; append subsequent experimental receipts elsewhere under the authorized campaign workflow.
