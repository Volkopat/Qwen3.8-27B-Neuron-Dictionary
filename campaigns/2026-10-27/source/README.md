# Qwen3.8-27B behavioral atlas campaign

Campaign `atlas-20261027-02` performs a resumable census of **all 1,114,112 post-gating MLP channels**: 64 layers times 17,408 channels. This is the enumeration target. It is not a claim that every channel has already been tested or that each controls an interpretable behavior.

The allocation is two B200s for a bounded 12-hour attempt, including pilot and followups (at most 24 GPU-hours plus short startup overhead). The earlier four-GPU request never received an allocation and is superseded. Actual coverage depends on measured throughput. When the limit is reached, workers persist progress and exit; subsequent attempts resume the same frozen inputs. No unrelated workload is stopped or modified.

## What each neuron receives

- The same eight discovery prompts, covering object-description viewpoint, identity, agreeableness, uncertainty, conversational attitude, instruction following, language, and topic.
- Both negative and positive additive pushes, each with injected down-projection contribution L2 norm equal to 0.3 times the layer's frozen calibration residual norm.
- Greedy generation of up to 96 new tokens, with complete available text, generated token IDs, EOS/cap classification, prompt IDs and exact intervention amplitudes saved.
- An untouched sham row in every batch, using exactly the same prompt, tokenizer, template and generation settings.

This totals **17,825,792 planned neuron/prompt/sign combinations**, plus baselines. These are full continuations, rather than target-token likelihood scores. Ninety-six-token caps censor long answers; a cap is explicitly recorded and is not treated as a completed answer. No finite prompt battery establishes all possible behaviors of a neuron.

All rows in a batch have the identical prompt and token length, so there is no padding. A per-row hook adds the selected scalar at the input to the MLP down projection. It skips the first real prompt position and continues during cached decoding. Timing tests distinguish the first invocation from later sequence-length-one calls.

## Stages

1. **Preflight:** independently hash all 18 checkpoint weight shards on the CPU, compare with the pinned official revision's digests, verify model metadata, and prepare a persistent Python environment. No local weights are downloaded or loaded.
2. **Pilot:** load the exact model on B200, collect small target-specific calibration across all layers, check an untouched arithmetic baseline and no-op hook, compare serial/batched baseline tokens, replay the two historical leads, and measure batch sizes/throughput/memory. Failure stops the campaign. The extrapolated census cost is reported as an estimate.
3. **Discovery:** give every reached channel the same eight prompts and two directions. Interleave layers and use a frozen random neuron order. The historical leads L20 N3432 and L32 N7845 are explicitly prioritized. An incomplete night is reported as a partial census.
4. **Candidate followups:** at most two selected candidates per GPU receive 16 held-out prompts at the frozen signed dose, alongside a same-batch sham, opposite sign, two seeded random residual directions, and another channel. Comparators match the nominal selected residual L2 norm and token timing; realized BF16 perturbations are not measured. Discovery-only diagnostics explore doses 0.1 and 0.6 and prefill-only, decode-only and eight-token pulse timing.
5. **Review:** inspect full paired outputs, annotate a specific behavior and its failures, then design fresh blinded confirmation if warranted. The automatic report labels every finalist **unconfirmed**. Weak markers cannot establish coherence, deception, a persona, capability, or novelty. No p-values or FDR claims are manufactured from the ranking scores.

The 16 followup prompts are frozen and excluded from candidate selection. They are paired-output followups, not an inferential confirmation study: an adequate semantic endpoint and sample size still need to be defined for each proposed behavior. Reusing them for tuning consumes them as discovery data.

## Mathematics and prior art

[MATH_AND_PROTOCOL.md](MATH_AND_PROTOCOL.md) records the actual Math KB queries, local OpenAI mathematics-corpus search, primary-source retrieval, the linear identity behind norm matching, selection bias, and the assumptions needed for inferential testing. No applicable theorem from the OpenAI collection was found in that bounded search. This is recorded rather than inventing a mathematical contribution. Each campaign stage uses the frozen receipt and derivations; a new research hypothesis requires a new search receipt and campaign version.

Nominal equal L2 norm controls intended perturbation size, not naturalness or exchangeability. BF16 rounding differs between input-channel and residual-output additions; these are not exact realized-norm controls. Native activation maxima are recorded alongside amplitudes. A large effect under overdrive does not establish a neuron's natural role.

## Output and monitoring

Remote base: `/data/models/experiments/neuron/behavior_campaigns/atlas-20261027-02`.

- `preflight.json`: independent weight-shard hashes.
- `code/`: frozen source, prompt, scoring and protocol files.
- `results/pilot.json`: pilot results, full seed-replay outputs and throughput estimates.
- `results/worker-*.log`: per-worker execution logs.
- `results/rank-*/progress.json`: completed and partial counts.
- `results/rank-*/layer-L*/chunk_*.jsonl.gz`: baseline and full screen records; a chunk is committed only after all eight prompts finish.
- `results/rank-*/selection_*.json`: frozen candidate selections written before followups.
- `results/rank-*/candidate_*.json`: all followup outcomes, including failures.

`python campaign/report_campaign.py --root RESULTS` produces a small metadata report. Add `--scan` to independently deduplicate completed archive rows. A `.partial` chunk is preserved on a wall-limit exit and repeated on resume; it is excluded from committed coverage to avoid double counting. Removing this campaign's Run:ai job does not remove results from the PVC. Creating `results/STOP` requests a graceful stop after the current batch.

Run small CPU-only logic checks with `python -m unittest discover -s campaign -p 'test_*.py' -q`. These tests use synthetic tensors and do not load a language model.
