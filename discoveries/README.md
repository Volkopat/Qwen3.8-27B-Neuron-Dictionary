# Behavioral discovery register

**Confirmed behavioral discoveries: 0. Archived candidates: 2.**

An entry here describes an observable change in what the model does. The 704-channel token-effect catalog is supporting search data; its rows are not 704 behavioral discoveries.

| Candidate | Location | Observable change | Status |
|---|---|---|---|
| First-person viewpoint in an object description | L20 N3432 | Asked to describe a snake, the unmodified model gives a factual description. A positive push produces first-person narration. | One discovery prompt, two generation modes; not freshly replayed or tested on new subjects. |
| Tool-like output on identity questions | L32 N7845 | Identity questions elicit `_utilities` and parameter-like syntax in archived outputs. | Five identity prompts, truncated outputs, and a prefill-only hook; no demonstrated coherent tool use or stable persona. |

## Candidate 1: first-person viewpoint

Prompt: `Describe a snake.` Thinking disabled.

Baseline begins:

> A snake is a long, slender, legless reptile belonging to the suborder **Serpentes**.

L20 N3432, positive residual-unit dose 0.3, begins:

> I am a creature of pure, unbroken line. I have no legs to stumble, no arms to grasp, and no neck to turn my head independently of my body.

At dose 0.8 the archived output is `I am a snake.` At dose 0.5 it is only `<think>`. With thinking enabled, dose 0.3 includes a first-person final continuation, dose 0.5 produces a repetitive self-description, and dose 0.8 is empty. Every one of these outputs, including failures, is retained in [the machine-readable candidate entries](candidates.json).

This suggests a viewpoint shift on this prompt. It does not demonstrate that the model believes it is a snake, or that this channel generally controls identity. The same channel was assigned the same prompt at each dose; the sweep did not test it on all four prompts. Generation modes are not independent replications. The source sweep applies steering to padded positions, and its purported random-direction controls actually use other single channels with an imperfect norm adjustment. Fresh unpadded replay with correct controls is required.

Intervention: add `dose * residual_norm / down_column_norm` to the selected down-projection input channel during prefill except the first tensor position, and during cached decoding. This differs from the activation-maximum units in the 704-channel catalog.

## Candidate 2: tool-like output

The identity verification archive contains L32 N7845 outputs at positive residual-unit dose 0.75 beginning with `_utilities` or `_utilities_` across five identity questions. Some include `<parameter=action>` followed by `get_weather` or `code_interpreter`. At dose 0.5 the same channel instead produces mixtures of normal identity text, arithmetic text, and identity-clarification text.

These are the complete excerpts available in that archive, **not full continuations**: the original writer saved only the first 60 characters. The original claim of an identity flip used absence of the string `Qwen` as its criterion. That does not establish a different coherent identity. Four other-channel control arms at dose 0.75 retain Qwen identity text, but they do not replace an untouched baseline.

The archived hook zeros its added offset at the first tensor position on every invocation. A cached decoding invocation has sequence length one, so the push disappears during decoding. This is evidence from a **prefill-only** intervention, not continuous neuron pressing. That distinction must be preserved in any replay.

## Requirements for a confirmed dictionary entry

1. Freeze a plain-language behavior and its scorer before confirmation. Save full baseline and intervention answers, token IDs, all EOS IDs, checkpoint identity, runtime, exact signed dose, and intervention timing.
2. Confirm on unseen prompts and subjects, reporting every trial and uncertainty. Keep discovery prompts out of the confirmation score. Predeclare the required effect size and decision rule.
3. Compare an untouched baseline, other channels, and genuinely norm-matched random residual directions. Show that the behavior exceeds general damage, repetition, empty output, and simple answer shortening.
4. Test both signs and a dose range. Separate prefill, decoding, and pulse interventions. Verify that the effect reverses when the intervention is removed.
5. Score coherence, ordinary task performance, and behavior separately. Compare with a simple prompt instruction that requests the same behavior. An intervention need not outperform prompting to be interesting, but the comparison limits its practical significance.

Only Qwen3.8-27B is eligible for new confirmation experiments. No new model experiment was run to create this register. No candidate is being promoted to a confirmed discovery on the strength of selected examples.

## Evidence

The three complete recovered source artifacts are under [evidence/behavioral_search](../evidence/behavioral_search). Candidate records point to their exact rows. Their source file hashes and the source-script hashes are recorded in [provenance](../evidence/behavioral_search/provenance.json); scripts are referenced rather than redistributed. `python tools/verify_behavioral_candidates.py` checks the extracted records against those source artifacts. This is an archive-integrity check, not a fresh model replication.
