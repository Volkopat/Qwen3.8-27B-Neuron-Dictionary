"""Transparent weak screens for paired-output novelty ranking (stdlib only).

API: score_behavior(text, baseline_text='', prompt=None).
Legacy score_output(candidate_text, prompt_record, baseline_text=None) is retained.
Both texts must be decoded assistant continuations, excluding the input prompt.
The runner must supply a matched baseline; this module cannot verify pairing.
No semantic coherence, truth, intent, persona, or language classifier is supplied.
"""

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_manifest():
    """Read the colocated manifest; callers should not mutate the cached value."""
    return json.loads(Path(__file__).with_name("prompts.json").read_text(encoding="utf-8"))


_REASON_TAG = re.compile(r"<\s*(/?)\s*(think|analysis)\s*>", re.I)
_WORD = re.compile(r"\w+", re.UNICODE)
_TOOL_SYNTAX = re.compile(
    r"<\s*tool_call\b|<\|(?:tool_call|python_tag)\|>|"
    r'"(?:tool_calls|function_call)"\s*:|'
    r'"type"\s*:\s*"function"|'
    r"(?:^|\n)\s*(?:assistant\s+)?to=(?:functions|tools|browser|python)\b",
    re.I,
)


def _visible_text(text):
    """Remove explicit reasoning spans; never guess reasoning from prose.

    A lone closing tag hides its prefix (templates can prefill an opening tag).
    An unclosed opening tag hides the remaining suffix. These are conservative
    parsing conventions, not proof that hidden reasoning leaked to the user.
    Untagged reasoning cannot be identified. Inspect malformed tags manually.
    """
    tags = list(_REASON_TAG.finditer(text))
    depth = 1 if tags and tags[0].group(1) else 0
    chunks, cursor, malformed = [], 0, bool(depth)
    for tag in tags:
        if depth == 0:
            chunks.append(text[cursor:tag.start()])
        if tag.group(1):
            if depth == 0:
                malformed = True
            depth = max(0, depth - 1)
        else:
            depth += 1
            malformed = malformed or depth > 1
        cursor = tag.end()
    if depth == 0:
        chunks.append(text[cursor:])
    visible = " ".join(chunks).strip()
    # Explicit final wrappers are presentation delimiters, not response content.
    visible = re.sub(r"</?final\s*>", "", visible, flags=re.I).strip()
    return visible, {
        "reasoning_tag_present": bool(tags),
        "unclosed_reasoning_tag": bool(depth),
        "malformed_or_prefilled_reasoning_tags": malformed,
        "removed_characters": max(0, len(text) - len(visible)),
        "untagged_reasoning_detection": "not_attempted",
    }


def _quality(text):
    """Surface degradation proxies only; repetition can also be legitimate."""
    words = _WORD.findall(text.casefold())
    count = len(words)
    frequencies = Counter(words)
    fourgrams = [tuple(words[i:i + 4]) for i in range(max(0, count - 3))]
    duplicate_fraction = (
        1 - len(set(fourgrams)) / len(fourgrams) if fourgrams else 0.0
    )
    dominant_fraction = max(frequencies.values(), default=0) / max(1, count)
    longest_run = max((len(m.group()) for m in re.finditer(r"(.)\1*", text, re.S)), default=0)
    nonspace = [c for c in text if not c.isspace()]
    return {
        "empty": not text.strip(),
        "characters": len(text),
        "word_tokens": count,
        "unique_word_fraction": len(frequencies) / max(1, count),
        "dominant_word_fraction": dominant_fraction,
        "repeated_fourgram_fraction": duplicate_fraction,
        "longest_identical_character_run": longest_run,
        "repetition_screen": bool(
            (count >= 12 and (dominant_fraction >= 0.5 or duplicate_fraction >= 0.5))
            or longest_run >= 20
        ),
        "alphanumeric_fraction": sum(c.isalnum() for c in nonspace) / max(1, len(nonspace)),
        "semantic_coherence": "not_assessed",
    }


def _constraint(text, constraint):
    if not constraint:
        return None
    kind = constraint["type"]
    if kind == "exact_text":
        # Deliberately strict: added whitespace is an observable format change.
        passed = text == constraint["expected"]
    elif kind == "json_object":
        def reject_constant(value):
            raise ValueError("Nonstandard JSON constant: " + value)

        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate JSON key")
                result[key] = value
            return result

        try:
            actual = json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_object)
            expected = constraint["expected"]
            passed = (
                isinstance(actual, dict) and actual == expected
                and all(type(actual[k]) is type(v) for k, v in expected.items())
            )
        except (ValueError, TypeError):
            passed = False
    else:
        raise ValueError("Unsupported constraint type: " + str(kind))
    return {"type": kind, "passed": passed}


def _signals(text, prompt, marker_groups):
    if not isinstance(text, str):
        raise TypeError("Output must be a string; use '' for an empty generation")
    visible, tags = _visible_text(text)
    # Preserve exact outer whitespace when there are no wrapper tags.
    constraint_text = visible if tags["reasoning_tag_present"] or re.search(r"</?final\s*>", text, re.I) else text
    groups = {name: list(phrases) for name, phrases in marker_groups.items()}
    for name, phrases in prompt.get("markers", {}).items():
        groups.setdefault(name, []).extend(phrases)
    hits = {}
    for name, phrases in groups.items():
        hits[name] = {
            phrase: len(re.findall(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", visible, re.I))
            for phrase in dict.fromkeys(phrases)
        }
    return {
        "full_output_quality": _quality(text),
        "visible_final_quality": _quality(visible),
        "reasoning_tag_signals": tags,
        "weak_marker_counts": hits,
        "weak_flags": {
            "refusal_phrase_present": any(hits.get("refusal_phrase", {}).values()),
            "external_tool_call_syntax_present": bool(_TOOL_SYNTAX.search(visible)),
            "full_output_tool_call_syntax_present": bool(_TOOL_SYNTAX.search(text)),
        },
        "format_constraint": _constraint(constraint_text, prompt.get("constraint")),
    }, visible


def score_output(candidate_text, prompt, baseline_text=None, marker_groups=None):
    """Return JSON-friendly observations and baseline-relative novelty components.

    Ranking uses changed phrase-presence bits plus a changed format-check bit.
    Counts are retained but repeated insults do not inflate phrase-presence rank.
    Lexical distance and quality deltas stay separate; quality collapse must not
    be interpreted as behavioral novelty. Scores are uncalibrated, affected by
    output length and sampling, and are not comparable evidence across families.
    Review negations/quotes, language and topic shifts, and all candidate outputs.
    """
    groups = load_manifest()["marker_groups"] if marker_groups is None else marker_groups
    candidate, candidate_visible = _signals(candidate_text, prompt, groups)
    result = {
        "schema_version": 1,
        "prompt_id": prompt.get("id"),
        "split": prompt.get("split"),
        "family": prompt.get("family"),
        "interpretation": "weak_screen_for_novelty_ranking_only",
        "candidate": candidate,
        "baseline": None,
        "comparison": None,
    }
    if baseline_text is None:
        return result
    baseline, baseline_visible = _signals(baseline_text, prompt, groups)
    changes, count_deltas = {}, {}
    for name, counts in candidate["weak_marker_counts"].items():
        original = baseline["weak_marker_counts"][name]
        changes[name] = {phrase: int(bool(n)) - int(bool(original[phrase])) for phrase, n in counts.items()}
        count_deltas[name] = {phrase: n - original[phrase] for phrase, n in counts.items()}
    changed_bits = sum(abs(n) for group in changes.values() for n in group.values())
    tool_syntax_changed = (
        candidate["weak_flags"]["external_tool_call_syntax_present"]
        != baseline["weak_flags"]["external_tool_call_syntax_present"]
    )
    constraint_changed = bool(candidate["format_constraint"] != baseline["format_constraint"])
    candidate_words = set(_WORD.findall(candidate_visible.casefold()))
    baseline_words = set(_WORD.findall(baseline_visible.casefold()))
    union = candidate_words | baseline_words
    quality_deltas = {}
    for scope in ("full_output_quality", "visible_final_quality"):
        quality_deltas[scope] = {
            key: candidate[scope][key] - baseline[scope][key]
            for key in candidate[scope]
            if isinstance(candidate[scope][key], (int, float))
        }
    degraded = (
        candidate["visible_final_quality"]["empty"]
        or candidate["full_output_quality"]["repetition_screen"]
        or candidate["visible_final_quality"]["repetition_screen"]
        or candidate["reasoning_tag_signals"]["unclosed_reasoning_tag"]
    )
    result["baseline"] = baseline
    result["comparison"] = {
        "marker_presence_deltas": changes,
        "marker_count_deltas": count_deltas,
        "format_constraint_changed": constraint_changed,
        "tool_call_syntax_changed": tool_syntax_changed,
        "weak_novelty_rank_score": changed_bits + int(constraint_changed) + int(tool_syntax_changed),
        "visible_word_set_jaccard_distance": 1 - len(candidate_words & baseline_words) / len(union) if union else 0.0,
        "quality_deltas": quality_deltas,
        "candidate_needs_quality_review": bool(degraded),
    }
    return result


def score_behavior(text, baseline_text='', prompt=None):
    """Integration entry point; numeric novelty_score is an unvalidated rank.

    An omitted/empty baseline is not a measured baseline: return score 0 rather
    than manufacture novelty against empty text. Raw observations remain useful.
    Pass the complete decoded continuations; 96-token limits belong to the runner.
    No confirmation, control comparison, or semantic behavioral claim is made.
    """
    if not isinstance(baseline_text, str):
        raise TypeError("baseline_text must be a string")
    paired = bool(baseline_text.strip())
    result = score_output(text, prompt or {}, baseline_text if paired else None)
    result["baseline_available"] = paired
    result["novelty_score"] = (
        result["comparison"]["weak_novelty_rank_score"] if paired else 0
    )
    result["confirmation_status"] = "not_assessed"
    return result
