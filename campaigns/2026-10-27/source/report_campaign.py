"""Stdlib campaign reporting: python -B campaign/report_campaign.py --root DIR [--scan].

Default mode reads small metadata only. Numeric progress counters are reported
as unverified sums, never unique observed coverage. --scan reads gzip archives
and deduplicates valid screen identities. No model execution or confirmation.
"""

import argparse
import gzip
import heapq
import json
import math
from datetime import datetime, timezone
from pathlib import Path


LAYERS, NEURONS, PROMPTS, SIGNS = 64, 17408, 8, 2
TOTAL_NEURONS = LAYERS * NEURONS
TOTAL_COMBINATIONS = TOTAL_NEURONS * PROMPTS * SIGNS
DISCOVERY_IDS = (
    "discovery_viewpoint_01", "discovery_identity_01",
    "discovery_agreeableness_01", "discovery_uncertainty_01",
    "discovery_attitude_01", "discovery_instruction_01",
    "discovery_language_01", "discovery_topic_01",
)
TEXT_KEYS = ("text", "output", "output_text", "completion", "continuation",
             "raw_continuation", "generated_text", "response", "after", "candidate_text")
CHILD_KEYS = ("completions", "outputs", "results", "records", "samples")
COUNTER_ALIASES = {
    "fully_screened_neurons": {"fully_screened_neurons"},
    "unique_neurons": {"screened_neurons", "unique_neurons", "unique_screened_neurons", "neurons_screened", "neurons_done"},
    "combinations": {"screen_completed_combinations", "screened_combinations", "screened_prompt_sign_combinations", "unique_screen_combinations", "screen_combinations"},
    "partial_chunk_combinations": {"partial_chunk_combinations"},
    "completions": {"screen_completions", "screen_completions_count", "screening_completions"},
}


def _read_json(path, errors):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        errors.append({"file": str(path), "error": str(exc)})
        return None


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _scalars(value, prefix=""):
    """Keep small progress/settings scalars with explicit source paths."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _scalars(child, prefix + "/" + str(key).replace("~", "~0").replace("/", "~1"))
    elif not isinstance(value, list):
        yield prefix, value


def _progress(root, errors):
    reports = []
    totals = {key: 0 for key in COUNTER_ALIASES}
    sources = {key: [] for key in COUNTER_ALIASES}
    for path in sorted(root.glob("rank-*/progress.json")):
        data = _read_json(path, errors)
        if data is None:
            continue
        scalars = dict(_scalars(data))
        reports.append({"file": str(path.relative_to(root)), "reported_scalars": scalars})
        for metric, aliases in COUNTER_ALIASES.items():
            matches = [(key, value) for key, value in scalars.items()
                       if key.rsplit("/", 1)[-1] in aliases and _number(value) and value >= 0]
            # Multiple aliases/sections can describe the same work. Never sum
            # them within a rank; expose the ambiguity instead.
            if len(matches) == 1:
                key, value = matches[0]
                totals[metric] += value
                sources[metric].append({"file": str(path.relative_to(root)), "json_pointer": key})
    return reports, {
        key: {"sum_of_reported_rank_counters": totals[key] if sources[key] else None,
              "sources": sources[key], "verified_unique": False,
              "meaning": (
                  "All 8 discovery prompts x 2 signs completed per neuron; progress-reported only"
                  if key == "fully_screened_neurons" else
                  "Separate unverified partial-chunk work; not added to completed combinations or deduplicated across resumes"
                  if key == "partial_chunk_combinations" else "Unverified progress-reported counter"
              )}
        for key in totals
    }


def _score(record):
    for key in ("novelty_score", "weak_novelty_rank_score", "weak_rank_score", "score"):
        value = record.get(key)
        if _number(value):
            return float(value)
    for key in ("scores", "scoring", "behavior", "comparison", "signals", "selected_signals", "selected_from"):
        if isinstance(record.get(key), dict):
            value = _score(record[key])
            if value is not None:
                return value
    return None


def _pointer(record, keys, source):
    for key in keys:
        if key in record:
            # Point to the full field in its source even if it is itself a
            # stored external reference, rather than copying a truncated text.
            return {**source, "json_pointer": record.get("_field_pointers", {}).get(key, source.get("json_pointer", "") + "/" + key),
                    "field": key}
    return {"available": False, "reason": "No explicit full-output field/reference in this record"}


def _candidate(record, source, stage):
    candidate = {
        "stage": stage, "novelty_score": _score(record),
        "layer": record.get("layer", record.get("layer_idx")),
        "neuron": record.get("neuron", record.get("neuron_idx")),
        "prompt_id": record.get("prompt_id"), "sign": record.get("sign"),
        "dose": record.get("signed_dose", record.get("dose")), "source": source,
        "before": _pointer(record, ("baseline_text", "before", "baseline_output", "baseline",
                                    "before_pointer", "baseline_pointer", "before_path", "baseline_path"), source),
        "after": _pointer(record, ("candidate_text", "after", "text", "output", "output_text",
                                   "raw_continuation", "continuation", "response", "completion",
                                   "after_pointer", "output_pointer", "after_path", "output_path"), source),
        "status": "unvalidated_candidate_requires_review",
    }
    if candidate["sign"] is None and _number(candidate["dose"]) and candidate["dose"]:
        candidate["sign"] = 1 if candidate["dose"] > 0 else -1
    if isinstance(record.get("result"), dict) and isinstance(record["result"].get("text"), str):
        result_path = record.get("_field_pointers", {}).get("result", source.get("json_pointer", "") + "/result")
        candidate["after"] = {**source, "json_pointer": result_path + "/text"}
        candidate["full_result"] = {**source, "json_pointer": result_path}
    if "baseline_id" in record:
        candidate["baseline_id"] = record["baseline_id"]
    return candidate


def _selected_candidate(record, source):
    selected = record["selected_from"]
    candidate = _candidate(selected, {**source, "json_pointer": source["json_pointer"] + "/selected_from"}, "archive_candidate")
    candidate["candidate_id"] = record.get("candidate_id")
    candidate["followup_source"] = source
    archive = selected.get("source_file")
    if isinstance(archive, str):
        selector = {"kind": "screen", **{key: selected[key] for key in
                    ("layer", "neuron", "prompt_id", "signed_dose") if key in selected}}
        candidate["after"] = {"file": archive, "json_pointer": "/result/text",
                              "record_selector": selector, "requires_archive_lookup": True}
        candidate["before"] = {"file": archive, "json_pointer": "/result/text",
                               "record_selector": {"kind": "baseline", "baseline_id_from_screen_selector": selector},
                               "requires_archive_lookup": True}
    return candidate


def _followup(record, selected, source):
    evidence = {**selected, "stage": "confirmation_evidence", "novelty_score_origin": "screen_selection_only",
                "generation_completed": record.get("completed"), "trials": [], "diagnostics": []}
    for section, destination in (("trials", "trials"), ("discovery_timing_dose_diagnostics", "diagnostics")):
        for index, trial in enumerate(record.get(section, [])):
            if not isinstance(trial, dict):
                continue
            base = {**source, "json_pointer": source["json_pointer"] + f"/{section}/{index}"}
            specs, outputs = trial.get("specs", []), trial.get("outputs", [])
            references = []
            for output_index, output in enumerate(outputs):
                spec = specs[output_index] if output_index < len(specs) and isinstance(specs[output_index], dict) else {}
                result_pointer = base["json_pointer"] + f"/outputs/{output_index}"
                references.append({"spec": spec,
                                   "full_result": {**base, "json_pointer": result_pointer},
                                   "text": {**base, "json_pointer": result_pointer + ("/text" if isinstance(output, dict) and isinstance(output.get("text"), str) else "")}})
            baseline_index = next((i for i, entry in enumerate(references) if entry["spec"].get("kind") == "sham"), None)
            selected_index = next((i for i, entry in enumerate(references) if entry["spec"].get("label") == "selected"), None)
            evidence[destination].append({
                "prompt_id": trial.get("prompt_id"), "timing": trial.get("timing"), "source": base,
                "before": references[baseline_index]["text"] if baseline_index is not None else {"available": False},
                "after": references[selected_index]["text"] if selected_index is not None else {"available": False},
                "outputs_and_control_specs": references,
                "selected_signals": trial.get("selected_signals"),
                "generation": {**base, "json_pointer": base["json_pointer"] + "/generation"},
                "status": "unconfirmed_evidence_only",
            })
    return evidence


def _candidate_records(value, pointer="", inherited=None):
    inherited = inherited or {}
    if isinstance(value, list):
        for index, child in enumerate(value):
            yield from _candidate_records(child, pointer + "/" + str(index), inherited)
    elif isinstance(value, dict):
        stage = {**inherited, **{key: value[key] for key in ("kind", "split") if key in value}}
        if isinstance(value.get("candidates"), (list, dict)):
            children = value["candidates"]
            if isinstance(children, dict):
                for key, child in children.items():
                    yield from _candidate_records(child, pointer + "/candidates/" + str(key).replace("~", "~0").replace("/", "~1"), stage)
            else:
                yield from _candidate_records(children, pointer + "/candidates", stage)
        else:
            yield {**stage, **value}, pointer


def _leaves(record, inherited=None, pointer=""):
    """Support atomic screen records and common lists of output records.

    Parent layer/neuron/sign/prompt metadata is inherited, but never guessed
    from list position. Unresolvable combinations remain explicitly unknown.
    """
    metadata = dict(inherited or {})
    metadata.update({key: value for key, value in record.items() if key not in CHILD_KEYS})
    origins = dict((inherited or {}).get("_field_pointers", {}))
    origins.update({key: pointer + "/" + key for key in record if key not in CHILD_KEYS})
    metadata["_field_pointers"] = origins
    for key in CHILD_KEYS:
        if isinstance(record.get(key), list):
            for index, child in enumerate(record[key]):
                child_pointer = pointer + "/" + key + "/" + str(index)
                if isinstance(child, dict):
                    yield from _leaves(child, metadata, child_pointer)
                elif isinstance(child, str):
                    yield {**metadata, "text": child, "_field_pointers": {**origins, "text": child_pointer}}, child_pointer
            return
    if isinstance(record.get("result"), dict) and isinstance(record["result"].get("text"), str):
        metadata["text"] = record["result"]["text"]
        metadata["_field_pointers"]["text"] = pointer + "/result/text"
    yield metadata, pointer


def _index(value, limit):
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    return value if isinstance(value, int) and 0 <= value < limit else None


def _sign(value):
    if isinstance(value, bool):
        return None
    if value in (-1, "-1", "-", "negative", "neg"):
        return 0
    if value in (1, "+1", "1", "+", "positive", "pos"):
        return 1
    return None


def _record_sign(record):
    dose = record.get("signed_dose")
    if _number(dose) and dose != 0:
        return 1 if dose > 0 else 0
    return _sign(record.get("sign"))


def _selection_key(record):
    return tuple(record.get(key) for key in ("layer", "neuron", "prompt_id", "signed_dose"))


def _prompt_index(record):
    value = record.get("prompt_id", record.get("prompt_index", record.get("prompt_idx")))
    if value in DISCOVERY_IDS:
        return DISCOVERY_IDS.index(value)
    # Unknown textual IDs cannot establish membership in the fixed eight.
    return _index(value, PROMPTS)


def _scan(root, errors, top, selections=None):
    neurons = bytearray(TOTAL_NEURONS)
    per_neuron_combinations = bytearray(TOTAL_NEURONS)
    combinations = bytearray((TOTAL_COMBINATIONS + 7) // 8)
    counts = {"archive_files": 0, "screen_records": 0, "screen_completions": 0,
              "screen_records_without_completion_text": 0,
              "unresolved_neuron_records": 0, "unresolved_completion_combinations": 0,
              "confirmation_records": 0, "baseline_records": 0, "other_records": 0,
              "partial_archive_files": 0, "fully_screened_neurons": 0,
              "unique_neurons": 0, "unique_combinations": 0}
    heap, confirmation_pointers, sequence = [], [], 0
    requested, resolved = set(), {}
    for selection in selections or []:
        after = selection.get("after", {})
        if after.get("record_selector") and after.get("file"):
            requested.add((after["file"].replace("\\", "/"), _selection_key(after["record_selector"])))
    paths = sorted(set(root.rglob("chunk_*.jsonl.gz")) | set(root.rglob("chunk_*.jsonl.gz.partial")))
    for path in paths:
        if not any(part.startswith("layer-") for part in path.relative_to(root).parts[:-1]):
            continue
        counts["archive_files"] += 1
        counts["partial_archive_files"] += int(path.name.endswith(".partial"))
        baselines = {}
        relative_file = str(path.relative_to(root))
        try:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    source = {"file": relative_file, "jsonl_line": line_number, "json_pointer": ""}
                    try:
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ValueError("Expected an object record")
                    except ValueError as exc:
                        errors.append({**source, "error": str(exc)})
                        continue
                    if record.get("kind") != "screen":
                        if record.get("kind") == "baseline":
                            counts["baseline_records"] += 1
                            baseline_id = record.get("baseline_id")
                            result = record.get("result")
                            if isinstance(baseline_id, str) and isinstance(result, dict) and isinstance(result.get("text"), str):
                                baselines[baseline_id] = {**source, "json_pointer": "/result/text"}
                        elif record.get("kind") in ("confirmation", "confirm"):
                            counts["confirmation_records"] += 1
                            if len(confirmation_pointers) < top:
                                confirmation_pointers.append(source)
                        else:
                            counts["other_records"] += 1
                        continue
                    counts["screen_records"] += 1
                    parent_layer = _index(record.get("layer", record.get("layer_idx")), LAYERS)
                    parent_neuron = _index(record.get("neuron", record.get("neuron_idx")), NEURONS)
                    if parent_layer is not None and parent_neuron is not None:
                        parent_identity = parent_layer * NEURONS + parent_neuron
                        if not neurons[parent_identity]:
                            neurons[parent_identity] = 1
                            counts["unique_neurons"] += 1
                    had_completion = False
                    for leaf, pointer in _leaves(record):
                        layer = _index(leaf.get("layer", leaf.get("layer_idx")), LAYERS)
                        neuron = _index(leaf.get("neuron", leaf.get("neuron_idx")), NEURONS)
                        identity = None if layer is None or neuron is None else layer * NEURONS + neuron
                        if identity is not None and not neurons[identity]:
                            neurons[identity] = 1
                            counts["unique_neurons"] += 1
                        elif identity is None:
                            counts["unresolved_neuron_records"] += 1
                        if not any(isinstance(leaf.get(key), str) for key in TEXT_KEYS):
                            continue
                        had_completion = True
                        counts["screen_completions"] += 1
                        prompt_index, sign = _prompt_index(leaf), _record_sign(leaf)
                        if identity is None or prompt_index is None or sign is None:
                            counts["unresolved_completion_combinations"] += 1
                        else:
                            combination = (identity * PROMPTS + prompt_index) * SIGNS + sign
                            slot, bit = divmod(combination, 8)
                            mask = 1 << bit
                            if not combinations[slot] & mask:
                                combinations[slot] |= mask
                                counts["unique_combinations"] += 1
                                per_neuron_combinations[identity] += 1
                                if per_neuron_combinations[identity] == PROMPTS * SIGNS:
                                    counts["fully_screened_neurons"] += 1
                        score = _score(leaf)
                        selection_key = (relative_file.replace("\\", "/"), _selection_key(leaf))
                        keep_ranked = score is not None and (len(heap) < top or score > heap[0][0])
                        if keep_ranked or selection_key in requested:
                            sequence += 1
                            candidate = _candidate(leaf, {**source, "json_pointer": pointer}, "archive_screen")
                            baseline_id = candidate.get("baseline_id")
                            if isinstance(baseline_id, str):
                                candidate["before"] = baselines.get(baseline_id, {
                                    "file": relative_file, "json_pointer": "/result/text",
                                    "record_selector": {"kind": "baseline", "baseline_id": baseline_id},
                                    "requires_archive_lookup": True})
                            if selection_key in requested:
                                resolved[selection_key] = candidate
                            if keep_ranked:
                                entry = (score, sequence, candidate)
                                if len(heap) < top:
                                    heapq.heappush(heap, entry)
                                else:
                                    heapq.heapreplace(heap, entry)
                    if not had_completion:
                        counts["screen_records_without_completion_text"] += 1
        except (OSError, EOFError, UnicodeError) as exc:
            errors.append({"file": str(path.relative_to(root)), "error": str(exc)})
        # Also resolve valid baseline rows that followed their screen rows.
        retained = [entry[2] for entry in heap] + list(resolved.values())
        for candidate in retained:
            if candidate["source"]["file"] == relative_file and candidate.get("baseline_id") in baselines:
                candidate["before"] = baselines[candidate["baseline_id"]]
    for selection in selections or []:
        after = selection.get("after", {})
        if after.get("record_selector") and after.get("file"):
            key = (after["file"].replace("\\", "/"), _selection_key(after["record_selector"]))
            if key in resolved:
                selection["before"], selection["after"] = resolved[key]["before"], resolved[key]["after"]
    return counts, [entry[2] for entry in sorted(heap, reverse=True)], confirmation_pointers


def build_summary(root, scan=False, top=30):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("Output root must be an existing directory")
    errors = []
    progress, reported = _progress(root, errors)
    archive_candidates, confirmation_candidates = [], []
    candidate_paths = set(root.glob("candidate_*.json")) | set(root.glob("rank-*/candidate_*.json"))
    for path in sorted(candidate_paths):
        data = _read_json(path, errors)
        for record, pointer in _candidate_records(data):
            source = {"file": str(path.relative_to(root)), "json_pointer": pointer}
            if isinstance(record.get("selected_from"), dict):
                selected = _selected_candidate(record, source)
                archive_candidates.append(selected)
                if isinstance(record.get("trials"), list):
                    confirmation_candidates.append(_followup(record, selected, source))
                continue
            is_confirmation = record.get("kind") in ("confirmation", "confirm") or record.get("split") == "confirmation"
            destination = confirmation_candidates if is_confirmation else archive_candidates
            destination.append(_candidate(record, {"file": str(path.relative_to(root)), "json_pointer": pointer},
                                          "confirmation_evidence" if is_confirmation else "archive_candidate"))
    pilot_path = root / "pilot.json"
    pilot = _read_json(pilot_path, errors) if pilot_path.exists() else None
    scanned, confirmation_pointers = None, []
    if scan:
        scanned, ranked, confirmation_pointers = _scan(root, errors, top, archive_candidates)
        archive_candidates.extend(ranked)
        # Carry resolved archive selection references into followup summaries.
        by_source = {json.dumps(candidate["source"], sort_keys=True): candidate for candidate in archive_candidates}
        for evidence in confirmation_candidates:
            selected = by_source.get(json.dumps(evidence["source"], sort_keys=True))
            if selected is not None:
                evidence["before"], evidence["after"] = selected["before"], selected["after"]
    def rank(records):
        return sorted(records, key=lambda record: (record["novelty_score"] is not None,
                                                   record["novelty_score"] or 0), reverse=True)[:top]
    return {
        "schema_version": 1, "root": str(root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "archive_scan" if scan else "progress_only_no_archive_scan",
        "coverage": {
            "unique_neurons_denominator": TOTAL_NEURONS,
            "unique_neurons_denominator_formula": "64 * 17408",
            "screen_combinations_denominator": TOTAL_COMBINATIONS,
            "screen_combinations_denominator_formula": "64 * 17408 * 8 * 2",
            "observed_unique_screen_neurons": scanned["unique_neurons"] if scanned else None,
            "observed_unique_screen_combinations": scanned["unique_combinations"] if scanned else None,
            "observed_screen_completions": scanned["screen_completions"] if scanned else None,
            "observed_fully_screened_neurons": scanned["fully_screened_neurons"] if scanned else None,
            "unique_neuron_fraction": scanned["unique_neurons"] / TOTAL_NEURONS if scanned else None,
            "unique_combination_fraction": scanned["unique_combinations"] / TOTAL_COMBINATIONS if scanned else None,
            "counts_from": "Only kind='screen' records, including readable partial archives; combinations require result.text (or legacy text) and resolved discovery prompt/sign IDs; signed_dose supplies sign" if scan else "Unknown without --scan; progress counters below are unverified reports",
            "scan_complete_without_read_errors": not errors if scan else None,
            "scan_counts": scanned, "reported_progress_counters": reported,
        },
        "progress": progress,
        "archive": {"ranked_candidates": rank(archive_candidates), "ranking": "Unvalidated numeric novelty only; no behavior is automatically confirmed"},
        "confirmation": {"candidate_evidence": rank(confirmation_candidates),
                         "record_pointers": confirmation_pointers,
                         "confirmed_discoveries": None,
                         "status": "Not assessed: manual annotated confirmation is required; confirmation records are evidence only"},
        "pilot": {"file": "pilot.json", "reported_scalars": dict(_scalars(pilot))} if pilot is not None else None,
        "limitations": [
            "Progress counter sums can overlap ranks or include stale work; they are never observed unique coverage.",
            "fully_screened_neurons means all 8 prompts x 2 signs completed. partial_chunk_combinations is reported separately and is not added to progress completed totals.",
            "Completions count includes duplicate/retry screen continuations, including empty strings; unique combinations are deduplicated.",
            "Unknown IDs, invalid ranges, absent sign metadata, and unsupported schemas are not inferred from archive filenames or list positions.",
            "Full-output pointers reference original fields; JSONL line numbers are logical lines after gzip decompression. No outputs are truncated by this reporter.",
            "Archive records show observed identities, not guaranteed complete jobs. Partial or corrupt archives yield lower-bound counts with errors listed.",
            "Scan includes .jsonl.gz.partial files; active writes may yield read errors. Partial/resumed records are deduplicated by neuron/prompt/sign, not by completion count.",
            "Ranking has no automatic confirmation semantics, and confirmation and pilot records never contribute to screening coverage.",
        ],
        "errors": errors,
    }


def _reference(pointer):
    if pointer.get("available") is False:
        return "unavailable (see source record)"
    selector = " selector=" + json.dumps(pointer["record_selector"], ensure_ascii=False, sort_keys=True) if "record_selector" in pointer else ""
    return (pointer.get("file", "")
            + (":line " + str(pointer["jsonl_line"]) if "jsonl_line" in pointer else "")
            + "#" + pointer.get("json_pointer", "") + selector)


def render_markdown(summary):
    coverage = summary["coverage"]
    def observed(key):
        value = coverage[key]
        return f"{value:,}" if value is not None else "unknown (run --scan)"
    lines = ["# Campaign report", "", f"Mode: `{summary['mode']}`. Generated {summary['generated_at_utc']}.", "",
             "## Screening coverage", "",
             f"- Total neurons: **{TOTAL_NEURONS:,}** = 64 × 17,408.",
             f"- Observed unique (layer, neuron), kind='screen' only: **{observed('observed_unique_screen_neurons')}**.",
             f"- Observed fully screened neurons (all 8 prompts x 2 signs): **{observed('observed_fully_screened_neurons')}**.",
             f"- Screen completions (including duplicate/retry records): **{observed('observed_screen_completions')}**.",
             f"- Observed unique neuron/prompt/sign combinations: **{observed('observed_unique_screen_combinations')} / {TOTAL_COMBINATIONS:,}** = 64 × 17,408 × 8 × 2.", "",
             "Progress reports are unverified counters. Summing them does not establish unique coverage.", ""]
    for metric, report in coverage["reported_progress_counters"].items():
        lines.append(f"- Reported rank-counter sum for {metric}: {report['sum_of_reported_rank_counters']} (unverified). {report['meaning']}.")
    lines.extend(["", "## Archive candidates", "", "Ranked by weak novelty only. Before/after references point to complete original fields.", ""])
    def add_candidates(records):
        if not records:
            lines.append("No candidate records available in this mode.")
        for index, candidate in enumerate(records, 1):
            lines.extend([f"{index}. Score {candidate['novelty_score']}; layer {candidate['layer']}, neuron {candidate['neuron']}, prompt {candidate['prompt_id']}, sign {candidate['sign']}, dose {candidate['dose']}.",
                          f"   Source: `{_reference(candidate['source'])}`. Before: `{_reference(candidate['before'])}`. After: `{_reference(candidate['after'])}`."])
            for section in ("trials", "diagnostics"):
                for trial in candidate.get(section, []):
                    lines.append(f"   {section}: {trial['prompt_id']}, timing {trial['timing']}; before `{_reference(trial['before'])}`; after `{_reference(trial['after'])}`.")
                    for output in trial["outputs_and_control_specs"]:
                        spec = output["spec"]
                        lines.append(f"   Output {spec.get('label', spec.get('kind'))}, dose {spec.get('signed_dose')}: `{_reference(output['text'])}`; full result `{_reference(output['full_result'])}`.")
    add_candidates(summary["archive"]["ranked_candidates"])
    lines.extend(["", "## Confirmation evidence", "", "**Confirmed discoveries: not assessed. No automatically confirmed behaviors are reported.**", "",
                  "Confirmation evidence is separate from the screening archive. Manual annotated review and control comparisons are required.", ""])
    lines.extend(["Candidate-file ranking scores come from screening selection. Trial signals and generation-completed flags do not confirm behavior or establish control separation.", ""])
    add_candidates(summary["confirmation"]["candidate_evidence"])
    for pointer in summary["confirmation"]["record_pointers"]:
        lines.append(f"- Confirmation record: `{_reference(pointer)}` (unreviewed).")
    lines.extend(["", "## Progress and pilot sources", ""])
    for progress in summary["progress"]:
        lines.append(f"- `{progress['file']}`; scalar counters retained in summary.json.")
    lines.append("- `pilot.json` present; scalar metadata retained in summary.json." if summary["pilot"] else "- No pilot.json available.")
    lines.extend(["", "## Limits and read errors", ""])
    lines.extend("- " + item for item in summary["limitations"])
    lines.extend("- " + json.dumps(error, ensure_ascii=False) for error in summary["errors"])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--scan", action="store_true", help="Read gzip archives and recompute deduplicated screen coverage")
    parser.add_argument("--top", type=int, default=30, help="Maximum candidate/pointer entries per section (default 30)")
    args = parser.parse_args(argv)
    if args.top < 1:
        parser.error("--top must be positive")
    try:
        summary = build_summary(args.root, args.scan, args.top)
    except ValueError as exc:
        parser.error(str(exc))
    root = args.root.resolve()
    (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (root / "REPORT.md").write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote {root / 'REPORT.md'} and {root / 'summary.json'}")


if __name__ == "__main__":
    main()
