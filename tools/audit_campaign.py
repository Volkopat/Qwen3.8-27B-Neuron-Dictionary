#!/usr/bin/env python3
"""Read-only, CPU/stdlib audit of the frozen atlas-20261027-02 archive.

Usage: python -B tools/audit_campaign.py --root ARCHIVE --output audit.json
                                       [--entries fully_screened.csv]

ARCHIVE contains code/, preflight.json and results/. A results/ root with a
sibling code/ is also accepted. No campaign code is imported or executed.
Compressed hashes and JSONL are read in bounded blocks/one line at a time.
Coverage bitmaps and per-neuron numeric arrays are bounded by the target's
1,114,112 channels; full continuations are retained only for the current chunk.
Exit 0 means the audit checks passed, including for a valid partial census;
exit 1 means integrity/schema errors. Missing followups are explicitly reported.
Neither outcome establishes behavioral success or independently rehashes weights.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys


LAYERS, CHANNELS, COMBOS = 64, 17408, 16
TOTAL = LAYERS * CHANNELS
FULL = (1 << COMBOS) - 1
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
CSV_FIELDS = [
    "layer", "neuron", "committed_combinations", "expected_combinations",
    "coverage_fraction", "changed_text_pairs", "changed_token_pairs",
    "needs_quality_review_pairs", "empty_pairs", "repetition_screen_pairs",
    "format_constraint_changed_pairs", "marker_delta_pairs", "weak_candidate_pairs",
    "cap_pairs", "eos_pairs", "eos_at_cap_pairs", "native_overdrive_recorded_pairs",
    "signed_amplitude_over_calibration_abs_max_min",
    "signed_amplitude_over_calibration_abs_max_max",
    "absolute_amplitude_over_calibration_abs_max_min",
    "absolute_amplitude_over_calibration_abs_max_max",
]


def number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def integer(x):
    return isinstance(x, int) and not isinstance(x, bool)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Audit:
    def __init__(self, root):
        self.root = root.resolve()
        self.base = self.root.parent if self.root.name == "results" else self.root
        self.results = self.root if self.root.name == "results" else self.root / "results"
        self.code = self.base / "code"
        self.errors, self.warnings = Counter(), Counter()
        self.examples = []
        self.counts = Counter()
        self.token_stats = {}
        self.chunks, self.partials = [], []
        self.runtime, self.progress, self.failures = [], [], []
        self.candidates, self.selections = [], []
        self.observed = array("H", [0]) * TOTAL
        self.coverage = array("H", [0]) * TOTAL
        self.duplicate_neuron = bytearray(TOTAL)
        self.metrics = {k: array("H", [0]) * TOTAL for k in CSV_FIELDS[5:17]}
        self.ranges = {k: array("d", [math.nan]) * TOTAL for k in CSV_FIELDS[17:]}
        self.discovery, self.confirmation, self.eos = [], [], {}

    def rel(self, path):
        try:
            return path.relative_to(self.base).as_posix()
        except ValueError:
            return str(path)

    def issue(self, code, path, detail, warning=False):
        (self.warnings if warning else self.errors)[code] += 1
        if len(self.examples) < 200:
            self.examples.append({"severity": "warning" if warning else "error",
                                  "code": code, "file": self.rel(path), "detail": str(detail)})

    def read(self, path, required=True):
        if not path.is_file():
            if required:
                self.issue("missing_file", path, "required receipt or artifact is absent")
            return None
        try:
            with path.open(encoding="utf-8-sig") as f:
                value = json.load(f)
            if not isinstance(value, dict):
                raise ValueError("JSON object required")
            return value
        except (OSError, ValueError) as exc:
            self.issue("invalid_json", path, exc)
            return None

    def check(self, condition, code, path, detail):
        if not condition:
            self.issue(code, path, detail)
        return bool(condition)

    def hash_map(self, receipt, path):
        if not isinstance(receipt, dict) or not receipt:
            self.issue("source_hash_receipt", path, "expected nonempty filename-to-SHA256 map")
            return 0
        checked = 0
        for name, expected in receipt.items():
            if not isinstance(name, str) or Path(name).name != name:
                self.issue("unsafe_source_path", path, name)
                continue
            target = self.code / name
            try:
                digest = sha256(target)
            except OSError as exc:
                self.issue("source_file_missing", target, exc)
                continue
            checked += 1
            self.check(digest == expected, "source_hash_mismatch", target, "frozen source digest differs")
        actual = {p.name for p in self.code.iterdir() if p.is_file() and p.suffix in {".py", ".json", ".md"}}
        self.check(set(receipt) == actual, "source_inventory_mismatch", path,
                   {"unreceipted": sorted(actual - set(receipt)), "missing": sorted(set(receipt) - actual)})
        return checked

    def setup(self):
        self.cfg = self.read(self.code / "config.json") or {}
        manifest = self.read(self.code / "prompts.json") or {}
        prompts = manifest.get("prompts", [])
        if not isinstance(prompts, list):
            prompts = []
        self.discovery = [p for p in prompts if isinstance(p, dict) and p.get("split") == "discovery"]
        self.confirmation = [p for p in prompts if isinstance(p, dict) and p.get("split") == "confirmation"]
        cfg = self.cfg
        self.check(cfg.get("model_id") == "Qwen/Qwen3.8-27B" and cfg.get("reference_revision") == REVISION,
                   "target_identity", self.code / "config.json", "expected the pinned Qwen3.8-27B target")
        structural = (cfg.get("layers") == LAYERS and cfg.get("channels_per_layer") == CHANNELS
                      and cfg.get("screen_signed_residual_doses") == [-0.3, 0.3]
                      and cfg.get("screen_max_new_tokens") == 96
                      and cfg.get("confirmation_max_new_tokens") == 192
                      and cfg.get("workers") == 2 and cfg.get("candidate_batches_per_worker") == 2)
        self.check(structural, "campaign_configuration", self.code / "config.json", "unexpected campaign02 dimensions or protocol")
        ids = [p.get("id") for p in self.discovery + self.confirmation]
        prompt_ok = (len(self.discovery) == 8 and len(self.confirmation) == 16
                     and all(isinstance(x, str) for x in ids) and len(set(ids)) == 24
                     and all(isinstance(p.get("prompt"), str) for p in self.discovery + self.confirmation))
        self.check(prompt_ok, "prompt_manifest", self.code / "prompts.json", "requires 8 discovery and 16 disjoint confirmation prompt IDs/texts")
        self.prompt_map = {p["id"]: p for p in self.discovery + self.confirmation if isinstance(p.get("id"), str)}
        self.combo = {(p["id"], d): i * 2 + j for i, p in enumerate(self.discovery)
                      for j, d in enumerate([-0.3, 0.3]) if isinstance(p.get("id"), str)}
        self.frozen = self.read(self.results / "frozen_sources.json")
        self.source_files_checked = self.hash_map(self.frozen, self.results / "frozen_sources.json") if self.code.is_dir() else 0
        self.pilot = self.read(self.results / "pilot.json") or {}
        self.check(self.pilot.get("status") == "PASS", "pilot_status", self.results / "pilot.json", "pilot is not PASS")
        self.calibration = self.read(self.results / "calibration.json") or {}
        self.check(self.calibration.get("prompts") == cfg.get("calibration_prompts")
                   and bool(cfg.get("calibration_prompts")), "calibration_prompts", self.results / "calibration.json", "calibration prompt provenance differs")
        self.calibration_hash = self.file_receipt(self.results / "calibration.npz", self.calibration.get("file_sha256"))
        self.preflight = self.read(self.base / "preflight.json") or {}
        shards = self.preflight.get("shards", [])
        expected_shards = self.read(self.code / "model_shards.json") or {}
        checked = 0
        names = set()
        for shard in shards if isinstance(shards, list) else []:
            if not isinstance(shard, dict):
                self.issue("preflight_shard", self.base / "preflight.json", "non-object shard")
                continue
            name = shard.get("name", shard.get("file"))
            digest = shard.get("sha256", shard.get("actual_sha256"))
            expected = expected_shards.get(name)
            good = (isinstance(name, str) and name not in names and isinstance(expected, str)
                    and digest == expected and shard.get("expected_sha256", shard.get("expected", expected)) == expected
                    and shard.get("matches", shard.get("match", True)) is True)
            self.check(good, "preflight_shard", self.base / "preflight.json", name)
            checked += int(good)
            names.add(name)
        self.preflight_checked = checked
        self.check(self.preflight.get("all_18_weight_shards_match") is True and checked == 18
                   and len(shards) == 18 and names == set(expected_shards),
                   "preflight_receipt", self.base / "preflight.json", "requires 18 distinct matching shard digest receipts")
        for path in sorted(self.results.glob("rank-*/runtime.json")):
            rt = self.read(path)
            if rt is None:
                continue
            self.runtime.append({"file": self.rel(path), **rt})
            self.check(rt.get("model_id") == cfg.get("model_id") and rt.get("reference_revision") == cfg.get("reference_revision"),
                       "runtime_identity", path, "runtime does not match pinned config")
            self.check(rt.get("source_sha256") == self.frozen and bool(self.frozen),
                       "runtime_source_hashes", path, "runtime differs from frozen sources")
            stops = rt.get("configured_eos_ids")
            if isinstance(stops, list) and stops and all(integer(v) and v >= 0 for v in stops):
                self.eos[path.parent.name] = set(stops)
            else:
                self.issue("runtime_eos_ids", path, "missing/invalid configured EOS IDs")
        for rank in range(2):
            self.check(any(Path(r["file"]).parent.name == f"rank-{rank}" for r in self.runtime),
                       "missing_runtime", self.results / f"rank-{rank}" / "runtime.json", "cannot verify rank identity or EOS configuration")
        return structural and prompt_ok

    def file_receipt(self, path, expected):
        try:
            actual = sha256(path)
        except OSError as exc:
            self.issue("hash_file_missing", path, exc)
            return False
        return self.check(isinstance(expected, str) and actual == expected,
                          "file_hash_mismatch", path, {"expected": expected, "actual": actual})

    def result(self, value, cap, stops, path, scope):
        stats = self.token_stats.setdefault(scope, Counter())
        stats["results"] += 1
        if not isinstance(value, dict):
            self.issue("result_schema", path, "missing result object")
            stats["invalid"] += 1
            return False
        ids, n = value.get("token_ids"), value.get("tokens")
        valid = (isinstance(ids, list) and all(integer(t) and t >= 0 for t in ids)
                 and integer(n) and n == len(ids) and 0 < n <= cap
                 and isinstance(value.get("text"), str))
        if not valid:
            self.issue("full_token_length", path, "text/IDs/length invalid, missing or beyond configured cap")
            stats["invalid"] += 1
            return False
        stats["tokens"] += n
        stats["at_token_cap"] += n == cap
        stop = value.get("stop_reason")
        stats["reported_stop_" + str(stop)] += 1
        stats["reported_eos_at_cap"] += value.get("eos_at_cap") is True
        if not stops:
            self.issue("eos_unverifiable", path, "rank runtime lacks EOS IDs")
            stats["invalid"] += 1
            return False
        first = next((i for i, token in enumerate(ids) if token in stops), None)
        classified = "eos" if first is not None else "cap" if n == cap else "other"
        valid = (stop == classified and (first is None or first == n - 1)
                 and value.get("eos_at_cap") is (classified == "eos" and n == cap))
        if not valid:
            self.issue("eos_classification", path, "stop_reason/eos_at_cap/padding disagrees with complete token IDs")
        elif classified == "other":
            self.issue("short_without_eos", path, "generation shorter than cap without a configured EOS")
            valid = False
        stats["invalid"] += not valid
        return valid

    def generation(self, value, path):
        return self.check(isinstance(value, dict) and isinstance(value.get("prompt_token_ids"), list)
                          and bool(value["prompt_token_ids"])
                          and all(integer(t) and t >= 0 for t in value["prompt_token_ids"])
                          and number(value.get("elapsed_seconds")) and value["elapsed_seconds"] >= 0
                          and integer(value.get("hook_invocations")) and value["hook_invocations"] > 0,
                          "generation_metadata", path, "missing full prompt token IDs, timing or hook count")

    def index(self, row):
        layer, neuron = row.get("layer"), row.get("neuron")
        if not integer(layer) or not integer(neuron) or not 0 <= layer < LAYERS or not 0 <= neuron < CHANNELS:
            return None
        return layer * CHANNELS + neuron

    def intervention(self, spec, neuron, dose, path):
        ok = (isinstance(spec, dict) and spec.get("neuron") == neuron
              and spec.get("signed_dose") == dose and number(spec.get("amplitude"))
              and number(spec.get("injected_l2_norm")) and spec["injected_l2_norm"] >= 0
              and spec.get("kind") in {"neuron", "sham"})
        if ok and spec["kind"] == "sham":
            ok = spec.get("label") == "zero_column" and spec["amplitude"] == 0 and spec["injected_l2_norm"] == 0
        if ok and spec["kind"] == "neuron":
            ok = (spec["amplitude"] * dose >= 0 and number(spec.get("calibration_abs_max"))
                  and spec["calibration_abs_max"] >= 0)
            ratio = spec.get("amplitude_over_calibration_abs_max")
            if ratio is not None:
                ok = ok and number(ratio) and math.isclose(ratio, spec["amplitude"] / max(spec["calibration_abs_max"], 1e-12), rel_tol=1e-8, abs_tol=1e-10)
        return self.check(ok, "intervention_schema", path, "neuron/dose/amplitude/calibration metadata disagree")

    def signal(self, signal, path):
        ok = (isinstance(signal, dict) and all(isinstance(signal.get(k), bool) for k in
              ("needs_quality_review", "empty", "repetition_screen", "format_constraint_changed"))
              and number(signal.get("weak_rank_score")) and number(signal.get("lexical_distance"))
              and integer(signal.get("visible_words")) and isinstance(signal.get("marker_presence_deltas"), dict)
              and signal.get("semantic_coherence") == "not_assessed")
        return self.check(ok, "weak_signal_schema", path, "missing or unexpected weak-screen fields")

    def chunk(self, path, partial=False):
        scope = "partial" if partial else "committed"
        info = {"file": self.rel(path), "compressed_bytes": path.stat().st_size,
                "screen_rows": 0, "baseline_rows": 0, "included_in_committed_coverage": False}
        (self.partials if partial else self.chunks).append(info)
        errors_before = sum(self.errors.values())
        rank = path.parent.parent.name
        stops = self.eos.get(rank)
        layer_match = re.fullmatch(r"layer-L(\d+)", path.parent.name)
        layer = int(layer_match[1]) if layer_match else None
        match = re.fullmatch(r"chunk_(\d+)\.jsonl\.gz(?:\.partial.*)?", path.name)
        block = int(match[1]) if match else None
        summary = None
        if not partial:
            summary = self.read(path.with_name(path.name + ".summary.json"))
            info["summary_sha256_verified"] = self.file_receipt(path, (summary or {}).get("sha256"))
        else:
            info["sha256"] = sha256(path)
        screens, baselines, seen, neurons = [], {}, set(), set()
        try:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                for line_no, line in enumerate(stream, 1):
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError("row must be an object")
                        if row.get("kind") == "baseline":
                            info["baseline_rows"] += 1
                            bid = row.get("baseline_id")
                            if not isinstance(bid, str) or bid in baselines:
                                self.issue("duplicate_or_invalid_baseline", path, line_no)
                                continue
                            baselines[bid] = row
                            p = self.prompt_map.get(row.get("prompt_id"), {})
                            self.check(row.get("layer") == layer and p.get("split") == "discovery"
                                       and row.get("prompt") == p.get("prompt"), "baseline_prompt", path, line_no)
                            self.generation(row.get("generation"), path)
                            self.result(row.get("result"), 96, stops, path, scope + "_baseline")
                        elif row.get("kind") == "screen":
                            info["screen_rows"] += 1
                            ix = self.index(row)
                            combo = self.combo.get((row.get("prompt_id"), row.get("signed_dose")))
                            if ix is None or combo is None:
                                self.issue("screen_identity", path, line_no)
                                continue
                            key = (ix, combo)
                            neurons.add(row["neuron"])
                            if key in seen:
                                self.issue("duplicate_screen_row", path, line_no)
                            seen.add(key)
                            if not partial:
                                bit = 1 << combo
                                if self.observed[ix] & bit:
                                    self.counts["duplicate_committed_screen_rows"] += 1
                                    self.duplicate_neuron[ix] = 1
                                    self.issue("global_duplicate_screen_row", path, {"line": line_no, "layer": row["layer"], "neuron": row["neuron"]})
                                self.observed[ix] |= bit
                            self.check(row["layer"] == layer and row.get("timing") == "continuous",
                                       "screen_layer_timing", path, line_no)
                            self.intervention(row.get("intervention"), row["neuron"], row["signed_dose"], path)
                            self.signal(row.get("signals"), path)
                            self.result(row.get("result"), 96, stops, path, scope + "_screen")
                            screens.append(row)
                        else:
                            self.issue("unknown_row_kind", path, line_no)
                    except (ValueError, TypeError, KeyError, OverflowError) as exc:
                        self.issue("malformed_row", path, {"line": line_no, "error": str(exc)})
        except (OSError, EOFError, UnicodeError) as exc:
            self.issue("gzip_read_failure", path, exc, warning=partial)
            info["read_failure"] = str(exc)
        for row in screens:
            base = baselines.get(row.get("baseline_id"))
            expected_id = f'L{layer}_C{block}_{row["prompt_id"]}_baseline'
            self.check(base is not None and base.get("prompt_id") == row["prompt_id"]
                       and base.get("layer") == row["layer"] and row.get("baseline_id") == expected_id,
                       "same_batch_baseline_join", path, {"baseline_id": row.get("baseline_id"), "neuron": row["neuron"]})
        info["distinct_screen_identities"] = len(seen)
        info["distinct_neurons"] = len(neurons)
        info["distinct_baselines"] = len(baselines)
        self.counts[scope + "_screen_rows"] += info["screen_rows"]
        self.counts[scope + "_baseline_rows"] += info["baseline_rows"]
        if partial:
            info["integrity_errors"] = sum(self.errors.values()) - errors_before
            return
        expected = {(layer * CHANNELS + n, c) for n in neurons for c in range(COMBOS)} if layer is not None else set()
        self.check(bool(neurons) and seen == expected and len(screens) == len(expected),
                   "chunk_16_combinations", path, "committed chunk lacks exactly 8 prompts x 2 signs per neuron")
        self.check(len(baselines) == 8 and {b.get("prompt_id") for b in baselines.values()} == {p["id"] for p in self.discovery},
                   "chunk_baselines", path, "requires one same-batch baseline for each discovery prompt")
        if summary is not None:
            self.check(summary.get("layer") == layer and summary.get("neurons") is not None
                       and sorted(summary["neurons"]) == sorted(neurons)
                       and len(summary["neurons"]) == len(neurons)
                       and summary.get("combinations") == len(expected)
                       and summary.get("prompt_ids") == sorted(p["id"] for p in self.discovery),
                       "summary_contents", path, "summary assignment/counts/prompt IDs disagree with rows")
        info["integrity_errors"] = sum(self.errors.values()) - errors_before
        if info["integrity_errors"]:
            return
        info["included_in_committed_coverage"] = True
        for row in screens:
            ix = self.index(row)
            self.coverage[ix] |= 1 << self.combo[(row["prompt_id"], row["signed_dose"])]
            result, sig = row["result"], row["signals"]
            baseline = baselines[row["baseline_id"]]["result"]
            values = {
                "changed_text_pairs": result["text"] != baseline["text"],
                "changed_token_pairs": result["token_ids"] != baseline["token_ids"],
                "needs_quality_review_pairs": sig["needs_quality_review"], "empty_pairs": sig["empty"],
                "repetition_screen_pairs": sig["repetition_screen"],
                "format_constraint_changed_pairs": sig["format_constraint_changed"],
                "marker_delta_pairs": bool(sig["marker_presence_deltas"]),
                "weak_candidate_pairs": (not sig["needs_quality_review"] and sig["weak_rank_score"] >= self.cfg.get("candidate_minimum_weak_score", 2)
                                         and sig["visible_words"] >= 8 and sig["lexical_distance"] >= .2),
                "cap_pairs": result["stop_reason"] == "cap", "eos_pairs": result["stop_reason"] == "eos",
                "eos_at_cap_pairs": result["eos_at_cap"],
            }
            ratio = row["intervention"].get("amplitude_over_calibration_abs_max")
            values["native_overdrive_recorded_pairs"] = number(ratio)
            for name, value in values.items():
                # Duplicate artifacts can otherwise overflow the compact arrays.
                self.metrics[name][ix] = min(65535, self.metrics[name][ix] + int(value))
            if number(ratio):
                for name, value in zip(CSV_FIELDS[17:], (ratio, ratio, abs(ratio), abs(ratio))):
                    old = self.ranges[name][ix]
                    self.ranges[name][ix] = value if math.isnan(old) else (min(old, value) if name.endswith("_min") else max(old, value))

    def followups(self):
        selected_ids = set()
        for path in sorted(self.results.glob("rank-*/selection_*.json")):
            selection = self.read(path)
            if selection is None:
                continue
            ident = selection.get("candidate_id")
            self.selections.append({"file": self.rel(path), "candidate_id": ident, "selected_from": selection.get("selected_from")})
            selected_ids.add((path.parent.name, ident))
            self.check(path.name == f"selection_{ident}.json" and selection.get("no_automatic_confirmation") is True
                       and selection.get("prompt_ids") == [p["id"] for p in self.confirmation],
                       "selection_metadata", path, "frozen identity, review guard or confirmation prompt IDs differ")
            if not (path.parent / f"candidate_{ident}.json").exists():
                self.issue("selection_without_followup", path, ident, warning=True)
        for path in sorted(self.results.glob("rank-*/candidate_*.json")):
            start_errors = sum(self.errors.values())
            data = self.read(path)
            if data is None:
                continue
            ident, selected = data.get("candidate_id"), data.get("selected_from", {})
            record = {"file": self.rel(path), "candidate_id": ident, "reported_completed": data.get("completed"),
                      "status": data.get("status"), "trial_count": 0, "six_arm_trials": 0,
                      "changed_selected_text_pairs": 0, "behavioral_success": "not_assessed"}
            self.candidates.append(record)
            selection_path = path.parent / f"selection_{ident}.json"
            frozen = self.read(selection_path)
            self.check(frozen is not None and all(data.get(k) == v for k, v in frozen.items()),
                       "candidate_frozen_selection", path, "candidate metadata differs from frozen selection")
            ix = self.index(selected) if isinstance(selected, dict) else None
            dose = selected.get("signed_dose") if isinstance(selected, dict) else None
            good_identity = ix is not None and number(dose) and dose in [-.3, .3]
            self.check(good_identity and path.name == f"candidate_{ident}.json" and data.get("primary_signed_dose") == dose
                       and data.get("no_automatic_confirmation") is True and data.get("status") == "needs_blinded_behavior_review",
                       "candidate_metadata", path, "invalid candidate identity/dose/review metadata")
            if good_identity:
                self.check(self.coverage[ix] == FULL and not self.duplicate_neuron[ix],
                           "candidate_source_coverage", path, "selected neuron is not in verified committed coverage")
                expected_ident = f'L{selected["layer"]}_N{selected["neuron"]}_{"plus" if dose > 0 else "minus"}'
                self.check(ident == expected_ident, "candidate_id", path, "identity differs from selected layer/neuron/sign")
                self.selected_source(selected, path)
            trials = data.get("trials", [])
            stops = self.eos.get(path.parent.name)
            seen = set()
            for trial in trials if isinstance(trials, list) else []:
                record["trial_count"] += 1
                if not isinstance(trial, dict):
                    self.issue("trial_schema", path, "expected object")
                    continue
                pid = trial.get("prompt_id")
                prompt = self.prompt_map.get(pid, {})
                self.check(pid not in seen and prompt.get("split") == "confirmation" and trial.get("prompt") == prompt.get("prompt")
                           and trial.get("timing") == "continuous", "trial_prompt_timing", path, pid)
                seen.add(pid)
                specs, outputs = trial.get("specs", []), trial.get("outputs", [])
                six = isinstance(specs, list) and isinstance(outputs, list) and len(specs) == len(outputs) == 6 and all(isinstance(s, dict) for s in specs)
                self.check(six, "six_arm_trial", path, pid)
                self.generation(trial.get("generation"), path)
                if not six:
                    continue
                record["six_arm_trials"] += 1
                for output in outputs:
                    self.result(output, 192, stops, path, "candidate_trial")
                if isinstance(outputs[0], dict) and isinstance(outputs[1], dict):
                    record["changed_selected_text_pairs"] += outputs[1].get("text") != outputs[0].get("text")
                self.check(specs[0].get("kind") == "sham" and specs[0].get("label") == "baseline",
                           "trial_baseline_arm", path, pid)
                if good_identity:
                    neuron = selected["neuron"]
                    self.intervention(specs[1], neuron, dose, path)
                    self.intervention(specs[2], neuron, -dose, path)
                    other = specs[3].get("neuron")
                    self.intervention(specs[3], other, dose, path)
                    self.check(integer(other) and 0 <= other < CHANNELS and other != neuron,
                               "other_channel_arm", path, pid)
                self.check([s.get("label") for s in specs[1:]] == ["selected", "reverse_sign", "other_channel", "random_residual_0", "random_residual_1"],
                           "trial_arm_labels", path, pid)
                norm = specs[1].get("injected_l2_norm")
                self.check(number(norm) and all(number(s.get("injected_l2_norm")) and math.isclose(s["injected_l2_norm"], norm, rel_tol=1e-6, abs_tol=1e-8) for s in specs[2:]),
                           "nominal_norm_matching", path, pid)
                self.check(all(s.get("kind") == "random_residual" and integer(s.get("direction_seed")) for s in specs[4:])
                           and specs[4].get("direction_seed") != specs[5].get("direction_seed"),
                           "random_residual_arms", path, pid)
                self.signal(trial.get("selected_signals"), path)
            diagnostics = data.get("discovery_timing_dose_diagnostics", [])
            timings = []
            for diag in diagnostics if isinstance(diagnostics, list) else []:
                if not isinstance(diag, dict):
                    self.issue("diagnostic_schema", path, "expected object")
                    continue
                timing = diag.get("timing")
                timings.append(timing)
                self.check(diag.get("prompt_id") == selected.get("prompt_id") and self.prompt_map.get(diag.get("prompt_id"), {}).get("split") == "discovery",
                           "diagnostic_prompt", path, timing)
                specs, outputs = diag.get("specs", []), diag.get("outputs", [])
                doses = [math.copysign(.1, dose), math.copysign(.6, dose)] if good_identity and timing == "continuous" else [dose]
                valid = (isinstance(specs, list) and isinstance(outputs, list) and len(specs) == len(outputs) == len(doses) + 1
                         and all(isinstance(s, dict) for s in specs) and specs[0].get("kind") == "sham")
                self.check(valid, "diagnostic_arms", path, timing)
                if valid and good_identity:
                    for spec, d in zip(specs[1:], doses):
                        self.intervention(spec, selected["neuron"], d, path)
                self.generation(diag.get("generation"), path)
                for output in outputs if isinstance(outputs, list) else []:
                    self.result(output, 192, stops, path, "candidate_diagnostic")
            complete = (len(seen) == 16 and seen == {p["id"] for p in self.confirmation}
                        and record["trial_count"] == record["six_arm_trials"] == 16
                        and timings == ["continuous", "prefill_only", "decode_only", "pulse_8"])
            self.check(data.get("completed") is complete, "candidate_completion_metadata", path, "completed flag disagrees with 16 trials and four diagnostics")
            record["integrity_errors"] = sum(self.errors.values()) - start_errors
            record["verified_complete_followup"] = complete and not record["integrity_errors"]
        self.check(len(self.candidates) <= 4, "candidate_budget", self.results, "more than four candidate artifacts")
        for rank in range(2):
            count = sum(Path(c["file"]).parent.name == f"rank-{rank}" for c in self.candidates)
            self.check(count <= 2, "rank_candidate_budget", self.results / f"rank-{rank}", count)
        if len(self.candidates) < 4:
            self.issue("followup_count_below_four", self.results, "stopped campaign may have fewer than four followups", warning=True)

    def selected_source(self, selected, path):
        """Join the frozen nomination to its exact archived screen row."""
        source = selected.get("source_file")
        if not isinstance(source, str):
            self.issue("candidate_source_file", path, "missing relative screen source_file")
            return
        target = (self.results / source).resolve()
        valid_path = (target.is_relative_to(self.results.resolve()) and target.name.endswith(".jsonl.gz")
                      and target.parent.parent == path.parent.resolve())
        if not self.check(valid_path, "candidate_source_file", path, source):
            return
        matched = []
        try:
            with gzip.open(target, "rt", encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row.get("kind") == "screen" and all(row.get(k) == selected.get(k) for k in
                            ("layer", "neuron", "prompt_id", "signed_dose")):
                        matched.append(row)
        except (OSError, EOFError, UnicodeError, ValueError, TypeError, AttributeError) as exc:
            self.issue("candidate_source_read", path, exc)
            return
        self.check(len(matched) == 1 and matched[0].get("signals", {}).get("weak_rank_score") == selected.get("weak_rank_score"),
                   "candidate_source_row", path, "frozen nomination must match exactly one committed screen row and its score")

    def metadata(self):
        for path in sorted(self.results.glob("rank-*/progress.json")):
            data = self.read(path)
            if data is not None:
                self.progress.append({"file": self.rel(path), "reported_only_not_coverage": True, **data})
        for path in sorted(self.results.rglob("*FAILED*.json")):
            self.failures.append({"file": self.rel(path), "record": self.read(path)})
        for path in sorted(self.results.glob("rank-*/assignment.json")):
            data = self.read(path)
            if data is not None:
                self.check(data.get("discovery_prompt_ids") == [p["id"] for p in self.discovery],
                           "assignment_prompts", path, "assignment differs from frozen discovery prompts")
        summaries = sorted(self.results.glob("rank-*/layer-L*/chunk_*.jsonl.gz.summary.json"))
        for path in summaries:
            if not path.with_name(path.name[:-len(".summary.json")]).is_file():
                self.issue("orphan_summary", path, "committed gzip is absent")
        self.counts["summary_files"] = len(summaries)

    def finish(self, entries):
        fully = 0
        observed_unique = 0
        committed_unique = 0
        rejected_duplicates = 0
        totals = Counter()
        per_layer = []
        writer = None
        stream = None
        if entries is not None:
            entries.parent.mkdir(parents=True, exist_ok=True)
            stream = entries.open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
        try:
            for layer in range(LAYERS):
                layer_full, layer_pairs = 0, 0
                for neuron in range(CHANNELS):
                    ix = layer * CHANNELS + neuron
                    observed_unique += self.observed[ix].bit_count()
                    if self.duplicate_neuron[ix]:
                        rejected_duplicates += bool(self.coverage[ix])
                        continue
                    pairs = self.coverage[ix].bit_count()
                    committed_unique += pairs
                    layer_pairs += pairs
                    if self.coverage[ix] != FULL:
                        continue
                    fully += 1
                    layer_full += 1
                    row = {"layer": layer, "neuron": neuron, "committed_combinations": COMBOS,
                           "expected_combinations": COMBOS, "coverage_fraction": 1.0}
                    for name, values in self.metrics.items():
                        row[name] = values[ix]
                        totals[name] += values[ix]
                    for name, values in self.ranges.items():
                        row[name] = "" if math.isnan(values[ix]) else values[ix]
                    if writer:
                        writer.writerow(row)
                per_layer.append({"layer": layer, "fully_screened_neurons": layer_full, "verified_committed_combinations": layer_pairs})
        finally:
            if stream:
                stream.close()
        stopped = ((self.results / "STOP").exists() or any(p.get("phase") == "wall_limit_or_stop" for p in self.progress))
        full_census = fully == TOTAL and not self.errors
        verified_candidates = sum(c["verified_complete_followup"] for c in self.candidates)
        return {
            "schema_version": 1, "audit_utc": datetime.now(timezone.utc).isoformat(),
            "archive_root": str(self.root), "audit_passed": not self.errors,
            "campaign_state": "full_census" if full_census else "failed_partial_census" if self.failures else "stopped_partial_census" if stopped else "partial_census",
            "full_census": full_census, "stop_marker_or_report": stopped,
            "interpretation": {
                "text_changes": "Exact decoded-text inequality to same-batch baseline; not behavioral success.",
                "weak_flags": "Stored quality, lexical, format and marker screens; not semantic labels or confirmed discoveries.",
                "token_caps": "Cap-ended continuations are censored; full saved token lists do not imply a finished answer.",
                "coverage": "Only intact summary-hashed committed chunks with complete 16-way rows and same-batch joins; duplicate neurons excluded. Partials never contribute.",
                "native_overdrive": "Signed/absolute amplitude divided by observed calibration absolute maximum only; not a natural-population range.",
                "provenance_limit": "Archive receipts checked against local frozen sources. Checkpoint weights, tokenizer execution and decoded text are not independently reproduced.",
                "norm_matching": "Recorded nominal L2 norms only; realized BF16 perturbations were not measured.",
            },
            "target": {k: self.cfg.get(k) for k in ("campaign_id", "model_id", "reference_revision", "metadata_sha256", "layers", "channels_per_layer", "screen_signed_residual_doses", "screen_max_new_tokens", "confirmation_max_new_tokens")},
            "provenance": {"source_files_hashed": self.source_files_checked,
                           "calibration_npz_hash_verified": self.calibration_hash,
                           "preflight_shard_digest_receipts_matched": self.preflight_checked,
                           "weight_files_rehashed": False, "runtime": self.runtime,
                           "pilot_status": self.pilot.get("status")},
            "coverage": {"planned_neurons": TOTAL, "planned_combinations": TOTAL * COMBOS,
                         "fully_screened_neurons": fully, "fraction_neurons_fully_screened": fully / TOTAL,
                         "verified_committed_combinations": committed_unique,
                         "observed_unique_committed_identities_including_invalid_chunks": observed_unique,
                         "neurons_excluded_for_duplicate_committed_identities": rejected_duplicates,
                         "committed_files": len(self.chunks), "verified_committed_files": sum(c["included_in_committed_coverage"] for c in self.chunks),
                         "partial_files_excluded": len(self.partials), "per_layer": per_layer, **dict(self.counts)},
            "paired_metrics_fully_screened_neurons_only": dict(totals),
            "token_validation_all_observed_rows_by_scope": self.token_stats,
            "followups": {"expected_candidate_budget": 4, "expected_trials_per_candidate": 16, "expected_arms_per_trial": 6,
                          "candidate_files": len(self.candidates), "verified_complete_followups": verified_candidates,
                          "four_candidate_16_trial_six_arm_metadata_complete": len(self.candidates) == verified_candidates == 4,
                          "confirmed_behavioral_discoveries": None, "behavioral_success_assessed": False,
                          "selections": self.selections, "candidates": self.candidates},
            "progress_reports_unverified": self.progress, "failure_receipts": self.failures,
            "committed_chunks": self.chunks, "partial_artifacts_retained_separately": self.partials,
            "errors": dict(self.errors), "warnings": dict(self.warnings),
            "issue_examples_first_200": self.examples,
            "entries_csv": str(entries.resolve()) if entries is not None else None,
        }

    def run(self, entries=None):
        ready = self.setup()
        self.metadata()
        if ready:
            for path in sorted(self.results.glob("rank-*/layer-L*/chunk_*.jsonl.gz")):
                try:
                    self.chunk(path)
                except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
                    self.issue("chunk_audit_failure", path, exc)
            for path in sorted(self.results.glob("rank-*/layer-L*/chunk_*.partial*")):
                if path.is_file():
                    try:
                        self.chunk(path, partial=True)
                    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
                        self.issue("partial_audit_failure", path, exc)
            self.followups()
        return self.finish(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entries", type=Path, help="CSV containing verified fully screened neurons only")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error("--root must be an existing local campaign archive directory")
    # Never overwrite source evidence, including when paths are passed relatively.
    base = args.root.resolve().parent if args.root.name == "results" else args.root.resolve()
    targets = [args.output] + ([args.entries] if args.entries is not None else [])
    if len({p.resolve() for p in targets}) != len(targets):
        parser.error("--output and --entries must be different files")
    for target in targets:
        resolved = target.resolve()
        if resolved.is_relative_to(base / "code") or resolved.is_relative_to(base / "results") or resolved == base / "preflight.json":
            parser.error("output paths must not overwrite campaign evidence in code/, results/ or preflight.json")
    try:
        audit = Audit(args.root)
        report = audit.run(args.entries)
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        report = {"schema_version": 1, "audit_passed": False, "full_census": False,
                  "fatal_error": f"{type(exc).__name__}: {exc}", "archive_root": str(args.root.resolve())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"audit_passed": report["audit_passed"], "output": str(args.output),
                      "coverage": report.get("coverage", {}).get("fully_screened_neurons"),
                      "error_counts": report.get("errors", report.get("fatal_error"))}))
    return 0 if report["audit_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
