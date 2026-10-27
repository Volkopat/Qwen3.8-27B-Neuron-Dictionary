"""Verify the compact public campaign package without downloading full outputs."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    root = Path(root)
    folder = root / 'campaigns/2026-10-27'
    evidence = folder / 'evidence'
    errors = []
    checks = 0

    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(message)

    def read(path):
        return json.loads(path.read_text(encoding='utf-8'))

    stop = read(folder / 'stop.json')
    audit = read(folder / 'verification.json')
    receipt = read(folder / 'archive.json')
    manifest = read(evidence / 'ARCHIVE_MANIFEST.json')['files']
    frozen = read(evidence / 'frozen_sources.json')
    config = read(folder / 'source/config.json')
    check(audit['audit_passed'] is True and audit['full_census'] is False, 'campaign must be an audited partial census')
    check(not audit['errors'] and not audit['failure_receipts'], 'campaign archive audit reports errors or worker failures')
    check(stop['model_id'] == config['model_id'] == 'Qwen/Qwen3.8-27B', 'campaign model differs')
    check(stop['revision'] == config['reference_revision'] == '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0', 'campaign revision differs')
    check(stop['stopped_by_user'] is True and stop['process_returncodes'] == [0, 0], 'campaign stop receipt differs')
    check(stop['confirmed_discoveries'] == 0, 'unsupported confirmed-discovery claim')
    check(stop['workers'] == config['workers'] == 2, 'campaign GPU worker count differs')
    check(abs(stop['wall_seconds'] / 3600 - stop['wall_hours']) < 1e-9, 'campaign wall-time arithmetic differs')
    check(abs(stop['approx_gpu_hours'] - 2 * stop['wall_hours']) < 1e-9, 'campaign GPU-time arithmetic differs')
    check(receipt['bytes'] == 76476621 and receipt['file_count'] == len(manifest) + 1 == 997, 'archive size or member count differs')
    check(receipt['sha256'] == 'eee2181ffbf8374392f3d7847fe213fbd1a7cd412aeace7e7826c318509dc143', 'campaign archive digest differs')
    check(sum(x['bytes'] for x in manifest.values()) == receipt['payload_bytes'], 'archive payload byte count differs')
    hashed = 0
    for name, digest in frozen.items():
        path = folder / 'source' / name
        check(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest, f'frozen campaign source changed: {name}')
        hashed += 1
    for name, info in manifest.items():
        if name.startswith('code/'):
            path = folder / 'source' / name[5:]
        elif name.startswith('results/'):
            path = evidence / name[8:]
        else:
            path = evidence / name
        if path.is_file():
            check(path.stat().st_size == info['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256'], f'copied campaign evidence changed: {name}')
            hashed += 1
    rows = []
    with (root / 'data/campaign_neurons.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    keys = set()
    layer_counts = Counter()
    totals = Counter()
    metric_names = list(audit['paired_metrics_fully_screened_neurons_only'])
    for row in rows:
        key = (int(row['layer']), int(row['neuron']))
        check(key not in keys and 0 <= key[0] < 64 and 0 <= key[1] < 17408, 'campaign index duplicate or invalid neuron key')
        keys.add(key)
        layer_counts[key[0]] += 1
        check(int(row['committed_combinations']) == int(row['expected_combinations']) == 16 and float(row['coverage_fraction']) == 1, f'incomplete campaign index row: {key}')
        for name in metric_names:
            value = int(row[name])
            check(0 <= value <= 16, f'invalid campaign descriptive count: {key}/{name}')
            totals[name] += value
        check(int(row['cap_pairs']) + int(row['eos_pairs']) == 16, f'campaign stop counts do not partition row: {key}')
        for name in ['signed_amplitude_over_calibration_abs_max_min', 'signed_amplitude_over_calibration_abs_max_max',
                     'absolute_amplitude_over_calibration_abs_max_min', 'absolute_amplitude_over_calibration_abs_max_max']:
            check(math.isfinite(float(row[name])), f'nonfinite campaign amplitude ratio: {key}')
    coverage = audit['coverage']
    check(len(rows) == len(keys) == coverage['fully_screened_neurons'] == stop['fully_screened_neurons'] == 30592, 'campaign fully screened count differs')
    check(len(rows) * 16 == coverage['verified_committed_combinations'] == stop['committed_screen_combinations'] == 489472, 'campaign committed combination count differs')
    check(dict(layer_counts) == {x['layer']: x['fully_screened_neurons'] for x in coverage['per_layer']}, 'campaign layer counts differ')
    check(dict(totals) == audit['paired_metrics_fully_screened_neurons_only'], 'campaign descriptive index totals differ')
    token_counts = audit['token_validation_all_observed_rows_by_scope']
    check(token_counts['partial_screen']['results'] == stop['partial_screen_combinations'] == 384, 'campaign partial count differs')
    check(token_counts['committed_screen']['tokens'] + token_counts['partial_screen']['tokens'] == stop['generated_screen_tokens'], 'campaign generated-token accounting differs')
    followups = audit['followups']
    check(followups['verified_complete_followups'] == stop['candidate_followups'] == 4, 'campaign followup count differs')
    check(followups['four_candidate_16_trial_six_arm_metadata_complete'] is True and followups['behavioral_success_assessed'] is False, 'campaign followup interpretation differs')
    examples = read(root / 'data/campaign_examples.json')
    check(examples['source_archive_sha256'] == receipt['sha256'] and len(examples['examples']) == 5, 'campaign example provenance differs')
    checked_examples = 0
    for example in examples['examples']:
        path, _, pointer = example['source'].partition('#')
        if path.endswith('.gz'):
            check(path in manifest, 'campaign weak-score example source missing from archive manifest')
            case = example['evidence']
            check(case['baseline']['baseline_id'] == case['intervention']['baseline_id'], 'campaign weak-score example baseline join differs')
            check(case['intervention']['signals']['weak_rank_score'] == 5, 'campaign weak-score example changed')
            continue
        source = read(evidence / path.removeprefix('results/'))
        for part in pointer.lstrip('/').split('/') if pointer else []:
            source = source[int(part)] if isinstance(source, list) else source[part]
        check(source == example['evidence'], f'campaign selected example changed: {example["id"]}')
        checked_examples += 1
    return {'status': 'FAIL' if errors else 'PASS', 'checks': checks, 'errors': errors,
            'scope': 'compact index, copied evidence, source fingerprints and audit receipts; full raw rows require audit_campaign.py',
            'fully_screened_neurons': len(rows), 'committed_combinations': len(rows) * 16,
            'partial_combinations': stop['partial_screen_combinations'], 'confirmed_discoveries': 0,
            'hashed_source_or_evidence_checks': hashed, 'exact_selected_example_joins': checked_examples}


if __name__ == '__main__':
    result = verify()
    print(json.dumps(result, indent=2))
    raise SystemExit(result['status'] != 'PASS')
