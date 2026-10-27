"""Check archived behavioral candidate records; this does not run the model."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    data = json.loads((root / 'discoveries/candidates.json').read_text(encoding='utf-8'))
    provenance = json.loads((root / 'evidence/behavioral_search/provenance.json').read_text(encoding='utf-8'))
    errors = []
    checked = 0
    if data['model_id'] != 'Qwen/Qwen3.8-27B' or data['confirmed_discoveries'] != 0:
        errors.append('incorrect model or unsupported confirmed-discovery count')
    if len(data['candidates']) != 2:
        errors.append('expected two archived candidates')
    for name, digest in provenance['artifact_files'].items():
        path = root / 'evidence/behavioral_search' / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            errors.append(f'changed source artifact: {name}')
    for candidate in data['candidates']:
        if candidate['status'] != 'unconfirmed_candidate' or candidate['confirmation_trials'] != 0:
            errors.append(f"unsupported confirmation claim: {candidate['id']}")
        for item in candidate['records']:
            path = (root / item['source_file']).resolve()
            if not path.is_relative_to(root.resolve()):
                errors.append('source path outside release')
                continue
            if path.suffix == '.jsonl':
                actual = json.loads(path.read_text(encoding='utf-8').splitlines()[item['line_1based'] - 1])
            else:
                actual = json.loads(path.read_text(encoding='utf-8'))['rows'][item['row_index_0based']]
            checked += 1
            if actual != item['record']:
                errors.append(f"source row mismatch: {candidate['id']}")
    return {'status': 'FAIL' if errors else 'PASS', 'candidate_count': len(data['candidates']),
            'confirmed_discoveries': data['confirmed_discoveries'], 'source_records_checked': checked,
            'scope': 'archive integrity only; no behavioral confirmation or fresh replay', 'errors': errors}


if __name__ == '__main__':
    result = verify()
    print(json.dumps(result, indent=2))
    raise SystemExit(result['status'] != 'PASS')
