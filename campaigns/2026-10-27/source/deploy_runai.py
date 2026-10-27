"""Transfer this small source package to the authorized PVC; no model files."""
import argparse
import base64
import hashlib
import io
import json
import subprocess
import time
import zipfile
from pathlib import Path

CODE = Path(__file__).resolve().parent


def execute(stage, project, code):
    boot = "import base64;exec(base64.b64decode('" + base64.b64encode(code.encode()).decode() + "'))"
    for attempt in range(3):
        result = subprocess.run(['runai', 'training', 'standard', 'exec', stage, '-p', project,
                                  '--', 'python3', '-u', '-c', boot], text=True, capture_output=True)
        if result.returncode == 0:
            break
        if 'EOF' not in result.stderr or attempt == 2:
            raise RuntimeError(result.stdout + result.stderr)
        time.sleep(1)
    if result.stdout.strip():
        print(result.stdout.strip())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', default='behavior-atlas-stage2-1027')
    p.add_argument('--project', default='dpatil2')
    p.add_argument('--remote', default='/data/models/experiments/neuron/behavior_campaigns/atlas-20261027-02')
    a = p.parse_args()
    files = sorted(p for p in CODE.iterdir() if p.is_file() and p.suffix in {'.py', '.md', '.json'})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for f in files:
            entry = zipfile.ZipInfo(f.name, date_time=(2026, 10, 27, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, f.read_bytes())
    payload = base64.b64encode(buf.getvalue()).decode()
    package_hash = hashlib.sha256(buf.getvalue()).hexdigest()
    (CODE / 'source_upload_receipt.log').write_text(json.dumps({'sha256': package_hash, 'files': [p.name for p in files]}) + '\n')
    parts = [payload[i:i + 5000] for i in range(0, len(payload), 5000)]
    for i, part in enumerate(parts):
        code = f"from pathlib import Path;p=Path({a.remote!r})/'upload-parts';p.mkdir(exist_ok=True);(p/{str(i)!r}).write_text({part!r});print('uploaded part {i}')"
        execute(a.stage, a.project, code)
    unpack = f"""from pathlib import Path
import base64,hashlib,io,json,zipfile
root=Path({a.remote!r});parts=root/'upload-parts'
blob=base64.b64decode(''.join((parts/str(i)).read_text() for i in range({len(parts)})))
assert hashlib.sha256(blob).hexdigest()=={package_hash!r}
dest=root/'code';dest.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(blob)) as archive:
 for name in archive.namelist():
  assert Path(name).name==name
  (dest/name).write_bytes(archive.read(name))
result={{'package_sha256':{package_hash!r},'files':{{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.is_file()}},'bytes':len(blob)}}
(root/'source_upload.json').write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps({{'event':'source_upload_complete','files':len(result['files']),'sha256':result['package_sha256']}}),flush=True)
"""
    execute(a.stage, a.project, unpack)
    print(json.dumps({'package_sha256': package_hash, 'files': len(files), 'parts': len(parts), 'remote': a.remote}))


if __name__ == '__main__':
    main()
