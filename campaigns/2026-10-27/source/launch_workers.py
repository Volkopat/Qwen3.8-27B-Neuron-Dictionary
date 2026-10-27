"""Launch the configured single-GPU replicas inside one bounded Run:ai job."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((CODE / 'config.json').read_text())
    args.root.mkdir(parents=True, exist_ok=True)
    frozen = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(CODE.iterdir())
              if p.is_file() and p.suffix in {'.py', '.json', '.md'}}
    receipt = args.root / 'frozen_sources.json'
    if receipt.exists() and json.loads(receipt.read_text()) != frozen:
        raise RuntimeError('source/config change on resume: use a new campaign directory')
    receipt.write_text(json.dumps(frozen, indent=2) + '\n')
    preflight = json.loads((CODE.parent / 'preflight.json').read_text())
    if not preflight.get('all_18_weight_shards_match'):
        raise RuntimeError('missing successful current weight-shard preflight')
    for shard in preflight['shards']:
        stat = (Path(config['model_path']) / shard['name']).stat()
        if stat.st_size != shard['bytes'] or stat.st_mtime_ns != shard['mtime_ns']:
            raise RuntimeError(f"checkpoint shard changed after preflight: {shard['name']}")
    if not (CODE / 'MATH_AND_PROTOCOL.md').is_file():
        raise RuntimeError('math/source protocol receipt is missing')
    import torch
    if torch.cuda.device_count() != config['workers']:
        raise RuntimeError(f"expected exactly {config['workers']} allocated GPU devices")
    devices = [torch.cuda.get_device_name(i) for i in range(config['workers'])]
    if any('B200' not in name for name in devices):
        raise RuntimeError(f'expected B200 devices, got {devices}')
    started = time.time()
    (args.root / 'STARTED_AT.json').write_text(json.dumps({'started_unix': started, 'wall_seconds': config['wall_seconds'], 'devices': devices}) + '\n')
    children = []
    streams = []
    original_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    print(json.dumps({'event': 'launching_workers', 'devices': devices, 'wall_seconds': config['wall_seconds']}), flush=True)
    try:
        for rank in range(config['workers']):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = original_visible[rank] if len(original_visible) == config['workers'] else str(rank)
            env['PYTHONUNBUFFERED'] = '1'
            env['OMP_NUM_THREADS'] = '4'
            stream = (args.root / f'worker-{rank}.log').open('a', buffering=1)
            streams.append(stream)
            child = subprocess.Popen([sys.executable, '-u', str(CODE / 'run_campaign.py'), '--root', str(args.root),
                                      '--rank', str(rank), '--workers', str(config['workers'])], env=env, stdout=stream, stderr=subprocess.STDOUT)
            children.append(child)
        last_heartbeat = 0
        while any(c.poll() is None for c in children):
            failed = [i for i, c in enumerate(children) if c.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f'worker failures: {failed}; inspect worker logs')
            if time.time() - started > config['wall_seconds'] + 300:
                raise RuntimeError('hard wall limit reached')
            if time.time() - last_heartbeat > 60:
                progress = []
                for path in sorted(args.root.glob('rank-*/progress.json')):
                    try:
                        item = json.loads(path.read_text())
                        progress.append({k: item.get(k) for k in ['rank', 'phase', 'fully_screened_neurons', 'screen_completed_combinations']})
                    except (ValueError, OSError):
                        pass
                print(json.dumps({'event': 'heartbeat', 'elapsed_seconds': time.time() - started, 'progress': progress}), flush=True)
                last_heartbeat = time.time()
            time.sleep(5)
        if any(c.returncode for c in children):
            raise RuntimeError('one or more workers failed')
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        until = time.monotonic() + 25
        for child in children:
            remaining = max(.1, until - time.monotonic())
            try:
                child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                child.kill()
        for stream in streams:
            stream.close()
        if (CODE / 'report_campaign.py').exists():
            subprocess.run([sys.executable, str(CODE / 'report_campaign.py'), '--root', str(args.root)], check=False)
    print(json.dumps({'event': 'workers_finished', 'returncodes': [c.returncode for c in children]}), flush=True)


if __name__ == '__main__':
    main()
