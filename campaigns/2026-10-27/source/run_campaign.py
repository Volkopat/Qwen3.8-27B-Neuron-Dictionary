"""Resumable, full-continuation census of Qwen3.8-27B MLP interventions.

Run only on the authorized remote GPUs. Discovery scores are weak screens;
neither a keyword match nor this program can certify a behavioral discovery.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from behavior_scoring import score_output
from interventions import Intervention, trim_generation

CODE = Path(__file__).resolve().parent
CFG = json.loads((CODE / 'config.json').read_text())
MANIFEST = json.loads((CODE / 'prompts.json').read_text(encoding='utf-8'))
DISCOVERY = [x for x in MANIFEST['prompts'] if x['split'] == 'discovery']
CONFIRMATION = [x for x in MANIFEST['prompts'] if x['split'] == 'confirmation']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def emit(event, **kw):
    print(json.dumps({'event': event, **kw}), flush=True)


def frozen_files():
    return {p.name: sha(p) for p in sorted(CODE.iterdir()) if p.is_file() and p.suffix in {'.py', '.json', '.md'}}


class Runner:
    def __init__(self, root, rank, workers, deadline):
        self.root, self.rank, self.workers, self.deadline = root, rank, workers, deadline
        self.out = root / f'rank-{rank}'
        self.out.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.screen_seconds = 0.0
        self.screen_tokens = 0
        self.completed_neurons = 0
        self.completed_combinations = 0
        self.partial_combinations = 0
        self.batches = 0
        self.candidates_run = 0
        self.candidate_ids = set()
        self.top_candidates = []
        self.columns = {}
        self.stop_ids = set()

    def expired(self):
        return time.time() >= self.deadline or (self.root / 'STOP').exists()

    def load(self):
        model_path = Path(CFG['model_path'])
        for name, expected in CFG['metadata_sha256'].items():
            if sha(model_path / name) != expected:
                raise RuntimeError(f'checkpoint fingerprint mismatch: {name}')
        config = json.loads((model_path / 'config.json').read_text())['text_config']
        if (config['num_hidden_layers'], config['hidden_size'], config['intermediate_size']) != (64, 5120, 17408):
            raise RuntimeError('checkpoint architecture differs from frozen target')
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(model_path), dtype=torch.bfloat16, local_files_only=True).eval().to('cuda:0')
        modules = [(n, m) for n, m in self.model.named_modules() if isinstance(m, torch.nn.ModuleList)
                   and len(m) and hasattr(m[0], 'mlp') and hasattr(m[0].mlp, 'down_proj')]
        self.block_path, self.blocks = max(modules, key=lambda pair: len(pair[1]))
        if len(self.blocks) != 64 or any(b.mlp.down_proj.weight.shape != (5120, 17408) for b in self.blocks):
            raise RuntimeError('actual intervenable module inventory differs')
        for value in [self.tokenizer.eos_token_id, self.model.generation_config.eos_token_id]:
            if value is not None:
                self.stop_ids.update(value if isinstance(value, (list, tuple)) else [value])
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = min(self.stop_ids)
        torch.manual_seed(CFG['seed'])
        self.runtime = {'model_id': CFG['model_id'], 'reference_revision': CFG['reference_revision'],
                        'torch': torch.__version__, 'transformers': transformers.__version__,
                        'python': platform.python_version(), 'cuda': torch.version.cuda,
                        'gpu': torch.cuda.get_device_name(0), 'precision': 'bfloat16',
                        'block_path': self.block_path, 'configured_eos_ids': sorted(self.stop_ids),
                        'weight_shard_verification': 'shared preflight SHA256 receipt',
                        'source_sha256': frozen_files()}
        atomic_json(self.out / 'runtime.json', self.runtime)
        emit('loaded', rank=self.rank, gpu=self.runtime['gpu'])

    def encode(self, prompt, rows=1, thinking=False):
        text = self.tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=thinking)
        ids = self.tokenizer([text], return_tensors='pt').to('cuda:0')
        return {k: v.expand(rows, -1).contiguous() for k, v in ids.items()}

    def generate(self, prompt, layer, specs, cap, timing='continuous', force_steps=False):
        enc = self.encode(prompt, len(specs))
        before = time.monotonic()
        with Intervention(self.blocks[layer].mlp.down_proj, specs, timing) as intervention:
            with torch.inference_mode():
                generated = self.model.generate(**enc, max_new_tokens=cap, do_sample=False, use_cache=True,
                         min_new_tokens=cap if force_steps else 0,
                         pad_token_id=self.pad_id, eos_token_id=sorted(self.stop_ids))
        if force_steps and generated.shape[1] - enc['input_ids'].shape[1] != cap:
            raise RuntimeError('memory-only full-window pilot ended before its required decode length')
        rows = generated[:, enc['input_ids'].shape[1]:].tolist()
        results = []
        for ids in rows:
            ids, stop = trim_generation(ids, self.stop_ids, cap)
            results.append({'text': self.tokenizer.decode(ids, skip_special_tokens=True), 'token_ids': ids,
                            'tokens': len(ids), 'stop_reason': stop,
                            'eos_at_cap': bool(stop == 'eos' and len(ids) == cap)})
        return results, {'elapsed_seconds': time.monotonic() - before,
                         'hook_invocations': intervention.calls,
                         'prompt_token_ids': enc['input_ids'][0].tolist()}

    def calibrate(self):
        accum = [{'norms': [], 'maximum': None} for _ in self.blocks]
        handles = []
        for layer, block in enumerate(self.blocks):
            def before(_m, inputs, layer=layer):
                x = inputs[0][0, 1:].detach().float()
                values = x.abs().amax(dim=0).cpu().numpy()
                old = accum[layer]['maximum']
                accum[layer]['maximum'] = values if old is None else np.maximum(old, values)

            def after(_m, _inputs, output, layer=layer):
                x = output[0] if isinstance(output, tuple) else output
                values = torch.linalg.vector_norm(x[0, 1:].detach().float(), dim=-1).cpu().tolist()
                accum[layer]['norms'].extend(values)
            handles.extend([block.mlp.down_proj.register_forward_pre_hook(before), block.register_forward_hook(after)])
        try:
            for prompt in CFG['calibration_prompts']:
                with torch.inference_mode():
                    self.model(**self.encode(prompt), use_cache=False, logits_to_keep=1)
        finally:
            for handle in handles:
                handle.remove()
        arrays = {}
        medians = []
        for layer, values in enumerate(accum):
            medians.append(float(np.median(values['norms'])))
            arrays[f'L{layer}_abs_max'] = values['maximum']
            arrays[f'L{layer}_down_norm'] = torch.linalg.vector_norm(
                self.blocks[layer].mlp.down_proj.weight.detach().float(), dim=0).cpu().numpy()
        arrays['residual_medians'] = np.array(medians)
        np.savez_compressed(self.root / 'calibration.npz', **arrays)
        atomic_json(self.root / 'calibration.json', {'prompts': CFG['calibration_prompts'],
                    'residual_norm_definition': 'median block-output L2 norm over real non-first prompt positions',
                    'activation_scale_definition': 'observed absolute maximum over calibration prompt positions only',
                    'calibration_tokens_by_layer': [len(x['norms']) for x in accum], 'residual_medians': medians,
                    'file_sha256': sha(self.root / 'calibration.npz'), 'scope': 'small frozen target-specific calibration, not a population estimate'})
        emit('calibrated', rank=self.rank, layers=len(medians))

    def read_calibration(self):
        with np.load(self.root / 'calibration.npz') as st:
            self.residual_norms = st['residual_medians'].copy()
            self.down_norms = [st[f'L{i}_down_norm'].copy() for i in range(64)]
            self.activation_maxima = [st[f'L{i}_abs_max'].copy() for i in range(64)]

    def spec(self, layer, neuron, signed_dose, label='selected'):
        norm = float(self.down_norms[layer][neuron])
        if norm <= 1e-8:
            return {'kind': 'sham', 'label': 'zero_column', 'neuron': int(neuron), 'signed_dose': signed_dose,
                    'amplitude': 0., 'injected_l2_norm': 0.}
        amplitude = signed_dose * float(self.residual_norms[layer]) / norm
        native = float(self.activation_maxima[layer][neuron])
        return {'kind': 'neuron', 'label': label, 'neuron': int(neuron), 'signed_dose': signed_dose,
                'amplitude': amplitude, 'injected_l2_norm': abs(amplitude) * norm,
                'norm_scope': 'nominal linear sub-update norm; BF16 realized perturbation not measured',
                'calibration_abs_max': native, 'amplitude_over_calibration_abs_max': amplitude / max(native, 1e-12)}

    def pilot(self):
        self.calibrate()
        self.read_calibration()
        health, _ = self.generate('What is 6 times 7?', 20, [{'kind': 'sham'}], 32)
        if '42' not in health[0]['text']:
            raise RuntimeError('unmodified arithmetic health check failed')
        samples = []
        for count in [4, 16, CFG['preferred_neurons_per_batch']]:
            specs = [self.spec(20, n, d) for n in range(count) for d in CFG['screen_signed_residual_doses']]
            specs.append({'kind': 'sham'})
            try:
                torch.cuda.reset_peak_memory_stats()
                result, info = self.generate('Describe a snake.', 20, specs, CFG['pilot_max_new_tokens'])
                tokens = sum(x['tokens'] for x in result)
                samples.append({'neurons_per_batch': count, 'rows': len(specs), 'tokens': tokens,
                                'seconds': info['elapsed_seconds'], 'tokens_per_second': tokens / info['elapsed_seconds'],
                                'peak_allocated_gb': torch.cuda.max_memory_allocated() / 1e9})
                emit('pilot_throughput', **samples[-1])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                samples.append({'neurons_per_batch': count, 'oom': True})
                break
        feasible = [s for s in samples if not s.get('oom')]
        if not feasible:
            raise RuntimeError('pilot found no viable batch size')
        chosen = max(feasible, key=lambda x: x['tokens_per_second'])
        # Short decoding does not exercise the census's full cache footprint.
        longest = max(DISCOVERY, key=lambda p: len(self.encode(p['prompt'])['input_ids'][0]))
        full_window = None
        while chosen['neurons_per_batch']:
            count = chosen['neurons_per_batch']
            specs = [self.spec(20, n, d) for n in range(count) for d in CFG['screen_signed_residual_doses']] + [{'kind': 'sham'}]
            try:
                torch.cuda.reset_peak_memory_stats()
                result, info = self.generate(longest['prompt'], 20, specs, CFG['screen_max_new_tokens'], force_steps=True)
                full_window = {'prompt_id': longest['id'], 'rows': len(specs), 'seconds': info['elapsed_seconds'],
                               'actual_generated_tokens': sum(x['tokens'] for x in result),
                               'forced_steps_per_row': CFG['screen_max_new_tokens'],
                               'memory_only_run_not_behavioral_evidence': True,
                               'peak_allocated_gb': torch.cuda.max_memory_allocated() / 1e9}
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                chosen = {**chosen, 'neurons_per_batch': count // 2}
        if full_window is None:
            raise RuntimeError('full-window pilot failed at every batch size')
        # The same sequence with and without sham hook must agree exactly.
        enc = self.encode('What is 6 times 7?')
        with torch.inference_mode():
            raw = self.model.generate(**enc, max_new_tokens=32, do_sample=False, use_cache=True,
                    pad_token_id=self.pad_id, eos_token_id=sorted(self.stop_ids))
        raw_ids, _ = trim_generation(raw[0, enc['input_ids'].shape[1]:].tolist(), self.stop_ids, 32)
        if raw_ids != health[0]['token_ids']:
            raise RuntimeError('no-op hook changes unmodified generation')
        seeds = []
        for layer, neuron, prompt in [(20, 3432, 'Describe a snake.'), (32, 7845, 'Who are you?')]:
            specs = [{'kind': 'sham'}, self.spec(layer, neuron, .3), self.spec(layer, neuron, -.3)]
            outputs, info = self.generate(prompt, layer, specs, 96)
            seeds.append({'layer': layer, 'neuron': neuron, 'prompt': prompt, 'specs': specs,
                          'outputs': outputs, 'generation': info, 'status': 'exploratory_seed_replay'})
        # Check batch effects explicitly; a difference is reported, never silently treated as behavior.
        batched, _ = self.generate('What is 6 times 7?', 20, [{'kind': 'sham'}] * 4, 32)
        batch_check = {'all_four_sham_token_sequences_equal_serial': all(x['token_ids'] == raw_ids for x in batched)}
        if not batch_check['all_four_sham_token_sequences_equal_serial']:
            raise RuntimeError('batch/serial baseline check failed; inspect numerical behavior before census')
        expected = CFG['layers'] * CFG['channels_per_layer'] * len(DISCOVERY) * 2
        atomic_json(self.root / 'pilot.json', {'status': 'PASS', 'throughput_samples': samples,
                    'neurons_per_batch': chosen['neurons_per_batch'], 'batch_equivalence_check': batch_check,
                    'full_window_memory_throughput_check': full_window,
                    'total_planned_neurons': 64 * 17408, 'total_planned_combinations': expected,
                    'worst_case_generated_tokens': expected * CFG['screen_max_new_tokens'],
                    'rough_wall_hours_at_allocated_gpu_count': expected * CFG['screen_max_new_tokens'] / chosen['tokens_per_second'] / CFG['workers'] / 3600,
                    'allocated_gpu_count_for_estimate': CFG['workers'],
                    'estimate_limit': 'short-pilot extrapolation excludes stopping, confirmation, I/O, prefill and long-cache effects',
                    'seed_replays': seeds, 'health_check': health[0], 'configured_wall_seconds': CFG['wall_seconds']})
        emit('pilot_pass', neurons_per_batch=chosen['neurons_per_batch'])

    def progress(self, phase, **extra):
        planned = 64 * 17408
        data = {'rank': self.rank, 'phase': phase, 'planned_total_neurons': planned,
                'planned_total_combinations': planned * len(DISCOVERY) * 2,
                'fully_screened_neurons': self.completed_neurons,
                'screen_completed_combinations': self.completed_combinations,
                'partial_chunk_combinations': self.partial_combinations,
                'screen_batches_this_process': self.batches, 'generated_screen_tokens_this_process': self.screen_tokens,
                'screen_seconds_this_process': self.screen_seconds,
                'candidate_batches_completed': self.candidates_run,
                'confirmed_discoveries': 0,
                'updated_unix': time.time(), 'elapsed_seconds_this_process': time.monotonic() - self.started,
                'top_candidates': self.top_candidates[:20], **extra}
        atomic_json(self.out / 'progress.json', data)

    def summarize_score(self, candidate, baseline, prompt):
        full = score_output(candidate['text'], prompt, baseline['text'])
        comparison = full['comparison']
        quality = full['candidate']['visible_final_quality']
        return {'weak_rank_score': comparison['weak_novelty_rank_score'],
                'lexical_distance': comparison['visible_word_set_jaccard_distance'],
                'needs_quality_review': comparison['candidate_needs_quality_review'],
                'visible_words': quality['word_tokens'],
                'marker_presence_deltas': {g: {k: v for k, v in d.items() if v} for g, d in comparison['marker_presence_deltas'].items() if any(d.values())},
                'format_constraint_changed': comparison['format_constraint_changed'],
                'empty': quality['empty'], 'repetition_screen': quality['repetition_screen'],
                'semantic_coherence': 'not_assessed'}

    def followup(self, candidate):
        layer, neuron, dose = candidate['layer'], candidate['neuron'], candidate['signed_dose']
        ident = f'L{layer}_N{neuron}_{"plus" if dose > 0 else "minus"}'
        path = self.out / f'candidate_{ident}.json'
        if path.exists() and json.loads(path.read_text()).get('completed'):
            return
        selected = self.spec(layer, neuron, dose)
        generator = np.random.default_rng(CFG['seed'] + layer * 17408 + neuron)
        other = int(generator.integers(17407))
        if other >= neuron:
            other += 1
        frozen = {'candidate_id': ident, 'selected_from': candidate, 'primary_signed_dose': dose,
                  'selected_unix': time.time(), 'status': 'needs_blinded_behavior_review',
                  'primary_endpoint': 'not supplied: this is an exploratory paired-output followup, not a significance test',
                  'no_automatic_confirmation': True, 'prompt_ids': [p['id'] for p in CONFIRMATION]}
        selection_path = self.out / f'selection_{ident}.json'
        if selection_path.exists():
            frozen = json.loads(selection_path.read_text())
        else:
            atomic_json(selection_path, frozen)
        trials = []
        for prompt in CONFIRMATION:
            if self.expired():
                break
            specs = [{'kind': 'sham', 'label': 'baseline'}, selected,
                     self.spec(layer, neuron, -dose, 'reverse_sign'), self.spec(layer, other, dose, 'other_channel')]
            for index in range(2):
                specs.append({'kind': 'random_residual', 'label': f'random_residual_{index}',
                              'direction_seed': CFG['seed'] + layer * 100000 + neuron * 3 + index,
                              'injected_l2_norm': selected['injected_l2_norm'],
                              'norm_scope': 'nominal float32 vector norm; BF16 realized perturbation not measured'})
            outputs, info = self.generate(prompt['prompt'], layer, specs, CFG['confirmation_max_new_tokens'])
            trials.append({'prompt_id': prompt['id'], 'prompt': prompt['prompt'], 'timing': 'continuous',
                           'specs': specs, 'outputs': outputs, 'generation': info,
                           'selected_signals': self.summarize_score(outputs[1], outputs[0], prompt)})
            # Timing and dose diagnostics use discovery prompts, so confirmation is not dose tuning.
            atomic_json(path, {**frozen, 'trials': trials, 'completed': False})
        diagnostics = []
        prompt = next(p for p in DISCOVERY if p['id'] == candidate['prompt_id'])
        for timing, doses in [('continuous', [math.copysign(.1, dose), math.copysign(.6, dose)]),
                              ('prefill_only', [dose]), ('decode_only', [dose]), ('pulse_8', [dose])]:
            if self.expired():
                break
            specs = [{'kind': 'sham'}] + [self.spec(layer, neuron, d) for d in doses]
            outputs, info = self.generate(prompt['prompt'], layer, specs, 192, timing)
            diagnostics.append({'prompt_id': prompt['id'], 'timing': timing, 'specs': specs, 'outputs': outputs, 'generation': info})
        atomic_json(path, {**frozen, 'trials': trials, 'discovery_timing_dose_diagnostics': diagnostics,
                          'completed': len(trials) == len(CONFIRMATION) and len(diagnostics) == 4})
        self.candidates_run += 1
        emit('candidate_followup', rank=self.rank, candidate=ident, trials=len(trials), status=frozen['status'])

    def validate_chunk(self, path, layer, neurons):
        summary_path = path.with_name(path.name + '.summary.json')
        digest = sha(path)
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if summary['sha256'] != digest or summary['neurons'] != neurons or summary['layer'] != layer:
                raise RuntimeError(f'completed chunk fingerprint or assignment changed: {path}')
            return
        expected = {(int(n), p['id'], d) for n in neurons for p in DISCOVERY for d in CFG['screen_signed_residual_doses']}
        observed = set()
        baselines = set()
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if row['kind'] == 'screen':
                    key = (row['neuron'], row['prompt_id'], row['signed_dose'])
                    if key in observed or row['layer'] != layer:
                        raise RuntimeError(f'duplicate or wrong-layer row in {path}')
                    observed.add(key)
                else:
                    baselines.add(row['prompt_id'])
        if observed != expected or baselines != {p['id'] for p in DISCOVERY}:
            raise RuntimeError(f'incomplete completed-chunk artifact: {path}')
        atomic_json(summary_path, {'sha256': digest, 'layer': layer, 'neurons': neurons,
                                  'combinations': len(observed), 'prompt_ids': sorted(baselines)})

    def screen(self):
        pilot = json.loads((self.root / 'pilot.json').read_text())
        chunk = pilot['neurons_per_batch']
        owned = list(range(self.rank, 64, self.workers))
        # Explicitly prioritize the two historical leads, then interleave all depths.
        owned.sort(key=lambda layer: (layer not in {20, 32}, layer))
        orders = {}
        for layer in owned:
            order = np.random.default_rng(CFG['seed'] + layer).permutation(17408).tolist()
            front = {20: [3432], 32: [7845]}.get(layer, [])
            orders[layer] = front + [n for n in order if n not in front]
        self.candidates_run = sum(1 for p in self.out.glob('candidate_*.json') if json.loads(p.read_text()).get('completed'))
        for selection_path in sorted(self.out.glob('selection_*.json')):
            selection = json.loads(selection_path.read_text())
            candidate_path = self.out / f"candidate_{selection['candidate_id']}.json"
            source = selection['selected_from']
            self.candidate_ids.add((source['layer'], source['neuron']))
            if not self.expired() and (not candidate_path.exists() or not json.loads(candidate_path.read_text()).get('completed')):
                self.followup(source)
        atomic_json(self.out / 'assignment.json', {'layers': owned, 'neuron_order_seed': CFG['seed'],
                    'prioritized_historical_leads': {'20': [3432], '32': [7845]}, 'neurons_per_chunk': chunk,
                    'assigned_neurons': len(owned) * 17408, 'discovery_prompt_ids': [p['id'] for p in DISCOVERY]})
        for block in range(math.ceil(17408 / chunk)):
            for layer in owned:
                neurons = orders[layer][block * chunk:(block + 1) * chunk]
                directory = self.out / f'layer-L{layer}'
                directory.mkdir(exist_ok=True)
                path = directory / f'chunk_{block:05d}.jsonl.gz'
                if path.exists():
                    self.validate_chunk(path, layer, neurons)
                    self.completed_neurons += len(neurons)
                    self.completed_combinations += len(neurons) * len(DISCOVERY) * 2
                    continue
                if self.expired():
                    self.progress('wall_limit_or_stop', resumable=True)
                    return
                temporary = path.with_name(path.name + '.partial')
                if temporary.exists():
                    temporary.replace(temporary.with_name(temporary.name + f'-preserved-{time.time_ns()}'))
                candidates = []
                finished = True
                self.partial_combinations = 0
                with gzip.open(temporary, 'wt', encoding='utf-8') as stream:
                    for prompt in DISCOVERY:
                        if self.expired():
                            finished = False
                            break
                        specs = [self.spec(layer, n, d) for n in neurons for d in CFG['screen_signed_residual_doses']]
                        specs.append({'kind': 'sham', 'label': 'same_batch_baseline'})
                        results, info = self.generate(prompt['prompt'], layer, specs, CFG['screen_max_new_tokens'])
                        baseline = results[-1]
                        base_id = f'L{layer}_C{block}_{prompt["id"]}_baseline'
                        stream.write(json.dumps({'kind': 'baseline', 'baseline_id': base_id, 'layer': layer,
                            'prompt_id': prompt['id'], 'prompt': prompt['prompt'], 'result': baseline, 'generation': info}, ensure_ascii=False) + '\n')
                        for spec, result in zip(specs[:-1], results[:-1]):
                            signals = self.summarize_score(result, baseline, prompt)
                            row = {'kind': 'screen', 'layer': layer, 'neuron': spec['neuron'],
                                   'prompt_id': prompt['id'], 'baseline_id': base_id, 'timing': 'continuous',
                                   'signed_dose': spec['signed_dose'], 'intervention': spec, 'result': result, 'signals': signals}
                            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                            if (not signals['needs_quality_review'] and signals['weak_rank_score'] >= CFG['candidate_minimum_weak_score']
                                    and signals['visible_words'] >= 8 and signals['lexical_distance'] >= .2):
                                candidates.append({'layer': layer, 'neuron': spec['neuron'], 'signed_dose': spec['signed_dose'],
                                    'prompt_id': prompt['id'], 'weak_rank_score': signals['weak_rank_score'],
                                    'source_file': str(path.relative_to(self.root)), 'status': 'unconfirmed_weak_screen_candidate'})
                        stream.flush()
                        self.partial_combinations += len(specs) - 1
                        self.screen_seconds += info['elapsed_seconds']
                        self.screen_tokens += sum(x['tokens'] for x in results[:-1])
                        self.batches += 1
                        self.progress('screening', active_layer=layer, active_chunk=block)
                        emit('screen_batch', rank=self.rank, layer=layer, chunk=block, prompt=prompt['id'],
                             rows=len(specs) - 1, seconds=round(info['elapsed_seconds'], 2),
                             fully_screened_neurons=self.completed_neurons)
                if not finished:
                    self.progress('wall_limit_or_stop', resumable=True, partial_file=str(temporary))
                    return
                temporary.replace(path)
                self.validate_chunk(path, layer, neurons)
                self.completed_neurons += len(neurons)
                self.completed_combinations += self.partial_combinations
                self.partial_combinations = 0
                candidates.sort(key=lambda x: -x['weak_rank_score'])
                self.top_candidates = sorted(self.top_candidates + candidates[:3], key=lambda x: -x['weak_rank_score'])[:20]
                self.progress('screening')
                if self.candidates_run < CFG['candidate_batches_per_worker']:
                    for candidate in candidates:
                        key = (candidate['layer'], candidate['neuron'])
                        if key in self.candidate_ids:
                            continue
                        self.candidate_ids.add(key)
                        self.followup(candidate)
                        break
        self.progress('complete', all_assigned_neurons_screened=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--rank', type=int, required=True)
    p.add_argument('--workers', type=int, default=CFG['workers'])
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    start_path = a.root / 'STARTED_AT.json'
    if a.rank == 0 and not start_path.exists():
        atomic_json(start_path, {'started_unix': time.time(), 'wall_seconds': CFG['wall_seconds']})
    else:
        for _ in range(1800):
            if start_path.exists():
                break
            time.sleep(1)
    start = json.loads(start_path.read_text())
    deadline = start['started_unix'] + start['wall_seconds']
    runner = Runner(a.root, a.rank, a.workers, deadline)
    try:
        if a.rank != 0:
            while not (a.root / 'pilot.json').exists():
                if (a.root / 'PILOT_FAILED.json').exists() or time.time() > deadline:
                    raise RuntimeError('pilot failed or wall budget expired before start')
                time.sleep(2)
        runner.load()
        if a.rank == 0:
            if not (a.root / 'pilot.json').exists():
                runner.pilot()
            elif json.loads((a.root / 'pilot.json').read_text())['status'] != 'PASS':
                raise RuntimeError('saved pilot is not PASS')
        runner.read_calibration()
        runner.screen()
    except Exception as error:
        record = {'rank': a.rank, 'error': repr(error), 'traceback': traceback.format_exc(), 'time': time.time()}
        atomic_json(runner.out / 'FAILED.json', record)
        if a.rank == 0 and not (a.root / 'pilot.json').exists():
            atomic_json(a.root / 'PILOT_FAILED.json', record)
        runner.progress('failed', error=repr(error))
        emit('failed', **record)
        raise


if __name__ == '__main__':
    main()
