"""Verify release data and recorded provenance without loading model weights."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,re,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def csv_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames or len(set(reader.fieldnames))!=len(reader.fieldnames):
            raise ValueError(f'invalid header: {path.name}')
        return list(reader)

def verify(root=ROOT,check_hashes=True):
    root=Path(root); errors=[];checks=0
    def check(condition,message):
        nonlocal checks
        checks+=1
        if not condition:errors.append(message)
    def close(a,b,tol=.0011):
        try:return math.isfinite(float(a)) and abs(float(a)-float(b))<=tol
        except (ValueError,TypeError):return False
    d=csv_rows(root/'data/dictionary.csv'); ex=csv_rows(root/'data/examples.csv')
    probes=csv_rows(root/'data/probes.csv');cells=csv_rows(root/'data/probe_effects.csv')
    source_d=csv_rows(root/'evidence/source_csv/neuron_dictionary.csv')
    source_e=csv_rows(root/'evidence/source_csv/neuron_examples.csv')
    D={r['entry_id']:r for r in d};E={r['entry_id']:r for r in ex}
    S={f"L{r['layer']}_N{r['neuron']}":r for r in source_d}
    SE={f"L{r['layer']}_N{r['neuron']}":r for r in source_e}
    check(len(d)==len(D)==704,'dictionary: expected 704 unique entries')
    check(len(ex)==len(E)==704,'examples: expected 704 unique entries')
    check(set(D)==set(E)==set(S)==set(SE),'dictionary/example/source key sets differ')
    check(len(probes)==16 and len({p['probe_id'] for p in probes})==16,'expected 16 unique probes')
    check(len(cells)==11264,'expected 11264 effect cells')
    C={(r['entry_id'],r['probe_id']):r for r in cells}
    check(len(C)==len(cells),'duplicate effect cells')
    probe_ids=[p['probe_id'] for p in probes]; Q={p['probe_id']:p['prompt'] for p in probes}
    counts={l:sum(int(r['layer'])==l for r in d) for l in [12,20,32,44]}
    check(counts=={12:256,20:192,32:128,44:128},'layer coverage differs')
    profiles={}; calibration={}
    for layer in counts:
        obj=json.loads((root/f'evidence/profiles/dict-L{layer}.json').read_text())
        arr=np.load(root/f'evidence/calibration/q38-27b_L{layer}.npz',allow_pickle=False)
        check(arr['abs_max'].shape==(17408,),f'L{layer}: calibration shape')
        check(bool(np.isfinite(arr['abs_max']).all()),f'L{layer}: non-finite calibration')
        calibration[layer]=arr['abs_max']
        check(obj['summary']['k']==4.0 and obj['summary']['behaviours']==16,f'L{layer}: protocol metadata')
        for e in obj['entries']:
            ident=f"L{layer}_N{e['neuron']}"
            check(ident not in profiles,f'duplicate raw entry {ident}')
            profiles[ident]=e
    check(set(profiles)==set(D),'raw profiles and dictionary keys differ')
    for ident,r in D.items():
        check(re.fullmatch(r'L\d+_N\d+',ident) is not None,f'invalid entry id {ident}')
        layer=int(r['layer']);neuron=int(r['neuron'])
        check(ident==f'L{layer}_N{neuron}' and 0<=neuron<17408,f'{ident}: index/identity mismatch')
        check(all(v not in ('',None) for v in r.values()),f'{ident}: blank cell')
        if ident not in profiles:continue
        e=profiles[ident];v=np.array(e['delta'],dtype=float)
        check(v.shape==(16,) and bool(np.isfinite(v).all()),f'{ident}: invalid delta vector')
        check(int(r['sign'])==1 and e['sign']==1.0,f'{ident}: unexpected sign')
        for f,val in [('mean_abs_first_token_delta_nats',float(np.mean(np.abs(v)))),('max_abs_first_token_delta_nats',float(np.max(np.abs(v)))),('effect_concentration',float(np.max(np.abs(v))/np.sum(np.abs(v))))]:
            check(close(r[f],val),f'{ident}: {f} does not match raw profile')
        check(close(r['additive_offset'],4*float(calibration[layer][neuron])),f'{ident}: calibration amplitude mismatch')
        check(close(r['additive_offset'],S[ident]['amplitude_units']),f'{ident}: source amplitude mismatch')
        check(close(r['mean_abs_first_token_delta_nats'],S[ident]['influence']),f'{ident}: source influence mismatch')
        chosen=r['largest_measured_probe']
        check(chosen in probe_ids and abs(abs(v[probe_ids.index(chosen)])-float(np.max(np.abs(v))))<=.0011,f'{ident}: wrong argmax probe beyond rounding tolerance')
        check(r['largest_measured_probe']==S[ident]['top_behaviour_INCIDENTAL'],f'{ident}: source probe mismatch')
        for p,value in zip(probe_ids,v):
            check((ident,p) in C and close(C[(ident,p)]['baseline_minus_intervention_logp_nats'],value,.000001),f'{ident}/{p}: effect mismatch')
    for ident,r in E.items():
        if ident not in D or ident not in SE:
            check(False,f'{ident}: example lacks matching dictionary/source row')
            continue
        s=SE[ident];cat=D[ident]
        check(r['probe_id']==cat['largest_measured_probe'],f'{ident}: example is not selected probe')
        check(r['prompt']==Q[r['probe_id']],f'{ident}: example prompt mismatch')
        differs=r['baseline_excerpt']!=r['intervention_excerpt']
        check(r['excerpt_differs']==str(differs).lower(),f'{ident}: wrong excerpt difference flag')
        check(r['source_changed_flag']==s['changed'].lower(),f'{ident}: original change flag mismatch')
        check(r['baseline_excerpt']==s['answer_untouched'] and r['intervention_excerpt']==s['answer_pushed'],f'{ident}: altered source excerpts')
        check(close(r['additive_offset'],cat['additive_offset'],.011),f'{ident}: rounded example amplitude mismatch')
        check(close(r['baseline_minus_intervention_logp_nats'],C[(ident,r['probe_id'])]['baseline_minus_intervention_logp_nats']),f'{ident}: signed example delta mismatch')
    changed=sum(r['excerpt_differs']=='true' for r in ex)
    check(changed==187,'changed excerpt count differs from 187')
    changed_layers={l:sum(r['excerpt_differs']=='true' and r['entry_id'].startswith(f'L{l}_') for r in ex) for l in counts}
    check(changed_layers=={12:43,20:80,32:30,44:34},'changed counts by layer differ')
    meta=json.loads((root/'data/model_metadata.json').read_text())
    reference=json.loads((root/'evidence/official_reference.json').read_text())
    check(meta['model_id']=='Qwen/Qwen3.8-27B','wrong model attribution')
    check(meta['model_id']==reference['model_id'] and meta['revision']==reference['revision'],'model reference revision mismatch')
    check(meta['metadata_sha256']==reference['metadata_sha256'],'model reference metadata fingerprints mismatch')
    check((meta['num_hidden_layers'],meta['hidden_size'],meta['intermediate_size'])==(64,5120,17408),'unexpected target model dimensions')
    check(meta['checkpoint_metadata_matches_official'] is True,'checkpoint metadata match not established')
    weights_path=root/'evidence/checkpoint_weights.json'
    check(meta['weight_shards_rehashed'] is True and weights_path.is_file(),'weight-shard verification evidence missing')
    if weights_path.is_file():
        weights=json.loads(weights_path.read_text())
        check(weights['model_id']==meta['model_id'] and weights['revision']==meta['revision'],'weight proof model/revision mismatch')
        check(len(weights['shards'])==18 and weights['all_shards_match'] is True,'not all 18 weight shards verified')
        check(len({r['file'] for r in weights['shards']})==18,'duplicate weight shard proof')
        for row in weights['shards']:
            check(row['sha256']==row['expected_sha256']==reference['weight_sha256'].get(row['file']) and row['matches'] is True,f"weight hash mismatch: {row['file']}")
    annotations={r['entry_id']:r for r in csv_rows(root/'data/descriptive_annotations.csv')}
    layer_stats=json.loads((root/'data/layer_statistics.json').read_text())
    check(set(annotations)==set(D),'annotation coverage mismatch')
    for layer in counts:
        entries=json.loads((root/f'evidence/profiles/dict-L{layer}.json').read_text())['entries']
        v=np.array([e['delta'] for e in entries]);mu=v.mean(axis=0);sd=v.std(axis=0,ddof=0);z=(v-mu)/(sd+1e-9)
        stats=layer_stats[str(layer)]
        check(bool(np.allclose(stats['mean_delta_by_probe'],mu)),f'L{layer}: reference mean mismatch')
        check(bool(np.allclose(stats['population_std_delta_by_probe'],sd)),f'L{layer}: reference std mismatch')
        bias=z[:,4:8].mean(axis=1)-z[:,:4].mean(axis=1)
        a=z[:,4:6].mean(axis=1)-z[:,:2].mean(axis=1);b=z[:,6:8].mean(axis=1)-z[:,2:4].mean(axis=1)
        corr=float(np.corrcoef(a,b)[0,1]);check(close(corr,stats['split_half_pearson_across_entries'],1e-9),f'L{layer}: split-half correlation mismatch')
        for i,e in enumerate(entries):
            ident=f"L{layer}_N{e['neuron']}";row=annotations.get(ident,{})
            check(close(row.get('standardized_arithmetic_minus_capitals'),bias[i]),f'{ident}: standardized contrast mismatch')
            check(close(S[ident]['cap_arith_bias'],bias[i]),f'{ident}: archived standardized contrast mismatch')
            n=int((np.abs(z[i])>3).sum())
            check(row.get('count_abs_empirical_z_gt_3')==str(n) and int(S[ident]['n_behaviours_above_z3'])==n,f'{ident}: empirical z count mismatch')
            check(close(S[ident]['bias_reliability'],corr),f'{ident}: archived layer-level correlation mismatch')
    fresh_examples=None
    example_path=root/'evidence/example_replay.json'
    if example_path.is_file():
        example=json.loads(example_path.read_text());fresh_examples=len(example['entries'])
        check(example['model_id']==meta['model_id'] and example['revision']==meta['revision'],'example replay identity mismatch')
        check(fresh_examples==6,'expected six selected fresh example pairs')
        for row in example['entries']:
            ident=row['entry_id'];src=SE.get(ident)
            check(src is not None,f'unknown fresh example {ident}')
            if src is None:continue
            for arm,col in [('baseline','answer_untouched'),('intervention','answer_pushed')]:
                generated=row[arm+'_12'];excerpt=generated['text'].strip().replace('\n',' ')[:70]
                check(excerpt==src[col] and row[arm+'_excerpt_matches'] is True,f'{ident}: fresh {arm} excerpt mismatch')
                check(len(generated['token_ids'])==generated['tokens']<=12,f'{ident}: invalid fresh token count')
                for cap in [12,96]:
                    g=row[f'{arm}_{cap}'];is_eos=bool(g['token_ids'] and g['token_ids'][-1] in example['configured_eos_ids'])
                    check(g['stop_reason']==('eos' if is_eos else 'cap'),f'{ident}: inconsistent EOS classification')
    replay_status='NOT_INCLUDED';replay_stats=None
    replay_path=root/'evidence/dictionary_replay.json'
    if replay_path.is_file():
        replay=json.loads(replay_path.read_text());summary=replay['summary'];replay_errors=[]
        check(summary['model_id']==meta['model_id'] and summary['revision_from_cache_ref']==meta['revision'],'dictionary replay identity mismatch')
        replay_keys=set()
        for row in replay['entries']:
            ident=f"L{row['layer']}_N{row['neuron']}";replay_keys.add(ident)
            check(ident in profiles,f'unknown replay entry {ident}')
            if ident not in profiles:continue
            check(row['archived_delta']==profiles[ident]['delta'],f'{ident}: replay archive reference mismatch')
            check(len(row['replayed_delta'])==16,f'{ident}: replay vector length')
            for a,b in zip(row['replayed_delta'],row['archived_delta']):
                check(math.isfinite(a),f'{ident}: nonfinite replay value');replay_errors.append(abs(a-b))
        check(len(replay_keys)==len(replay['entries'])==704 and replay_keys==set(D),'fresh dictionary replay coverage incomplete')
        v=np.array(replay_errors);tol=summary['delta_tolerance_nats']
        replay_stats={'entries':len(replay_keys),'cells':len(v),'max_abs_error_nats':float(v.max()),'median_abs_error_nats':float(np.median(v)),'fraction_within_declared_tolerance':float(np.mean(v<=tol)),'declared_tolerance_nats':tol}
        check(close(summary['max_abs_error_nats'],replay_stats['max_abs_error_nats'],1e-9),'replay max error summary mismatch')
        check(close(summary['median_abs_error_nats'],replay_stats['median_abs_error_nats'],1e-9),'replay median error summary mismatch')
        check(close(summary['fraction_cells_within_tolerance'],replay_stats['fraction_within_declared_tolerance'],1e-9),'replay tolerance summary mismatch')
        comparison_path=root/'evidence/replay_comparison.json'
        if comparison_path.exists():
            comparison=json.loads(comparison_path.read_text())
            av=np.array([r['archived_delta'] for r in replay['entries']]);bv=np.array([r['replayed_delta'] for r in replay['entries']]);mask=np.abs(av)>=1
            derived={'cells':int(v.size),'within_0_15_nats':int((v<=.15).sum()),'outside_0_15_nats':int((v>.15).sum()),'median_abs_error_nats':float(np.median(v)),'max_abs_error_nats':float(v.max()),'pearson_all_cells':float(np.corrcoef(av.ravel(),bv.ravel())[0,1]),'sign_agreement_all':float(np.mean(np.sign(av)==np.sign(bv))),'sign_agreement_archived_abs_ge_1':float(np.mean(np.sign(av[mask])==np.sign(bv[mask])))}
            for name,value in derived.items():check(close(comparison.get(name),value,1e-9),f'replay comparison mismatch: {name}')
        replay_status='COMPLETED_WITH_DIFFERENCES' if bool((v>tol).any()) else 'COMPLETED_WITHIN_DECLARED_TOLERANCE'
    hash_count=0
    if check_hashes:
        manifest_path=root/'evidence/SHA256SUMS.json'
        check(manifest_path.exists(),'hash manifest missing')
        if manifest_path.exists():
            manifest=json.loads(manifest_path.read_text())
            for name,expected in manifest['files'].items():
                p=(root/name).resolve()
                check(p.is_relative_to(root.resolve()),f'unsafe manifest path {name}')
                check(p.is_file(),f'missing manifest file {name}')
                if p.is_file():
                    check(hashlib.sha256(p.read_bytes()).hexdigest()==expected,f'hash mismatch: {name}');hash_count+=1
            candidates={str(p.relative_to(root)).replace('\\','/') for p in root.rglob('*') if p.is_file() and not any(x in {'.git','__pycache__','.pytest_cache','.venv'} for x in p.relative_to(root).parts) and str(p.relative_to(root)).replace('\\','/') not in {'evidence/SHA256SUMS.json','evidence/verification_report.json'}}
            check(candidates==set(manifest['files']),'untracked or missing release files relative to manifest')
    return {'status':'PASS' if not errors else 'FAIL','scope':'archival data, arithmetic, joins, calibration, provenance and file integrity; GPU numerical agreement has its own explicit status','checks':checks,'errors':errors,'dictionary_entries':len(d),'effect_cells':len(cells),'examples':len(ex),'changed_excerpt_pairs':changed,'changed_excerpt_fraction':changed/len(ex),'layer_counts':counts,'changed_by_layer':changed_layers,'max_concentration':max(float(x['effect_concentration']) for x in d),'max_abs_delta_nats':max(float(x['max_abs_first_token_delta_nats']) for x in d),'hashed_files_checked':hash_count,'fresh_example_pairs_checked':fresh_examples,'gpu_replay_status':replay_status,'gpu_replay_comparison':replay_stats}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path)
    args=p.parse_args()
    try:result=verify(args.root)
    except Exception as e:result={'status':'FAIL','errors':[f'{type(e).__name__}: {e}']}
    text=json.dumps(result,indent=2)
    if args.output:args.output.write_text(text+'\n',encoding='utf-8',newline='\n')
    print(text);return 0 if result['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
