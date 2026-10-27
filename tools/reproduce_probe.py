"""Replay one archived intervention on a local Qwen3.8-27B checkpoint, on a GPU host."""
from __future__ import annotations
import argparse,csv,hashlib,json,platform,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def read(name):
    with (ROOT/'data'/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True,help='Local checkpoint directory on the remote GPU host')
    p.add_argument('--entry',required=True,help='Example: L20_N2302')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-new-tokens',type=int,default=12)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--dry-run',action='store_true',help='Inspect planned intervention without importing torch')
    a=p.parse_args()
    catalog={r['entry_id']:r for r in read('dictionary.csv')}
    if a.entry not in catalog:p.error('unknown entry')
    r=catalog[a.entry];probes={r['probe_id']:r for r in read('probes.csv')};probe=probes[r['largest_measured_probe']]
    if a.max_new_tokens<1:p.error('--max-new-tokens must be positive')
    plan={'model_id':'Qwen/Qwen3.8-27B','entry':r,'probe':probe,'max_new_tokens':a.max_new_tokens,'device':a.device}
    if a.dry_run:print(json.dumps(plan,indent=2));return
    cfg_path=a.model/'config.json'
    metadata=json.loads((ROOT/'data/model_metadata.json').read_text())
    if not cfg_path.is_file():p.error('config.json is missing from checkpoint directory')
    fingerprints={}
    for name,expected in metadata['metadata_sha256'].items():
        path=a.model/name
        if not path.is_file():p.error(f'missing checkpoint metadata file: {name}')
        fingerprints[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        if fingerprints[name]!=expected:p.error(f'checkpoint metadata fingerprint differs for {name}')
    digest=fingerprints['config.json']
    cfg=json.loads(cfg_path.read_text());tc=cfg.get('text_config',cfg)
    if (tc.get('num_hidden_layers'),tc.get('hidden_size'),tc.get('intermediate_size'))!=(64,5120,17408):p.error('unexpected model dimensions')
    import torch,transformers
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    torch.manual_seed(0)
    started=time.monotonic()
    tok=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(str(a.model),dtype=torch.bfloat16,local_files_only=True).eval().to(a.device)
    candidates=[(name,m) for name,m in model.named_modules() if isinstance(m,torch.nn.ModuleList) and len(m) and hasattr(m[0],'mlp') and hasattr(m[0].mlp,'down_proj')]
    if not candidates:raise RuntimeError('text decoder blocks not found')
    block_path,blocks=max(candidates,key=lambda pair:len(pair[1]))
    if len(blocks)!=64:raise RuntimeError('unexpected decoder block count')
    module=blocks[int(r['layer'])].mlp.down_proj
    text=tok.apply_chat_template([{'role':'user','content':probe['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    enc=tok([text],return_tensors='pt').to(a.device)
    target_ids=tok(probe['target_text'],add_special_tokens=False).input_ids
    stop_ids=set()
    for value in [tok.eos_token_id,model.generation_config.eos_token_id]:
        if value is not None:stop_ids.update(value if isinstance(value,(tuple,list)) else [value])
    def run(offset):
        def hook(_module,inputs):
            x=inputs[0].clone();n=int(r['neuron'])
            if x.shape[1]>1:x[:,1:,n]+=offset
            else:x[...,n]+=offset
            return (x,)
        handle=module.register_forward_pre_hook(hook)
        try:
            with torch.inference_mode():
                logits=model(**enc,use_cache=False).logits[0,-1,:].float()
                logp=float(torch.log_softmax(logits,dim=-1)[target_ids[0]])
                g=model.generate(**enc,max_new_tokens=a.max_new_tokens,do_sample=False,use_cache=True)
            ids=g[0,enc['input_ids'].shape[1]:].tolist()
        finally:handle.remove()
        return {'first_target_token_logp':logp,'text':tok.decode(ids,skip_special_tokens=True),'generated_token_ids':ids,'generated_tokens':len(ids),'stop_reason':'eos' if ids and ids[-1] in stop_ids else ('max_new_tokens' if len(ids)>=a.max_new_tokens else 'other'),'ended_with_eos_at_cap':bool(ids and ids[-1] in stop_ids and len(ids)==a.max_new_tokens)}
    base=run(0.0);pushed=run(float(r['additive_offset']))
    result={**plan,'metadata_sha256':fingerprints,'reference_revision':metadata['revision'],'historical_run_checkpoint_revision_logged':False,'weight_shards_rehashed_this_run':False,'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(torch.device(a.device))},'hook_path':f"{block_path}.{r['layer']}.mlp.down_proj input",'prompt_token_ids':enc['input_ids'][0].tolist(),'target_token_ids':target_ids,'configured_eos_token_ids':sorted(stop_ids),'baseline':base,'intervention':pushed,'baseline_minus_intervention_logp_nats':base['first_target_token_logp']-pushed['first_target_token_logp'],'intervention_minus_baseline_logp_nats':pushed['first_target_token_logp']-base['first_target_token_logp'],'elapsed_seconds':time.monotonic()-started}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(a.output),'entry':a.entry,'baseline_minus_intervention_logp_nats':result['baseline_minus_intervention_logp_nats'],'baseline_stop':base['stop_reason'],'intervention_stop':pushed['stop_reason']}))
if __name__=='__main__':main()
