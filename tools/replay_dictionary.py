"""Independent archival replay; Qwen3.8-27B only. No checkpoint edits."""
import os,time,json,hashlib,pathlib,platform,csv,sys
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model',type=pathlib.Path,required=True)
parser.add_argument('--output-dir',type=pathlib.Path,required=True)
args=parser.parse_args()
ROOT=pathlib.Path(__file__).resolve().parents[1]
MODEL=args.model
OUT=args.output_dir
OUT.mkdir(exist_ok=True)
ITEMS=[('cap_france','What is the capital of France?','Paris'),('cap_japan','What is the capital of Japan?','Tokyo'),('cap_italy','What is the capital of Italy?','Rome'),('cap_spain','What is the capital of Spain?','Madrid'),('mul_12x12','What is 12 times 12?','144'),('mul_13x13','What is 13 times 13?','169'),('mul_9x9','What is 9 times 9?','81'),('mul_6x7','What is 6 times 7?','42'),('colour_sky','What colour is the sky on a clear day?','Blue'),('colour_grass','What colour is grass?','Green'),('spider_legs','How many legs does a spider have?','8'),('week_days','How many days are in a week?','7'),('planet_closest','Which planet is closest to the sun?','Mercury'),('symbol_gold','What is the chemical symbol for gold?','Au'),('symbol_water','What is the chemical symbol for water?','H'),('identity','Who are you?','Qwen')]
torch.manual_seed(0)
started=time.monotonic()
assert hashlib.sha256((MODEL/'config.json').read_bytes()).hexdigest()=='191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab'
cfg=json.loads((MODEL/'config.json').read_text());tc=cfg['text_config']
assert (tc['num_hidden_layers'],tc['hidden_size'],tc['intermediate_size'])==(64,5120,17408)
info={'model_id':'Qwen/Qwen3.8-27B','revision_from_cache_ref':'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0','torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'config_sha256':hashlib.sha256((MODEL/'config.json').read_bytes()).hexdigest(),'scope':'archival replay, not an independent new scientific sample','delta_tolerance_nats':0.15,'wallclock_limit_seconds':1200}
print(json.dumps({'event':'environment',**info}),flush=True)
tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
model=Qwen3_5ForConditionalGeneration.from_pretrained(str(MODEL),dtype=torch.bfloat16,local_files_only=True).eval().to('cuda:0')
blocks=max((m for m in model.modules() if isinstance(m,torch.nn.ModuleList) and len(m) and hasattr(m[0],'mlp') and hasattr(m[0].mlp,'down_proj')),key=len)
assert len(blocks)==64
encs=[]
for name,q,target in ITEMS:
 text=tok.apply_chat_template([{'role':'user','content':q}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 enc=tok([text],return_tensors='pt').to('cuda:0')
 ids=tok(target,add_special_tokens=False).input_ids
 encs.append((name,enc,ids[0]))

def values():
 vals=[]
 for _,e,tid in encs:
  with torch.inference_mode():o=model(**e,use_cache=False)
  vals.append(float(torch.log_softmax(o.logits[0,-1,:].float(),dim=-1)[tid]))
 return vals

base=values()
records=[]; errors=[]; entry_count=0
for layer in [12,20,32,44]:
 raw=json.loads((ROOT/f'evidence/profiles/dict-L{layer}.json').read_text())
 st=np.load(ROOT/f'evidence/calibration/q38-27b_L{layer}.npz')
 module=blocks[layer].mlp.down_proj
 for row in raw['entries']:
  if time.monotonic()-started>1200:break
  n=row['neuron'];amp=4*float(st['abs_max'][n])
  def hook(_m,inputs,n=n,amp=amp):
   x=inputs[0].clone()
   if x.shape[1]>1:x[:,1:,n]+=amp
   else:x[...,n]+=amp
   return (x,)
  h=module.register_forward_pre_hook(hook)
  try:actual=[b-v for b,v in zip(base,values())]
  finally:h.remove()
  err=[abs(a-b) for a,b in zip(actual,row['delta'])];errors.extend(err);entry_count+=1
  records.append({'layer':layer,'neuron':n,'amplitude':amp,'archived_delta':row['delta'],'replayed_delta':actual,'max_abs_error_nats':max(err)})
  if entry_count%32==0:print(json.dumps({'event':'progress','entries':entry_count,'elapsed_seconds':round(time.monotonic()-started,1),'max_abs_delta_error':max(errors)}),flush=True)
 if time.monotonic()-started>1200:break
summary={**info,'entries_replayed':entry_count,'expected_entries':704,'probe_cells_replayed':len(errors),'max_abs_error_nats':max(errors) if errors else None,'median_abs_error_nats':float(np.median(errors)) if errors else None,'fraction_cells_within_tolerance':float(np.mean(np.array(errors)<=.15)) if errors else None,'all_entries_replayed':entry_count==704,'elapsed_seconds':round(time.monotonic()-started,1),'baseline_first_token_logp':base,'target_token_ids':{n:tid for n,_,tid in encs}}
result={'summary':summary,'entries':records}
(OUT/'dictionary_replay.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'event':'completed',**summary}),flush=True)
