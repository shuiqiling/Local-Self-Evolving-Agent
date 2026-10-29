"""CPU original-record to navigation-index LoRA, independently versioned from actor policies."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import resource
import time

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--resume',action='store_true');args=p.parse_args()
root=args.root.resolve();work=root/'training/index-v2';base=root/'models/Qwen3-4B-Instruct-2507';initial=root/'training/index-v1/adapter'
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel

torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(20261029);random.seed(20261029)
initial_sha=hashlib.sha256((initial/'adapter_model.safetensors').read_bytes()).hexdigest()
assert initial_sha=='53302c678b3c9af7b771613eb1fc890b4eacdbfa329efd23b4837367fc6a193e'
manifest=json.loads((work/'data/manifest.json').read_text());train=work/'data/train.jsonl'
assert hashlib.sha256(train.read_bytes()).hexdigest()==manifest['splits']['train']['sha256']
assert not (work/'adapter').exists(),'Completed index adapter exists; refusing to overwrite'
report={'version':'index-v2','task':'source_to_navigation_index','started_at':datetime.now(timezone.utc).isoformat(),
    'base_weights_frozen':True,'base_merged':False,'initial_adapter':'index-v1','initial_adapter_sha256':initial_sha,
    'train_sha256':manifest['splits']['train']['sha256'],'prompt_sha256':manifest['prompt_sha256'],
    'dtype':'float32','threads':8,'learning_rate_peak':.00015,'critical_literal_token_weight':5,'condition_token_weight':5,'stop_token_weight':4,'training_target_max_chars':72,
    'steps':[],'completed':False}


def save():
    tmp=work/'training-report.tmp';tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2));tmp.replace(work/'training-report.json')


def log(event,**v):print(json.dumps({'event':event,**v},ensure_ascii=False),flush=True)


tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True);dataset=[]
for line in train.read_text().splitlines():
    row=json.loads(line);messages=row['messages'];prefix=tokenizer.apply_chat_template(messages[:-1],tokenize=True,add_generation_prompt=True)
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
    enc=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'];assert ids[:len(prefix)]==prefix
    assert len(ids)<=1536,(row['id'],len(ids))
    content=messages[-1]['content'];offset=text.rfind(content);assert offset>=0
    critical=[(offset+m.start(),offset+m.end()) for m in re.finditer(r'[A-Za-z]+[-_/][A-Za-z0-9_.-]*\d[A-Za-z0-9_.-]*|\d+(?:\.\d+)?|低于|等于|禁止|未|不|更正|作废|仅|失败|计划|通过',content)]
    phrases=[]
    for fragment in row['critical_fragments']:
        pos=content.find(fragment);assert pos>=0,(row['id'],fragment)
        phrases.append((offset+pos,offset+pos+len(fragment)))
    assert len(row['expected']['summary'])<=72 and row['expected']['summary'].endswith('。')
    positions=[];targets=[];weights=[]
    for i,(a,b) in enumerate(enc['offset_mapping']):
        if i<len(prefix):continue
        positions.append(i-1);targets.append(ids[i])
        weight=5. if any(b>lo and a<hi for lo,hi in critical+phrases) else 1.
        if ids[i]==tokenizer.eos_token_id or i>=len(ids)-4:weight=max(weight,4.)
        weights.append(weight)
    dataset.append({'id':row['id'],'input':torch.tensor([ids]),'positions':torch.tensor(positions),
                    'targets':torch.tensor(targets),'weights':torch.tensor(weights)})
report.update(train_rows=len(dataset),optimizer_steps_target=len(dataset),sequence_tokens={'min':min(x['input'].numel() for x in dataset),'max':max(x['input'].numel() for x in dataset),'mean':sum(x['input'].numel() for x in dataset)/len(dataset)})
checkpoint=None
if args.resume and (work/'latest-checkpoint.json').exists():
    checkpoint=json.loads((work/'latest-checkpoint.json').read_text());old=json.loads((work/'training-report.json').read_text())
    assert old['train_sha256']==report['train_sha256'];report=old;report['steps']=report['steps'][:checkpoint['step']]
save();log('loading_base',**report['sequence_tokens']);tick=time.monotonic()
model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,torch_dtype=torch.float32,low_cpu_mem_usage=True,attn_implementation='sdpa')
model.config.use_cache=False
model=PeftModel.from_pretrained(model,work/checkpoint['directory'] if checkpoint else initial,is_trainable=True)
trainable=[v for v in model.parameters() if v.requires_grad]
assert all('lora_' in name for name,v in model.named_parameters() if v.requires_grad)
report['trainable_parameters']=sum(v.numel() for v in trainable);assert report['trainable_parameters']==1867776
optimizer=torch.optim.AdamW(trainable,lr=.00015,weight_decay=.01)
if checkpoint:
    state=torch.load(work/checkpoint['directory']/'training_state.pt',map_location='cpu',weights_only=False)
    optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng'])
model.train();optimizer.zero_grad(set_to_none=True);report['load_seconds']=time.monotonic()-tick
save();log('training_started',rows=len(dataset),parameters=report['trainable_parameters']);started=time.monotonic()
try:
    for step in range(checkpoint['step'] if checkpoint else 0,len(dataset)):
        item=dataset[step];tick=time.monotonic();lr=.00015*min(1.,(step+1)/6)*(.25+.75*(1-step/len(dataset)))
        for group in optimizer.param_groups:group['lr']=lr
        logits=model(input_ids=item['input'],logits_to_keep=item['positions'],use_cache=False).logits[0]
        losses=F.cross_entropy(logits.float(),item['targets'],reduction='none');loss=(losses*item['weights']).sum()/item['weights'].sum()
        assert torch.isfinite(loss);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True))
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        entry={'step':step+1,'sample_id':item['id'],'loss':float(loss.detach()),'lr':lr,'gradient_norm':norm,'tokens':item['input'].numel(),'seconds':round(time.monotonic()-tick,3),'peak_rss_gib':round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024,3)}
        report['steps'].append(entry);report['elapsed_training_seconds']=time.monotonic()-started;save();log('step',**entry)
        if (step+1)%16==0 or step+1==len(dataset):
            name=f'checkpoint-{step+1:03d}';folder=work/name;folder.mkdir(exist_ok=False);model.save_pretrained(folder,safe_serialization=True)
            torch.save({'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state()},folder/'training_state.pt')
            tmp=work/'latest-checkpoint.tmp';tmp.write_text(json.dumps({'step':step+1,'directory':name}));tmp.replace(work/'latest-checkpoint.json')
    model.save_pretrained(work/'adapter',safe_serialization=True)
    report['adapter_files']={f.name:{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in (work/'adapter').iterdir() if f.is_file()}
    report.update(completed=True,finished_at=datetime.now(timezone.utc).isoformat());save();log('completed')
except BaseException as exc:
    report['error']=type(exc).__name__+': '+str(exc);save();raise
