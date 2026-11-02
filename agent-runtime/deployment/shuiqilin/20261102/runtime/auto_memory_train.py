"""Resumable CPU NF4 QLoRA for the unified Agent's compact workflow memories."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import random
import resource
import sqlite3
import sys
import time
import uuid

parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
W=Path(__file__).resolve().parent;R=Path('/home/ubuntu/AI-Native/embodied-agent');BASE=R/'models/Qwen3-30B-A3B-bnb-4bit'
from auto_memory_data import sha,save
manifest=json.loads((W/'data/manifest.json').read_text());raw=(W/'data/train.jsonl').read_text();assert sha(raw)==manifest['train_sha256']
rows=[json.loads(line) for line in raw.splitlines()];assert rows
run_config=json.loads((W/'run-config.json').read_text()) if (W/'run-config.json').exists() else {}
if run_config.get('training_content')=='teacher_only':
    from teacher_training import validate_dataset
    validate_dataset(manifest,rows)
    assert args.preflight or not manifest.get('preview_only'),'Preview data must never run gradient training'
from light_lora_policy import training_plan
plan=training_plan(run_config,len(rows))
import torch
import torch.nn.functional as F
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training,PeftModel

torch.set_num_threads(plan['threads']);torch.set_num_interop_threads(1);torch.manual_seed(20261030);random.seed(20261030)
tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True,trust_remote_code=False)
vocab_size=len(tokenizer)
dataset=[]
for row in rows:
    if 'input_ids' in row:
        ids=row['input_ids'];start=row['target_start']
        assert isinstance(start,int) and 0<start<len(ids) and all(isinstance(t,int) and 0<=t<vocab_size for t in ids)
        prefix=ids[:start];suffix=ids[start:]
    else:
        prefix=tokenizer.apply_chat_template(row['messages'][:-1],tokenize=True,add_generation_prompt=True,enable_thinking=False)
        suffix=tokenizer.encode(row['messages'][-1]['content']+tokenizer.eos_token,add_special_tokens=False)
    assert 0<len(prefix) and len(prefix)+len(suffix)<=plan['max_sequence_tokens'],(row['id'],len(prefix)+len(suffix))
    dataset.append({'id':row['id'],'input_ids':torch.tensor([prefix+suffix]),'positions':torch.arange(len(prefix)-1,len(prefix)+len(suffix)-1),'targets':torch.tensor(suffix),'tokens':len(prefix)+len(suffix)})
download=json.loads((R/'experiments/qwen30b-nf4-20261030-r1/download-report.json').read_text());assert download['completed'] and download['all_sha256_verified']
for name,meta in download['files'].items():assert (BASE/name).stat().st_size==meta['bytes']
if args.preflight:
    print(json.dumps({'passed':True,'rows':len(rows),'training_plan':plan,'sequence_tokens':[x['tokens'] for x in dataset],'max_tokens':max(x['tokens'] for x in dataset),'method':'CPU NF4 double quantization, rank8 q/v all48 layers, FP32 compute','initial_adapter':run_config.get('parent_adapter') or 'fresh base'}));raise SystemExit()

OUT=W/'training-report.json';checkpoint=W/'latest-checkpoint.json';tick0=time.monotonic()
report={'completed':False,'phase':'loading','method':'CPU bitsandbytes NF4 double quantization + PEFT rank8 q_proj/v_proj all48 layers; FP32 compute',
    'cpu_only':True,'base':str(BASE),'base_frozen':True,'data_manifest':manifest,'train_rows':len(rows),'steps':[],
    'sequence_tokens':{'min':min(x['tokens'] for x in dataset),'max':max(x['tokens'] for x in dataset)},'initial_adapter':'fresh base','production_modified':False,'training_plan':plan}
initial=run_config.get('parent_adapter')
if initial:
    parent=Path(initial['peft']);assert sha((parent/'adapter_model.safetensors').read_bytes())==initial['peft_sha256']
    cfg=json.loads((parent/'adapter_config.json').read_text());assert cfg['r']==8 and set(cfg['target_modules'])=={'q_proj','v_proj'}
    report['initial_adapter']=initial
resume=json.loads(checkpoint.read_text()) if args.resume and checkpoint.exists() else None
if resume:
    report=json.loads(OUT.read_text());assert report['data_manifest']['train_sha256']==manifest['train_sha256'];assert not report['completed']
    report['discarded_uncheckpointed_steps']=len(report['steps'])-resume['step'];report['steps']=report['steps'][:resume['step']]
else:assert not OUT.exists(),'Inspect prior run; use --resume'

def resources():
    mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','SwapFree:'))}
    temps=[]
    for p in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
        try:
            value=float(p.read_text())/1000
            if 0<value<120:temps.append(value)
        except (OSError,ValueError):pass
    return {'available_gib':mem['MemAvailable']/1024**2,'swap_free_gib':mem.get('SwapFree',0)/1024**2,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'max_temperature_c':max(temps,default=None)}
def progress(phase=None,**extra):
    if phase:report['phase']=phase
    report.update(extra);report['elapsed_seconds_current_process']=round(time.monotonic()-tick0,3);report['resources']=resources();save(OUT,report)
def busy():
    for dbpath in [R/'experiments/qwen30b-memory-20261030-r1/data/memory.sqlite3',R/'data/memory.sqlite3']:
        with sqlite3.connect('file:'+str(dbpath)+'?mode=ro',uri=True) as db:
            if db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('running','queued')").fetchone()[0]:return True
    return False
def checkpoint_at(step,optimizer):
    directory=W/'checkpoints'/f'step-{step:03d}-{uuid.uuid4().hex[:8]}';directory.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(directory);torch.save({'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'order':order},directory/'optimizer.pt')
    save(checkpoint,{'step':step,'directory':str(directory),'data_sha256':manifest['train_sha256']})
    progress(checkpoint=str(directory))
def loss_for(item):
    logits=model(input_ids=item['input_ids'],logits_to_keep=item['positions'],use_cache=False).logits[0]
    return F.cross_entropy(logits.float(),item['targets'])

try:
    assert not busy(),'Foreground jobs arrived before load; retry when idle'
    assert resources()['available_gib']>=23,'NF4 load requires23GiB currently available'
    progress()
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float32)
    model=AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,trust_remote_code=False,quantization_config=quant,
        device_map={'':'cpu'},torch_dtype=torch.float32,low_cpu_mem_usage=True,attn_implementation='sdpa')
    modules=[m for m in model.modules() if isinstance(m,bnb.nn.Linear4bit)];assert len(modules)==18672
    assert all(m.weight.device.type=='cpu' and m.weight.quant_state.quant_type=='nf4' for m in modules)
    for m in modules:m.compute_dtype=torch.float32
    model.config.use_cache=False;model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
    if resume:model=PeftModel.from_pretrained(model,resume['directory'],is_trainable=True)
    elif initial:model=PeftModel.from_pretrained(model,initial['peft'],is_trainable=True)
    else:model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,target_modules=['q_proj','v_proj'],lora_dropout=0.,bias='none',task_type='CAUSAL_LM'))
    params=[p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in params)==3342336
    assert all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
    optimizer=torch.optim.AdamW(params,lr=plan['learning_rate'],weight_decay=.01)
    order=list(range(len(dataset)));random.Random(20261030).shuffle(order)
    start=0
    if resume:
        saved=torch.load(Path(resume['directory'])/'optimizer.pt',map_location='cpu',weights_only=True)
        optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['torch_rng']);order=saved['order'];start=resume['step']
    report.update(trainable_parameters=sum(p.numel() for p in params),quantized_linear_count=len(modules),gradient_checkpointing=model.is_gradient_checkpointing)
    from window_memory_data import optimizer_groups
    groups=optimizer_groups(order,plan['max_steps']) if plan.get('gradient_accumulation') else [[i] for i in order]
    progress('training',max_steps=len(groups),micro_batches=len(order));model.train()
    for step in range(start,len(groups)):
        tick=time.monotonic();optimizer.zero_grad(set_to_none=True);loss_sum=0.;total_tokens=0;total_targets=0
        lr=plan['learning_rate']*min(1,(step+1)/plan['warmup_steps'])*max(.2,1-step/len(groups))
        for group in optimizer.param_groups:group['lr']=lr
        for micro,index in enumerate(groups[step]):
            mem=resources()
            if busy() or mem['available_gib']<6 or (mem['max_temperature_c'] is not None and mem['max_temperature_c']>=90):
                # Discard partial accumulated gradients; checkpoint only completed updates.
                optimizer.zero_grad(set_to_none=True)
                if not checkpoint.exists() or json.loads(checkpoint.read_text())['step']!=step:checkpoint_at(step,optimizer)
                progress('yielded',yield_reason='foreground_job_or_memory_or_temperature',resumable=True);raise SystemExit(75)
            item=dataset[index];loss=loss_for(item);assert torch.isfinite(loss)
            (loss/len(groups[step])).backward()
            loss_sum+=float(loss.detach());total_tokens+=item['tokens'];total_targets+=item['targets'].numel()
            if plan.get('gradient_accumulation'):progress('training',current_optimizer_step=step+1,current_micro_batch=micro+1,micro_batches_in_step=len(groups[step]))
        norm=float(torch.nn.utils.clip_grad_norm_(params,plan['gradient_clip']));assert math.isfinite(norm) and norm>0
        optimizer.step()
        entry={'step':step+1,'row_id':item['id'],'tokens':total_tokens,'target_tokens':total_targets,'micro_batches':len(groups[step]),'loss':loss_sum/len(groups[step]),'gradient_norm':norm,'lr':lr,'seconds':round(time.monotonic()-tick,3)}
        report['steps'].append(entry);progress('training',last_step=entry);print(json.dumps(entry),flush=True)
        if (step+1)%(1 if plan.get('gradient_accumulation') else plan['checkpoint_every'])==0 or step+1==len(groups):checkpoint_at(step+1,optimizer)
    model.save_pretrained(W/'adapter');tokenizer.save_pretrained(W/'tokenizer')
    progress('completed',completed=True,adapter=str(W/'adapter'),adapter_sha256=sha((W/'adapter/adapter_model.safetensors').read_bytes()),step_count=len(report['steps']))
except SystemExit:raise
except BaseException as error:progress('failed',error=repr(error));raise
