"""CPU style LoRA with frozen base, completion-only loss, and durable checkpoints."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import resource
import time

parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--steps',type=int,default=32)
parser.add_argument('--accumulate',type=int,default=2)
parser.add_argument('--threads',type=int,default=8)
parser.add_argument('--dtype',choices=['float32','bfloat16'],default='float32')
parser.add_argument('--version',default='concise-v1')
parser.add_argument('--response-only',action='store_true')
parser.add_argument('--target-down',action='store_true')
parser.add_argument('--learning-rate',type=float,default=2e-4)
args=parser.parse_args()
assert re.fullmatch(r'concise-v[1-9][0-9]*',args.version)
root=args.root.resolve();work=root/'training'/args.version;out=work/'adapter'
work.mkdir(parents=True,exist_ok=True)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import torch
import transformers
import peft
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import LoraConfig,get_peft_model

torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
torch.manual_seed(20261028);random.seed(20261028)
base=root/'models/Qwen3-4B-Instruct-2507'
report={'started_at':datetime.now(timezone.utc).isoformat(),'base':str(base),
        'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,
        'dtype':args.dtype,'threads':args.threads,'optimizer_steps_target':args.steps,
        'version':args.version,'response_only_loss':args.response_only,'learning_rate_peak':args.learning_rate,
        'gradient_accumulation':args.accumulate,'training_kind':'Supervised style LoRA, no continual-learning automation',
        'base_weights_frozen':True,'base_weights_merged':False,'steps':[]}
def save():
    p=work/'training-report.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2));tmp.replace(p)
def log(event,**v):print(json.dumps({'event':event,**v},ensure_ascii=False),flush=True)
save();tick=time.monotonic()
tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
dataset=[]
for row in [json.loads(line) for line in (work/'data/train.jsonl').read_text().splitlines()]:
    messages=row['messages'];prefix=tokenizer.apply_chat_template(messages[:-1],tokenize=True,add_generation_prompt=True)
    full=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False)
    assert full[:len(prefix)]==prefix,'Chat prefix mismatch'
    assert len(full)<=1536, (row['id'],len(full))
    labels=[-100]*len(prefix)+full[len(prefix):]
    if args.response_only and row['expected']['response']:
        rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
        encoded=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
        assert encoded['input_ids']==full
        content=messages[-1]['content'];position=rendered.rfind(content)
        assert position>=0
        start=position+content.index('"response":')+len('"response":')
        end=start+len(json.dumps(row['expected']['response'],ensure_ascii=False))+1
        labels=[token if b>start and a<end else -100 for token,(a,b) in zip(full,encoded['offset_mapping'])]
        assert any(label!=-100 for label in labels)
    dataset.append({'id':row['id'],'input_ids':torch.tensor([full]),'labels':torch.tensor([labels])})
report['data_sha256']=hashlib.sha256((work/'data/train.jsonl').read_bytes()).hexdigest()
report['train_rows']=len(dataset);report['sequence_tokens']={'min':min(x['input_ids'].numel() for x in dataset),'max':max(x['input_ids'].numel() for x in dataset)}
log('loading_base',**report['sequence_tokens']);save()
model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,torch_dtype=getattr(torch,args.dtype),
                                         low_cpu_mem_usage=True,attn_implementation='sdpa')
model.config.use_cache=False
# Only upper layers are adapted; frozen lower layers have no backward graph.
layers=list(range(model.config.num_hidden_layers-8,model.config.num_hidden_layers))
config=LoraConfig(task_type='CAUSAL_LM',r=8,lora_alpha=16,lora_dropout=0.05,bias='none',
                  target_modules=['q_proj','v_proj','o_proj']+(['down_proj'] if args.target_down else []),layers_to_transform=layers)
model=get_peft_model(model,config)
trainable=[p for p in model.parameters() if p.requires_grad]
assert trainable and all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
report['lora']={'rank':8,'alpha':16,'dropout':.05,'layers':layers,'target_modules':config.target_modules if isinstance(config.target_modules,list) else sorted(config.target_modules),
                'trainable_parameters':sum(p.numel() for p in trainable),'total_parameters':sum(p.numel() for p in model.parameters())}
report['load_seconds']=time.monotonic()-tick
optimizer=torch.optim.AdamW(trainable,lr=args.learning_rate,weight_decay=.01)
model.train();optimizer.zero_grad(set_to_none=True);save();log('training_started',**report['lora'],load_seconds=report['load_seconds'])
started=time.monotonic()
try:
    for step in range(args.steps):
        losses=[];lengths=[];t=time.monotonic()
        lr=args.learning_rate*min(1.,(step+1)/3)*(.3+.7*(1-step/max(args.steps,1)))
        for group in optimizer.param_groups:group['lr']=lr
        for micro in range(args.accumulate):
            item=dataset[(step*args.accumulate+micro)%len(dataset)]
            loss=model(input_ids=item['input_ids'],labels=item['labels'],use_cache=False).loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            (loss/args.accumulate).backward();losses.append(float(loss.detach()));lengths.append(item['input_ids'].numel())
            log('microstep',step=step+1,micro=micro+1,loss=losses[-1],tokens=lengths[-1],seconds=round(time.monotonic()-t,3))
        norm=float(torch.nn.utils.clip_grad_norm_(trainable,1.))
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        row={'step':step+1,'loss':sum(losses)/len(losses),'lr':lr,'gradient_norm':norm,
             'seconds':round(time.monotonic()-t,3),'sequence_tokens':lengths,
             'peak_rss_gib':round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024,3)}
        report['steps'].append(row);report['elapsed_training_seconds']=time.monotonic()-started;save();log('step',**row)
        if (step+1)%8==0 or step+1==args.steps:
            folder=work/f'checkpoint-{step+1:03d}';model.save_pretrained(folder,safe_serialization=True)
            log('checkpoint_saved',path=str(folder))
    model.save_pretrained(out,safe_serialization=True)
    report['adapter_files']={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in out.iterdir() if p.is_file()}
    report['completed']=True;report['finished_at']=datetime.now(timezone.utc).isoformat();save();log('completed',adapter=str(out))
except BaseException as exc:
    report['error']=type(exc).__name__+': '+str(exc);save();raise
