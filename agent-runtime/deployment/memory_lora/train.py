"""Continue the existing concise adapter on memory-policy decisions; never merge the base."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import time

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--threads', type=int, default=8)
p.add_argument('--lr', type=float, default=.00025)
p.add_argument('--resume', action='store_true')
args = p.parse_args()
root = args.root.resolve(); work = root/'training/memory-v1'
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
torch.manual_seed(20261029); random.seed(20261029)
base = root/'models/Qwen3-4B-Instruct-2507'
initial = root/'training/concise-v2/adapter'
initial_sha = hashlib.sha256((initial/'adapter_model.safetensors').read_bytes()).hexdigest()
assert initial_sha == '48e105c01a479b393e690d88beb2bcb56b7c19cd7500a7635a45dfd44a13814e'
manifest = json.loads((work/'data/manifest.json').read_text())
train_path = work/'data/train.jsonl'
assert hashlib.sha256(train_path.read_bytes()).hexdigest() == manifest['splits']['train']['sha256']
assert not (work/'adapter').exists(), 'Completed adapter already exists; refusing to overwrite it'
report = {'version': 'memory-v1', 'started_at': datetime.now(timezone.utc).isoformat(),
    'training_kind': 'Synthetic supervised memory-policy LoRA; continued concise-v2, full assistant decision loss',
    'base': str(base), 'base_weights_frozen': True, 'base_weights_merged': False,
    'initial_adapter': str(initial), 'initial_adapter_sha256': initial_sha, 'dtype': 'float32',
    'threads': args.threads, 'learning_rate_peak': args.lr, 'action_token_weight': 4,
    'card_id_token_weight': 2, 'train_sha256': manifest['splits']['train']['sha256'],
    'spec_sha256': manifest['spec_sha256'], 'steps': [], 'completed': False}


def save():
    path = work/'training-report.json'; temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2)); temp.replace(path)


def log(event, **values):
    print(json.dumps({'event': event, **values}, ensure_ascii=False), flush=True)


tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
dataset = []
for line in train_path.read_text().splitlines():
    row = json.loads(line); messages = row['messages']
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded['input_ids']; assert ids[:len(prefix)] == prefix
    assert len(ids) <= 1536, (row['id'], len(ids))
    content = messages[-1]['content']; offset = rendered.rfind(content); assert offset >= 0
    ranges = []
    for field, weight in (('action', 4.), ('card_ids', 2.)):
        marker = '"'+field+'":'
        if marker not in content: continue
        start = content.index(marker)+len(marker)
        value = row['expected'][field] if field == 'action' else row['expected']['arguments'].get(field)
        if value is not None:
            ranges.append((offset+start, offset+start+len(json.dumps(value, ensure_ascii=False, separators=(',', ':'))), weight))
    positions = []; targets = []; weights = []
    for i, (a, b) in enumerate(encoded['offset_mapping']):
        if i < len(prefix): continue
        positions.append(i-1); targets.append(ids[i])
        weights.append(max([w for lo, hi, w in ranges if b > lo and a < hi] or [1.]))
    assert targets and min(positions) >= 0
    dataset.append({'id': row['id'], 'input_ids': torch.tensor([ids]), 'positions': torch.tensor(positions),
                    'targets': torch.tensor(targets), 'weights': torch.tensor(weights)})
report.update(train_rows=len(dataset), optimizer_steps_target=len(dataset),
    sequence_tokens={'min': min(x['input_ids'].numel() for x in dataset),
                     'max': max(x['input_ids'].numel() for x in dataset),
                     'mean': sum(x['input_ids'].numel() for x in dataset)/len(dataset)})
checkpoint = None
if args.resume and (work/'latest-checkpoint.json').exists():
    checkpoint = json.loads((work/'latest-checkpoint.json').read_text())
    old = json.loads((work/'training-report.json').read_text())
    assert old['train_sha256'] == report['train_sha256'] and old['initial_adapter_sha256'] == initial_sha
    report = old; report['steps'] = report['steps'][:checkpoint['step']]
    report['resumed_at'] = datetime.now(timezone.utc).isoformat()
save(); log('loading_base', **report['sequence_tokens'])
tick = time.monotonic()
base_model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True,
    torch_dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation='sdpa')
base_model.config.use_cache = False
source = work/checkpoint['directory'] if checkpoint else initial
model = PeftModel.from_pretrained(base_model, source, is_trainable=True)
trainable = [v for v in model.parameters() if v.requires_grad]
assert trainable and all('lora_' in n for n, v in model.named_parameters() if v.requires_grad)
report['trainable_parameters'] = sum(v.numel() for v in trainable)
assert report['trainable_parameters'] == 1867776
optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=.01)
if checkpoint:
    state = torch.load(work/checkpoint['directory']/'training_state.pt', map_location='cpu', weights_only=False)
    optimizer.load_state_dict(state['optimizer']); torch.set_rng_state(state['torch_rng'])
    random.setstate(state['python_rng'])
model.train(); optimizer.zero_grad(set_to_none=True)
report['load_seconds'] = time.monotonic()-tick
save(); log('training_started', rows=len(dataset), trainable=report['trainable_parameters'], load_seconds=report['load_seconds'])
started = time.monotonic()
try:
    for step in range(checkpoint['step'] if checkpoint else 0, len(dataset)):
        item = dataset[step]; tick = time.monotonic()
        lr = args.lr*min(1., (step+1)/6)*(.25+.75*(1-step/len(dataset)))
        for group in optimizer.param_groups: group['lr'] = lr
        # Compute vocabulary logits only at positions that predict assistant completion tokens.
        logits = model(input_ids=item['input_ids'], logits_to_keep=item['positions'], use_cache=False).logits[0]
        losses = F.cross_entropy(logits.float(), item['targets'], reduction='none')
        loss = (losses*item['weights']).sum()/item['weights'].sum()
        assert torch.isfinite(loss), 'Nonfinite training loss'
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.))
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        entry = {'step': step+1, 'sample_id': item['id'], 'loss': float(loss.detach()), 'lr': lr,
            'gradient_norm': norm, 'tokens': item['input_ids'].numel(), 'seconds': round(time.monotonic()-tick, 3),
            'peak_rss_gib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024, 3)}
        report['steps'].append(entry); report['elapsed_this_run_seconds'] = time.monotonic()-started
        save(); log('step', **entry)
        if (step+1)%16 == 0 or step+1 == len(dataset):
            name = f'checkpoint-{step+1:03d}'; folder = work/name
            folder.mkdir(exist_ok=False)
            model.save_pretrained(folder, safe_serialization=True)
            torch.save({'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
                        'python_rng': random.getstate()}, folder/'training_state.pt')
            temp = work/'latest-checkpoint.tmp'; temp.write_text(json.dumps({'step': step+1, 'directory': name}))
            temp.replace(work/'latest-checkpoint.json'); log('checkpoint_saved', directory=name)
    model.save_pretrained(work/'adapter', safe_serialization=True)
    report['adapter_files'] = {f.name: {'bytes': f.stat().st_size, 'sha256': hashlib.sha256(f.read_bytes()).hexdigest()}
                              for f in (work/'adapter').iterdir() if f.is_file()}
    report.update(completed=True, finished_at=datetime.now(timezone.utc).isoformat())
    save(); log('completed', adapter=str(work/'adapter'))
except BaseException as exc:
    report.update(error=type(exc).__name__+': '+str(exc)); save(); raise
