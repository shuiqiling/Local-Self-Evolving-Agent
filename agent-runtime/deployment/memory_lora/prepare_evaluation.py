"""Convert the trained adapter and start an isolated, dual-adapter evaluation server."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request

p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); args = p.parse_args()
root = args.root.resolve(); work = root/'training/memory-v1'; dest = root/'adapters/memory-v1'
report = json.loads((work/'training-report.json').read_text()); assert report.get('completed')
dest.mkdir(exist_ok=True)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


for name, info in report['adapter_files'].items():
    assert sha(work/'adapter'/name) == info['sha256']
inspect = '''import json,torch,sys
from safetensors.torch import load_file
a=load_file(sys.argv[1]);b=load_file(sys.argv[2])
assert set(a)==set(b) and all(torch.isfinite(x).all() for x in a.values())
delta=sum(float((a[k].float()-b[k].float()).norm()) for k in a)
assert delta>0
print(json.dumps({'tensors':len(a),'parameters':sum(x.numel() for x in a.values()),'delta_norm_sum':delta,'finite':True}))
'''
check = subprocess.run([str(root/'training/concise-v2/venv/bin/python'), '-c', inspect,
    str(work/'adapter/adapter_model.safetensors'), str(root/'training/concise-v2/adapter/adapter_model.safetensors')],
    capture_output=True, text=True, check=True)
gguf = dest/'memory-v1-f16.gguf'; assert not gguf.exists(), 'Candidate already converted; inspect before replacing'
subprocess.run(['/home/ubuntu/ai/hf-venv/bin/python', '/home/ubuntu/ai/tools/llama.cpp/convert_lora_to_gguf.py',
    str(work/'adapter'), '--base', str(root/'models/Qwen3-4B-Instruct-2507'), '--outtype', 'f16', '--outfile', str(gguf)], check=True)
base_unit = Path.home()/'.config/systemd/user/lea-model.service'
command = next(s.removeprefix('ExecStart=') for s in base_unit.read_text().splitlines() if s.startswith('ExecStart='))
tokens = shlex.split(command); assert not any(s.startswith('--lora') for s in tokens)
assert tokens[tokens.index('--model')+1] == str(root/'models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf')
assert tokens[tokens.index('--port')+1] == '18801'
tokens[tokens.index('--port')+1] = '18805'
old = root/'adapters/concise-v2/concise-v2-f16.gguf'
tokens += ['--lora-scaled', str(old)+':0', '--lora-scaled', str(gguf)+':0']
unit = Path.home()/'.config/systemd/user/lea-memory-lora-eval.service'; assert not unit.exists()
unit.write_text('[Unit]\nDescription=Isolated Qwen memory adapter evaluation\n[Service]\nType=simple\n'
    f'WorkingDirectory={root}\nExecStart={shlex.join(tokens)}\nRestart=no\n'
    f'StandardOutput=append:{work}/evaluation-server.log\nStandardError=append:{work}/evaluation-server.log\n')
subprocess.run(['systemctl','--user','daemon-reload'], check=True)
subprocess.run(['systemctl','--user','start','lea-memory-lora-eval.service'], check=True)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for _ in range(90):
    try:
        with opener.open('http://127.0.0.1:18805/lora-adapters', timeout=3) as response:
            adapters = json.load(response)
        if len(adapters) == 2: break
    except Exception: pass
    time.sleep(1)
else: raise TimeoutError('Isolated evaluation server failed to load both adapters')
assert [a['id'] for a in adapters] == [0,1] and all(a['scale'] == 0 for a in adapters), adapters
manifest = {'version':'memory-v1','enabled_in_production':False,'state':'isolated_evaluation_only',
    'created_at':datetime.now(timezone.utc).isoformat(),'gguf':{'path':str(gguf),'bytes':gguf.stat().st_size,'sha256':sha(gguf)},
    'initial_adapter':'concise-v2','training_report_sha256':sha(work/'training-report.json'),
    'weight_check':json.loads(check.stdout),'evaluation_adapters':adapters,'evaluation_port':18805,
    'production_service_modified':False,'base_merged':False}
identity_path=work/'identity.json'
if identity_path.exists():
    manifest['identity']=json.loads(identity_path.read_text())
(dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(json.dumps(manifest,ensure_ascii=False),flush=True)
