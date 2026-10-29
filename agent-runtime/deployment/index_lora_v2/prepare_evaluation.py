"""Prepare an index-only candidate without changing any production service."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
root=args.root.resolve();work=root/'training/index-v2';dest=root/'adapters/index-v2';dest.mkdir(exist_ok=True)
r=json.loads((work/'training-report.json').read_text());assert r.get('completed')
for name,info in r['adapter_files'].items():assert hashlib.sha256((work/'adapter'/name).read_bytes()).hexdigest()==info['sha256']
gguf=dest/'index-v2-f16.gguf';assert not gguf.exists()
check='''import json,torch,sys
from safetensors.torch import load_file
a=load_file(sys.argv[1]);b=load_file(sys.argv[2]);assert set(a)==set(b) and all(torch.isfinite(x).all() for x in a.values())
delta=sum(float((a[k].float()-b[k].float()).norm()) for k in a);assert delta>0
print(json.dumps({'finite':True,'delta_norm_sum':delta,'parameters':sum(x.numel() for x in a.values())}))
'''
inspection=subprocess.run([str(root/'training/concise-v2/venv/bin/python'),'-c',check,str(work/'adapter/adapter_model.safetensors'),str(root/'training/index-v1/adapter/adapter_model.safetensors')],capture_output=True,text=True,check=True)
subprocess.run(['/home/ubuntu/ai/hf-venv/bin/python','/home/ubuntu/ai/tools/llama.cpp/convert_lora_to_gguf.py',str(work/'adapter'),'--base',str(root/'models/Qwen3-4B-Instruct-2507'),'--outtype','f16','--outfile',str(gguf)],check=True)
unit=Path.home()/'.config/systemd/user/lea-model.service'
tokens=shlex.split(next(x.removeprefix('ExecStart=') for x in unit.read_text().splitlines() if x.startswith('ExecStart=')))
assert not any(x.startswith('--lora') for x in tokens)
assert tokens[tokens.index('--port')+1]=='18801';tokens[tokens.index('--port')+1]='18807'
tokens+=['--lora-scaled',str(root/'adapters/index-v1/index-v1-f16.gguf')+':0','--lora-scaled',str(gguf)+':0']
evalunit=Path.home()/'.config/systemd/user/lea-index-lora-v2-eval.service';assert not evalunit.exists()
evalunit.write_text('[Unit]\nDescription=Index-only LoRA isolated evaluation\n[Service]\nType=simple\n'+f'WorkingDirectory={root}\nExecStart={shlex.join(tokens)}\nRestart=no\nStandardOutput=append:{work}/evaluation-server.log\nStandardError=append:{work}/evaluation-server.log\n')
subprocess.run(['systemctl','--user','daemon-reload'],check=True);subprocess.run(['systemctl','--user','start','lea-index-lora-v2-eval.service'],check=True)
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for _ in range(90):
    try:
        with opener.open('http://127.0.0.1:18807/lora-adapters',timeout=3) as f:adapters=json.load(f)
        if len(adapters)==2:break
    except Exception:pass
    time.sleep(1)
else:raise TimeoutError('Index evaluation server not ready')
assert [a['id'] for a in adapters]==[0,1] and all(a['scale']==0 for a in adapters)
manifest={'version':'index-v2','role':'indexer_only','production_enabled':False,'state':'isolated_evaluation',
    'initial_adapter':'index-v1','baseline':'previous index-v1, same prompt and caps',
    'weight_check':json.loads(inspection.stdout),'gguf':{'path':str(gguf),'bytes':gguf.stat().st_size,'sha256':hashlib.sha256(gguf.read_bytes()).hexdigest()},'evaluation_port':18807}
(dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps(manifest),flush=True)
