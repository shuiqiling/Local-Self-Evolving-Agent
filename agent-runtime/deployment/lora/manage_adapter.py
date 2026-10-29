"""Convert a verified adapter; stage disabled, activate after an explicit review file."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import re
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--version',default='concise-v1')
p.add_argument('mode',choices=['prepare','activate','rollback']);args=p.parse_args()
assert re.fullmatch(r'concise-v[1-9][0-9]*',args.version)
version=args.version;root=args.root.resolve();work=root/'training'/version;dest=root/'adapters'/version;dest.mkdir(parents=True,exist_ok=True)
unit=Path.home()/'.config/systemd/user/lea-model.service'
dropin=unit.with_name(unit.name+'.d')/'50-style-lora.conf'
agent_dropin=unit.with_name('lea-agent.service.d')/'50-style-lora.conf'
config_path=root/'runtime-config.json';manifest_path=dest/'manifest.json'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def run(*v):return subprocess.run(list(map(str,v)),check=True)
def api(path):
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open('http://127.0.0.1:18801'+path,timeout=5) as f:return json.load(f)
def restart():
    run('systemctl','--user','daemon-reload');run('systemctl','--user','restart','lea-model.service')
    for _ in range(90):
        try:
            if api('/health').get('status')=='ok':return api('/lora-adapters')
        except Exception:pass
        time.sleep(1)
    raise TimeoutError('Model service did not become ready')
def write(v):manifest_path.write_text(json.dumps(v,ensure_ascii=False,indent=2))

base_command=next(line.removeprefix('ExecStart=') for line in unit.read_text().splitlines() if line.startswith('ExecStart='))
assert '--lora' not in base_command,'Base service already has an adapter; inspect before replacing'
tokens=shlex.split(base_command)
assert tokens[tokens.index('--model')+1]==str(root/'models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf')
gguf=dest/(version+'-f16.gguf')
if args.mode=='prepare':
    training=json.loads((work/'training-report.json').read_text());assert training.get('completed'),training.get('error')
    source=work/'adapter'
    for name,entry in training['adapter_files'].items():assert sha(source/name)==entry['sha256'],name
    inspect_code="""import sys,json,torch
from safetensors.torch import load_file
w=load_file(sys.argv[1]);assert w and all(torch.isfinite(t).all() for t in w.values())
b=[t for n,t in w.items() if 'lora_B' in n];assert b and sum(float(t.float().norm()) for t in b)>0
print(json.dumps({'tensors':len(w),'parameters':sum(t.numel() for t in w.values()),'lora_B_norm_sum':sum(float(t.float().norm()) for t in b),'finite':True}))
"""
    inspection=subprocess.run([str(work/'venv/bin/python'),'-c',inspect_code,str(source/'adapter_model.safetensors')],capture_output=True,text=True,check=True)
    weight_check=json.loads(inspection.stdout)
    run('/home/ubuntu/ai/hf-venv/bin/python','/home/ubuntu/ai/tools/llama.cpp/convert_lora_to_gguf.py',source,
        '--base',root/'models/Qwen3-4B-Instruct-2507','--outtype','f16','--outfile',gguf)
    assert not dropin.exists(),'Refusing to replace an existing adapter drop-in'
    config=json.loads(config_path.read_text());assert not config.get('lora',{}).get('enabled'),config
    (work/'runtime-config.before-lora.json').write_bytes(config_path.read_bytes())
    (work/'lea-model.before-lora.service').write_bytes(unit.read_bytes())
    record={'version':version,'enabled':False,'state':'candidate_staged_disabled','kind':'manual_style_lora',
        'gguf':{'path':str(gguf),'bytes':gguf.stat().st_size,'sha256':sha(gguf)},
        'peft_adapter_path':str(source),'base_repository':'Qwen/Qwen3-4B-Instruct-2507',
        'base_training_precision':training['dtype'],'base_inference_precision':'Q4_K_M',
        'lora':training['lora'],'training_report_sha256':sha(work/'training-report.json'),
        'weight_check':weight_check,
        'created_at':datetime.now(timezone.utc).isoformat()}
    write(record);dropin.parent.mkdir(parents=True,exist_ok=True)
    dropin.write_text('[Service]\nExecStart=\nExecStart='+base_command+' --lora-scaled '+str(gguf)+':0\n')
    try:
        loaded=restart();assert len(loaded)==1 and loaded[0]['scale']==0,loaded
        record['loaded_for_evaluation']=loaded;write(record)
    except BaseException:
        dropin.unlink(missing_ok=True);restart();raise
elif args.mode=='activate':
    record=json.loads(manifest_path.read_text());assert sha(gguf)==record['gguf']['sha256']
    review=json.loads((work/'review.json').read_text());assert review.get('accepted') is True,'Adapter has not passed review'
    smoke=work/'smoke-report.json'
    assert sha(smoke)==review['smoke_report_sha256'],'Reviewed runtime smoke report changed'
    assert json.loads(smoke.read_text()).get('passed'),'Real runtime checks did not pass'
    for split in ['dev','test']:
        result=work/('evaluation-'+split)/'report.json'
        assert sha(result)==review['evaluation_sha256'][split], 'Reviewed results changed'
        assert json.loads(result.read_text()).get('completed')
    assert dropin.exists() and '--lora-scaled '+str(gguf)+':0' in dropin.read_text()
    previous_config=config_path.read_bytes()
    assert not agent_dropin.exists(),'Inspect existing primary adapter settings before replacing'
    agent_dropin.parent.mkdir(parents=True,exist_ok=True)
    agent_dropin.write_text('[Service]\nEnvironment=LEA_PRIMARY_ADAPTER_ID=0\nEnvironment=LEA_PRIMARY_ADAPTER_VERSION='+version+'\n')
    config=json.loads(previous_config);config['lora']={'enabled':True,'version':version,'scope':'primary_only','path':str(gguf),'sha256':record['gguf']['sha256'],'review':str(work/'review.json')}
    config_path.write_text(json.dumps(config,indent=2))
    try:
        loaded=api('/lora-adapters');assert len(loaded)==1 and loaded[0]['id']==0 and loaded[0]['scale']==0,loaded
        run('systemctl','--user','daemon-reload');run('systemctl','--user','restart','lea-agent.service')
    except BaseException:
        agent_dropin.unlink(missing_ok=True);config_path.write_bytes(previous_config)
        run('systemctl','--user','daemon-reload');run('systemctl','--user','restart','lea-agent.service');raise
    record.update(enabled=True,state='active_after_regression',scope='primary_only',loaded=loaded,review_sha256=sha(work/'review.json'),activated_at=datetime.now(timezone.utc).isoformat());write(record)
else:
    if dropin.exists():
        assert str(gguf) in dropin.read_text(),'Unrecognized drop-in'
        dropin.unlink()
    if agent_dropin.exists():
        assert 'LEA_PRIMARY_ADAPTER_VERSION='+version in agent_dropin.read_text()
        agent_dropin.unlink()
    config=json.loads(config_path.read_text());config['lora']={'enabled':False,'version':version}
    config_path.write_text(json.dumps(config,indent=2));loaded=restart();assert loaded==[],loaded
    run('systemctl','--user','restart','lea-agent.service')
    record=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    record.update(enabled=False,state='rolled_back',rollback_at=datetime.now(timezone.utc).isoformat());write(record)
print(json.dumps(json.loads(manifest_path.read_text()),ensure_ascii=False),flush=True)
