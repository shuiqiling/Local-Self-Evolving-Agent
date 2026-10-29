"""Export this style pilot's own artifacts; never include production memory DBs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--version',default='concise-v2');p.add_argument('--verify-live',action='store_true')
args=p.parse_args();assert re.fullmatch(r'concise-v[1-9][0-9]*',args.version)
root=args.root.resolve();work=root/'training'/args.version
if args.verify_live:
    live=work/'live-report.json'
    if not live.exists():subprocess.run([str(root/'venv/bin/python'),str(work/'verify_live.py'),str(root),args.version],check=True)
    assert json.loads(live.read_text()).get('passed')
assert json.loads((work/'review.json').read_text()).get('accepted')
files=[]
for version,label in [('concise-v1','v1'),(args.version,'v2')]:
    base=root/'training'/version
    for name in ['training-report.json','review.json','activation-report.json','smoke-report.json','live-report.json']:
        if (base/name).exists():files.append((base/name,label+'/'+name))
    for split in ['dev','test']:
        src=base/('evaluation-'+split)/'report.json'
        if src.exists():files.append((src,label+'/evaluation-'+split+'.json'))
    files.append((root/'adapters'/version/'manifest.json',label+'/manifest.json'))
    if version==args.version:
        for src in base.glob('*.py'):files.append((src,label+'/scripts/'+src.name))
        for src in (base/'data').iterdir():
            if src.is_file():files.append((src,label+'/data/'+src.name))
        for src in (base/'adapter').iterdir():
            if src.is_file():files.append((src,label+'/peft/'+src.name))
        for src in base.glob('*.log'):files.append((src,label+'/logs/'+src.name))
        gguf=root/'adapters'/version/(version+'-f16.gguf');files.append((gguf,label+'/'+gguf.name))
        for name in ['config.py','model.py']:files.append((root/'app/lea'/name,label+'/runtime/'+name))
metadata={'files':{arc:{'bytes':src.stat().st_size,'sha256':hashlib.sha256(src.read_bytes()).hexdigest()} for src,arc in files}}
archive=work/'final-evidence.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
    for src,arc in files:z.write(src,arc)
    z.writestr('metadata.json',json.dumps(metadata,indent=2))
result={'bundle':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
(work/'bundle-status.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
