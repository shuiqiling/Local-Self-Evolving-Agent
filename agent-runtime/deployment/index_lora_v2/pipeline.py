"""Continue v1, evaluate frozen new questions, then wait for source-based review."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
root=args.root.resolve();work=root/'training/index-v2'
state={'phase':'waiting_for_training','started_at':datetime.now(timezone.utc).isoformat(),'production_modified':False}


def save():
    p=work/'pipeline-status.tmp';p.write_text(json.dumps(state,ensure_ascii=False,indent=2));p.replace(work/'pipeline-status.json')


def run(phase,script,*extra):
    state['phase']=phase;save();subprocess.run([str(root/'venv/bin/python'),'-u',str(work/script),'--root',str(root),*extra],check=True)


save()
try:
    while True:
        p=work/'training-report.json'
        if p.exists():
            r=json.loads(p.read_text())
            if r.get('completed'):break
            if r.get('error'):raise RuntimeError(r['error'])
        active=subprocess.run(['systemctl','--user','show','lea-index-lora-v2.service','-p','ActiveState','--value'],capture_output=True,text=True,check=True).stdout.strip()
        if active=='failed':raise RuntimeError('Training service failed')
        time.sleep(20)
    run('preparing_evaluation','prepare_evaluation.py');run('development_evaluation','evaluate.py','--split','dev')
    d=json.loads((work/'evaluation-dev/report.json').read_text())['summary'];old,new=d['index-v1'],d['index-v2']
    if any(new[k]<old[k] for k in ('schema_ok','identity_ok','literal_supported','key_values_present')):
        state.update(phase='development_review_required',completed=True,development_summary=d);save();raise SystemExit(0)
    run('heldout_evaluation','evaluate.py','--split','test');run('role_integration','verify_role.py')
    state.update(phase='awaiting_source_review',completed=True,finished_at=datetime.now(timezone.utc).isoformat());save()
except Exception as exc:
    state.update(phase='error',error=type(exc).__name__+': '+str(exc));save();raise
finally:subprocess.run(['systemctl','--user','stop','lea-index-lora-v2-eval.service'],check=False)
