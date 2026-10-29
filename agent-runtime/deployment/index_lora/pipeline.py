"""Index trainer post-processing and immutable evaluation; production stays unchanged."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root.resolve();work=root/'training/index-v1'
state={'phase':'waiting_for_training','started_at':datetime.now(timezone.utc).isoformat(),'production_modified':False}


def save():
    tmp=work/'pipeline-status.tmp';tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2));tmp.replace(work/'pipeline-status.json')


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
        active=subprocess.run(['systemctl','--user','show','lea-index-lora-v1.service','-p','ActiveState','--value'],capture_output=True,text=True,check=True).stdout.strip()
        if active=='failed':raise RuntimeError('Index training service failed')
        time.sleep(20)
    run('preparing_evaluation','prepare_evaluation.py');run('development_evaluation','evaluate.py','--split','dev')
    summary=json.loads((work/'evaluation-dev/report.json').read_text())['summary'];old,new=summary['base'],summary['index-v1']
    if any(new[k]<old[k] for k in ('schema_ok','literal_supported','coverage_ok','identity_ok','finished')):
        state.update(phase='development_review_required',completed=True,development_summary=summary);save();raise SystemExit(0)
    run('heldout_evaluation','evaluate.py','--split','test')
    run('role_integration','verify_role.py')
    state.update(phase='awaiting_source_review',completed=True,finished_at=datetime.now(timezone.utc).isoformat());save()
except Exception as exc:
    state.update(phase='error',error=type(exc).__name__+': '+str(exc));save();raise
finally:subprocess.run(['systemctl','--user','stop','lea-index-lora-eval.service'],check=False)
