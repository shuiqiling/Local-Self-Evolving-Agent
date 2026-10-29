"""Durable post-training conversion/evaluation; no automatic production promotion."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
root=args.root.resolve();work=root/'training/memory-v1';status=work/'pipeline-status.json'
state={'started_at':datetime.now(timezone.utc).isoformat(),'phase':'waiting_for_training','production_modified':False}


def save():
    tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2));tmp.replace(status)


def run(phase,script,*arguments):
    state['phase']=phase;save()
    subprocess.run([str(root/'venv/bin/python'),'-u',str(work/script),'--root',str(root),*arguments],check=True)


save()
try:
    while True:
        report=work/'training-report.json'
        if report.exists():
            report=json.loads(report.read_text())
            if report.get('completed'):break
            if report.get('error'):raise RuntimeError(report['error'])
        unit=subprocess.run(['systemctl','--user','show','lea-memory-lora-v1.service','-p','ActiveState','--value'],capture_output=True,text=True,check=True).stdout.strip()
        if unit=='failed':raise RuntimeError('Training service failed')
        time.sleep(20)
    run('preparing_evaluation','prepare_evaluation.py')
    run('development_evaluation','evaluate.py','--split','dev')
    dev=json.loads((work/'evaluation-dev/report.json').read_text())['summary']
    old,new=dev['concise-v2'],dev['memory-v1']
    if (new['correct_action']<old['correct_action'] or new['missed_required_reads']>old['missed_required_reads'] or
            new['correct_read_selection']<old['correct_read_selection']):
        state.update(phase='development_regression_review_required',completed=True,development_summary=dev)
        save()
        raise SystemExit(0)  # Do not spend the independent held-out set on a known dev regression.
    run('heldout_evaluation','evaluate.py','--split','test')
    run('runtime_evaluation','runtime_evaluate.py')
    state.update(phase='awaiting_result_review',completed=True,finished_at=datetime.now(timezone.utc).isoformat());save()
except Exception as exc:
    state.update(phase='error',error=type(exc).__name__+': '+str(exc));save();raise
finally:
    # The temporary inference server must not remain a new CPU/RAM consumer.
    subprocess.run(['systemctl','--user','stop','lea-memory-lora-eval.service'],check=False)
