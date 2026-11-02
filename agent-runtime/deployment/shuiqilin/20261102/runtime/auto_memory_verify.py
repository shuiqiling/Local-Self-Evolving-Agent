"""Verify the actual deployed Agent; roll back a candidate on memory regression."""
from pathlib import Path
import json
import re
import sqlite3
import subprocess
import time

from auto_memory_data import save
from auto_memory_eval import request

R=Path('/home/ubuntu/AI-Native/embodied-agent');A=R/'auto-lora-shuiqilin';U=R/'experiments/shuiqilin-unified-20261030-r1'
W=R/'experiments/shuiqilin-memoryqlora-20261030-r1';OUT=W/'production-verification.json';assert not OUT.exists()
DATA=R/'experiments/qwen30b-memory-20261030-r1/data'
assert json.loads((A/'installation-report.json').read_text())['completed']
report={'completed':False,'memory_checks':[],'rolled_back':False}
CASES=[
    ('port','RX-742当前通信端口是多少？按之前的更正记录简答。',lambda s:'19237' in s),
    ('condition','RX-742当前46摄氏度、盖板关闭，是否允许启动？按原规程回答。',lambda s:bool(re.search(r'不能|不允许|禁止|不满足|不符合|不可',s))),
    ('unknown','RX-742记录中的电池容量是多少mAh？没记录请说明。',lambda s:bool(re.search(r'未.{0,15}(?:提供|记录|提及)|没有|无法|未知',s)) and not re.search(r'\d+\s*mAh',s,re.I))]

def api(path,data=None):return request('http://127.0.0.1:18815',path,data,timeout=30)
def system(*args):return subprocess.run(['systemctl','--user',*args],check=True,capture_output=True,text=True,timeout=120)
def memory(label):
    values=[]
    for kind,text,judge in CASES:
        tick=time.monotonic();job=api('/api/tasks/task_8b7a2be435a84199/messages',{'text':text,'request_id':'auto-memory-'+label+'-'+kind})
        for _ in range(300):
            outcome=api('/api/jobs/'+job['id'])
            if outcome['status'] in ('completed','error','canceled'):break
            time.sleep(1)
        answer=(outcome.get('result') or {}).get('response','')
        values.append({'kind':kind,'job':outcome,'answer':answer,'seconds':round(time.monotonic()-tick,3),'passed':outcome['status']=='completed' and judge(answer)})
        report['memory_checks']=values;save(OUT,report)
    return values

def rollback():
    previous=json.loads((W/'rollback-state.json').read_text());drop=Path.home()/'.config/systemd/user/lea-qwen30b-trial.service.d/80-memory-lora.conf'
    system('stop','lea-shuiqilin.service')
    if previous['backend_dropin'] is None:drop.unlink(missing_ok=True)
    else:drop.write_text(previous['backend_dropin'])
    if previous['active_manifest'] is None:(A/'active.json').unlink(missing_ok=True)
    else:(A/'active.json').write_text(previous['active_manifest'])
    # Keep the verified automatic learner interface; revert only the weights.
    system('daemon-reload');system('restart','lea-qwen30b-trial.service')
    for _ in range(420):
        try:
            if request('http://127.0.0.1:18813','/health',timeout=3).get('status')=='ok':break
        except OSError:pass
        time.sleep(1)
    else:raise TimeoutError('Base model did not restore')
    system('start','lea-shuiqilin.service')
    for _ in range(60):
        try:api('/api/model-profile');return
        except OSError:time.sleep(1)
    raise TimeoutError('Controller did not restore')

try:
    initial=api('/api/model-profile');report['initial_profile']=initial
    checks=memory('initial')
    if not all(c['passed'] for c in checks) and initial['primary_adapter']:
        report['candidate_memory_failures']=checks;rollback();report['rolled_back']=True
        report['memory_checks']=memory('base-restoration')
    final=api('/api/model-profile');health=api('/api/health')
    with sqlite3.connect(U/'production-before.sqlite3') as old,sqlite3.connect(DATA/'memory.sqlite3') as live:
        old.row_factory=live.row_factory=sqlite3.Row;archived=[dict(r) for r in old.execute('SELECT * FROM events ORDER BY seq')]
        intact=all(dict(live.execute('SELECT * FROM events WHERE id=?',(e['id'],)).fetchone())==e for e in archived)
        readonly=[r['id'] for r in old.execute("SELECT id FROM tasks WHERE allowed_tools='[]'")]
        restrictions=all(live.execute('SELECT allowed_tools FROM tasks WHERE id=?',(t,)).fetchone()[0]=='[]' for t in readonly)
        handoffs=live.execute("SELECT COUNT(*) FROM events WHERE kind='handoff_completed'").fetchone()[0]
    # Deployment verification windows are not user experience triggers.
    save(A/'cursor.json',{'handoffs':handoffs,'reason':'Start counting new user windows after deployment verification'})
    checks2={'single_decision_agent':final['decision_agents']==1,'automatic_lora_interface':final['automatic_lora']=='idle_and_evidence_gated',
        'model_ready':health['model']['ready'],'embedding_ready':health['embedding']['ready'],
        'original_archive_unchanged':intact,'readonly_restrictions_preserved':restrictions,
        'memory_retention':all(c['passed'] for c in report['memory_checks']),
        'worker_enabled':system('is-enabled','lea-shuiqilin-autolora.service').stdout.strip()=='enabled',
        'worker_active':system('is-active','lea-shuiqilin-autolora.service').stdout.strip()=='active'}
    report.update(completed=True,passed=all(checks2.values()),checks=checks2,final_profile=final,health=health,original_events=len(archived),readonly_tasks=len(readonly))
    if report['rolled_back']:save(A/'post-validation-rollback.json',{'work':str(W),'reason':'Memory regression','candidate_retained':True,'current_adapter':final['primary_adapter']})
finally:save(OUT,report)
