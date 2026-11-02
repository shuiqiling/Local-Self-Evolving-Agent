"""Enable idle consolidation around the same Agent after the bootstrap finishes."""
from dataclasses import asdict
from datetime import datetime,timezone,timedelta
from pathlib import Path
import json
import os
import sqlite3
import subprocess
import sys
import time

from auto_memory_data import sha,save
from lea.consolidation import ConsolidationPolicy
from lea.store import Store
from auto_memory_eval import request
from auto_memory_policy import runtime_fingerprint

R=Path('/home/ubuntu/AI-Native/embodied-agent');U=R/'experiments/shuiqilin-unified-20261030-r1';A=R/'auto-lora-shuiqilin'
W=Path(sys.argv[1]) if len(sys.argv)>1 else R/'experiments/shuiqilin-memoryqlora-20261030-r1'
DATA=R/'experiments/qwen30b-memory-20261030-r1/data';OUT=A/'installation-report.json';assert not OUT.exists()
bootstrap=json.loads((W/'pipeline-report.json').read_text());assert bootstrap['completed'],'Finish current run before installing'
report={'completed':False,'bootstrap_activated':bootstrap['activated'],'bootstrap_work':str(W)}
def system(*args):return subprocess.run(['systemctl','--user',*args],capture_output=True,text=True,check=True,timeout=120)

try:
    with sqlite3.connect(DATA/'memory.sqlite3') as db:
        assert db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('running','queued')").fetchone()[0]==0,'Foreground Agent is busy'
        seq=db.execute('SELECT MAX(seq) FROM events').fetchone()[0]
        windows=db.execute("SELECT COUNT(*) FROM events WHERE kind='handoff_completed'").fetchone()[0]
    config={'enabled':True,'source_code':str(A/'source'),'app_pythonpath':str(U/'app')+':'+str(U),
        'live_data':str(DATA),'source_paths':[str(U/'data/memory.sqlite3'),str(DATA/'memory.sqlite3')],
        'epoch_live_db':str(DATA/'memory.sqlite3'),'epoch_min_seq':seq,'runtime_epoch':'unified-auto-memory-v1',
        'runtime_fingerprint':runtime_fingerprint(U),
        'last_completed_run':str(W),'last_training_report':str(W/'training-report.json'),
        'policy':asdict(ConsolidationPolicy()),'five_windows_is_a_check_not_permission_to_bypass_quality':True,
        'independent_acceptance_api':'POST /api/tasks/{task_id}/tool-acceptance; freeze owner cases before any implementation; not a model tool'}
    save(A/'config.json',config);save(A/'cursor.json',{'handoffs':windows})
    started=json.loads((W/'dispatch.json').read_text())['dispatched_at']
    day=datetime.fromtimestamp(started,timezone(timedelta(hours=8))).date().isoformat()
    save(A/'attempts.json',[{'local_day':day,'unit':'lea-shuiqilin-memoryqlora-r1','work':str(W),'kind':'owner_authorized_bootstrap'}])
    body=('这是当前水麒麟工具接口的流程记忆。缺少必要细节时用内置action=read_memory，复制request_id和选中索引；外部工具使用action=tool。'
          '需要资料时web.search返回候选，web.open读正文；不能把搜索摘要说成读过网页。'
          '自己编写纯Python工具后dev.submit保存，dev.test必须复制当前name和revision。代码或自测改动后旧测试失效。'
          '只有当前版本真实测试通过才dev.publish。测试失败时依据输入、预期和实际输出修正，原样重复失败代码不能解决问题。'
          '资料或API语义不清楚时再检索核实，不要无目的重复搜索。'
          'dev.run已经成功返回本轮单次调用的结果时直接简短回答，不重复相同调用；多次调用任务按各自要求执行。'
          '无法完成时明确尚未完成，不得宣称已测试或发布。流程记忆不等于当前任务已经执行。')
    store=Store(DATA);task=store.create_task('水麒麟当前开发流程的来源归档',[],[],[])
    manifest=json.loads((W/'data/manifest.json').read_text())
    origins=[json.loads(s)['origin'] for s in (W/'data/train.jsonl').read_text().splitlines() if json.loads(s)['origin']['kind']=='verified_memory_policy']
    event=store.append(task['id'],'runtime','message',{'text':body,'memory_kind':'owner_authorized_procedure_consolidation','source_type':'operator_procedure_synthesis','source_origins':origins,
        'synthesis':'Current ABI and owner-authorized workflow distilled from verified receipts; not evidence of completed actions or of general autonomous learning'})
    save(A/'policy-memory.json',{'validated':True,'body':body,'body_sha256':sha(body),'task_id':task['id'],'event_ref':event['ref'],'sources':origins})
    unit=Path.home()/'.config/systemd/user/lea-shuiqilin.service';before=unit.read_text()
    if 'unified_agent_v11:create_app' in before:
        (A/'controller-before-auto.service').write_text(before)
        unit.write_text(before.replace('unified_agent_v11:create_app','unified_auto_agent:create_app'))
        system('daemon-reload');system('restart','lea-shuiqilin.service')
    else:assert 'unified_auto_agent:create_app' in before
    for _ in range(60):
        try:
            profile=request('http://127.0.0.1:18815','/api/model-profile',timeout=4)
            if profile.get('automatic_lora')=='idle_and_evidence_gated':break
        except OSError:pass
        time.sleep(1)
    else:raise TimeoutError('Unified controller does not show automatic LoRA status')
    worker=Path.home()/'.config/systemd/user/lea-shuiqilin-autolora.service';assert not worker.exists()
    worker.write_text('[Unit]\nDescription=Shuiqilin idle memory consolidation worker (no decision Agent)\nWants=lea-shuiqilin.service\nAfter=lea-shuiqilin.service\n'
        '[Service]\nWorkingDirectory='+str(A/'source')+'\nEnvironment=PYTHONPATH='+str(U/'app')+':'+str(U)+':'+str(A/'source')+'\n'
        'Environment=OPENBLAS_NUM_THREADS=1\nEnvironment=OMP_NUM_THREADS=1\nExecStart='+str(R/'venv/bin/python')+' -u '+str(A/'source/auto_memory_worker.py')+'\n'
        'Restart=on-failure\nRestartSec=10\nNice=19\nCPUQuota=25%\nMemoryMax=512M\nNoNewPrivileges=true\nUMask=0077\n[Install]\nWantedBy=default.target\n')
    system('daemon-reload');system('enable','--now',worker.name)
    for _ in range(30):
        if (A/'worker-status.json').exists():break
        time.sleep(1)
    else:raise TimeoutError('Worker has no readiness report')
    state=json.loads((A/'worker-status.json').read_text())
    assert state['phase'] in ('waiting_for_idle','waiting_for_five_new_windows','waiting_for_independent_evidence','daily_update_budget_reached'),state
    report.update(completed=True,profile=profile,worker_state=state,guide_event_ref=event['ref'],new_runtime_epoch_min_seq=seq,
        policy=asdict(ConsolidationPolicy()),original_archive_events_preserved=True,bootstrap_training_manifest=manifest)
except BaseException as error:report['error']=repr(error);raise
finally:save(OUT,report)
