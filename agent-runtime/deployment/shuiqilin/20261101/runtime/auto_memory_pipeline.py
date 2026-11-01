"""Bounded learner: actual CPU QLoRA, paired GGUF checks, live tool test, rollback."""
from pathlib import Path
import fcntl
import json
import os
import re
import shlex
import shutil
import socket
import sqlite3
import subprocess
import sys
import time

from auto_memory_data import sha,save
from auto_memory_eval import evaluate,request

R=Path('/home/ubuntu/AI-Native/embodied-agent');U=R/'experiments/shuiqilin-unified-20261030-r1'
W=Path(__file__).resolve().parent;AUTO=R/'auto-lora-shuiqilin';AUTO.mkdir(exist_ok=True)
DATA=R/'experiments/qwen30b-memory-20261030-r1/data'
OUT=W/'pipeline-report.json';STATUS=AUTO/'state.json'
report=json.loads(OUT.read_text()) if OUT.exists() else {'completed':False,'phase':'preflight','activated':False,'work':str(W),'bootstrap_authorized_by_owner':True}
assert not report['completed'],'Run already complete; do not duplicate'
lock=(AUTO/'learner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
server=None;controller=None

def progress(phase,**kwargs):
    report.update(phase=phase,**kwargs);save(OUT,report)
    save(STATUS,{'phase':phase,'work':str(W),'run_id':W.name,'activated':report.get('activated',False),'last_update':time.time(),
        'summary':kwargs.get('summary'),'training_report':str(W/'training-report.json')})
def system(*args,check=True):return subprocess.run(['systemctl','--user',*args],capture_output=True,text=True,check=check,timeout=180)
def idle():
    for p in [DATA/'memory.sqlite3',R/'data/memory.sqlite3']:
        with sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True) as db:
            if db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]:return False
    return True
def wait_idle():
    while not idle():progress('waiting_for_foreground_jobs');time.sleep(10)
def cpu_quota(percent):
    name=Path('/proc/self/cgroup').read_text().strip().split('/')[-1]
    if name.startswith('lea-shuiqilin-') and name.endswith('.service'):
        system('set-property','--runtime',name,'CPUQuota='+str(percent)+'%')
def env_for_train():return {**os.environ,'PYTHONPATH':str(R/'experiments/qwen30b-nf4-20261030-r1/deps'),
    'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','CUDA_VISIBLE_DEVICES':'','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'6'}
def stop(process):
    if process is not None:
        process.terminate()
        try:process.wait(timeout=30)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
def command():
    pid=system('show','lea-qwen30b-trial.service','-p','MainPID','--value').stdout.strip()
    return [os.fsdecode(x) for x in Path('/proc/'+pid+'/cmdline').read_bytes().split(b'\0') if x]
def health(base,process):
    for _ in range(420):
        assert process.poll() is None,'Isolated process exited; inspect its log'
        try:
            if request(base,'/health',timeout=3).get('status')=='ok':return
        except OSError:pass
        time.sleep(1)
    raise TimeoutError('Isolated backend not ready')

def live_tool(adapter_id,loaded_ids):
    global controller
    from lea.store import Store,dump
    snapshot=W/'live-data';snapshot.mkdir()
    with sqlite3.connect('file:'+str(DATA/'memory.sqlite3')+'?mode=ro',uri=True) as src,sqlite3.connect(snapshot/'memory.sqlite3') as dst:src.backup(dst)
    for name in ('tasks','workspaces'):
        if (DATA/name).exists():shutil.copytree(DATA/name,snapshot/name)
    manifest={**candidate,'id':adapter_id,'loaded_ids':loaded_ids,'purpose':'isolated_evaluation'};save(W/'candidate-manifest.json',manifest)
    environment={**os.environ,'PYTHONPATH':str(U/'app')+':'+str(U),'LEA_DATA_DIR':str(snapshot),
        'LEA_MEMORY_BACKEND':'hybrid','LEA_EMBEDDING_URL':'http://127.0.0.1:18803',
        'LEA_EMBEDDING_FINGERPRINT':'421a27e58d165478cc7acb984a688c2aa41404968b0203e7cd743ece44c54340',
        'SQ_MODEL_URL':'http://127.0.0.1:18816','SQ_ADAPTER_MANIFEST':str(W/'candidate-manifest.json'),'SQ_ALLOW_CANDIDATE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    controller=subprocess.Popen([str(R/'venv/bin/python'),'-m','uvicorn','unified_auto_agent:create_app','--factory','--host','127.0.0.1','--port','18819'],cwd=U,env=environment,stdout=(W/'live-controller.log').open('w'),stderr=subprocess.STDOUT)
    base='http://127.0.0.1:18819'
    for _ in range(60):
        assert controller.poll() is None
        try:request(base,'/api/model-profile',timeout=3);break
        except OSError:time.sleep(1)
    else:raise TimeoutError('Isolated controller not ready')
    from unified_tools import EXTRA_HELP
    task=request(base,'/api/tasks',{'goal':'开发一个经过测试的数字区间工具','completion_criteria':['通过自测和独立验收','登记后实际调用工具'],'allowed_tools':list(EXTRA_HELP)})
    cases=[({'n':n,'low':lo,'high':hi},expect) for n,lo,hi,expect in [(-2,0,5,0),(10,0,5,5),(3,0,5,3),(0,0,0,0),(1.25,0,2,1.25),(-9,-5,-1,-5),(-3,-5,-1,-3),(7,7,9,7),(9,7,9,9),(4,8,2,None)]]
    cases += [({},0),({'n':2},1)]
    store=Store(snapshot)
    with store.db(True) as db:db.execute('INSERT INTO unified_contracts VALUES(?,?,?)',(task['id'],'bench_clip',dump([{'input':i,'expected':e} for i,e in cases])))
    text='请开发bench_clip工具。run(arguments)读取数字n、low、high，缺省分别为0、0、1；若low>high返回None，否则把n限制在闭区间[low,high]内。自己编写代码及至少3个不同自测，实际测试当前版本，通过后登记；最后dev.run调用n=9,low=0,high=5，再依据真实回执简答。'
    tick=time.monotonic();job=request(base,'/api/tasks/'+task['id']+'/messages',{'text':text,'request_id':'qlora-heldout-clip-v1'})
    for _ in range(1200):
        outcome=request(base,'/api/jobs/'+job['id'])
        if outcome['status'] in ('completed','error','canceled'):break
        time.sleep(1)
    operations=store.operations(task['id']);events=store.events(task['id'],limit=10000)
    with store.db() as db:library=[dict(x) for x in db.execute('SELECT * FROM unified_library WHERE task_id=?',(task['id'],))]
    published=any(json.loads(x['test_report']).get('passed') and json.loads(x['test_report']).get('independent',{}).get('passed') for x in library)
    called=any(o['tool']=='dev.run' and (o.get('result') or {}).get('value',{}).get('passed') and (o['result']['value'].get('result')==5) for o in operations)
    authored=any(e['kind']=='primary_decision' and e['body'].get('tool')=='dev.submit' and any(sha(e['body'].get('arguments',{}).get('code',''))==x['code_sha'] for x in library) for e in events)
    result={'completed':True,'passed':outcome['status']=='completed' and published and called and authored,'job':outcome,'task_id':task['id'],
        'published':library,'operations':operations,'decisions':[e['body'] for e in events if e['kind']=='primary_decision'],'owner_cases':len(cases),'seconds':round(time.monotonic()-tick,3),
        'scope':'One held-out numeric tool authored by the same Agent; workflow preconditions remain active. Evaluation-only archive is not automatic training evidence.'}
    retention=[]
    for label,question,check in [
        ('port','RX-742当前通信端口是多少？按之前的更正记录简答。',lambda s:'19237' in s),
        ('condition','RX-742当前46摄氏度、盖板关闭，是否允许启动？按原规程回答。',lambda s:bool(re.search(r'不能|不允许|禁止|不满足|不符合|不可',s))),
        ('unknown','RX-742记录中的电池容量是多少mAh？没记录请说明。',lambda s:bool(re.search(r'未.{0,15}(?:提供|记录|提及)|没有|无法|未知',s)) and not re.search(r'\d+\s*mAh',s,re.I))]:
        memory_job=request(base,'/api/tasks/task_8b7a2be435a84199/messages',{'text':question,'request_id':'qlora-memory-'+label})
        for _ in range(300):
            memory_outcome=request(base,'/api/jobs/'+memory_job['id'])
            if memory_outcome['status'] in ('completed','error','canceled'):break
            time.sleep(1)
        answer=(memory_outcome.get('result') or {}).get('response','')
        retention.append({'id':label,'job':memory_outcome,'passed':memory_outcome['status']=='completed' and check(answer)})
    result['same_agent_memory_retention']=retention;result['passed']=result['passed'] and all(r['passed'] for r in retention)
    save(W/'live-tool-check.json',result);stop(controller);controller=None;return result

def activate(manifest):
    wait_idle();progress('activating_validated_adapter')
    unit=Path.home()/'.config/systemd/user/lea-shuiqilin.service';previous_unit=unit.read_text()
    drop=Path.home()/'.config/systemd/user/lea-qwen30b-trial.service.d/80-memory-lora.conf';drop.parent.mkdir(exist_ok=True)
    previous_drop=drop.read_text() if drop.exists() else None
    active=AUTO/'active.json';previous_active=active.read_text() if active.exists() else None
    save(W/'rollback-state.json',{'agent_unit':previous_unit,'backend_dropin':previous_drop,'active_manifest':previous_active})
    argv=command();filtered=[];i=0
    while i<len(argv):
        if argv[i] in ('--lora','--lora-scaled'):i+=2
        elif argv[i]=='--lora-init-without-apply':i+=1
        else:filtered.append(argv[i]);i+=1
    filtered += ['--lora-scaled',manifest['gguf']+':0']
    try:
        system('stop','lea-shuiqilin.service')
        save(active,manifest)
        drop.write_text('[Service]\nExecStart=\nExecStart='+shlex.join(filtered).replace('%','%%')+'\n')
        assert 'unified_agent_v11:create_app' in previous_unit or 'unified_auto_agent:create_app' in previous_unit
        unit.write_text(previous_unit.replace('unified_agent_v11:create_app','unified_auto_agent:create_app'))
        system('daemon-reload');system('restart','lea-qwen30b-trial.service')
        for _ in range(420):
            try:
                if request('http://127.0.0.1:18813','/health',timeout=3).get('status')=='ok':break
            except OSError:pass
            time.sleep(1)
        else:raise TimeoutError('Production backend restoration timed out')
        system('start','lea-shuiqilin.service')
        for _ in range(90):
            try:
                p=request('http://127.0.0.1:18815','/api/model-profile',timeout=3)
                h=request('http://127.0.0.1:18815','/api/health',timeout=8)
                if p['primary_adapter']==manifest['version'] and h['model']['ready']:return p
            except OSError:pass
            time.sleep(1)
        raise TimeoutError('Validated adapter controller health did not pass')
    except BaseException:
        system('stop','lea-shuiqilin.service',check=False)
        unit.write_text(previous_unit)
        if previous_drop is None:drop.unlink(missing_ok=True)
        else:drop.write_text(previous_drop)
        if previous_active is None:active.unlink(missing_ok=True)
        else:active.write_text(previous_active)
        system('daemon-reload');system('restart','lea-qwen30b-trial.service');system('start','lea-shuiqilin.service');report['activation_rolled_back']=True;raise

try:
    progress('preflight')
    assert (U/'unified_auto_agent.py').exists()
    config=json.loads((W/'run-config.json').read_text()) if (W/'run-config.json').exists() else {}
    report['bootstrap_authorized_by_owner']=config.get('mode')=='owner_bootstrap'
    if not (W/'preflight.log').exists():
        with (W/'preflight.log').open('w') as log:subprocess.run([str(R/'training/concise-v2/venv/bin/python'),str(W/'auto_memory_train.py'),'--preflight'],cwd=W,env=env_for_train(),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    parent=config.get('parent_adapter')
    if parent:assert json.loads((AUTO/'active.json').read_text())['gguf_sha256']==parent['gguf_sha256'],'Training parent changed'
    if config.get('mode')=='automatic':
        from copy import deepcopy
        from lea.consolidation import ConsolidationPolicy,assess
        from auto_memory_data import mine_sources
        from auto_memory_policy import current_epoch
        from auto_memory_policy import runtime_fingerprint
        assert runtime_fingerprint(U)==config['runtime_fingerprint'],'Runtime changed before consolidation'
        observations,_,_=mine_sources(config['source_paths'])
        observations=current_epoch(observations,config)
        gate=assess(observations,ConsolidationPolicy())
        assert not [r for r in gate['reasons'] if r!='parametric_value_not_measured'],'Independent evidence gate changed'
        guide=json.loads((AUTO/'policy-memory.json').read_text());assert guide['validated'] and sha(guide['body'])==guide['body_sha256']
        cases=[c for c in json.loads((W/'data/evaluation.json').read_text()) if c['kind'] not in ('repair','retention')]
        enriched=deepcopy(cases)
        for c in enriched:c['messages'].insert(1,{'role':'system','content':'已验证的流程记忆，仅作参考，当前请求与实际状态优先：'+guide['body']})
        progress('automatic_memory_value_probe')
        wait_idle();cpu_quota(1600)
        baseline=evaluate(W,'http://127.0.0.1:18813',1. if parent else None,'memory-probe-without',cases_override=cases,before_case=wait_idle)
        conditioned=evaluate(W,'http://127.0.0.1:18813',1. if parent else None,'memory-probe-with',cases_override=enriched,before_case=wait_idle)
        probe={'provenance_verified':True,'baseline_passed':baseline['scores']['workflow']['passed'],'with_memory_passed':conditioned['scores']['workflow']['passed'],'guide_sha256':guide['body_sha256']}
        eligibility=assess(observations,ConsolidationPolicy(),probe);save(W/'automatic-eligibility.json',eligibility)
        if eligibility['status']!='accept':
            progress('completed',completed=True,summary='automatic_training_delayed_no_measured_memory_value',eligibility=eligibility);raise SystemExit(0)
    primary=('lea-agent.service','lea-model.service','lea-native-kv.service','lea-embedding.service')
    report['primary_pids_before']={u:system('show',u,'-p','MainPID','--value').stdout.strip() for u in primary}
    while not (W/'training-report.json').exists() or not json.loads((W/'training-report.json').read_text()).get('completed'):
        wait_idle();cpu_quota(500);progress('training')
        cmd=[str(R/'training/concise-v2/venv/bin/python'),'-u',str(W/'auto_memory_train.py')]
        if (W/'latest-checkpoint.json').exists():cmd.append('--resume')
        with (W/'training.log').open('a') as log:exit_code=subprocess.run(cmd,cwd=W,env=env_for_train(),stdout=log,stderr=subprocess.STDOUT,timeout=5400).returncode
        if exit_code==75:progress('yielded_to_foreground');time.sleep(30);continue
        assert exit_code==0,'Trainer failed; inspect training.log and checkpoints'
    wait_idle();cpu_quota(1600);progress('converting')
    gguf=W/'adapter-f16.gguf'
    if not gguf.exists():
        from auto_memory_policy import export_base_config
        export=export_base_config(R/'models/Qwen3-30B-A3B-bnb-4bit/config.json',W/'export-base-config');save(W/'export-config-report.json',export)
        with (W/'conversion.log').open('w') as log:subprocess.run(['/home/ubuntu/ai/hf-venv/bin/python','/home/ubuntu/ai/tools/llama.cpp/convert_lora_to_gguf.py',str(W/'adapter'),
            '--base',export['export_directory'],'--outtype','f16','--outfile',str(gguf)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    assert gguf.stat().st_size>1000000
    candidate={'model':'Qwen3-30B-A3B-Q4_K_M','version':'memory-workflow-'+W.name,'gguf':str(gguf),'gguf_sha256':sha(gguf.read_bytes()),
        'peft':str(W/'adapter'),'peft_sha256':sha((W/'adapter/adapter_model.safetensors').read_bytes()),'base_merged':False}
    save(W/'candidate.json',candidate);progress('loading_isolated_gguf_candidate')
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',18816))!=0,'18816 is already in use'
    argv=command();argv[argv.index('--port')+1]='18816';clean=[];i=0
    while i<len(argv):
        if argv[i] in ('--lora','--lora-scaled'):i+=2
        elif argv[i]=='--lora-init-without-apply':i+=1
        else:clean.append(argv[i]);i+=1
    adapters=[parent['gguf'],str(gguf)] if parent else [str(gguf)]
    for path in adapters:clean+=['--lora-scaled',path+':0']
    server=subprocess.Popen(clean,env={**os.environ,'LD_LIBRARY_PATH':str(R/'runtime/q4-vnni-20261029-r1')},stdout=(W/'candidate-server.log').open('w'),stderr=subprocess.STDOUT)
    health('http://127.0.0.1:18816',server)
    loaded=request('http://127.0.0.1:18816','/lora-adapters');assert [a['path'] for a in loaded]==adapters
    request('http://127.0.0.1:18816','/lora-adapters',[{'id':i,'scale':0.} for i in range(len(adapters))])
    progress('paired_baseline_evaluation')
    before=evaluate(W,'http://127.0.0.1:18816',1. if parent else 0.,'paired-before',adapter_id=0,adapter_count=len(adapters),before_case=wait_idle)
    progress('paired_candidate_evaluation')
    after=evaluate(W,'http://127.0.0.1:18816',1.,'paired-after',adapter_id=len(adapters)-1,adapter_count=len(adapters),before_case=wait_idle)
    before_by={r['id']:r for r in before['rows']};regressed=[r['id'] for r in after['rows'] if before_by[r['id']]['passed'] and not r['passed']]
    gain=sum(r['passed'] for r in after['rows'] if r['kind']!='retention')-sum(r['passed'] for r in before['rows'] if r['kind']!='retention')
    gate=after['scores']['workflow']['passed']>=8 and after['scores']['retention']['passed']==4 and gain>0 and not regressed
    acceptance={'completed':True,'accepted':False,'paired_before':before['scores'],'paired_after':after['scores'],'decision_gain':gain,'regressed_cases':regressed,
        'preconditions_met':gate,'same_loaded_gguf_server':True,'cache_prompt':False,'scope':'16 frozen policy/repair/retention cases plus one held-out live tool if eligible; not general programming accuracy'}
    if gate:
        progress('heldout_live_tool_test');live=live_tool(len(adapters)-1,list(range(len(adapters))));acceptance['live_tool_passed']=live['passed'];acceptance['accepted']=live['passed']
    save(W/'acceptance-report.json',acceptance);stop(server);server=None
    report.update(candidate=candidate,acceptance=acceptance)
    if acceptance['accepted']:
        if config.get('mode')=='automatic':assert runtime_fingerprint(U)==config['runtime_fingerprint'],'Runtime changed during training; rollout refused'
        current_parent=json.loads((AUTO/'active.json').read_text()).get('gguf_sha256') if (AUTO/'active.json').exists() else None
        assert current_parent==(parent['gguf_sha256'] if parent else None),'Training parent changed; candidate retained without rollout'
        candidate.update(accepted=True,acceptance_report=str(W/'acceptance-report.json'),acceptance_sha256=sha((W/'acceptance-report.json').read_bytes()),id=0,loaded_ids=[0])
        profile=activate(candidate);report.update(activated=True,active_profile=profile)
        selected=config.get('selected_episodes') or [o['id'] for o in json.loads((W/'data/observations.json').read_text()) if o['independently_verified']]
        consumed=json.loads((AUTO/'consumed.json').read_text()) if (AUTO/'consumed.json').exists() else []
        save(AUTO/'consumed.json',sorted(set(consumed+selected)))
    report['primary_pids_after']={u:system('show',u,'-p','MainPID','--value').stdout.strip() for u in primary}
    assert report['primary_pids_before']==report['primary_pids_after'],'Unrelated inference services changed'
    if (AUTO/'config.json').exists():
        worker_config=json.loads((AUTO/'config.json').read_text());worker_config.update(last_completed_run=str(W),last_training_report=str(W/'training-report.json'));save(AUTO/'config.json',worker_config)
    progress('completed',completed=True,summary='accepted_and_activated' if report['activated'] else 'candidate_retained_gate_failed')
except SystemExit:raise
except BaseException as error:
    progress('failed',error=repr(error));raise
finally:
    stop(controller);stop(server)
