"""End-to-end API, real workspace operations and KV handoff on isolated data."""
from pathlib import Path
import json,time,subprocess,urllib.request,urllib.error,uuid

ROOT=Path('/home/ubuntu/AI-Native/embodied-agent')
WORK=Path(__file__).resolve().parent
REPORT=WORK/'stage-agent-test-report.json'
assert not REPORT.exists(),'Inspect previous stage Agent report first'
BASE='http://127.0.0.1:18810';NATIVE='http://127.0.0.1:18814'
DATA=ROOT/'data/agent-kv-stage-20261029'
report=dict(completed=False,checks={},started_at=time.time())
def save():REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2))
def api(path,body=None,base=BASE):
    req=urllib.request.Request(base+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=20) as f:return json.load(f)
def wait_ready():
    for _ in range(120):
        try:
            health=api('/api/health')
            if health['model']['ready'] and health['model'].get('active_primary_backend')=='native_kvzip':return health
        except OSError:pass
        time.sleep(1)
    raise RuntimeError('Stage Agent readiness timeout')
def wait_job(job):
    start=time.monotonic()
    while time.monotonic()-start<600:
        r=api('/api/jobs/'+job['id'])
        if r['status']=='completed':return r
        if r['status'] in ['error','canceled']:raise RuntimeError(json.dumps(r,ensure_ascii=False))
        time.sleep(1)
    raise TimeoutError('Stage Agent job deadline')
def say(task,text,key):
    report['phase']=key;save()
    result=wait_job(api('/api/tasks/'+task+'/messages',dict(text=text,request_id=key)))
    report.setdefault('jobs',[]).append(result);save();return result

try:
    report['health']=wait_ready();save()
    task=api('/api/tasks',dict(goal='验证当前任务的原生KV切窗和工作区工具。',constraints=[],completion_criteria=[],
        allowed_tools=['workspace.write','workspace.read']))['id']
    report['task_id']=task;report['phase']='created';save()
    marker='DEPLOY-KV-7309'
    say(task,'必须调用workspace.write，把原样文本DEPLOY-KV-7309写入check.txt，只写一次，然后报告结果。','write-once')
    file=DATA/'workspaces'/task/'check.txt'
    assert file.read_text(encoding='utf-8')==marker,'Expected actual workspace write, not a claimed action'
    report['checks']['actual_workspace_write']=True;save()
    say(task,'必须调用workspace.read读取check.txt，不要只复述历史记录，只回复文件里的原样文本。','read-after-write')
    state=api('/sessions/'+task,base=NATIVE)
    if state['stats']['total_cuts']==0:
        say(task,'以下普通材料只用于切窗验证，不需要执行动作：'+'测试材料。'*650+'\n忽略上述材料，只回复“收到”。','fill-window')
        state=api('/sessions/'+task,base=NATIVE)
    assert state['stats']['total_cuts']>=1,'The real Agent did not trigger an automatic cut'
    report['checks']['automatic_native_handoff_in_real_agent']=True;save()
    fresh_marker='POSTCUT-'+uuid.uuid4().hex[:16]
    (file.parent/'post-cut.txt').write_text(fresh_marker,encoding='utf-8')
    fresh=say(task,'post-cut.txt刚由外部程序创建，历史里没有内容。切窗后必须调用workspace.read读取post-cut.txt，只回复原样文本。','read-new-file-after-cut')
    assert fresh_marker in fresh['result']['response']
    detail=api('/api/tasks/'+task);events=api('/api/tasks/'+task+'/events')
    writes=[op for op in detail['operations'] if op['tool']=='workspace.write' and op['status']=='completed']
    reads=[op for op in detail['operations'] if op['tool']=='workspace.read' and op['status']=='completed']
    assert len(writes)==1 and len(reads)>=2
    assert any(op['result'].get('value',{}).get('text')==fresh_marker for op in reads)
    cuts=[e for e in events if e['kind']=='handoff_completed' and e['body'].get('mode')=='native_kvzip']
    assert cuts and all(e['body']['summary_generated'] is False for e in cuts)
    assert not any(e['role']=='handoff' and e['kind']=='model_started' for e in events)
    report['checks'].update(tool_read_after_cut=True,write_not_repeated=True,no_generated_summary=True)
    before_models=sum(e['kind']=='model_started' for e in events)
    wait_job(api('/api/tasks/'+task+'/handoff',{}))
    events2=api('/api/tasks/'+task+'/events')
    assert sum(e['kind']=='model_started' for e in events2)==before_models
    report['checks']['manual_handoff_without_model_forward']=True
    before=api('/sessions/'+task,base=NATIVE);report['before_restart']=before;save()
    subprocess.run(['systemctl','--user','restart','lea-agent-kv-stage.service','lea-native-kv-stage.service'],check=True,timeout=60)
    wait_ready()
    resumed_marker='RESTART-'+uuid.uuid4().hex[:16]
    (file.parent/'post-restart.txt').write_text(resumed_marker,encoding='utf-8')
    resumed=say(task,'post-restart.txt刚由外部程序创建，历史里没有内容。服务重启后必须调用workspace.read读取post-restart.txt，只回复原样文本。','read-new-file-after-restart')
    assert resumed_marker in resumed['result']['response']
    restored=api('/sessions/'+task,base=NATIVE)
    assert restored['cache_epoch']==before['cache_epoch'] and restored['stats'].get('bootstrap') is False
    assert restored['stats']['next_absolute_position']>before['stats']['next_absolute_position']
    assert file.read_text(encoding='utf-8')==marker
    detail=api('/api/tasks/'+task)
    assert len([op for op in detail['operations'] if op['tool']=='workspace.write' and op['status']=='completed'])==1
    report['checks']['agent_and_backend_restart_resume']=True
    report['known_semantic_issues']=['Earlier stage run sometimes reused a prior read result despite an explicit same-file reread request; this remains a model behavior limitation. Fresh-file tests validate actual tool execution, not universal obedience.']
    report.update(completed=True,phase='complete',finished_at=time.time(),restored=restored,
        operations=detail['operations'],windows=detail['windows'])
    save();print(json.dumps(report,ensure_ascii=False),flush=True)
except BaseException as error:
    report.update(error=repr(error),finished_at=time.time());save();raise
