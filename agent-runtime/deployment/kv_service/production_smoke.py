"""Exercise the formal API on a dedicated deployment-validation task."""
from pathlib import Path
import json,time,urllib.request,uuid

ROOT=Path('/home/ubuntu/AI-Native/embodied-agent');WORK=Path(__file__).resolve().parent
REPORT=WORK/'production-smoke.json'
assert not REPORT.exists(),'Inspect the existing formal verification first'
report=dict(completed=False,checks={},started_at=time.time())
def save():REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2))
def api(path,body=None,port=18800):
    req=urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=15) as f:return json.load(f)
def job(value):
    start=time.monotonic()
    while time.monotonic()-start<600:
        result=api('/api/jobs/'+value['id'])
        if result['status']=='completed':return {**result,'wall_seconds':time.monotonic()-start}
        if result['status'] in ['error','canceled']:raise RuntimeError(json.dumps(result,ensure_ascii=False))
        time.sleep(1)
    raise TimeoutError('Formal verification job timeout')
def say(text,key):
    report['phase']=key;save()
    result=job(api('/api/tasks/'+task+'/messages',dict(text=text,request_id='native-deploy-'+key)))
    report.setdefault('jobs',[]).append(result);save();return result
try:
    report['health']=api('/api/health');assert report['health']['model']['active_primary_backend']=='native_kvzip'
    task=api('/api/tasks',dict(goal='原生KV正式上线验证，仅处理当前请求。',constraints=[],completion_criteria=[],allowed_tools=['workspace.read','workspace.write']))['id']
    report['task_id']=task;save()
    marker='LIVE-KV-'+uuid.uuid4().hex[:12]
    say('必须调用workspace.write，把原样文本'+marker+'写入live-check.txt，只写一次，然后报告结果。','write')
    folder=ROOT/'data/workspaces'/task
    assert (folder/'live-check.txt').read_text()==marker
    report['checks']['formal_actual_write']=True;save()
    # Unknown contents prove actual access to the file, rather than a remembered
    # answer. It also keeps the verification confined to this new test workspace.
    postcut='CUT-'+uuid.uuid4().hex[:16]
    (folder/'fresh-cut.txt').write_text(postcut)
    state=api('/sessions/'+task,port=18804)
    if state['stats']['total_cuts']==0:
        say('以下普通材料用于切窗验证，不需要动作：'+'普通测试资料。'*650+'\n只回复收到。','fill')
    state=api('/sessions/'+task,port=18804)
    assert state['stats']['total_cuts']>=1
    result=say('fresh-cut.txt刚由外部程序创建，历史里没有内容。必须调用workspace.read读取fresh-cut.txt，只回复原样文本。','read-after-cut')
    assert postcut in result['result']['response']
    detail=api('/api/tasks/'+task);events=api('/api/tasks/'+task+'/events')
    assert len([o for o in detail['operations'] if o['tool']=='workspace.write' and o['status']=='completed'])==1
    assert any(o['tool']=='workspace.read' and o['status']=='completed' and (o['result'] or {}).get('value',{}).get('text')==postcut for o in detail['operations'])
    cuts=[c for e in events if e['kind']=='native_kv_checkpoint' for c in e['body'].get('cuts') or []]
    assert cuts and all(c['after']==614 and c['old_tokens_recomputed']==0 and c['retained_kv_bitwise_equal'] for c in cuts)
    assert not any(e['role']=='handoff' and e['kind']=='model_started' for e in events)
    report['checks'].update(formal_automatic_cut=True,formal_tool_read_after_cut=True,write_not_repeated=True,no_text_summary=True,retained_kv_bitwise_equal=True,old_tokens_not_recomputed=True)
    count=sum(e['kind']=='model_started' for e in events)
    t=time.monotonic();report['manual_handoff']=job(api('/api/tasks/'+task+'/handoff',{}));report['manual_handoff_wall_s']=time.monotonic()-t
    events=api('/api/tasks/'+task+'/events')
    assert sum(e['kind']=='model_started' for e in events)==count
    report['checks']['formal_manual_handoff_without_generation']=True
    report.update(completed=True,phase='verified',finished_at=time.time(),cuts=cuts,session=api('/sessions/'+task,port=18804),operations=detail['operations'],windows=detail['windows'],model_calls=[e['body'] for e in events if e['kind']=='model_result'])
    save();print(json.dumps(report,ensure_ascii=False),flush=True)
except BaseException as e:
    report.update(error=repr(e),finished_at=time.time());save();raise
