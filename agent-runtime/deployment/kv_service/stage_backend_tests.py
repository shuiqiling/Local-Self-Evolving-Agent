"""Real-model functional tests, confined to the dedicated stage KV directory."""
from pathlib import Path
import hashlib,json,time,subprocess,urllib.request,urllib.error,os

ROOT=Path('/home/ubuntu/AI-Native/embodied-agent')
WORK=Path(__file__).resolve().parent
REPORT=WORK/'stage-backend-test-report.json'
assert not REPORT.exists(),'Inspect existing test report before rerun'
BASE='http://127.0.0.1:18814'
STATE=Path(os.environ.get('LEA_KV_STATE_DIR',str(ROOT/'data/native-kv-stage-20261029-r6')))
report=dict(completed=False,checks={},started_at=time.time())

def save():REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2))
def request(path,body=None,origin=None):
    headers={'Content-Type':'application/json'}
    if origin:headers['Origin']=origin
    req=urllib.request.Request(BASE+path,data=json.dumps(body).encode() if body is not None else None,headers=headers)
    try:
        with urllib.request.urlopen(req,timeout=180) as response:return response.status,json.load(response)
    except urllib.error.HTTPError as error:return error.code,json.load(error)
def body(task,key,text,records=None,limit=180):
    schema=Decision.model_json_schema();schema['properties']['action']['enum']=['respond']
    schema['properties']['response']['maxLength']=200
    payload=dict(goal='原生会话功能验证',completion_criteria=[],constraints=[],tools={},handoff_state={},
        records=records or [],current_request=text,environment={},operation_status=[],pending_operations=[],
        memory_queries_remaining=0,memory_reads_remaining=0,answer_max_chars=200,_native_through_seq=1)
    return dict(task_id=task,request_id=hashlib.sha256(key.encode()).hexdigest(),
        messages=messages(PRIMARY,payload,layout='stable_v1'),output_schema=schema,limit=limit)
def infer(value):
    status,result=request('/complete',value)
    assert status==200,(status,result)
    Decision.model_validate_json(result['content'])
    return result

try:
    import sys
    sys.path.insert(0,str(ROOT/'app'))
    from lea.schemas import Decision
    from lea.model import messages
    from lea.runtime import PRIMARY
    for _ in range(90):
        try:
            if request('/health')[0]==200:break
        except OSError:pass
        time.sleep(1)
    else:raise RuntimeError('Stage backend did not become ready')
    a=body('stage_alpha','a1','记住本会话工艺代码QA-6172，只回复已记住。',[
        dict(ref='w_stage:e_a1',kind='message',role='user',text='本会话工艺代码为QA-6172。')])
    first=infer(a);report['first']=first;save()
    repeated=infer(a)
    assert repeated['replayed'] and repeated['next_absolute_position']==first['next_absolute_position']
    report['checks']['request_replay_no_append']=True
    b=infer(body('stage_beta','b1','记住本会话工艺代码RB-4905，只回复已记住。',[
        dict(ref='w_stage:e_b1',kind='message',role='user',text='本会话工艺代码为RB-4905。')]))
    alpha=infer(body('stage_alpha','a2','本会话工艺代码是什么？只回答代码。'))
    beta=infer(body('stage_beta','b2','本会话工艺代码是什么？只回答代码。'))
    assert 'QA-6172' in json.loads(alpha['content'])['response'],alpha
    assert 'RB-4905' in json.loads(beta['content'])['response'],beta
    report['checks']['task_isolation']=True;report['isolation']=[alpha,beta];save()
    long=infer(body('stage_alpha','a3','这些材料仅用于切窗验证，只回复“阶段完成”。',[
        dict(ref='w_stage:e_long',kind='message',role='user',text='普通材料：'+' x'*3300)]))
    assert long['total_cuts']>=1 and long['physical_tokens']<=3072
    assert long['cuts'] and all(c['before']==3072 and c['after']==614 and c['old_tokens_recomputed']==0
        and c['decode_calls']==0 and c['attention_callbacks']==0 and c['retained_kv_bitwise_equal'] for c in long['cuts'])
    report['checks']['automatic_real_cut']=True;report['automatic_cut']=long;save()
    before=request('/sessions/stage_alpha')[1]
    invalid=body('stage_alpha','a-invalid','只回复“这是一条完整的回答”。',limit=1)
    status,result=request('/complete',invalid)
    assert status==502,(status,result)
    after=request('/sessions/stage_alpha')[1]
    assert before['checkpoint_sha256']==after['checkpoint_sha256']
    assert before['stats']['next_absolute_position']==after['stats']['next_absolute_position']
    report['checks']['invalid_generation_rollback']=True;save()
    denied=request('/complete',a,origin='https://untrusted.invalid')[0]
    assert denied==403
    report['checks']['browser_origin_rejected']=True
    cancelled=body('stage_alpha','a-canceled','只回复取消验证。')
    assert request('/cancel',dict(task_id=cancelled['task_id'],request_id=cancelled['request_id']))[0]==200
    assert request('/complete',cancelled)[0]==502
    assert request('/sessions/stage_alpha')[1]['checkpoint_sha256']==before['checkpoint_sha256']
    report['checks']['canceled_before_commit_rollback']=True;save()
    subprocess.run(['systemctl','--user','restart','lea-native-kv-stage.service'],check=True,timeout=45)
    for _ in range(100):
        try:
            if request('/health')[0]==200:break
        except OSError:pass
        time.sleep(1)
    else:raise RuntimeError('Stage restart readiness timeout')
    restored=infer(body('stage_alpha','a-resume','只回复“恢复完成”。'))
    assert restored['cache_epoch']==before['cache_epoch'] and not restored['bootstrap']
    assert restored['cache_tokens_before']==before['stats']['physical_tokens']
    assert restored['next_absolute_position']>before['stats']['next_absolute_position']
    assert restored['input_tokens']<1500,'Restart unexpectedly replayed the full system/history'
    report['checks']['restart_restores_kv_without_history_replay']=True;report['restored']=restored;save()
    # Tamper only with a disposable stage manifest, then restore its exact bytes.
    folder=STATE/'sessions'/hashlib.sha256(b'stage_beta').hexdigest();manifest=folder/'CURRENT.json'
    assert STATE.name.startswith('native-kv-stage-')
    original=manifest.read_bytes();m=json.loads(original);m['kv']['sha256']='0'*64
    try:
        manifest.write_text(json.dumps(m))
        status,result=request('/complete',body('stage_beta','b-corrupt','只回复校验测试。'))
        assert status==502 and 'checksum' in str(result)
    finally:manifest.write_bytes(original)
    valid=infer(body('stage_beta','b-after-corrupt','只回复“校验恢复”。'))
    assert valid['cache_epoch']==b['cache_epoch']
    report['checks']['corrupt_checkpoint_rejected']=True
    report['completed']=True;report['finished_at']=time.time();save();print(json.dumps(report,ensure_ascii=False),flush=True)
except BaseException as error:
    report.update(error=repr(error),finished_at=time.time());save();raise
