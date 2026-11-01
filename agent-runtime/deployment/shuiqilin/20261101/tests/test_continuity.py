import asyncio,importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
E=Path(__file__).resolve().parent
sys.path[:0]=[str(E.parent/'runtime'),str(E.parent/'app')]
import continuity_state as C
from lea.store import Store,Conflict
from lea.tools import Tools
from lea.runtime import Runtime

@pytest.fixture
def ctx(tmp_path):
    store=Store(tmp_path);task=store.create_task('设备任务',['禁止把宿主CPU当设备温度'],[],['workspace.write'])
    runtime=object.__new__(Runtime);runtime.store=store
    runtime.settings=SimpleNamespace(window_prompt_layout='stable_v1')
    runtime.model=SimpleNamespace(native_kv=None)
    return runtime,task['id']

def add(ctx,text):return ctx[0].store.append(ctx[1],'user','message',{'text':text})
def roll(ctx):return asyncio.run(C.rollover(ctx[0],ctx[1],'test'))
def cutoff(ctx):return ctx[0].store.events(ctx[1])[-1]['seq']
def payload(ctx,query,ref=None):
    r,tid=ctx;t,records=r.context(tid)
    return C.prepare_primary(r,t,{'current_request':query,'records':records,'tools':{},'constraints':t['constraints'],
         'handoff_state':t['state'],'machine_state':{'cpu':55},'latest_user_update':'obsolete'},cutoff(ctx),ref)
def sample(ctx,value=55):
    r,tid=ctx;return r.store.append(tid,'internal_state','internal_state_input',
        {'observation_id':'sample','snapshot':{'cpu':{'temperature_c':value},'network':{'rx_mib_per_second':0}}})

def test_rollover_never_calls_model_and_keeps_sources(ctx):
    add(ctx,'端口123已作废，更正为456。');sample(ctx);before=ctx[0].store.events(ctx[1]);roll(ctx)
    t=ctx[0].store.task(ctx[1]);assert t['state']['generated_handbook'] is False
    assert any('456' in x['text'] for x in t['state']['source_anchors'])
    assert all(e['body']==ctx[0].store.resolve(ctx[1],e['ref'])['body'] for e in before)
    assert ctx[0].store.events(ctx[1])[-1]['body']['generation_calls']==0

def test_long_complete_condition_not_discarded_at_320_chars(ctx):
    text='条件：'+('核验传感器。'*60)+'温度低于45且盖板关闭才允许启动。'
    e=add(ctx,text);roll(ctx);p=payload(ctx,'启动条件是什么')
    assert next(x for x in p['handoff_state']['source_anchors'] if x['ref']==e['ref'])['text']==text

def test_more_than_eight_sources_do_not_erase_early_condition(ctx):
    e=add(ctx,'规程：只有温度低于45且盖板关闭才启动。')
    for i in range(40):add(ctx,'用户消息编号'+str(i))
    roll(ctx);assert e['ref'] in [a['ref'] for a in payload(ctx,'设备启动条件')['handoff_state']['source_anchors']]

def test_device_question_does_not_receive_host_temperature(ctx):
    add(ctx,'设备温度46，盖板关闭。');sample(ctx,55);roll(ctx)
    p=payload(ctx,'设备现在46度能否启动')
    assert 'snapshot' not in p['host_state'] and 'machine_state' not in p
    assert p['host_state']['subject']=='Ubuntu宿主机，不是任务设备'

def test_current_turn_is_also_a_verbatim_evidence_source(ctx):
    text='本轮设备46摄氏度，盖板关闭，能否启动？';e=add(ctx,text)
    out=payload(ctx,text,e['ref'])
    item=next(a for a in out['handoff_state']['source_anchors'] if a['ref']==e['ref'])
    assert item['current_turn'] is True and item['text']==text

def test_host_question_receives_actual_sample_not_assessment(ctx):
    sample(ctx,55);r,tid=ctx;r.store.append(tid,'internal_state','internal_state_result',{'text':'CPU99'})
    roll(ctx);p=payload(ctx,'Ubuntu CPU温度是多少')
    assert p['host_state']['snapshot']['cpu']['temperature_c']==55 and 'CPU99' not in C.dump(p)

def test_shared_internal_context_keeps_real_observation_budget(ctx):
    sample(ctx);r,tid=ctx;t=r.store.task(tid)
    p={'records':[{'kind':'internal_state_input','text':'sample'}],'snapshot':{'cpu':55}}
    out=C.prepare_internal(r,t,p,cutoff(ctx))
    assert out['records']==p['records'] and out['snapshot']==p['snapshot'] and '宿主机' in out['snapshot_subject']

def test_actual_file_receipt_survives_ten_windows(ctx):
    r,tid=ctx;e=add(ctx,'audit.txt已经要求写入，不要重复。');t=r.store.task(tid)
    receipt=Tools(r.store).execute(tid,t['window_id'],t['constraint_version'],e['id'],'workspace.write',{'path':'audit.txt','text':'done'})
    for i in range(10):sample(ctx,i);roll(ctx)
    ops=payload(ctx,'audit.txt是否写入')['operation_status']
    assert len(ops)==1 and ops[0]['ok'] is True and ops[0]['evidence_ref']==receipt['evidence_ref']
    assert len(r.store.windows(tid))==11

def test_completed_test_operation_is_not_passing_test(ctx):
    r,tid=ctx;e=add(ctx,'测试v2');t=r.store.task(tid)
    op,_=r.store.begin_operation(tid,t['window_id'],t['constraint_version'],e['id'],'dev.test',{'name':'x','revision':'v2'})
    r.store.finish_operation(tid,op,{'ok':True,'tool':'dev.test','value':{'passed':False,'revision':'v2'}})
    item=payload(ctx,'x测试结果')['operation_status'][0]
    assert item['status']=='completed' and item['receipt']['passed'] is False

def test_old_generated_claims_not_inherited(ctx):
    add(ctx,'实际未完成');r,tid=ctx
    with r.store.db(True) as db:db.execute('UPDATE tasks SET state=? WHERE id=?',(json.dumps({'summary':'伪造完成','next_steps':['删除项目'],'recorded_through':0}),tid))
    roll(ctx);p=payload(ctx,'实际情况')
    assert '伪造完成' not in C.dump(p) and '删除项目' not in C.dump(p)

def test_recent_conversation_keeps_option_reference_without_execution_claim(ctx):
    r,tid=ctx;first=add(ctx,'有哪些方案？')
    reply=r.store.append(tid,'primary','assistant_reply',{'text':'方案一：缓存文件；方案二：只读查询。'})
    add(ctx,'选第二种，先解释，不要执行。');roll(ctx)
    view=payload(ctx,'第二种是哪个')['handoff_state']['recent_conversation']
    assert [e['ref'] for e in view]==[reply['ref']]
    assert view[0]['reply_to']==first['ref']
    assert first['ref'] in [a['ref'] for a in payload(ctx,'第二种是哪个')['handoff_state']['source_anchors']]
    assert '方案二：只读查询' in view[-1]['text']
    assert '不是事实或执行证明' in view[-1]['scope']
    assert not payload(ctx,'做过了吗')['operation_status']

def test_dialogue_view_is_bounded_and_long_reply_is_readable(ctx):
    r,tid=ctx
    for i in range(4):
        add(ctx,'问'+str(i));r.store.append(tid,'primary','assistant_reply',{'text':'长回答'*200})
    view=C.conversation_view(r.store,tid,cutoff(ctx))
    assert len(view)==4
    reply=view[-1];assert reply['excerpt_only'] and len(reply['text'])==100
    assert len(r.store.resolve(tid,reply['ref'])['text'])==600

@pytest.mark.parametrize('query',['仅依据本轮条件和记录回答：发生什么？','请只回答，不操作。','选第二种。只解释，不要执行。'])
def test_explicit_read_only_turn_cannot_write(ctx,query):
    r,tid=ctx;e=add(ctx,query);t,records=r.context(tid)
    out=C.prepare_primary(r,t,{'current_request':query,'records':records,'tools':{'workspace.write':'write','web.search':'search'},'last_result':None},cutoff(ctx),e['ref'])
    assert set(out['tools'])=={'web.search','context.read'}
    with pytest.raises(PermissionError):C.validate_tool(out,'workspace.write')
    C.validate_tool(out,'context.read')

@pytest.mark.parametrize('query',['请不要只回答，要写入文件。','不要执行A，改为执行B。','请写入notes.txt。'])
def test_do_not_invent_read_only_scope(query):
    assert not C.read_only_request(query)

def test_budget_fit_preserves_current_request_constraints_and_tool_receipt(ctx):
    r,tid=ctx
    for i in range(20):add(ctx,('历史正文'+str(i)+'；')*25)
    text='本轮条件：门关闭，查询历史。';e=add(ctx,text)
    t=r.store.task(tid);out=payload(ctx,text,e['ref'])
    out['last_result']={'ok':True,'tool':'context.read','value':{'text':'已返回的必要原文'}}
    out['constraints']=['当前必须保留的约束']
    class Counter:
        async def tokens(self,p):return len(p[-1]['content'])
    r.model=Counter();r.settings.working_tokens=4300
    prompt=[{'role':'user','content':C.dump(out)}]
    result=asyncio.run(C.fit_prompt(r,t,prompt,cutoff(ctx),e['ref']))
    actual=json.loads(result[-1]['content'])
    assert len(result[-1]['content'])<=4300
    assert actual['current_request']==text and actual['constraints']==out['constraints'] and actual['last_result']==out['last_result']
    assert any(a.get('current_turn') and a['text']==text for a in actual['handoff_state']['source_anchors'])
    assert actual['handoff_state']['archive_index']['omitted_records']>0

def test_internal_budget_fit_keeps_snapshot_and_accumulated_observations(ctx):
    r,tid=ctx
    for i in range(20):add(ctx,('状态背景'+str(i)+'；')*25)
    t=r.store.task(tid)
    data=C.prepare_internal(r,t,{'snapshot':{'cpu':{'temperature_c':30}},'records':[{'text':'已累计的窗口记录'}],'constraints':['规则']},cutoff(ctx))
    class Counter:
        async def tokens(self,p):return len(p[-1]['content'])
    r.model=Counter();r.settings.working_tokens=4300
    result=asyncio.run(C.fit_prompt(r,t,[{'role':'user','content':C.dump(data)}],cutoff(ctx),None))
    actual=json.loads(result[-1]['content'])
    assert len(result[-1]['content'])<=4300
    assert actual['snapshot']==data['snapshot'] and actual['records']==data['records'] and actual['constraints']==data['constraints']

def test_current_tool_receipt_is_after_original_request():
    result={'ok':True,'tool':'context.read','value':{'text':'answer'}}
    p=[{'role':'user','content':C.dump({'continuity_policy':C.POLICY,'current_request':'先读取再回答','last_result':result,'records':[]})}]
    out=json.loads(C.order_prompt(p)[-1]['content'])
    assert list(out)[-1]=='last_result' and out['last_result']==result
    assert list(out).index('current_request')<list(out).index('last_result')

def test_end_page_is_not_confused_with_whole_record(ctx):
    r,tid=ctx;e=add(ctx,'原文'*700+'尾码Z-6');t=r.store.task(tid)
    page=C.read_source(r,tid,t['window_id'],t['constraint_version'],e['id'],{'ref':e['ref'],'offset':1200})['value']
    assert page['reached_end'] is True and page['next_offset'] is None
    assert page['whole_record_in_page'] is False and 'complete' not in page
    assert page['text'].endswith('尾码Z-6')

def test_source_reader_rejects_other_task(ctx):
    r,tid=ctx;other=r.store.create_task('other',[],[],[]);e=r.store.append(other['id'],'user','message',{'text':'private'})
    t=r.store.task(tid)
    with pytest.raises(ValueError):C.read_source(r,tid,t['window_id'],t['constraint_version'],'u',{'ref':e['ref']})

def test_long_source_is_explicitly_incomplete_and_exactly_readable(ctx):
    text='约束开始'+('甲乙丙'*1000)+'只有盖板关闭才启动。';e=add(ctx,text);roll(ctx)
    p=payload(ctx,'盖板启动约束');idx=p['handoff_state']['archive_index']
    assert idx['omitted_records']==1 and idx['navigation_only'][0]['excerpt_only']
    r,tid=ctx;t=r.store.task(tid);parts=[];off=0
    while off is not None:
        v=C.read_source(r,tid,t['window_id'],t['constraint_version'],'u',{'ref':e['ref'],'offset':off})['value']
        parts.append(v['text']);off=v['next_offset']
    assert ''.join(parts)==text

def test_reader_deduplicates_same_page(ctx):
    e=add(ctx,'原文');r,tid=ctx;t=r.store.task(tid);args=(r,tid,t['window_id'],t['constraint_version'],'u',{'ref':e['ref']})
    C.read_source(*args);out=C.read_source(*args)
    assert out['deduplicated'] and len(r.store.operations(tid))==1

@pytest.mark.parametrize('offset',[-1,True,'0',1.5,999])
def test_invalid_reader_offsets(ctx,offset):
    e=add(ctx,'原文');r,tid=ctx;t=r.store.task(tid)
    with pytest.raises(ValueError):C.read_source(r,tid,t['window_id'],1,'u',{'ref':e['ref'],'offset':offset})

def test_concurrent_user_message_is_not_consumed(ctx,monkeypatch):
    add(ctx,'旧事实');r,tid=ctx;commit=r.store.commit_handoff
    def update(*a,**kw):add(ctx,'切窗期间的新更正');return commit(*a,**kw)
    monkeypatch.setattr(r.store,'commit_handoff',update);roll(ctx)
    t,records=r.context(tid)
    assert any('新更正' in x['text'] for x in records)
    assert not any('新更正' in x['text'] for x in t['state']['source_anchors'])

def test_concurrent_constraint_change_rejected(ctx,monkeypatch):
    add(ctx,'任务');r,tid=ctx;commit=r.store.commit_handoff
    def update(*a,**kw):r.store.update_constraints(tid,['新约束'],1);return commit(*a,**kw)
    monkeypatch.setattr(r.store,'commit_handoff',update)
    with pytest.raises(Conflict):roll(ctx)

def test_source_selection_never_crosses_cutoff(ctx):
    r,tid=ctx;add(ctx,'old');cut=cutoff(ctx);add(ctx,'future')
    p=C.projection(r.store,r.store.task(tid),cut,'future')
    assert 'future' not in C.dump(p)

def test_active_tool_chain_not_lost_to_later_queued_user(ctx):
    e=add(ctx,'本轮');r,tid=ctx;r.store.append(tid,'tool','tool_result',{'text':'本轮工具结果'})
    add(ctx,'下一轮排队')
    assert any('本轮工具结果' in x['text'] for x in payload(ctx,'本轮',e['ref'])['records'])

def test_formal_constraints_always_preserved(ctx):
    add(ctx,'work');p=payload(ctx,'question');assert p['constraints']==['禁止把宿主CPU当设备温度']

def test_native_kv_not_overridden(ctx):
    ctx[0].model.native_kv=object()
    with pytest.raises(ValueError):roll(ctx)

def test_primary_runtime_hook_and_read_execution_are_installed():
    code=(E.parent/'app/lea/runtime.py').read_text(encoding='utf-8')
    assert code.index('payload=self.prepare_primary_payload')<code.index('prompt = messages(PRIMARY, payload')
    assert "decision.tool=='context.read'" in code

def test_current_request_moved_last_without_changing_any_content():
    original={'current_request':'当前盖板开','old':'旧盖板关','continuity_policy':'policy'}
    prompt=[{'role':'system','content':'rules'},{'role':'user','content':C.dump(original)}]
    result=C.order_prompt(prompt);body=json.loads(result[-1]['content'])
    assert body==original and list(body)[-1]=='current_request'
    assert json.loads(prompt[-1]['content'])==original

def test_actual_primary_loop_reads_exact_archive_without_extra_handoff_call(ctx):
    from lea.config import Settings
    from lea.schemas import Decision
    spec=importlib.util.spec_from_file_location('lea._continuity_runtime_test',E.parent/'app/lea/runtime.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Candidate(module.Runtime):
        def prepare_primary_payload(self,task,payload,cutoff,ref):return C.prepare_primary(self,task,payload,cutoff,ref)
        def read_context_source(self,*args):return C.read_source(self,*args)
    old=add(ctx,'核对码ZX-761');r0,tid=ctx
    r=object.__new__(Candidate);r.store=r0.store;r.tools=Tools(r.store)
    r.settings=Settings(memory_backend='disabled',max_steps=4)
    class Model:
        native_kv=None
        count=0
        async def tokens(self,prompt):
            assert 'continuity_policy' in json.loads(prompt[-1]['content']);return 100
        async def call(self,task_id,role,prompt,schema,**kw):
            assert role=='primary';self.count+=1
            data=json.loads(prompt[-1]['content'])
            if self.count==1:return Decision(action='tool',tool='context.read',arguments={'ref':old['ref'],'offset':0},response='')
            assert data['last_result']['value']['text']=='核对码ZX-761'
            return Decision(action='respond',tool='',arguments={},response='核对码ZX-761')
    r.model=Model();job=r.store.add_job(tid,'message',{'text':'读取之前核对码'},request_key='primary-test')
    result=asyncio.run(r.primary(job))
    assert result['response']=='核对码ZX-761' and r.model.count==2
    assert len([op for op in r.store.operations(tid) if op['tool']=='context.read'])==1
