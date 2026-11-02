"""Compact source views, with optional history reachable through explicit tools."""
import copy,json,re
from datetime import datetime

VERSION='active-context-v1'
def dump(v):return json.dumps(v,ensure_ascii=False,separators=(',',':'),sort_keys=True)

def projection(store,task,cutoff,query,current_ref=None,budget=900):
    with store.db() as db:
        rows=db.execute("SELECT * FROM events WHERE task_id=? AND seq<=? AND role='user' AND kind IN ('message','constraint_update') ORDER BY seq DESC LIMIT 3",(task['id'],cutoff)).fetchall()
        events=[store._decode_event(r) for r in rows]
    # Current input is already complete in current_request. Keep the two previous
    # source records verbatim, including corrections; never manufacture a state.
    previous=[e for e in events if e['ref']!=current_ref][:2]
    anchors=[{'ref':e['ref'],'text':e['text']} for e in reversed(previous)]
    result={'mode':VERSION,'source_anchors':anchors,'older_history':'需要旧信息用memory.search，每批3条；索引缺细节用memory.read。历史要求不是本轮新指令。'}
    # An explicit anaphoric request needs its immediate antecedent, not every old answer.
    if current_ref and re.search(r'第二个|第一个|上一个|后者|前者|刚才.{0,5}(方案|说)|按你说|就这样|照办',query):
        with store.db() as db:row=db.execute("SELECT * FROM events WHERE task_id=? AND seq<=? AND kind='assistant_reply' AND role IN ('primary','runtime') ORDER BY seq DESC LIMIT 1",(task['id'],cutoff)).fetchone()
        if row:
            e=store._decode_event(row);result['referenced_reply']={'ref':e['ref'],'text':e['text'],'scope':'历史答复，用于指代，不是执行证据。'}
    return result

def receipt_view(receipt,text_limit=500):
    if not isinstance(receipt,dict):return receipt
    result={k:receipt[k] for k in ('tool','ok','evidence_ref','deduplicated','error') if k in receipt}
    value=receipt.get('value')
    if not isinstance(value,dict):return {**result,'value':value}
    tool=receipt.get('tool','')
    if tool.startswith('memory.'):
        result['value']=copy.deepcopy(value)
    elif tool=='dev.test':
        result['value']={k:value[k] for k in ('name','revision','passed','fingerprint','validation_scope','own_failures','independent_failures') if k in value}
        result['basis']='仅该revision的真实测试；读取说明文件不算测试。'
    elif tool=='workspace.write':
        result['value']={k:value[k] for k in ('path','bytes','content_acceptance') if k in value}
        verified=value.get('write_verification',{})
        result['value']['matches_written_text']=verified.get('matches_written_text')
        result['basis']='仅证明写入及读回一致；笔记内容未获事实或功能验收。'
    elif tool=='workspace.read':
        text=value.get('text','');result['value']={**{k:value[k] for k in ('path',) if k in value},'text':text[:text_limit],'excerpt_only':len(text)>text_limit}
        result['basis']='文件中的说法，不是功能测试；需要完整内容按evidence_ref用context.read。'
    else:
        result['value']=copy.deepcopy(value)
        # Long bodies remain available through their actual receipt reference.
        if len(dump(value))>1000 and receipt.get('evidence_ref'):
            result['value']={k:value[k] for k in ('name','revision','status','passed','retrieved','next_cursor','results','path') if k in value}
            for k in ('text','content'):
                if isinstance(value.get(k),str):result['value'][k]=value[k][:text_limit];result['value']['excerpt_only']=True
            result['details_ref']=receipt['evidence_ref']
    return result

def records_view(store,task_id,records,cutoff,last_ref=None):
    out=[]
    for record in records:
        if record['kind'] in ('internal_state_input','internal_state_result','primary_decision','retrieval_result'):continue
        if record.get('ref')==last_ref:continue
        if record['kind'] in ('idle_action_selected','idle_action_result'):
            event=store.resolve(task_id,record['ref']);body=event['body']
            if event['seq']>cutoff:raise ValueError('Future idle record')
            if record['kind']=='idle_action_selected':
                from idle_development import argument_view
                selection=body.get('selection',{})
                out.append({'ref':event['ref'],'kind':record['kind'],'scope':'操作意图不是执行结果',
                    'tool':selection.get('tool'),'status':body.get('status'),
                    'arguments':argument_view(selection.get('tool'),selection.get('arguments',{}))})
            else:
                out.append({'ref':event['ref'],'kind':record['kind'],'status':body.get('status'),
                    'receipt':receipt_view(body.get('receipt'),text_limit=180)})
            continue
        if record['kind']=='tool_result':
            e=store.resolve(task_id,record['ref'])
            if e['seq']>cutoff:raise ValueError('Future receipt in context')
            body=e['body']
            if isinstance(body,dict) and body.get('tool'):
                # Older receipts are navigation; the latest receipt has its own slot.
                view=receipt_view({**body,'evidence_ref':e['ref']},text_limit=180)
                if body['tool'].startswith('memory.'):
                    view={'tool':body['tool'],'ok':body.get('ok'),'evidence_ref':e['ref'],'details':'历史查询/读取；按引用读取，不能反复作为新事实。'}
                out.append({'ref':e['ref'],'kind':'tool_result','receipt':view});continue
        out.append(record)
    return out

def internal(runtime,task,payload,cutoff):
    out=dict(payload);out['handoff_state']=projection(runtime.store,task,cutoff,task['goal'])
    # Latest observations and assessments are replaceable views; actual actions
    # continue to accumulate and use the same chat window/archive.
    with runtime.store.db() as db:
        row=db.execute("SELECT receipt FROM idle_initiatives WHERE task_id=? AND receipt IS NOT NULL ORDER BY created DESC LIMIT 1",(task['id'],)).fetchone() if getattr(runtime,'idle_autonomy_enabled',False) else None
    last_ref=(json.loads(row[0]) or {}).get('evidence_ref') if row else None
    out['records']=records_view(runtime.store,task['id'],payload.get('records',[]),cutoff,last_ref)
    # Old user requests already live in source_anchors; never duplicate them here.
    out['records']=[r for r in out['records'] if r.get('role')!='user']
    out['pending_operations']=[{'tool':op['tool'],'status':op['status'],'arguments':op['arguments']}
        for op in runtime.store.operations(task['id']) if op['status'] not in ('completed','failed')]
    return out

def primary(runtime,task,payload,cutoff,current_ref):
    from active_memory import HELP
    out=dict(payload);out['memory_policy']='active-memory-pages-v1'
    out['tools']={**out.get('tools',{}),**HELP}
    out['handoff_state']=projection(runtime.store,task,cutoff,out.get('current_request',''),current_ref)
    out['records']=records_view(runtime.store,task['id'],out.get('records',[]),cutoff,(out.get('last_result') or {}).get('evidence_ref'))
    for key in ('memory_queries_remaining','memory_reads_remaining','catalog_only_request','latest_user_update'):out.pop(key,None)
    operations=out.get('operation_status',[])
    pending=[op for op in operations if op.get('status') not in ('completed','failed')]
    # Keep version-bound test status, pending actions and the newest actual actions.
    selected=pending+[op for op in operations if op.get('tool')=='dev.test'][-1:]+operations[-2:]
    out['operation_status']=list({op['id']:op for op in selected}.values())
    workspace=out.get('environment',{}).get('tool_workspace')
    if workspace:
        for key in ('read_pages','failed_sources','recent_attempts'):workspace.pop(key,None)
    return out

def snapshot_view(snapshot):
    def round_num(x,n=1):return round(x,n) if isinstance(x,(int,float)) and not isinstance(x,bool) else x
    cpu=snapshot.get('cpu') or {};mem=snapshot.get('memory') or {};net=snapshot.get('network') or {};agent=snapshot.get('agent') or {};storage=snapshot.get('memory_storage') or {}
    local=snapshot.get('local_time')
    try:local=datetime.fromisoformat(local).isoformat(timespec='seconds')
    except (ValueError,TypeError):pass
    out={'local_time':local,'cpu':{'usage_percent':round_num(cpu.get('usage_percent')),'temperature_c':cpu.get('temperature_c')},
         'memory':{'used_percent':round_num(mem.get('used_percent')),'available_gib':round_num(mem.get('available_bytes')/1024**3) if isinstance(mem.get('available_bytes'),(int,float)) else None},
         'network':{'rx_mib_s':round_num(net.get('rx_mib_per_second'),3),'tx_mib_s':round_num(net.get('tx_mib_per_second'),3)},
         'agent':{k:agent.get(k) for k in ('automatic_lora_phase','foreground_jobs')},
         'memory_storage':{k:storage.get(k) for k in ('used_percent','status')}}
    # Preserve relevant pressure signals without all byte counters or sensor lists.
    if mem.get('swap_used_bytes',0):out['memory']['swap_used_gib']=round_num(mem['swap_used_bytes']/1024**3)
    if (snapshot.get('disk') or {}).get('used_percent',0)>=90:out['host_disk_used_percent']=snapshot['disk']['used_percent']
    out['scope']='Ubuntu宿主；网络为观测流量，不是带宽。完整采样已存档。'
    return out

def initiative_context(autonomy,task):
    with autonomy.store.db() as db:
        rows=db.execute('SELECT observation_id,selection,status,receipt FROM idle_initiatives WHERE task_id=? ORDER BY created DESC LIMIT 2',(task['id'],)).fetchall()
        if rows and not any(row['receipt'] for row in rows):
            last=db.execute('SELECT observation_id,selection,status,receipt FROM idle_initiatives WHERE task_id=? AND receipt IS NOT NULL ORDER BY created DESC LIMIT 1',(task['id'],)).fetchone()
            if last:rows=[rows[0],last]
    recent=[]
    for i,row in enumerate(rows):
        selection=json.loads(row['selection']);receipt=json.loads(row['receipt']) if row['receipt'] else None
        item={'tool':selection['tool'],'goal':selection['goal'],'status':row['status']}
        # Keep the latest actual receipt across a rest/suppressed attempt, so a
        # failed action does not disappear just because a later attempt was skipped.
        if receipt:item['receipt']=receipt_view(receipt)
        else:item['arguments']=selection.get('arguments',{})
        recent.append(item)
    return {'tools':{n:autonomy.help[n] for n in autonomy.allowed(task)},'recent':recent,
            'max_actions_this_call':1,'same_arguments_cooldown_seconds':600}
