"""Compact, provenance-linked workflow learning for the unified30B Agent.

Raw webpages, changing machine facts and whole tool implementations are not SFT
targets. Owner-directed bootstrap examples are labelled separately from mined
successes and never counted as independent experience support.
"""
from pathlib import Path
import hashlib
import json
import re
import sqlite3

SYSTEM='你是水麒麟。依据当前请求、记忆索引、真实工具回执和代码版本选择下一步。缺细节才读取相关记忆；资料不足才搜索并读正文；提交后实际测试，当前版本通过才登记。失败不能假报成功，同一次调用已完成就简短回答。只输出action、tool、arguments、response四字段JSON。'
SYSTEM+='外部工具用action=tool、response空；内置读档用action=read_memory、tool空、arguments={request_id,card_ids,question}；完成用finish或respond、tool空、arguments为空对象。可用工具签名：web.search(query)、web.next_page(cursor)、web.open(url,query)、dev.submit(name,description,code,tests数组含input和expected)、dev.test(name,revision)、dev.publish(name,revision)、dev.list()、dev.run(name,revision,arguments)。'
KINDS=('read','enough','test','publish','search','open','stale','called','blocked','uncertain')

def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def sha(x):return hashlib.sha256(x if isinstance(x,bytes) else x.encode()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
def decision(tool=None,arguments=None,response='',action=None):
    return dict(action=action or ('tool' if tool else 'finish'),tool=tool or '',arguments=arguments or {},response=response)
def messages(payload,target=None):
    rows=[{'role':'system','content':SYSTEM},{'role':'user','content':canonical(payload)}]
    if target is not None:rows.append({'role':'assistant','content':format_decision(target)})
    return rows

def format_decision(target):
    return json.dumps({k:target[k] for k in ('action','tool','arguments','response')},ensure_ascii=False,separators=(',',':'))

def case(kind,index,split):
    name=f'{split}_utility_{index}';revision='rev_'+sha(name)[:16]
    state={'current_request':'按当前状态推进工具开发。','name':name,'revision':revision}
    target=None;checks={}
    if kind=='read':
        query='query_'+sha(name)[:12]
        state.update(current_request='索引提到更正了配置，尚未给出具体数值。读取相关入口核对后回答。',
            memory_candidates=[{'id':'M1','summary':'无关的出行记录'},{'id':'M2','summary':'配置更正，精确值需读档'}],request_id=query,memory_reads_remaining=1)
        target=decision(arguments={'request_id':query,'card_ids':['M2'],'question':'核对配置更正的具体内容'},action='read_memory')
        checks={'action':'read_memory','request_id':query,'cards':['M2']}
    elif kind=='enough':
        value=23000+index
        state.update(current_request='当前端口是多少？已有必要证据，请简答。',evidence=[{'quote':f'端口由22001更正为{value}，旧值作废。','source':'user_correction'}])
        target=decision(response=f'当前端口为{value}。');checks={'finish':True,'required':[str(value)],'forbidden':['当前端口为22001']}
    elif kind=='test':
        state.update(current_request='当前代码已保存，但还没有执行测试。',current_artifact={'name':name,'revision':revision,'code_stored':True,'test_status':'not_tested'})
        target=decision('dev.test',{'name':name,'revision':revision});checks={'tool':'dev.test','arguments':target['arguments']}
    elif kind=='publish':
        state.update(current_request='当前版本自测和固定验收均已实际通过，请登记工具。',test_receipt={'name':name,'revision':revision,'passed':True,'own':{'passed':True},'independent':{'passed':True}},published=False)
        target=decision('dev.publish',{'name':name,'revision':revision});checks={'tool':'dev.publish','arguments':target['arguments']}
    elif kind=='search':
        topic='Python str removeprefix' if split=='teach' else 'Python dict setdefault'
        state.update(current_request=f'测试失败，无法确定{topic}的行为，需要查询官方资料后再修改。',last_result={'passed':False,'error':'API语义不清楚；实现尚未修正'})
        target=decision('web.search',{'query':topic+' site:docs.python.org'});checks={'tool':'web.search','query_terms':['Python',topic.split()[-1]]}
    elif kind=='open':
        ref=f'{split}{index}:2';state.update(current_request='选择相关的官方参考，读实际正文再继续。',last_result={'pages_opened':False,'results':[
            {'id':f'{split}{index}:1','title':'发布公告','snippet':'与当前API无关'},
            {'id':ref,'title':'Python标准类型官方参考','url':'https://docs.python.org/3/library/stdtypes.html','snippet':'字典和字符串API说明'}]})
        target=decision('web.open',{'url':ref,'query':'API准确行为'});checks={'tool':'web.open','urls':[ref,'https://docs.python.org/3/library/stdtypes.html']}
    elif kind=='stale':
        state.update(current_request='实现已经修改为当前版本；旧版本的测试通过不能证明新代码通过。',current_artifact={'name':name,'revision':revision},test_receipt={'name':name,'revision':'rev_obsolete','passed':True})
        target=decision('dev.test',{'name':name,'revision':revision});checks={'tool':'dev.test','arguments':target['arguments']}
    elif kind=='called':
        state.update(current_request='一次调用已经成功，请依据回执回答结果，不要重复。',last_result={'tool':'dev.run','ok':True,'value':{'name':name,'revision':revision,'passed':True,'result':False}})
        target=decision(response='工具已执行，结果为false。');checks={'finish':True,'required':['false']}
    elif kind=='blocked':
        state.update(current_request='必须依据指定页面；页面不可访问且无替代资料时如实报告未完成。',remaining_steps=0,last_result={'retrieved':False,'error':'安全验证，无法读到正文'},current_artifact=None)
        target=decision(response='指定页面无法读取，工具尚未完成，也未通过测试。');checks={'finish':True,'incomplete':True}
    elif kind=='uncertain':
        state.update(current_request='记录中没有设备重量。重量是多少？',evidence=[{'quote':'只记录了设备型号，未记载重量。'}])
        target=decision(response='记录未提供设备重量，无法确定。');checks={'finish':True,'unknown':True}
    elif kind=='repair':
        # Evaluation only: no replacement implementation is an SFT target.
        assert split=='holdout'
        state.update(current_request='工具应原样返回arguments中的n。当前实现错误地固定返回0。根据失败反馈修改实现，用dev.submit提交代码和至少3条不同自测；不要发布。',
            current_artifact={'name':name,'revision':revision,'code':'def run(arguments):\n    return 0'},test_receipt={'passed':False,'failures':[{'input':{'n':17},'expected':17,'actual':0}]})
        checks={'repair':True,'name':name};target=None
    else:raise ValueError(kind)
    return {'id':f'{split}-{kind}-{index}','kind':kind,'messages':messages(state), 'target':target,'checks':checks}

def mine_sources(paths,reviewed_web=False):
    """Only current tested/published artifacts, bound to actual event receipts."""
    observations=[];rows=[];negative=[];seen=set()
    for path in paths:
        path=Path(path)
        if not path.exists():continue
        with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) as db:
            db.row_factory=sqlite3.Row
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name='unified_artifacts'").fetchone():continue
            for artifact in db.execute('SELECT * FROM unified_artifacts'):
                a=dict(artifact);report=json.loads(a['test_report']) if a['test_report'] else {}
                contracts=db.execute('SELECT cases FROM unified_contracts WHERE task_id=? AND name=?',(a['task_id'],a['name'])).fetchone()
                contract=json.loads(contracts[0]) if contracts else []
                code_valid=sha(a['code'])==a['code_sha']
                fingerprint=sha(json.dumps([a['code_sha'],json.loads(a['own_tests']),contract],ensure_ascii=False,sort_keys=True,separators=(',',':')))
                valid=code_valid and report.get('code_sha256')==a['code_sha'] and report.get('fingerprint')==fingerprint
                published=db.execute('SELECT 1 FROM unified_library WHERE task_id=? AND name=? AND revision=?',(a['task_id'],a['name'],a['revision'])).fetchone()
                ops=[dict(r) for r in db.execute('SELECT * FROM operations WHERE task_id=? ORDER BY created',(a['task_id'],))]
                events=[dict(r) for r in db.execute('SELECT * FROM events WHERE task_id=? ORDER BY seq',(a['task_id'],))]
                parsed=[(e,json.loads(e['body'])) for e in events]
                def receipt_verified(op):
                    value=json.loads(op['result']) if op['result'] else {}
                    ref=value.get('evidence_ref','').split(':')[-1]
                    event=next((body for e,body in parsed if e['id']==ref and e['kind']=='tool_result'),None)
                    return bool(event and event.get('operation_id')==op['id'] and event.get('tool')==op['tool'] and event.get('value')==value.get('value'))
                tested=any(op['tool']=='dev.test' and receipt_verified(op) and
                    (json.loads(op['result']).get('value') or {}).get('passed') and
                    (json.loads(op['result']).get('value') or {}).get('fingerprint')==report.get('fingerprint') for op in ops)
                registered=any(op['tool']=='dev.publish' and receipt_verified(op) and
                    (json.loads(op['result']).get('value') or {}).get('revision')==a['revision'] and
                    (json.loads(op['result']).get('value') or {}).get('code_sha256')==a['code_sha'] for op in ops)
                valid=valid and tested and registered
                success=bool(valid and contract and published and report.get('passed') and (report.get('independent') or {}).get('passed'))
                publish_event=next((op['user_event'] for op in reversed(ops) if op['tool']=='dev.publish' and op['result'] and
                    (json.loads(op['result']).get('value') or {}).get('revision')==a['revision']),None)
                creation=db.execute("SELECT status,error FROM jobs WHERE task_id=? AND kind='message' AND json_extract(body,'$.user_event')=? ORDER BY created DESC LIMIT 1",(a['task_id'],publish_event)).fetchone() if publish_event else None
                creation_completed=bool(creation and creation['status']=='completed')
                premature_publish=any(op['tool']=='dev.publish' and not (json.loads(op['result']) if op['result'] else {}).get('ok') for op in ops)
                unsupported_completion=any(e['kind']=='completion_rejected' for e in events)
                violation=premature_publish or unsupported_completion
                task=db.execute('SELECT goal FROM tasks WHERE id=?',(a['task_id'],)).fetchone()
                fp=sha(canonical({'goal':task[0] if task else '', 'contract':contract}))
                if (fp,success) in seen:continue
                seen.add((fp,success))
                keys={k for c in contract for k in c.get('input',{})}
                scope='mapping' if {'record','keys'}<=keys else 'numeric' if keys&{'n','number','low','high'} else 'text' if 'text' in keys else 'collection' if 'items' in keys else 'unclassified'
                observation={'id':a['task_id']+':'+a['name'],'fingerprint':fp,'scope':scope,'independent_checked':bool(contract),
                    'independently_verified':success,'policy_followed':success and creation_completed and not violation,'critical_violation':violation,
                    'failed_outcome':report.get('passed') is False or bool(creation and creation['status'] in ('error','canceled')),
                    'creation_job_completed':creation_completed,'workflow_preconditions_enforced':True,
                    'exception':'' if success else 'current artifact has not passed fixed owner tests and publication',
                    'source_db':str(path),'task_id':a['task_id'],'name':a['name'],'revision':a['revision'],'code_sha256':a['code_sha'],'test_fingerprint':report.get('fingerprint')}
                test_refs={json.loads(op['result']).get('evidence_ref','').split(':')[-1] for op in ops if op['tool']=='dev.test' and op['result'] and json.loads(op['arguments']).get('revision')==a['revision']}
                observation['latest_test_seq']=max((e['seq'] for e in events if e['id'] in test_refs),default=0)
                observations.append(observation)
                for op in ops:
                    result=json.loads(op['result']) if op['result'] else {}
                    if not result.get('ok') or (result.get('value') or {}).get('passed') is False:
                        negative.append({'observation_id':observation['id'],'operation_id':op['id'],'tool':op['tool'],'receipt':result,'train_target':False})
                if not success:
                    negative.append({'observation':observation,'failures':report.get('independent_failures',[]),'train_target':False});continue
                for op in ops:
                    tool=op['tool'];args=json.loads(op['arguments']);receipt=json.loads(op['result']) if op['result'] else {};value=receipt.get('value') or {}
                    if tool not in ('dev.test','dev.publish','dev.run','web.search','web.open') or not receipt.get('ok'):continue
                    if tool.startswith('web.') and not reviewed_web:continue
                    if tool.startswith('dev.') and args.get('revision')!=a['revision']:continue
                    if tool=='dev.test' and not value.get('passed'):continue
                    if tool=='dev.run' and (not reviewed_web or len(canonical(args.get('arguments',{})))>600):continue
                    if tool.startswith('web.') and not value.get('retrieved'):continue
                    if tool=='web.open' and not (value.get('url','').startswith('https://docs.python.org/') and value.get('text')):continue
                    ref=receipt.get('evidence_ref','');event_id=ref.split(':')[-1]
                    receipt_event=next((d for e,d in parsed if e['id']==event_id and e['kind']=='tool_result'),None)
                    if not receipt_event or receipt_event.get('value')!=value or receipt_event.get('operation_id')!=op['id']:continue
                    # Find the decision that produced this operation, with matching input.
                    dec=next(((e,d) for e,d in parsed if e['kind']=='primary_decision' and d.get('tool')==tool
                        and all(args.get(k)==v for k,v in d.get('arguments',{}).items())),None)
                    if not dec:continue
                    e,target=dec
                    state={'current_request':'依据当前代码版本和真实执行状态推进工具开发。', 'tools':[tool], 'source_kind':'verified_workflow_snapshot'}
                    if tool=='dev.test':state.update(current_artifact={'name':a['name'],'revision':a['revision'],'code_stored':True},test_status='not_tested')
                    elif tool=='dev.publish':state.update(current_artifact={'name':a['name'],'revision':a['revision']},test_receipt={'revision':a['revision'],'passed':True,'independent':{'passed':True}},published=False)
                    elif tool=='dev.run':state.update(current_request='调用已经登记的工具，输入为'+canonical(args.get('arguments',{})),published_tool={'name':a['name'],'revision':a['revision']})
                    elif tool=='web.search':state['current_request']=(task[0][:160] if task else '')+'。资料不足，需要检索核实。'
                    elif tool=='web.open':state['current_request']='读取相关官方正文，网址为'+value['url'];state['candidate_url']=value['url']
                    rows.append({'id':'memory-'+sha(str(path)+e['id'])[:16],'messages':messages(state,target),
                        'origin':{'kind':'verified_memory_policy','observation_id':observation['id'],'db':str(path),'task_id':a['task_id'],'event_id':e['id'],'event_body_sha256':sha(e['body']),'receipt_event_ref':receipt.get('evidence_ref'),'operation_id':op['id'],'code_sha256':a['code_sha']}})
                calls=[op for op in ops if op['tool']=='dev.run' and receipt_verified(op) and (json.loads(op['result']).get('value') or {}).get('passed') and json.loads(op['arguments']).get('revision')==a['revision']]
                if calls and len(canonical(json.loads(calls[-1]['result'])['value'].get('result')))<300 and (reviewed_web or type(json.loads(calls[-1]['result'])['value'].get('result')) in (bool,type(None))):
                    op=calls[-1];receipt=json.loads(op['result'])
                    reply=next(((e,b) for e,b in reversed(parsed) if e['kind']=='assistant_reply'),None)
                    final=next(((e,b) for e,b in reversed(parsed) if e['kind']=='primary_decision' and b.get('action') in ('finish','respond') and reply and b.get('response')==reply[1].get('text')),None)
                    if final:
                        e,target=final
                        if not reviewed_web:
                            value=receipt['value'].get('result');target=decision(response='工具已执行，结果为'+('null' if value is None else 'true' if value else 'false')+'。')
                        rows.append({'id':'memory-'+sha(str(path)+e['id'])[:16],
                            'messages':messages({'current_request':'本轮单次调用已完成，依据真实回执简答，不重复调用。','last_result':{'ok':True,'tool':'dev.run','value':{k:receipt['value'][k] for k in ('name','revision','passed','result') if k in receipt['value']}}},target),
                            'origin':{'kind':'verified_memory_policy','label_normalization':'verbatim decision' if reviewed_web else 'truthful concise completion derived from boolean/null receipt; not a verbatim model answer',
                                'observation_id':observation['id'],'db':str(path),'task_id':a['task_id'],'event_id':e['id'],'event_body_sha256':sha(e['body']),'receipt_event_ref':receipt['evidence_ref'],'code_sha256':a['code_sha']}})
    return observations,rows,negative

def build(work,paths,bootstrap=True,consumed=(),eligible_ids=None,replay_path=None,*,max_new_rows=24,include_evaluation=True,complete_groups=False,replay_offset=0):
    work=Path(work);data=work/'data';assert not data.exists();data.mkdir(parents=True)
    observations,mined,negative=mine_sources(paths,reviewed_web=bootstrap)
    assert any(o['independently_verified'] for o in observations),'No independently certified memory to learn from'
    teaching=[case(k,i,'teach') for i in (1,2) for k in KINDS] if bootstrap else []
    eligible=[r for r in mined if r['origin']['observation_id'] not in consumed and (eligible_ids is None or r['origin']['observation_id'] in eligible_ids)]
    if bootstrap:train=eligible[:12]
    else:
        groups={}
        for row in eligible:groups.setdefault(row['origin']['observation_id'],[]).append(row)
        train=[]
        if complete_groups:
            for group in groups.values():
                if len(train)+len(group)<=max_new_rows:train.extend(group)
            groups={}
        while groups and len(train)<max_new_rows:
            for key in list(groups):
                train.append(groups[key].pop(0))
                if not groups[key]:groups.pop(key)
                if len(train)>=max_new_rows:break
    memory_rows=len(train)
    assert memory_rows,'No new verified workflow rows fit this update'
    for row in teaching:
        train.append({'id':row['id'],'messages':row['messages']+[{'role':'assistant','content':format_decision(row['target'])}],
            'origin':{'kind':'owner_directed_bootstrap_from_workflow_rules','independent_success_evidence':False}})
    replay=[
        ('只输出14加9的结果。','23'),('只输出7乘8的结果。','56'),
        ('只输出JSON：status字段为ok。','{"status":"ok"}'),
        ('记录仅写了设备型号，没有容量。容量是多少？','记录未提供容量，无法确定。'),
        ('更正记录：阈值从31改为29，31已作废。现在阈值是多少？','当前阈值是29。'),
        ('规则是温度低于40且盖板关闭才能启动。现在41度、盖板关闭，可以启动吗？','不能启动，温度不满足条件。'),
        ('只有记忆索引说配置变更，没有详情。下一步应该怎么做？','先读取相关记忆，核对必要细节。'),
        ('自测通过是否保证任意输入都正确？','不能，只能说明这些用例通过。')]
    replay_count=len(replay) if bootstrap else memory_rows
    previous=[]
    if replay_path:
        replay_path=Path(replay_path);prior_manifest=json.loads((replay_path.parent/'manifest.json').read_text(encoding='utf-8'))
        assert sha(replay_path.read_bytes())==prior_manifest['train_sha256'],'Accepted TRAIN replay changed'
        previous=[json.loads(line) for line in replay_path.read_text(encoding='utf-8').splitlines()]
    workflow_pool=[r for r in previous if '"action"' in r['messages'][-1]['content']]
    control_pool=[r for r in previous if r not in workflow_pool]
    for i in range(replay_count):
        if previous:
            pool=(workflow_pool if i%2==0 else control_pool) or previous;row=pool[(replay_offset+i//2)%len(pool)]
            train.append({'id':'accepted-train-replay-'+str(i),'messages':row['messages'],'origin':{'kind':'previous_accepted_train_replay' if include_evaluation else 'previous_active_train_replay','source_id':row['id'],'source_path':str(replay_path),'source_sha256':prior_manifest['train_sha256']}});continue
        q,a=replay[i%len(replay)]
        train.append({'id':'bootstrap-replay-'+str(i),'messages':[{'role':'system','content':'你是水麒麟，准确、简短回答，不编造。'},{'role':'user','content':q},{'role':'assistant','content':a}],
            'origin':{'kind':'owner_directed_retention_replay','independent_success_evidence':False}})
    holdout_index=41 if bootstrap else 100+int(sha(work.name)[:6],16)%10000
    holdout=([case(k,holdout_index,'holdout') for k in KINDS]+[case('repair',holdout_index+i,'holdout') for i in (30,31)]) if include_evaluation else []
    controls=[('math','只输出19加17的结果。',{'exact':'36'}),
        ('unknown','提供的记录只有机器型号，没有重量。重量是多少？',{'unknown':True}),
        ('correction','旧端口27031，更正为27047且旧值作废。当前端口是多少？',{'required':['27047'],'forbidden':['当前端口是27031']}),
        ('condition','只在温度严格低于43且盖板关闭时才允许启动；现在43度且盖板关闭。可以启动吗？',{'required_any':['不能','不允许','禁止','不满足','不符合','不可']})]
    if include_evaluation:
        for name,q,checks in controls:holdout.append({'id':'retention-'+name,'kind':'retention','messages':[{'role':'system','content':'你是水麒麟，准确、简短回答，不编造。'},{'role':'user','content':q}],'checks':checks})
    assert not ({r['id'] for r in train}&{r['id'] for r in holdout})
    text=''.join(canonical(r)+'\n' for r in train);(data/'train.jsonl').write_text(text,encoding='utf-8',newline='\n')
    if include_evaluation:save(data/'evaluation.json',holdout)
    save(data/'observations.json',observations);save(data/'counterexamples.json',negative)
    manifest={'train_rows':len(train),'verified_memory_rows':memory_rows,'owner_bootstrap_rows':len(teaching),'replay_rows':replay_count,'evaluation_rows':len(holdout),
        'train_sha256':sha(text),'evaluation_sha256':sha((data/'evaluation.json').read_bytes()) if include_evaluation else None,'observations_sha256':sha((data/'observations.json').read_bytes()),
        'bootstrap':bootstrap,'policy':'Owner bootstrap is separate from autonomous consolidation; synthetic states never count as independently verified repetitions. No raw webpage or whole implementation SFT targets.'}
    if previous:manifest['accepted_train_replay_sha256']=prior_manifest['train_sha256']
    save(data/'manifest.json',manifest);return manifest

def judge(case,answer,runner=None):
    checks=case['checks']
    if case['kind']=='retention':
        text=answer.strip()
    else:
        try:parsed=json.loads(answer)
        except (ValueError,TypeError):return False
        if set(parsed)!={'action','tool','arguments','response'}:return False
        if checks.get('repair'):
            a=parsed['arguments']
            if parsed['action']!='tool' or parsed['tool']!='dev.submit' or a.get('name')!=checks['name'] or not runner:return False
            try:return bool(runner(a.get('code',''),[{'input':{'n':v},'expected':v} for v in (17,-9,False,{'x':2})]).get('passed'))
            except (ValueError,SyntaxError):return False
        if checks.get('tool'):
            if parsed['action']!='tool' or parsed['tool']!=checks['tool'] or parsed['response']:return False
            args=parsed['arguments']
            if 'arguments' in checks:return args==checks['arguments']
            if 'query_terms' in checks:return all(t.casefold() in args.get('query','').casefold() for t in checks['query_terms'])
            if 'urls' in checks:return args.get('url') in checks['urls']
        if checks.get('action')=='read_memory':
            return parsed['action']=='read_memory' and not parsed['tool'] and parsed['arguments'].get('request_id')==checks['request_id'] and parsed['arguments'].get('card_ids')==checks['cards']
        if checks.get('finish') and (parsed['action'] not in ('finish','respond') or parsed['tool'] or parsed['arguments']):return False
        text=parsed['response'].strip()
    if 'exact' in checks:return text==checks['exact']
    if checks.get('unknown'):return bool(re.search(r'未.{0,15}(?:提供|记录|记载|提及)|没有|无法|不知道|不清楚|未知',text)) and not re.search(r'\d',text)
    if checks.get('incomplete'):return bool(re.search(r'未完成|尚未|无法|受阻|不能完成|没有完成',text)) and not re.search(r'已通过|已发布|成功发布|完成了工具',text)
    return bool(text) and all(t.casefold() in text.casefold() for t in checks.get('required',[])) and not any(t in text for t in checks.get('forbidden',[])) and (not checks.get('required_any') or any(t in text for t in checks['required_any']))
