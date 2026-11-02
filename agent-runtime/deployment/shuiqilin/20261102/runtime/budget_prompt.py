"""Count the final model prompt, and avoid duplicating version-bound tool evidence."""
import copy,json,re

NOTICE='当前测试失败。按此真实反例修正代码，禁止原样重提；无法修正就报告尚未完成：'

def prepare(prompt):
    from unified_schema import development_request
    if any(x.get('role')=='system' and x.get('content','').startswith(NOTICE) for x in prompt):return prompt
    payload=json.loads(prompt[-1]['content'])
    if not development_request(payload.get('current_request','')):return prompt
    failed=[]
    for a in payload.get('environment',{}).get('tool_workspace',{}).get('current_artifacts',[]):
        status=a.get('test_status')
        if not isinstance(status,dict) or status.get('passed') is not False:continue
        if status.get('details_from')=='last_result.value':status=payload['last_result']['value']
        examples=status.get('independent_failures') or status.get('own_failures') or []
        if examples:failed.append({'name':a['name'],'revision':a['revision'],'failures':examples[:1]})
    if not failed:return prompt
    return [prompt[0],{'role':'system','content':NOTICE+json.dumps(failed[:1],ensure_ascii=False)},*prompt[1:]]

def compact_environment(store,task_id,environment,last_result,cutoff):
    out=copy.deepcopy(environment);workspace=out.get('tool_workspace')
    if not isinstance(workspace,dict):return out
    operations=store.operations(task_id)
    for artifact in workspace.get('current_artifacts',[]):
        ref=None
        for op in reversed(operations):
            value=(op.get('result') or {}).get('value') or {}
            if op['tool']!='dev.submit' or value.get('revision')!=artifact.get('revision') or value.get('name')!=artifact.get('name'):continue
            with store.db() as db:
                row=db.execute("SELECT * FROM events WHERE task_id=? AND seq<=? AND kind='tool_request' AND json_extract(body,'$.operation_id')=? ORDER BY seq LIMIT 1",(task_id,cutoff,op['id'])).fetchone()
            if not row:continue
            event=store._decode_event(row);args=event['body']['arguments']
            if args.get('tests')!=artifact.get('own_tests'):continue
            code=args.get('code','');shown=artifact.get('code','')
            if code[:1600]!=shown:continue
            ref=event['ref'];artifact['definition_ref']=ref
            artifact['own_test_count']=len(artifact.pop('own_tests'))
            published=False
            for publication in operations:
                receipt=publication.get('result') or {};v=receipt.get('value') or {}
                if (publication['tool']=='dev.publish' and receipt.get('ok') and v.get('status')=='published'
                    and v.get('name')==artifact.get('name') and v.get('revision')==artifact.get('revision')
                    and isinstance(artifact.get('test_status'),dict) and artifact['test_status'].get('passed') is True):
                    evidence=store.resolve(task_id,receipt['evidence_ref'])
                    if evidence['seq']<=cutoff:published=True;break
            if artifact.get('code_truncated') or len(shown)>800 or published:
                artifact.pop('code',None);artifact['code_ref']={'ref':ref,'field':'/arguments/code'}
                artifact['code_omitted']=True
                if published:artifact['code_scope']='已验证发布；使用工具无需重复源码，修改前按code_ref读取原文。'
            break
        status=artifact.get('test_status');last=last_result or {};value=last.get('value') or {}
        if not isinstance(status,dict) or not isinstance(value,dict) or last.get('tool')!='dev.test':continue
        if value.get('name')!=artifact.get('name') or value.get('revision')!=artifact.get('revision'):continue
        if not status.get('fingerprint') or status['fingerprint']!=value.get('fingerprint'):continue
        if status.get('passed') is not value.get('passed'):continue
        # Only remove exact duplicates of the latest complete receipt.
        if all(status.get(k)==value.get(k) for k in ('own_failures','independent_failures')):
            status.pop('own_failures',None);status.pop('independent_failures',None)
            status['details_from']='last_result.value'
    return out

def source_text(event,field=None):
    if field is None or field=='':return event['text']
    if not isinstance(field,str) or not field.startswith('/') or len(field)>256 or len(field.split('/'))>13:
        raise ValueError('field must be a bounded JSON pointer, e.g. /arguments/code')
    value=event['body']
    for raw in field[1:].split('/'):
        if re.search(r'~(?![01])',raw):raise ValueError('Invalid JSON pointer escape')
        key=raw.replace('~1','/').replace('~0','~')
        if isinstance(value,dict) and key in value:value=value[key]
        elif isinstance(value,list) and re.fullmatch(r'0|[1-9][0-9]*',key) and int(key)<len(value):value=value[int(key)]
        else:raise ValueError('Source field does not exist')
    return value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,separators=(',',':'))
