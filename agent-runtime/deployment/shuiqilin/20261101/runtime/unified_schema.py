"""Versioned tool affordances for the same chat decision loop, not another Agent."""
import json
import re


def development_request(text):
    if re.search(r'之前|以前|曾经|过去|哪些|哪个|什么工具',text) and not re.search(r'^(?:请|帮我|开始|继续|尝试|现在|你来)+(?:开发|编写|实现|写)',text):
        return False
    if re.search(r'区别|是什么|为什么|解释|介绍|如何使用|怎么使用',text) and not re.search(r'^(?:请|帮我|开始|继续|尝试|现在|你来)+(?:开发|编写|实现|写)',text):
        return False
    return bool(re.search(r'开发|编写|写.{0,6}(?:工具|函数)|实现.{0,6}工具',text))


def string(limit=200):
    return {'type':'string','maxLength':limit}


def object_schema(fields):
    return {'type':'object','properties':fields,'required':list(fields),'additionalProperties':False}


def tool_variant(tool,fields):
    return object_schema({'action':{'type':'string','const':'tool'},
        'tool':{'type':'string','const':tool},'arguments':object_schema(fields),
        'response':{'type':'string','const':''}})


def ordinary_schema(payload):
    """Keep built-in memory actions distinct from externally authorized tools."""
    empty={'type':'string','const':''}
    variants=[object_schema({'action':{'type':'string','enum':['finish','respond']},
        'tool':empty,'arguments':object_schema({}),'response':string(200)})]
    allowed=list(payload.get('tools',{}))
    if allowed:
        variants.append(object_schema({'action':{'type':'string','const':'tool'},
            'tool':{'type':'string','enum':allowed},'arguments':{'type':'object','additionalProperties':True},'response':empty}))
    if payload.get('memory_reads_remaining',0)>0:
        latest=payload.get('last_result') or {}
        if latest.get('request_id'):
            variants.append(object_schema({'action':{'type':'string','const':'read_memory'},'tool':empty,'response':empty,
                'arguments':object_schema({'request_id':{'type':'string','const':latest['request_id']},
                    'card_ids':{'type':'array','minItems':1,'maxItems':3,'items':string(100)},'question':string(200)})}))
    if payload.get('memory_queries_remaining',0)>0:
        variants.append(object_schema({'action':{'type':'string','const':'recall'},'tool':empty,'response':empty,
            'arguments':object_schema({'question':string(200)})}))
    variants.append(object_schema({'action':{'type':'string','const':'handoff'},'tool':empty,
        'response':empty,'arguments':object_schema({})}))
    return {'oneOf':variants}


def workflow_schema(store, context):
    if not context or context['role']!='primary':return None
    payload=context['payload'];request=payload.get('current_request','')
    allowed=set(payload.get('tools',{}));task_id=context['task_id']
    if not development_request(request):
        # A single explicitly requested invocation ends after its actual receipt.
        # Multi-call tasks retain the ordinary loop; no fabricated tool result.
        if 'dev.run' in request and not re.search(r'分别|多次|两次|两种|两个|逐个|每个|每次',request):
            with store.db() as db:
                current=db.execute("SELECT body FROM jobs WHERE task_id=? AND kind='message' AND status='running' AND json_extract(body,'$.text')=? ORDER BY created LIMIT 1",(task_id,request)).fetchone()
            user_event=json.loads(current[0]).get('user_event') if current else None
            called=any(op['user_event']==user_event and op['tool']=='dev.run' and
                (op.get('result') or {}).get('value',{}).get('passed') for op in store.operations(task_id)) if user_event else False
            if called:
                return object_schema({'action':{'type':'string','enum':['finish','respond']},
                    'tool':{'type':'string','const':''},'arguments':object_schema({}),'response':string(200)})
        return ordinary_schema(payload)
    task=store.task(task_id)
    with store.db() as db:
        job=db.execute("SELECT body FROM jobs WHERE task_id=? AND kind='message' AND status='running' AND json_extract(body,'$.text')=? ORDER BY created LIMIT 1",(task_id,request)).fetchone()
        contracts=[r[0] for r in db.execute('SELECT name FROM unified_contracts WHERE task_id=?',(task_id,))]
        artifacts=[dict(r) for r in db.execute('SELECT * FROM unified_artifacts WHERE task_id=? ORDER BY created DESC',(task_id,))]
        published=[dict(r) for r in db.execute('SELECT name,revision FROM unified_library WHERE task_id=?',(task_id,))]
    if not job:return None
    user_event=json.loads(job[0]).get('user_event')
    with store.db() as db:
        start=db.execute('SELECT seq FROM events WHERE id=? AND task_id=?',(user_event,task_id)).fetchone()
        repeats=db.execute("SELECT COUNT(*) FROM events WHERE task_id=? AND seq>? AND kind='tool_attempt' AND json_extract(body,'$.tool')='dev.submit' AND json_extract(body,'$.deduplicated')=1",(task_id,start[0] if start else 0)).fetchone()[0]
    if repeats>=3:
        return object_schema({'action':{'type':'string','const':'finish'},'tool':{'type':'string','const':''},
            'arguments':object_schema({}),'response':string(200)})
    operations=store.operations(task_id)
    def outputs(tool):return [(op.get('result') or {}).get('value',{}) for op in operations if op['tool']==tool]
    criteria=' '.join(task['criteria'])
    searched=any(v.get('retrieved') and v.get('results') for v in outputs('web.search'))
    read=any(v.get('retrieved') and v.get('text') for v in outputs('web.open'))
    named=re.search(r'(?:开发|编写|实现)(?:一个|名为)?\s*([a-z][a-z0-9_]{1,39})',request)
    name=named[1] if named else contracts[0] if contracts else artifacts[0]['name'] if artifacts else None
    artifact=next((a for a in artifacts if a['name']==name),None)
    tests=json.loads(artifact['test_report']) if artifact and artifact['test_report'] else None
    registered=bool(artifact and any(p['name']==name and p['revision']==artifact['revision'] for p in published))
    invoked=bool(artifact and any(v.get('passed') and v.get('name')==name and v.get('revision')==artifact['revision'] for v in outputs('dev.run')))
    # Research is bounded before an implementation attempt. Failed tests reopen
    # research; executable code and all test values remain model-generated.
    if ('搜索' in criteria or '网页' in criteria) and not searched:
        choices=['web.search']
    elif ('正文' in criteria or '读取相关网页' in criteria) and not read:
        choices=['web.open','web.next_page','web.search']
    elif not artifact:
        choices=['dev.submit'] if read else ['dev.submit','web.search','web.open','web.next_page']
    elif not tests:
        choices=['dev.test','dev.submit']
    elif not tests.get('passed'):
        latest=payload.get('last_result') or {}
        researched=latest.get('tool')=='web.open' and latest.get('value',{}).get('retrieved')
        execution_errors=any(case.get('error') for part in ('own','independent') for case in (tests.get(part) or {}).get('cases',[]))
        choices=['dev.submit','web.search','web.open'] if execution_errors and not researched else ['dev.submit']
    elif not registered:
        choices=['dev.publish']
    elif '实际调用' in criteria and not invoked:
        choices=['dev.run']
    else:
        return object_schema({'action':{'type':'string','enum':['finish','respond']},
            'tool':{'type':'string','const':''},'arguments':object_schema({}),'response':string(200)})
    props={
        'web.search':{'query':string(200)},'web.open':{'url':string(2000),'query':string(120)},
        'web.next_page':{'cursor':string(80)},
        'dev.submit':{'name':{'type':'string',**({'const':name} if name else {'maxLength':40})},
            'description':string(200),'code':string(8000),
            'tests':{'type':'array','minItems':3,'maxItems':8,'items':object_schema({
                'input':{'type':'object','additionalProperties':True},'expected':{}})}},
        'dev.test':{'name':{'type':'string','const':name},'revision':{'type':'string','const':artifact['revision'] if artifact else ''}},
        'dev.publish':{'name':{'type':'string','const':name},'revision':{'type':'string','const':artifact['revision'] if artifact else ''}},
        'dev.run':{'name':{'type':'string','const':name},'revision':{'type':'string','const':artifact['revision'] if artifact else ''},
                   'arguments':{'type':'object','additionalProperties':True}},
    }
    variants=[tool_variant(tool,props[tool]) for tool in choices if tool in allowed]
    # Retain an honest stopping option after repeated execution failures or a
    # near-exhausted budget. Runtime still rejects unsupported success claims.
    failures=sum(not (op.get('result') or {}).get('ok') or (
        op['tool'].startswith('web.') and not (op.get('result') or {}).get('value',{}).get('retrieved')) for op in operations)
    if failures>=3 or payload.get('step',0)>=13:
        variants.append(object_schema({'action':{'type':'string','const':'finish'},
            'tool':{'type':'string','const':''},'arguments':object_schema({}),'response':string(200)}))
    return {'oneOf':variants} if variants else None
