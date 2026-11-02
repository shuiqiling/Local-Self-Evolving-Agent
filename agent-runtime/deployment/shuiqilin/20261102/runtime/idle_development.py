"""Real development affordances for idle activities; no second decision model."""
import copy,hashlib,json
VERSION='idle-development-v1'
FIELDS={
 'dev.submit':{'name':{'type':'string','pattern':'[a-z][a-z0-9_]{1,39}','maxLength':40},
   'description':{'type':'string','maxLength':200},'code':{'type':'string','maxLength':8000},
   'tests':{'type':'array','minItems':3,'maxItems':8,'items':{'type':'object','properties':{
       'input':{'type':'object','additionalProperties':True},'expected':{}},'required':['input','expected'],'additionalProperties':False}}},
 'dev.publish':{'name':{'type':'string','maxLength':40},'revision':{'type':'string','maxLength':80}},
}
HELP={
 'dev.submit':'{name,description,code,tests:[{input,expected}]} 保存自己写的def run(arguments)纯Python；3～8个不同自测；保存不等于通过。',
 'dev.publish':'{name,revision} 把当前代码和测试均通过的版本登记入自己的本地工具库；不向互联网发布。',
}
POLICY=('遇到反复的确定性处理，先查看已有工具，值得复用时可自己编写小工具。不要仅因重复失败就造工具。'
        '开发流程dev.submit→dev.test→修正失败并重新submit/test→dev.publish→dev.run实际使用；名字和revision复制真实回执。'
        '函数为def run(arguments)，只能做JSON输入输出的纯Python；不导入、不联网、不读写文件、不调用其他工具。'
        '编写至少3个不同自测，包含正常和边界条件。自测通过不等于独立验收；未经计时不能声称提速。'
        '开发也耗时间；有可复用版本就先用。activity时间不足时保留草稿，下一次依据真实开发状态继续。')

def enabled(aut):return bool(aut.options.get('development_enabled'))
def rows(aut,task):
    with aut.store.db() as db:
        drafts=[dict(r) for r in db.execute('SELECT * FROM unified_artifacts WHERE task_id=? ORDER BY created DESC LIMIT 8',(task['id'],))]
        library=[dict(r) for r in db.execute('SELECT name,revision,description,task_id FROM unified_library ORDER BY created DESC LIMIT 50')]
    return drafts,library

def variants(aut,task,name,fields):
    """Only existing versions can be tested/run; only actual passes published."""
    if not enabled(aut):return [] if name in FIELDS else [fields]
    if name not in ('dev.test','dev.publish','dev.run'):return [fields]
    drafts,library=rows(aut,task);out=[]
    if name=='dev.run':targets=library
    else:
        targets=[]
        for draft in drafts:
            artifact=aut.runtime.tools.artifact(task['id'],draft['name']);report=artifact['test_report']
            tested=report and report.get('fingerprint')==aut.runtime.tools.fingerprint(artifact)
            if name=='dev.test' and not tested:targets.append(draft)
            if name=='dev.publish' and tested and report.get('passed') and not any(p['name']==draft['name'] and p['revision']==draft['revision'] for p in library):targets.append(draft)
    for target in targets:
        out.append({**copy.deepcopy(fields),'name':{'type':'string','const':target['name']},'revision':{'type':'string','const':target['revision']}})
    return out

def context(aut,task):
    drafts,library=rows(aut,task);visible=[]
    for row in drafts[:2]:
        artifact=aut.runtime.tools.artifact(task['id'],row['name']);test=artifact['test_report']
        valid=test and test.get('fingerprint')==aut.runtime.tools.fingerprint(artifact)
        published=any(p['name']==row['name'] and p['revision']==row['revision'] for p in library)
        item={'name':row['name'],'revision':row['revision'],'description':row['description'],
              'tests_current':bool(valid),'passed':test.get('passed') if valid else None,'published':published}
        if valid:
            item['validation_scope']=test.get('validation_scope')
            item['failures']=(test.get('own_failures',[])+test.get('independent_failures',[]))[:3]
            if not test.get('passed'):
                item['code_excerpt']=row['code'][:1000];item['code_truncated']=len(row['code'])>1000
        with aut.store.db() as db:
            event=db.execute("SELECT window_id,id FROM events WHERE task_id=? AND kind='idle_action_selected' AND json_extract(body,'$.selection.tool')='dev.submit' AND json_extract(body,'$.selection.arguments.name')=? ORDER BY seq DESC LIMIT 1",(task['id'],row['name'])).fetchone()
        if event:item['source_ref']=event['window_id']+':'+event['id']
        visible.append(item)
    with aut.store.db() as db:
        attempts=[dict(r) for r in db.execute('SELECT selection,status FROM idle_initiatives WHERE task_id=? ORDER BY created DESC LIMIT 16',(task['id'],))]
    counts={}
    for item in attempts:
        selection=json.loads(item['selection']);tool=selection.get('tool')
        if tool:
            key=(tool,item['status']);counts[key]=counts.get(key,0)+1
    from lea.selfdev_runner import METHODS
    return {'drafts':visible,'published_tools':library[:8],'supported_methods':sorted(METHODS),
            'recent_attempt_counts':[{'tool':k[0],'status':k[1],'count':n} for k,n in counts.items() if n>1],
            'scope':'真实版本/测试账本。重复意图或失败不是进展。新建工具仅能纯Python处理JSON；现有工作区与网页工具负责I/O。',
            'continuation':'有未测试/失败/未发布草稿时可按真实状态续做；旧任务要求不是新的执行命令。'}

def output_limit(aut,task,default):
    return max(default,1536) if enabled(aut) and 'dev.submit' in aut.allowed(task) else default

def argument_view(tool,args):
    if tool!='dev.submit':return args
    code=args.get('code','')
    return {'name':args.get('name'),'description':args.get('description'),'code_chars':len(code),
            'code_sha256':hashlib.sha256(code.encode()).hexdigest(),'own_tests':len(args.get('tests',[])),
            'code_scope':'完整代码和自测在原始意图事件；需要时用context.read读取。'}
