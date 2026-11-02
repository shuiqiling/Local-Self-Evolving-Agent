"""Development functions inside the memory/chat Agent; no developer model call."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
from urllib.parse import urlsplit, urlunsplit, unquote

from lea.tools import Tools as BaseTools
from lea.store import dump, now
from lea.web import WebClient
from lea.selfdev_runner import validate

EXTRA_HELP = {
    'web.search': '{"query":"简短关键词"} 搜索并返回5个候选，尚未读取正文',
    'web.next_page': '{"cursor":"原样复制next_cursor"} 返回另外5个候选',
    'web.open': '{"url":"候选id或公开网址","query":"相关主题"} 读取网页正文节选',
    'dev.submit': '{"name":"工具名","description":"用途","code":"def run(arguments):...","tests":[{"input":{},"expected":null}]} 保存自己编写的纯Python工具；至少3个不同测试。禁止导入、联网、文件读写',
    'dev.test': '{"name":"工具名","revision":"submit返回的revision"} 实际执行自测和独立验收；结果仅绑定此版本',
    'dev.publish': '{"name":"工具名","revision":"已测试版本"} 当前版本真实测试通过后登记为可复用工具',
    'dev.list': '{} 列出已登记的工具及版本',
    'dev.run': '{"name":"工具名","revision":"已发布版本","arguments":{}} 执行已登记工具',
}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def search_view(result):
    view={k:result[k] for k in ('status','retrieved','query','next_cursor','has_more','message','error_kind') if k in result}
    view.update(pages_opened=False,untrusted_web_content=True)
    view['results']=[{'id':r['id'],'title':r.get('title','')[:90],
        'domain':urlsplit(r.get('url','')).hostname,'snippet':r.get('snippet','')[:90],
        **({'url':r['url']} if len(r.get('url',''))<=180 else {})} for r in result.get('results',[])[:5]]
    return view


class UnifiedTools(BaseTools):
    def __init__(self, store, web_factory=WebClient, executor=None):
        super().__init__(store)
        self.web_factory, self.executor = web_factory, executor
        self.browsers = {}
        with store.db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS unified_artifacts(
                  task_id TEXT,name TEXT,revision TEXT,description TEXT,code TEXT,code_sha TEXT,
                  own_tests TEXT,test_report TEXT,created TEXT,PRIMARY KEY(task_id,name));
                CREATE TABLE IF NOT EXISTS unified_contracts(task_id TEXT,name TEXT,cases TEXT,PRIMARY KEY(task_id,name));
                CREATE TABLE IF NOT EXISTS unified_library(
                  name TEXT,revision TEXT,task_id TEXT,description TEXT,code TEXT,code_sha TEXT,
                  test_report TEXT,created TEXT,PRIMARY KEY(name,revision));
            ''')

    def browser(self, task_id):
        if task_id not in self.browsers:
            self.browsers[task_id] = self.web_factory()
        return self.browsers[task_id]

    def execute(self, task_id, window_id, constraint_version, user_event, tool, arguments, role='primary'):
        task = self.store.task(task_id)
        if (role=='primary' and tool in ('dev.test','dev.publish') and tool in task['allowed_tools']
                and isinstance(arguments,dict)):
            try:
                artifact = self.artifact(task_id,arguments['name'],arguments['revision'])
            except (ValueError,KeyError) as error:
                result = {'ok':False,'tool':tool,'error':str(error)}
                self.store.append(task_id,'runtime','tool_precondition_rejected',result)
                return result
            # Idempotency includes the currently bound owner-test contract.
            # A cached pass cannot substitute for a different revision/contract.
            arguments = {**arguments,'test_fingerprint':self.fingerprint(artifact)}
        result=super().execute(task_id,window_id,constraint_version,user_event,tool,arguments,role)
        self.store.append(task_id,'runtime','tool_attempt',{'tool':tool,'ok':result.get('ok'),
            'deduplicated':result.get('deduplicated',False),'operation_id':result.get('operation_id'),
            **({'observation_revalidated':True,'observed_at':result.get('observed_at')} if result.get('observation_revalidated') else {})})
        if tool=='dev.submit' and result.get('deduplicated'):
            current=self.artifact(task_id,arguments['name'])
            if current['test_report'] and not current['test_report'].get('passed'):
                return {'ok':False,'tool':tool,'error':'此版本已测试失败，代码和自测没有变化。请根据当前test_status中的失败用例修改后重新提交。',
                        'revision':current['revision'],'deduplicated':True}
        if tool.startswith('web.') and result.get('deduplicated'):
            return {'ok':False,'tool':tool,'error':'相同操作本轮已执行，未再次请求网络。请依据已有结果继续下一步或换来源。',
                    'previous_receipt':{k:(result.get('value') or {}).get(k) for k in ('status','retrieved','url')},
                    'operation_id':result.get('operation_id'),'deduplicated':True}
        return result

    def artifact(self, task_id, name, revision=None):
        with self.store.db() as db:
            row = db.execute('SELECT * FROM unified_artifacts WHERE task_id=? AND name=?', (task_id,name)).fetchone()
        if row is None:
            raise ValueError('Submit this tool in the current task first')
        value = dict(row)
        if revision is not None and value['revision'] != revision:
            raise ValueError('Stale revision; use the current submitted revision')
        if digest(value['code']) != value['code_sha']:
            raise ValueError('Tool code integrity mismatch')
        value['own_tests'] = json.loads(value['own_tests'])
        value['test_report'] = json.loads(value['test_report']) if value['test_report'] else None
        return value

    def contracts(self, task_id, name):
        with self.store.db() as db:
            row = db.execute('SELECT cases FROM unified_contracts WHERE task_id=? AND name=?', (task_id,name)).fetchone()
        return json.loads(row[0]) if row else []

    def fingerprint(self, artifact):
        return digest(dump([artifact['code_sha'],artifact['own_tests'],self.contracts(artifact['task_id'],artifact['name'])]))

    def run_code(self, code, cases):
        if self.executor:
            return self.executor(code, cases)
        runner = Path(__file__).parent/'app/lea/selfdev_runner.py'
        try:
            process = subprocess.run(['/usr/bin/python3','-I','-S',str(runner)],
                input=json.dumps({'code':code,'cases':cases},ensure_ascii=False),text=True,
                capture_output=True,timeout=8)
        except subprocess.TimeoutExpired:
            return {'passed':False,'error':'Execution timed out'}
        if process.returncode:
            return {'passed':False,'error':'Tool runner failed','exit_code':process.returncode}
        try:
            return json.loads(process.stdout)
        except ValueError:
            return {'passed':False,'error':'Tool runner did not return valid JSON'}

    def observe(self, task_id):
        result = super().observe(task_id)
        with self.store.db() as db:
            rows = [dict(row) for row in db.execute(
                'SELECT name,revision,description,code,own_tests,test_report FROM unified_artifacts WHERE task_id=? ORDER BY created DESC LIMIT 2', (task_id,))]
            library = [dict(row) for row in db.execute(
                'SELECT name,revision,description FROM unified_library ORDER BY created DESC LIMIT 8')]
        for row in rows:
            row['own_tests'] = json.loads(row['own_tests'])
            report = json.loads(row['test_report']) if row['test_report'] else None
            row['test_status'] = ({k:report.get(k) for k in ('passed','revision','fingerprint','validation_scope','own_failures','independent_failures')} if report else 'not_tested')
            if report:
                row['test_status']['own_expectations_source']='model_proposed; check against user requirement'
                row['test_status']['independent_expectations_source']='fixed owner contract' if report.get('independent') else None
            row.pop('test_report')
            row['code_truncated'] = len(row['code']) > 1600
            row['code'] = row['code'][:1600]
        result['tool_workspace'] = {'current_artifacts':rows,'published_tools':library,
                                    'executor':'pure Python subprocess; no second Agent'}
        pages=[];failed=[]
        for operation in self.store.operations(task_id):
            value=(operation.get('result') or {}).get('value',{})
            if operation['tool']=='web.open' and value.get('retrieved') and value.get('text'):
                pages.append({'url':value.get('url'),'title':value.get('title'),'text':value['text'][:700],
                              'excerpt_only':True,'kind':'untrusted_external_excerpt'})
            elif operation['tool']=='web.open':
                failed.append({'url':value.get('url',operation['arguments'].get('url')),
                               'status':value.get('status'),'reason':str(value.get('error_detail') or value.get('message',''))[:160]})
        result['tool_workspace']['read_pages']=pages[-2:]
        result['tool_workspace']['failed_sources']=failed[-4:]
        with self.store.db() as db:
            attempts=[json.loads(r[0]) for r in db.execute("SELECT body FROM events WHERE task_id=? AND kind='tool_attempt' ORDER BY seq DESC LIMIT 6",(task_id,))]
        result['tool_workspace']['recent_attempts']=list(reversed(attempts))
        last_submit=next((op for op in reversed(self.store.operations(task_id)) if op['tool']=='dev.submit'),None)
        if last_submit and not (last_submit.get('result') or {}).get('ok'):
            args=last_submit['arguments']
            result['tool_workspace']['last_rejected_submission']={'not_saved':True,'code':args.get('code','')[:1600],
                'own_tests':args.get('tests',[]),'error':last_submit['result'].get('error')}
        return result

    def _execute(self, task_id, operation_id, tool, arguments):
        if tool.startswith('web.'):
            browser = self.browser(task_id)
            if tool == 'web.search':
                return search_view(browser.search(arguments['query']))
            if tool == 'web.next_page':
                return search_view(browser.next_page(arguments['cursor']))
            if tool == 'web.open':
                original=arguments['url']
                target=browser.references.get(original,original)
                focus=arguments.get('query','')
                parts=urlsplit(target)
                if parts.scheme in ('http','https') and parts.fragment:
                    target=urlunsplit((parts.scheme,parts.netloc,parts.path,parts.query,''))
                    focus=focus or unquote(parts.fragment)
                result = browser.open(target,query=focus)
                if result.get('text'):
                    result = {**{k:result[k] for k in ('status','retrieved','url','title','fetched_at','message','error_kind') if k in result},'text':result['text'][:1400],
                              'excerpt_chars':min(1400,len(result['text'])),
                              'truncated':result.get('truncated',False) or len(result['text'])>1400,'untrusted_web_content':True}
                return result
        if not tool.startswith('dev.'):
            return super()._execute(task_id,operation_id,tool,arguments)
        if tool == 'dev.list':
            with self.store.db() as db:
                return {'tools':[dict(r) for r in db.execute('SELECT name,revision,description,task_id FROM unified_library ORDER BY created DESC LIMIT 50')]}
        name = arguments['name']
        if not isinstance(name,str) or not re.fullmatch(r'[a-z][a-z0-9_]{1,39}',name):
            raise ValueError('Tool name must use lowercase letters, digits and underscores')
        if tool == 'dev.submit':
            code, tests, description = arguments['code'],arguments['tests'],arguments['description']
            try:
                validate(code)
            except SyntaxError as error:
                raise ValueError('Python syntax error: '+str(error)) from error
            if not isinstance(description,str) or not 1<=len(description)<=200:
                raise ValueError('Expected a short description')
            if not isinstance(tests,list) or not 3<=len(tests)<=8:
                raise ValueError('Provide3–8 own tests')
            if any(not isinstance(t,dict) or set(t)!={'input','expected'} or not isinstance(t['input'],dict) for t in tests):
                raise ValueError('Each test requires a JSON object input and expected value')
            if len({dump(t['input']) for t in tests})<3 or len(dump(tests))>10000:
                raise ValueError('Provide at least3 distinct bounded test inputs')
            revision = 'rev_'+digest(dump([task_id,name,code,tests]))[:16]
            with self.store.db(True) as db:
                existing=db.execute('SELECT revision,test_report FROM unified_artifacts WHERE task_id=? AND name=?',(task_id,name)).fetchone()
                if existing and existing['revision']==revision:
                    old_report=json.loads(existing['test_report']) if existing['test_report'] else None
                    return {'status':'unchanged','name':name,'revision':revision,'code_sha256':digest(code),
                            'tested':old_report is not None,'test_passed':old_report.get('passed') if old_report else None,
                            'message':'实现与自测均未改变，保留原有测试结论。失败版本需要修改后重新提交。'}
                db.execute('INSERT OR REPLACE INTO unified_artifacts VALUES(?,?,?,?,?,?,?,?,?)',
                           (task_id,name,revision,description,code,digest(code),dump(tests),None,now()))
            return {'status':'saved','name':name,'revision':revision,'code_sha256':digest(code),'tested':False}
        revision = arguments['revision']
        if tool == 'dev.run':
            with self.store.db() as db:
                row = db.execute('SELECT * FROM unified_library WHERE name=? AND revision=?',(name,revision)).fetchone()
            if row is None:
                raise ValueError('Tool version is not published')
            value = dict(row)
            if digest(value['code']) != value['code_sha']:
                raise ValueError('Published code integrity mismatch')
            payload = arguments['arguments']
            if not isinstance(payload,dict):
                raise ValueError('Tool input must be a JSON object')
            output = self.run_code(value['code'],[{'input':payload}])
            return {'name':name,'revision':revision,'passed':output.get('passed',False),
                    'result':output['cases'][0].get('actual') if output.get('passed') else None,'execution':output}
        artifact = self.artifact(task_id,name,revision)
        fingerprint = self.fingerprint(artifact)
        if tool == 'dev.test':
            contract = self.contracts(task_id,name)
            try:
                own = self.run_code(artifact['code'],artifact['own_tests'])
                independent = self.run_code(artifact['code'],contract) if contract else None
            except subprocess.TimeoutExpired:
                own, independent = {'passed':False,'error':'Execution timed out'},None
            failures = [{'case':i+1,**case,**actual} for i,(case,actual) in enumerate(zip(artifact['own_tests'],own.get('cases',[]))) if not actual.get('passed')]
            report = {'name':name,'revision':revision,'code_sha256':artifact['code_sha'],'fingerprint':fingerprint,
                'passed':bool(own.get('passed') and (not contract or independent and independent.get('passed'))),
                'own':own,'own_failures':failures,'independent':independent,
                'validation_scope':'self_and_owner_tests' if contract else 'selftests_only'}
            if independent and not independent.get('passed'):
                report['independent_failures']=[{'case':i+1,**case,**actual}
                    for i,(case,actual) in enumerate(zip(contract,independent.get('cases',[]))) if not actual.get('passed')][:3]
                report['independent_feedback_scope']='Frozen owner tests; at most3 failing examples disclosed for debugging'
            with self.store.db(True) as db:
                db.execute('UPDATE unified_artifacts SET test_report=? WHERE task_id=? AND name=? AND revision=?',
                           (dump(report),task_id,name,revision))
            return report
        if tool == 'dev.publish':
            report = artifact['test_report']
            if not report or not report['passed'] or report['fingerprint'] != fingerprint:
                raise ValueError('Current code and test contract must pass actual tests first')
            with self.store.db(True) as db:
                db.execute('INSERT OR IGNORE INTO unified_library VALUES(?,?,?,?,?,?,?,?)',
                    (name,revision,task_id,artifact['description'],artifact['code'],artifact['code_sha'],dump(report),now()))
            return {'status':'published','name':name,'revision':revision,'code_sha256':artifact['code_sha'],
                    'validation_scope':report['validation_scope'],'author':'Shuiqilin chat Agent'}
        raise ValueError('Unknown development tool')
