import bootstrap
import asyncio,copy,json,os,unittest
import test_idle_regression as prior
from idle_autonomy import FIELDS
from idle_sessions import IdleSessions
from idle_development import context,variants,output_limit,argument_view
from lea.selfdev_runner import validate
from types import SimpleNamespace

CODE="""def run(arguments):
    result = []
    for value in arguments.get('labels', []):
        value = value.strip().lower()
        if value and value not in result:
            result.append(value)
    return result
"""
CASES=[{'input':{'labels':[' A ','a','B']},'expected':['a','b']},
       {'input':{'labels':[]},'expected':[]},{'input':{'labels':['',' b ','A','b']},'expected':['b','a']}]

def local_fixture_executor(code,cases):
    tree=validate(code);scope={};exec(compile(tree,'<trusted-unit-fixture>','exec'),scope)
    checks=[]
    for c in cases:
        try:
            actual=scope['run'](copy.deepcopy(c['input']));checks.append({'passed':'expected' not in c or actual==c['expected'],'actual':actual})
        except Exception as error:checks.append({'passed':False,'error':str(error)})
    return {'passed':all(c['passed'] for c in checks),'cases':checks}

class Development(unittest.TestCase):
    def setUp(self):
        self.f=prior.InitiativeTests();self.f.setUp();f=self.f
        with f.store.db(True) as db:db.execute('UPDATE tasks SET allowed_tools=? WHERE id=?',(json.dumps(f.task['allowed_tools']+['dev.submit','dev.publish']),f.task['id']))
        f.task=f.store.task(f.task['id']);f.config['idle_autonomy'].update(session_enabled=True,development_enabled=True,activity_budget_seconds=300,max_consecutive_no_progress=2)
        (f.auto/'config.json').write_text(json.dumps(f.config));f.aut.options=dict(f.config['idle_autonomy'])
        f.gate.sessions=IdleSessions(f.gate)
        if os.name=='nt':f.r.tools.executor=local_fixture_executor
    def tearDown(self):self.f.tearDown()
    def output(self,tool='',args=None,kind='tool'):
        return {'summary':'根据真实开发回执继续','attention_needed':False,'suggestion':'验证当前版本',
                'initiative':{'kind':kind,'goal':'把重复标签清洗封装并实际使用','tool':tool,'arguments':args or {}}}
    def submission(self,code=CODE,tests=CASES):return {'name':'labels_clean','description':'去空白、小写、去空项、保序去重','code':code,'tests':copy.deepcopy(tests)}
    def direct(self,tool,args):
        e=self.f.store.append(self.f.task['id'],'internal_state','idle_action_selected',{'selection':{'tool':tool,'arguments':args}})
        task=self.f.store.task(self.f.task['id']);return self.f.r.tools.execute(task['id'],task['window_id'],task['constraint_version'],e['id'],tool,args)
    def serve(self,fn):
        async def post(path,**kw):
            self.f.http.calls.append(kw['json']);value=fn(len(self.f.http.calls),kw['json'])
            return SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'choices':[{'finish_reason':'stop','message':{'content':json.dumps(value)}}]})
        self.f.http.post=post

    def test_full_idle_pipeline_with_actual_current_version_receipts(self):
        def decide(n,body):
            if n==1:return self.output('dev.submit',self.submission())
            a=self.f.r.tools.artifact(self.f.task['id'],'labels_clean');args={'name':a['name'],'revision':a['revision']}
            current=json.loads(body['messages'][-1]['content'])['initiative_context']
            self.assertEqual(current['development']['drafts'][0]['revision'],a['revision'])
            if n==2:return self.output('dev.test',args)
            if n==3:return self.output('dev.publish',args)
            if n==4:return self.output('dev.run',{**args,'arguments':{'labels':[' C ','c',' A ']}})
            self.assertEqual(current['recent'][0]['receipt']['value']['result'],['c','a'])
            return self.output(kind='done',args={'evidence_refs':self.f.gate.sessions.current['refs'][-2:]})
        self.serve(decide);result=asyncio.run(self.f.gate.deliver())
        self.assertEqual(result['activity']['stop_reason'],'goal_complete');self.assertEqual(result['activity']['actions'],4)
        self.assertEqual([o['tool'] for o in self.f.store.operations(self.f.task['id'])],['dev.submit','dev.test','dev.publish','dev.run'])
        self.assertTrue(all(c['max_tokens']==1536 for c in self.f.http.calls))

    def test_no_nonexistent_test_or_premature_publish_in_schema(self):
        self.assertEqual(variants(self.f.aut,self.f.task,'dev.test',FIELDS['dev.test']),[])
        self.assertEqual(variants(self.f.aut,self.f.task,'dev.publish',FIELDS['dev.publish']),[])
        with self.assertRaises(ValueError):self.f.aut.parse(json.dumps(self.output('dev.test',{'name':'fake','revision':'fake'})),self.f.task)

    def test_failed_tests_cannot_publish_and_new_revision_must_retest(self):
        first=self.direct('dev.submit',self.submission('def run(arguments):\n    return []'))['value']
        bad=self.direct('dev.test',{'name':first['name'],'revision':first['revision']});self.assertFalse(bad['value']['passed'])
        self.assertEqual(variants(self.f.aut,self.f.task,'dev.publish',FIELDS['dev.publish']),[])
        self.assertFalse(self.direct('dev.publish',{'name':first['name'],'revision':first['revision']})['ok'])
        second=self.direct('dev.submit',self.submission())['value'];self.assertNotEqual(first['revision'],second['revision'])
        self.assertFalse(self.direct('dev.publish',{'name':second['name'],'revision':second['revision']})['ok'])
        self.assertEqual(variants(self.f.aut,self.f.task,'dev.test',FIELDS['dev.test'])[0]['revision']['const'],second['revision'])

    def test_changed_independent_contract_invalidates_old_pass(self):
        a=self.direct('dev.submit',self.submission())['value'];args={'name':a['name'],'revision':a['revision']}
        self.assertTrue(self.direct('dev.test',args)['value']['passed'])
        with self.f.store.db(True) as db:db.execute('INSERT INTO unified_contracts VALUES(?,?,?)',(self.f.task['id'],a['name'],json.dumps([{'input':{'labels':['x']},'expected':['different']}])))
        self.assertEqual(variants(self.f.aut,self.f.task,'dev.publish',FIELDS['dev.publish']),[])
        self.assertFalse(self.direct('dev.publish',args)['ok'])

    def test_current_draft_available_after_activity_restart(self):
        a=self.direct('dev.submit',self.submission())['value'];view=context(self.f.aut,self.f.task)
        self.assertEqual(view['drafts'][0]['revision'],a['revision']);self.assertFalse(view['drafts'][0]['tests_current'])
        self.assertIn('source_ref',view['drafts'][0])

    def test_publication_needs_task_permission_even_if_feature_enabled(self):
        task={**self.f.task,'allowed_tools':['workspace.read']}
        self.assertNotIn('dev.submit',self.f.aut.allowed(task));self.assertNotIn('dev.publish',self.f.aut.allowed(task))
        self.assertEqual(output_limit(self.f.aut,task,512),512)

    def test_json_expected_values_are_supported_but_test_shape_is_checked(self):
        for expected in [None,True,False,3,1.5,[],{},'x']:
            data=self.submission();data['tests'][0]['expected']=expected
            self.f.aut.parse(json.dumps(self.output('dev.submit',data)),self.f.task)
        data=self.submission();data['tests']=data['tests'][:2]
        with self.assertRaises(ValueError):self.f.aut.parse(json.dumps(self.output('dev.submit',data)),self.f.task)

    def test_untrusted_code_cannot_import_or_modify_host(self):
        result=self.direct('dev.submit',self.submission('import os\ndef run(arguments):\n    return os.listdir()'))
        self.assertFalse(result['ok'])

    def test_code_is_archived_but_not_duplicated_in_activity_context(self):
        result=self.direct('dev.submit',self.submission());self.assertTrue(result['ok'])
        view=argument_view('dev.submit',self.submission());self.assertNotIn('code',view);self.assertEqual(view['code_chars'],len(CODE))
        from active_context import records_view
        events=self.f.store.events(self.f.task['id']);cutoff=events[-1]['seq']
        records=[{'ref':e['ref'],'kind':e['kind'],'role':e['role'],'text':e['text']} for e in events if e['kind']=='idle_action_selected']
        compact=records_view(self.f.store,self.f.task['id'],records,cutoff)
        self.assertNotIn('def run',json.dumps(compact));self.assertIn('code_sha256',json.dumps(compact))

    def test_failed_functional_test_is_not_progress_or_successful_tool_count(self):
        def decide(n,body):
            if n==1:return self.output('dev.submit',self.submission('def run(arguments):\n    return []'))
            if n==2:
                a=self.f.r.tools.artifact(self.f.task['id'],'labels_clean');return self.output('dev.test',{'name':a['name'],'revision':a['revision']})
            activity=json.loads(body['messages'][-1]['content'])['initiative_context']['activity']
            self.assertEqual(activity['consecutive_no_progress'],1);self.assertEqual(activity['verified_tool_counts']['dev.test'],0)
            return self.output(kind='rest')
        self.serve(decide);asyncio.run(self.f.gate.deliver())

if __name__=='__main__':unittest.main()
