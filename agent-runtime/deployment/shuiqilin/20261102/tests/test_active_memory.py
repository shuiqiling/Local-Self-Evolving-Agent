import asyncio,copy,gc,hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import bootstrap
import numpy as np
from lea.store import Store
from lea.config import Settings
from lea.evidence import EvidenceMemory
from lea.embedding import eligible,tokens
from navigation_runtime import memory_type
import navigation_contract as C
from active_memory import ActiveMemory,FIELDS
import active_context as V

class Embedder:
    model_key='unit-test-embedding'
    calls=[]
    async def encode(self,texts,query=False):
        self.calls.append((texts,query));out=[]
        for text in texts:
            v=np.zeros(128,dtype=np.float32)
            for token in tokens(text):v[int(hashlib.sha256(token.encode()).hexdigest()[:8],16)%128]+=1
            out.append(v/max(float(np.linalg.norm(v)),1))
        return out

class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.store=Store(Path(self.tmp.name)/'data')
        self.task=self.store.create_task('查询旧记忆',[],[],[]);self.events=[]
        for i in range(7):self.events.append(self.store.append(self.task['id'],'user','message',{'text':f'测试规则项目X{i}：仅在温度低于{40+i}度时启动，未验证室外使用。'}))
        self.embedder=Embedder();self.embedder.calls=[]
        self.settings=Settings(data_dir=self.store.root,retrieval_top_k=6)
        self.memory=memory_type(EvidenceMemory)(self.store,None,self.settings,self.embedder,policy={'semantic_min':.9,'keyword_min':.25})
        self.memory.navigator=SimpleNamespace(get=lambda e:C.direct(e['text']))
        self.r=SimpleNamespace(store=self.store,memory=self.memory);self.api=ActiveMemory(self.r)
        self.index();self.origin=self.store.append(self.task['id'],'user','message',{'text':'查一下测试规则'})
    def tearDown(self):gc.collect();self.tmp.cleanup()
    def index(self):
        self.memory.index.sync(self.task['id']);asyncio.run(self.memory.index.encode_pending(self.memory.index.pending(1000)))
    def search(self,q='测试规则'):
        return asyncio.run(self.api.search(self.task['id'],q,self.origin['id']))
    def test_three_per_page_frozen_nonduplicating_repeatable_across_new_records(self):
        a=self.search();self.assertEqual(len(a['items']),3);self.assertNotIn('evidence',a)
        before=len(self.embedder.calls);self.store.append(self.task['id'],'user','message',{'text':'测试规则新项目：必须先取得新许可。'});self.index()
        b=self.api.page(self.task['id'],a['next_cursor']);again=self.api.page(self.task['id'],a['next_cursor'])
        self.assertEqual(b,again);self.assertEqual(len(b['items']),3)
        c=self.api.page(self.task['id'],b['next_cursor']);self.assertEqual(len(c['items']),1);self.assertIsNone(c['next_cursor'])
        ids=[x['index_id'] for p in (a,b,c) for x in p['items']];self.assertEqual(len(set(ids)),7)
        self.assertEqual(self.memory.settings.retrieval_top_k,6)
    def test_empty_result_not_padded(self):
        result=self.search('unrelated_zzx987');self.assertEqual(result['items'],[]);self.assertIsNone(result['next_cursor'])
    def test_cursor_and_index_cannot_cross_task_or_be_invented(self):
        first=self.search();other=self.store.create_task('other',[],[],[])
        for tid,cursor in [(other['id'],first['next_cursor']),(self.task['id'],'mp_forged')]:
            with self.assertRaises(ValueError):self.api.page(tid,cursor)
        with self.assertRaises(ValueError):asyncio.run(self.api.read(other['id'],first['items'][0]['index_id'],'温度'))
        with self.assertRaises(ValueError):asyncio.run(self.api.read(self.task['id'],'mi_forged','温度'))
    def test_detail_is_only_selected_record_and_retains_condition(self):
        item=self.search()['items'][0];answer=asyncio.run(self.api.read(self.task['id'],item['index_id'],'何时启动'))
        self.assertEqual(len(answer['evidence']),1);quote=answer['evidence'][0]
        self.assertIn('仅在温度低于',quote['text']);self.assertIn('未验证室外使用',quote['text'])
        self.assertEqual(self.store.resolve(self.task['id'],quote['ref'])['text'],quote['text'])
    def test_long_detail_uses_only_selected_record_vectors_and_keeps_correction(self):
        text='普通背景说明。'*180+'测试规则特项：旧值20度已作废，更正为25度，仅限室内。'+'后续备注。'*30
        event=self.store.append(self.task['id'],'user','message',{'text':text});self.index()
        page=self.search('测试规则特项 更正 室内')
        selected=None
        while True:
            for item in page['items']:
                with self.store.db() as db:g=db.execute('SELECT search_id,card_id FROM active_memory_grants WHERE index_id=?',(item['index_id'],)).fetchone()
                _,_,cards=self.api.result(self.task['id'],g['search_id'])
                if next(c for c in cards if c['card_id']==g['card_id'])['_sources'][0]['event_ref']==event['ref']:selected=item
            if selected or not page['next_cursor']:break
            page=self.api.page(self.task['id'],page['next_cursor'])
        self.assertIsNotNone(selected)
        answer=asyncio.run(self.api.read(self.task['id'],selected['index_id'],'特项更正值 室内'))
        self.assertEqual(answer['method'],'scoped_vector_keyword');e=answer['evidence'][0]
        self.assertEqual(e['ref'],event['ref']);self.assertIn('旧值20度已作废',e['text']);self.assertIn('更正为25度，仅限室内',e['text'])
    def test_tool_operations_have_real_receipts_and_are_not_reindexed(self):
        result=asyncio.run(self.api.execute(self.task['id'],self.task['window_id'],self.task['constraint_version'],self.origin['id'],'memory.search',{'query':'测试规则'}))
        self.assertTrue(result['ok']);e=self.store.resolve(self.task['id'],result['evidence_ref']);self.assertFalse(eligible(e))
        previous=asyncio.run(self.api.execute(self.task['id'],self.task['window_id'],self.task['constraint_version'],self.origin['id'],'memory.search',{'query':'测试规则'}))
        self.assertTrue(previous['deduplicated']);self.assertEqual(result['value'],previous['value'])
    def runtime(self,model):
        from unified_auto_agent import AutoMemoryRuntime
        from unified_tools import UnifiedTools,EXTRA_HELP
        from lea.tools import TOOL_HELP
        TOOL_HELP.update(EXTRA_HELP)
        r=object.__new__(AutoMemoryRuntime);r.store=self.store;r.memory=self.memory;r.settings=self.settings
        r.tools=UnifiedTools(self.store);r.model=model;r.locks=defaultdict(asyncio.Lock)
        return r
    def test_real_primary_loop_search_next_read_and_no_automatic_lookup(self):
        from lea.schemas import Decision
        owner=self
        class Model:
            calls=[]
            async def tokens(self,prompt):return 100
            async def call(self,tid,role,prompt,schema,**kw):
                p=json.loads(prompt[-1]['content']);self.calls.append(p);n=len(self.calls)
                if n==1:
                    owner.assertIsNone(p['last_result']);owner.assertIn('memory.search',p['tools'])
                    tool,args='memory.search',{'query':'测试规则'}
                elif n==2:tool,args='memory.next',{'cursor':p['last_result']['value']['next_cursor']}
                elif n==3:tool,args='memory.read',{'index_id':p['last_result']['value']['items'][0]['index_id'],'query':'启动条件'}
                else:
                    owner.assertIn('仅在温度低于',p['last_result']['value']['evidence'][0]['text'])
                    return Decision(action='respond',tool='',arguments={},response='已按来源读到启动条件。')
                return Decision(action='tool',tool=tool,arguments=args,response='')
        model=Model();runtime=self.runtime(model)
        job=self.store.add_job(self.task['id'],'message',{'text':'查询测试规则，翻到第二批并读取一条原文。'})
        self.store.set_job(job['id'],'running')
        result=asyncio.run(runtime.primary(job));self.assertEqual(result['response'],'已按来源读到启动条件。')
        operations=self.store.operations(self.task['id']);self.assertEqual([o['tool'] for o in operations],['memory.search','memory.next','memory.read'])
        with self.store.db() as db:self.assertEqual(db.execute("SELECT count(*) FROM events WHERE kind='retrieval_result'").fetchone()[0],1)
    def test_twenty_idle_samples_share_window_then_memory_tool_executes(self):
        from idle_autonomy import IdleAutonomy
        from idle_state_input import StateInput
        from shared_context import SharedChatContext
        owner=self;selection={'kind':'rest','goal':'等待有价值的行动','tool':'','arguments':{}}
        class Model:
            adapter={'id':0,'version':'fixture'}
            async def tokens(self,prompt):return len(json.dumps(prompt,ensure_ascii=False))//2
        class HTTP:
            generation_lock=asyncio.Lock()
            calls=[]
            async def post(self,path,**kw):
                self.calls.append(kw['json']);output={'summary':'CPU29度','suggestion':'暂无需处理','attention_needed':False,'initiative':selection.copy()}
                return SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'choices':[{'finish_reason':'stop','message':{'content':json.dumps(output,ensure_ascii=False)}}]})
        model=Model();runtime=self.runtime(model);http=HTTP();auto=Path(self.tmp.name)/'auto';auto.mkdir()
        (auto/'config.json').write_text(json.dumps({'min_new_handoffs':10,'memory_teacher':{'enabled':False},'idle_autonomy':{'enabled':True}}))
        aut=IdleAutonomy(runtime,auto,{'enabled':True});shared=SharedChatContext(runtime,http,aut)
        gate=StateInput(self.store.path,Path(self.tmp.name),auto,Model.adapter,http,
            {'enabled':True,'context_mode':'shared_chat','idle_databases':[str(self.store.path)],'min_idle_seconds':0,'automatic_interval_seconds':0,'max_output_tokens':512},
            sampler=lambda _:{'cpu':{'temperature_c':29},'memory':{},'network':{}},lease_acquirer=lambda _:SimpleNamespace(close=lambda:None),shared_context=shared)
        async def run():
            for _ in range(20):owner.assertEqual((await gate.deliver())['status'],'completed')
            owner.assertEqual(len(self.store.windows(self.task['id'])),1)
            selection.update(kind='tool',goal='查记忆',tool='memory.search',arguments={'query':'测试规则'})
            return await gate.deliver()
        result=asyncio.run(run());self.assertEqual(result['initiative_execution']['status'],'executed')
        self.assertEqual(len(result['initiative_execution']['receipt']['value']['items']),3)
        with self.store.db() as db:self.assertEqual(db.execute("SELECT count(*) FROM events WHERE kind='internal_state_input'").fetchone()[0],21)
        current=json.loads(http.calls[-1]['messages'][-1]['content']);self.assertNotIn('request',current)

class ContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.store=Store(Path(self.tmp.name)/'data');self.task=self.store.create_task('测试目标',['必须先验证；不得发布'],[],[])
    def tearDown(self):gc.collect();self.tmp.cleanup()
    def test_latest_correction_and_conditions_are_whole_and_old_index_not_injected(self):
        self.store.append(self.task['id'],'user','message',{'text':'旧上限20；必须先测试。'})
        self.store.append(self.task['id'],'user','message',{'text':'更正：20作废，改为25；仅室内，测试要求仍然有效。'})
        current=self.store.append(self.task['id'],'user','message',{'text':'现在是多少？'})
        out=V.projection(self.store,self.task,current['seq'],'现在是多少',current['ref'])
        rendered=V.dump(out);self.assertIn('20作废，改为25',rendered);self.assertIn('必须先测试',rendered)
        self.assertNotIn('archive_index',out);self.assertNotIn('recent_conversation',out)
    def test_anaphora_keeps_previous_answer_as_unverified(self):
        self.store.append(self.task['id'],'primary','assistant_reply',{'text':'方案一是A，方案二是B。'})
        current=self.store.append(self.task['id'],'user','message',{'text':'用第二个方案'})
        out=V.projection(self.store,self.task,current['seq'],'用第二个方案',current['ref']);self.assertIn('方案二是B',out['referenced_reply']['text']);self.assertIn('不是执行证据',out['referenced_reply']['scope'])
    def test_state_keeps_null_and_cpu_identity_without_mutating_raw(self):
        sample={'local_time':'2026-11-02T09:00:00.123456+08:00','cpu':{'temperature_c':29,'usage_percent':1.2345,'temperature_sensors':[{'device':'wifi','celsius':114}]},'memory':{'available_bytes':None},'network':{}}
        saved=copy.deepcopy(sample);out=V.snapshot_view(sample)
        self.assertEqual(sample,saved);self.assertEqual(out['cpu']['temperature_c'],29);self.assertIsNone(out['memory']['available_gib'])
        self.assertNotIn('114',V.dump(out));self.assertEqual(out['local_time'],'2026-11-02T09:00:00+08:00')
    def test_old_samples_and_assessments_not_repeated_but_tool_failures_retained(self):
        events=[self.store.append(self.task['id'],'internal_state',kind,{'text':'旧状态'}) for kind in ('internal_state_input','internal_state_result')]
        events.append(self.store.append(self.task['id'],'tool','tool_result',{'tool':'dev.test','ok':True,'value':{'name':'tool_a','revision':'v2','passed':False,'independent_failures':[{'actual':3,'expected':4}]}}))
        records=[{'ref':e['ref'],'kind':e['kind'],'role':e['role'],'text':self.store.resolve(self.task['id'],e['ref'])['text']} for e in events]
        result=V.records_view(self.store,self.task['id'],records,events[-1]['seq']);self.assertEqual(len(result),1)
        self.assertFalse(result[0]['receipt']['value']['passed']);self.assertEqual(result[0]['receipt']['value']['revision'],'v2')
    def test_read_write_success_is_not_functional_verification(self):
        read=V.receipt_view({'tool':'workspace.read','ok':True,'value':{'text':'测试已通过'}})
        write=V.receipt_view({'tool':'workspace.write','ok':True,'value':{'content_acceptance':'not_evaluated','write_verification':{'matches_written_text':True}}})
        self.assertIn('不是功能测试',read['basis']);self.assertIn('未获事实或功能验收',write['basis'])
    def test_latest_real_receipt_survives_multiple_rest_or_suppressed_choices(self):
        from idle_autonomy import IdleAutonomy
        aut=IdleAutonomy(SimpleNamespace(store=self.store),Path(self.tmp.name),{})
        selection={'tool':'dev.test','goal':'检验v2','arguments':{'name':'test','revision':'v2'}}
        receipt={'tool':'dev.test','ok':True,'evidence_ref':'original:test','value':{'revision':'v2','passed':False}}
        with self.store.db(True) as db:
            db.execute('INSERT INTO idle_initiatives VALUES(?,?,?,?,?,?,?)',('one',self.task['id'],V.dump(selection),'a','executed',V.dump(receipt),1))
            for i in range(2,5):db.execute('INSERT INTO idle_initiatives VALUES(?,?,?,?,?,?,?)',(str(i),self.task['id'],V.dump({'tool':'','goal':'rest','arguments':{}}),str(i),'rest',None,i))
        value=aut.context(self.task);latest=next(r['receipt'] for r in value['recent'] if r.get('receipt'))
        self.assertEqual(latest['value']['revision'],'v2');self.assertFalse(latest['value']['passed'])
    def test_three_memory_tools_available_in_ordinary_and_development_schema(self):
        from unified_schema import ordinary_schema,tool_variant
        from idle_autonomy import IdleAutonomy
        schema=ordinary_schema({'tools':{k:'' for k in FIELDS},'memory_policy':'active-memory-pages-v1'})
        self.assertEqual(set(v['properties']['tool'].get('const') for v in schema['oneOf'] if v['properties']['action'].get('const')=='tool'),set(FIELDS))
        self.assertNotIn('recall',V.dump(schema));self.assertNotIn('read_memory',V.dump(schema))
        aut=IdleAutonomy(SimpleNamespace(store=self.store),Path(self.tmp.name),{})
        self.assertTrue(set(FIELDS)<=set(aut.allowed(self.task)))

if __name__=='__main__':unittest.main()
