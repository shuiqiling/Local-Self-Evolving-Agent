import bootstrap
import lea.runtime
import asyncio
import copy
import gc
import json
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lea.store import Store
from lea.runtime import Runtime
from lea.config import Settings
from lea.tools import TOOL_HELP
from unified_tools import UnifiedTools,EXTRA_HELP
from shared_context import SharedChatContext
import idle_state_input as idle
from idle_autonomy import IdleAutonomy,VERSION,validate
TOOL_HELP.update(EXTRA_HELP)


class InitiativeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();root=Path(self.tmp.name)
        self.store=Store(root/'data');self.task=self.store.create_task('共享用户任务',['保留原文'],[],['workspace.list','workspace.read','workspace.write','web.search','dev.list','dev.run','dev.test'])
        self.store.append(self.task['id'],'user','message',{'text':'记住编号Q-831。'})
        self.r=object.__new__(Runtime);self.r.store=self.store;self.r.locks=defaultdict(asyncio.Lock)
        self.r.settings=Settings(data_dir=self.store.root,working_tokens=5000)
        class Model:
            adapter={'id':0,'version':'fixture'}
            async def tokens(self,prompt):return 100
        self.r.model=Model();self.r.tools=UnifiedTools(self.store)
        import continuity_state as C
        self.r.read_context_source=lambda *a:C.read_source(self.r,*a)
        self.r.handoff=lambda tid,reason:C.rollover(self.r,tid,reason)
        self.output={'summary':'CPU29度，当前可空闲学习','attention_needed':False,'suggestion':'做一个小步骤',
                     'initiative':{'kind':'tool','goal':'记录一个分页边界想法','tool':'workspace.write','arguments':{'path':'self_notes/paging.txt','text':'分页末尾应限制在原文长度内；这是待验证想法。'}}}
        owner=self
        class HTTP:
            callback=None
            def __init__(self):self.calls=[];self.generation_lock=asyncio.Lock()
            async def post(self,path,**kw):
                self.calls.append(kw['json'])
                if self.callback:self.callback()
                return SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'choices':[{'finish_reason':'stop','message':{'content':json.dumps(owner.output,ensure_ascii=False)}}]})
        self.http=HTTP();self.auto=root/'auto';self.auto.mkdir()
        self.config={'min_new_handoffs':10,'memory_teacher':{'enabled':False},'idle_autonomy':{'enabled':True}}
        (self.auto/'config.json').write_text(json.dumps(self.config))
        self.aut=IdleAutonomy(self.r,self.auto,self.config['idle_autonomy']);self.shared=SharedChatContext(self.r,self.http,autonomy=self.aut)
        self.gate=idle.StateInput(self.store.path,root,self.auto,Model.adapter,self.http,
            {'enabled':True,'context_mode':'shared_chat','idle_databases':[str(self.store.path)],'min_idle_seconds':0,'automatic_interval_seconds':0,'max_output_tokens':512},
            sampler=lambda _:{'cpu':{'temperature_c':29,'temperature_sensors':[]},'memory':{},'network':{}},
            lease_acquirer=lambda _:SimpleNamespace(close=lambda:None),shared_context=self.shared)

    def tearDown(self):gc.collect();self.tmp.cleanup()
    def run_once(self):return asyncio.run(self.gate.deliver())

    def test_one_inference_real_write_and_next_round_receipt(self):
        first=self.run_once();self.assertEqual(first['initiative_execution']['status'],'executed')
        self.assertEqual(len(self.http.calls),1)
        self.assertEqual(self.r.tools.path(self.task['id'],'self_notes/paging.txt').read_text(encoding='utf-8'),self.output['initiative']['arguments']['text'])
        self.assertTrue(first['initiative_execution']['receipt']['value']['write_verification']['matches_written_text'])
        self.assertTrue(self.gate.last()['initiative_execution']['receipt']['ok'])
        self.output['initiative']={'kind':'rest','goal':'等下次再继续','tool':'','arguments':{}}
        second=self.run_once();self.assertEqual(second['initiative_execution']['status'],'rest')
        current=json.loads(self.http.calls[-1]['messages'][-1]['content'])
        self.assertTrue(current['initiative_context']['recent'][0]['receipt']['value']['matches_written_text'])
        self.assertEqual(first['context_window_id'],second['context_window_id'])
        background=json.loads(self.http.calls[-1]['messages'][-2]['content'])['shared_context']
        self.assertIn('Q-831',json.dumps(background,ensure_ascii=False))

    def test_self_action_not_a_fake_user_turn_or_idle_reset(self):
        before=idle.activity([self.store.path]);self.run_once();after=idle.activity([self.store.path])
        self.assertEqual(before['episode_key'],after['episode_key'])
        with self.store.db() as db:
            op=db.execute('SELECT user_event FROM operations').fetchone()
            source=db.execute('SELECT role,kind FROM events WHERE id=?',(op[0],)).fetchone()
            self.assertEqual(tuple(source),('internal_state','idle_action_selected'))
        self.store.add_job(self.task['id'],'message',{'text':'现在先回答我'})
        self.assertNotEqual(after['episode_key'],idle.activity([self.store.path])['episode_key'])

    def test_same_action_suppressed_across_observations(self):
        self.run_once();second=self.run_once()
        self.assertEqual(second['initiative_execution']['status'],'duplicate_suppressed')
        self.assertEqual(len(self.store.operations(self.task['id'])),1)

    def test_rest_executes_nothing(self):
        self.output['initiative']={'kind':'rest','goal':'暂时没有有价值的小事','tool':'','arguments':{}}
        result=self.run_once();self.assertEqual(result['initiative_execution']['status'],'rest')
        self.assertEqual(self.store.operations(self.task['id']),[])

    def test_new_user_during_inference_preempts_action(self):
        self.http.callback=lambda:self.store.add_job(self.task['id'],'message',{'text':'先停下回答我'})
        self.assertEqual(self.run_once()['status'],'preempted_by_foreground_activity')
        self.assertEqual(self.store.operations(self.task['id']),[])

    def test_foreground_arrives_after_inference_before_admission(self):
        original=self.aut.execute
        async def arrive(result,*args):
            self.store.add_job(self.task['id'],'message',{'text':'新用户任务'})
            return await original(result,*args)
        self.aut.execute=arrive
        result=self.run_once();self.assertEqual(result['initiative_execution']['status'],'preempted_by_foreground_activity')
        self.assertEqual(self.store.operations(self.task['id']),[])

    def test_training_priority_checked_again_before_action(self):
        original=self.aut.execute
        async def training(result,*args):
            with patch('training_priority.status',return_value={'hold_background_inference':True}):return await original(result,*args)
        self.aut.execute=training
        result=self.run_once();self.assertEqual(result['initiative_execution']['status'],'yielding_to_lora')
        self.assertEqual(self.store.operations(self.task['id']),[])

    def test_existing_note_is_not_overwritten(self):
        p=self.r.tools.path(self.task['id'],'self_notes/paging.txt');p.parent.mkdir();p.write_text('existing')
        result=self.run_once();self.assertEqual(result['initiative_execution']['status'],'failed')
        self.assertEqual(p.read_text(),'existing');self.assertEqual(self.store.operations(self.task['id']),[])

    def test_write_outside_own_notes_rejected(self):
        for path in ['novel_usage.txt','self_notes/../novel_usage.txt','/tmp/x','self_notes/nested/a.txt']:
            with self.subTest(path=path):
                output=copy.deepcopy(self.output);output['initiative']['arguments']['path']=path
                with self.assertRaises(ValueError):self.aut.parse(json.dumps(output),self.task)

    def test_forbidden_tools_and_malformed_arguments_rejected(self):
        for name,args in [('dev.publish',{}),('dev.submit',{}),('shell',{}),('workspace.write',{'path':'self_notes/a.txt'}),('context.read',{'ref':'x','offset':True,'field':''})]:
            with self.subTest(tool=name):
                output=copy.deepcopy(self.output);output['initiative'].update(tool=name,arguments=args)
                with self.assertRaises(ValueError):self.aut.parse(json.dumps(output),self.task)

    def test_tool_failure_is_kept_as_failure_receipt(self):
        self.output['initiative'].update(tool='workspace.read',arguments={'path':'missing.txt'})
        result=self.run_once();execution=result['initiative_execution']
        self.assertEqual(execution['status'],'executed');self.assertFalse(execution['receipt']['ok'])
        self.assertIn('error',execution['receipt'])

    def test_memory_source_read_is_actual_archived_text(self):
        source=next(e for e in self.store.events(self.task['id']) if e['kind']=='message')
        self.output['initiative'].update(tool='context.read',arguments={'ref':source['ref'],'offset':0,'field':''})
        result=self.run_once();receipt=result['initiative_execution']['receipt']
        self.assertTrue(receipt['ok']);self.assertIn('Q-831',receipt['value']['text'])
        self.assertEqual(receipt['value']['ref'],source['ref'])

    def test_changed_model_intent_cannot_execute_uncommitted_action(self):
        original=self.aut.execute
        async def forged(result,*args):
            result=copy.deepcopy(result);result['model_output']['initiative']['arguments']['path']='self_notes/forged.txt'
            return await original(result,*args)
        self.aut.execute=forged
        result=self.run_once();self.assertEqual(result['status'],'failed')
        self.assertEqual(self.store.operations(self.task['id']),[])

    def test_interrupted_action_never_automatically_retried(self):
        self.run_once()
        with self.store.db(True) as db:db.execute("UPDATE idle_initiatives SET status='running',created=0")
        self.shared.autonomy=IdleAutonomy(self.r,self.auto,{'enabled':True})
        result=self.run_once();self.assertEqual(result['initiative_execution']['status'],'duplicate_suppressed')
        self.assertEqual(len(self.store.operations(self.task['id'])),1)


if __name__=='__main__':unittest.main()
