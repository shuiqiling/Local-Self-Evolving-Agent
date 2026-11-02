import bootstrap
import asyncio,copy,json,time,threading,unittest
from types import SimpleNamespace
from unittest.mock import patch,AsyncMock
import test_idle_regression as prior
import idle_state_input as idle
from idle_sessions import IdleSessions

class Sessions(unittest.TestCase):
    def setUp(self):
        self.f=prior.InitiativeTests();self.f.setUp();f=self.f
        f.config['idle_autonomy'].update(session_enabled=True,activity_budget_seconds=300,max_consecutive_no_progress=2)
        (f.auto/'config.json').write_text(json.dumps(f.config));f.aut.options=dict(f.config['idle_autonomy'])
        f.gate.sessions=IdleSessions(f.gate);self.owner=f
    def tearDown(self):self.f.tearDown()
    def output(self,tool='',args=None,kind='tool',goal='验证短笔记写入与读回'):
        return {'summary':'依据当前回执继续','attention_needed':False,'suggestion':'继续验证或停止',
                'initiative':{'kind':kind,'goal':goal,'tool':tool,'arguments':args or {}}}
    def done(self):return self.output(args={'evidence_refs':self.f.gate.sessions.current['refs'][-2:]},kind='done')
    def serve(self,fn):
        async def post(path,**kw):
            self.f.http.calls.append(kw['json']);n=len(self.f.http.calls)
            result=fn(n,kw['json'])
            if asyncio.iscoroutine(result):result=await result
            return SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'choices':[{'finish_reason':'stop','message':{'content':json.dumps(result)}}]})
        self.f.http.post=post
    def run_session(self):return asyncio.run(self.f.gate.deliver())

    def test_write_read_done_same_activity_real_receipts(self):
        f=self.f
        def decide(n,body):
            current=json.loads(body['messages'][-1]['content'])['initiative_context']
            if n==1:return self.output('workspace.write',{'path':'self_notes/check.txt','text':'只保存已经验证的回执。'})
            self.assertTrue(current['recent'][0]['receipt']['ok'])
            self.assertEqual(current['activity']['goal'],'验证短笔记写入与读回')
            if n==2:return self.output('workspace.read',{'path':'self_notes/check.txt'})
            self.assertEqual(current['recent'][0]['receipt']['value']['text'],'只保存已经验证的回执。')
            return self.done()
        self.serve(decide);out=self.run_session()
        self.assertEqual(out['activity']['stop_reason'],'goal_complete');self.assertEqual(out['activity']['actions'],2)
        self.assertEqual(out['activity']['decisions'],3);self.assertFalse(out['activity']['independent_goal_acceptance'])
        self.assertEqual(len(f.store.operations(f.task['id'])),2)
        with f.store.db() as db:self.assertEqual(db.execute('select count(distinct context_window_id) from internal_state_observations').fetchone()[0],1)

    def test_failure_then_corrected_action_resets_no_progress(self):
        def decide(n,body):
            if n==1:return self.output('workspace.read',{'path':'missing.txt'})
            if n==2:
                context=json.loads(body['messages'][-1]['content'])['initiative_context']
                self.assertFalse(context['recent'][0]['receipt']['ok']);self.assertEqual(context['activity']['consecutive_no_progress'],1)
                return self.output('workspace.write',{'path':'self_notes/fixed.txt','text':'缺失文件，创建新的笔记。'})
            self.assertEqual(self.f.gate.sessions.current['no_progress'],0);return self.done()
        self.serve(decide);out=self.run_session();self.assertEqual(out['activity']['stop_reason'],'goal_complete')

    def test_repeated_action_stops_after_two_no_progress(self):
        p=self.f.r.tools.path(self.f.task['id'],'source.txt');p.write_text('unchanged')
        self.serve(lambda *_:self.output('workspace.read',{'path':'source.txt'}))
        out=self.run_session();self.assertEqual(out['activity']['stop_reason'],'consecutive_no_progress')
        self.assertEqual(len(self.f.http.calls),3);self.assertEqual(len(self.f.store.operations(self.f.task['id'])),1)

    def test_same_content_different_paths_is_not_new_evidence(self):
        for n in range(1,4):self.f.r.tools.path(self.f.task['id'],f'{n}.txt').write_text('same observation')
        self.serve(lambda n,_:self.output('workspace.read',{'path':f'{n}.txt'}))
        out=self.run_session();self.assertEqual(out['activity']['stop_reason'],'consecutive_no_progress');self.assertEqual(len(self.f.http.calls),3)

    def test_invalid_completion_refs_cannot_certify_completion(self):
        self.serve(lambda *_:self.output(args={'evidence_refs':['old:forged']},kind='done'))
        out=self.run_session();self.assertEqual(out['activity']['stop_reason'],'consecutive_no_progress')
        self.assertEqual(out['activity']['actions'],0)

    def test_rest_stops_without_second_inference(self):
        self.serve(lambda *_:self.output(kind='rest'));out=self.run_session()
        self.assertEqual(out['activity']['stop_reason'],'model_rest');self.assertEqual(len(self.f.http.calls),1)

    def test_training_arrives_during_model_request_cancel_and_release(self):
        pending=False;cancelled=False;closed=[]
        self.f.gate.lease_acquirer=lambda _:SimpleNamespace(close=lambda:closed.append(True))
        async def decide(n,_):
            nonlocal pending,cancelled
            if n==1:return self.output('workspace.list',{'path':'.'})
            pending=True
            try:await asyncio.sleep(20)
            except asyncio.CancelledError:cancelled=True;raise
        self.serve(decide)
        with patch('training_priority.status',side_effect=lambda *_:{'hold_background_inference':pending,'enabled':True}):out=self.run_session()
        self.assertEqual(out['activity']['stop_reason'],'yielding_to_lora');self.assertTrue(cancelled)
        self.assertEqual(len(closed),2);self.assertEqual(len(self.f.store.operations(self.f.task['id'])),1)

    def test_foreground_arrives_during_second_inference(self):
        async def decide(n,_):
            if n==1:return self.output('workspace.list',{'path':'.'})
            self.f.store.add_job(self.f.task['id'],'message',{'text':'先回答用户'})
            await asyncio.sleep(20)
        self.serve(decide);out=self.run_session()
        self.assertEqual(out['activity']['stop_reason'],'preempted_by_foreground_activity');self.assertEqual(len(self.f.store.operations(self.f.task['id'])),1)

    def test_budget_cancels_generation(self):
        clock=[0.];self.f.gate.sessions.clock=lambda:clock[0]
        async def decide(n,_):
            if n==1:return self.output('workspace.list',{'path':'.'})
            clock[0]=301.;await asyncio.sleep(20)
        self.serve(decide);out=self.run_session()
        self.assertEqual(out['activity']['stop_reason'],'time_budget');self.assertEqual(len(self.f.store.operations(self.f.task['id'])),1)

    def test_training_between_steps_prevents_next_generation(self):
        pending=False;original=self.f.r.tools.execute
        def action(*args,**kwargs):
            nonlocal pending
            receipt=original(*args,**kwargs);pending=True;return receipt
        self.f.r.tools.execute=action;self.serve(lambda *_:self.output('workspace.list',{'path':'.'}))
        with patch('training_priority.status',side_effect=lambda *_:{'hold_background_inference':pending,'enabled':True}):out=self.run_session()
        self.assertEqual(out['activity']['stop_reason'],'yielding_to_lora');self.assertEqual(len(self.f.http.calls),1)

    def test_inflight_write_finishes_receipt_before_foreground_yield(self):
        original=self.f.r.tools.execute
        def action(*args,**kwargs):
            self.f.store.add_job(self.f.task['id'],'message',{'text':'用户抢占'})
            return original(*args,**kwargs)
        self.f.r.tools.execute=action
        self.serve(lambda *_:self.output('workspace.write',{'path':'self_notes/drain.txt','text':'只执行一次'}))
        out=self.run_session();self.assertEqual(out['activity']['stop_reason'],'preempted_by_foreground_activity')
        self.assertTrue(self.f.aut.record(out['id'])['receipt']['ok']);self.assertEqual(len(self.f.http.calls),1)

    def test_concurrent_request_does_not_start_second_activity(self):
        async def scenario():
            started=asyncio.Event();release=asyncio.Event()
            async def decide(*_):started.set();await release.wait();return self.output(kind='rest')
            self.serve(decide);first=asyncio.create_task(self.f.gate.deliver());await started.wait()
            second=await self.f.gate.deliver();release.set();await first
            self.assertEqual(second['status'],'state_delivery_in_progress')
        asyncio.run(scenario());self.assertEqual(len(self.f.http.calls),1)

    def test_restart_marks_activity_interrupted_never_resumes(self):
        with self.f.store.db(True) as db:db.execute("INSERT INTO idle_activity_sessions(id,task_id,status,started,goal,budget_seconds) VALUES('stale',?,'running',0,'old',300)",(self.f.task['id'],))
        sessions=IdleSessions(self.f.gate);self.assertEqual(sessions.last()['stop_reason'],'process_restarted_no_automatic_resume')
        self.assertEqual(len(self.f.http.calls),0)

    def test_real_query_origin_does_not_become_foreground_activity(self):
        from lea.evidence import EvidenceMemory
        from lea.retrieval import HybridMemory
        from lea.memory import Memory
        before=idle.activity([self.f.store.path]);intent=self.f.store.append(self.f.task['id'],'internal_state','idle_action_selected',{'selection':{}})
        memory=object.__new__(EvidenceMemory);memory.store=self.f.store;memory.strategy='hybrid_rerank'
        memory._prepare_evidence=lambda *_:None
        result={'request_id':'query_fixture','timings':{},'evidence':[]}
        with patch.object(HybridMemory,'query',new=AsyncMock(return_value=result)):
            asyncio.run(memory.query(self.f.task['id'],'查历史',exclude_event_id=intent['id']))
        after=idle.activity([self.f.store.path]);self.assertEqual(before['episode_key'],after['episode_key'])
        self.f.store.append(self.f.task['id'],'retriever','retrieval_result',{'request_id':'external'})
        self.assertNotEqual(after['episode_key'],idle.activity([self.f.store.path])['episode_key'])

    def test_user_activity_between_steps_cannot_rebind_activity_to_new_task(self):
        original=self.f.r.tools.execute
        def action(*args,**kwargs):
            receipt=original(*args,**kwargs);self.f.store.create_task('new owner task',[],[],[]);return receipt
        self.f.r.tools.execute=action;self.serve(lambda *_:self.output('workspace.list',{'path':'.'}))
        out=self.run_session();self.assertEqual(out['activity']['stop_reason'],'preempted_by_foreground_activity');self.assertEqual(len(self.f.http.calls),1)

if __name__=='__main__':unittest.main()
