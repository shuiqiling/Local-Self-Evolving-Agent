"""Receipt-driven, bounded idle activities; each decision still admits one tool."""
import asyncio
import hashlib
import json
import time
import uuid

VERSION='idle-activity-v1'

def dump(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))

TRANSIENT={'evidence_ref','initiative_event_ref','operation_id','query_id','request_id','index_id',
           'next_cursor','cursor','search_id','retrieved_at','fetched_at','checked_at','seconds','elapsed_seconds'}
def semantic(value):
    if isinstance(value,dict):return {k:semantic(v) for k,v in value.items() if k not in TRANSIENT}
    if isinstance(value,list):return [semantic(v) for v in value]
    return value

class IdleSessions:
    def __init__(self,gate):
        self.gate=gate;self.aut=gate.shared_context.autonomy;self.store=self.aut.store
        self.lock=asyncio.Lock();self.current=None;self.clock=time.monotonic
        with self.store.db(True) as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS idle_activity_sessions(
              id TEXT PRIMARY KEY,task_id TEXT NOT NULL,status TEXT NOT NULL,started REAL NOT NULL,
              finished REAL,goal TEXT NOT NULL,stop_reason TEXT,budget_seconds REAL NOT NULL,
              decisions INTEGER NOT NULL DEFAULT 0,actions INTEGER NOT NULL DEFAULT 0,
              no_progress INTEGER NOT NULL DEFAULT 0,last_observation_id TEXT,completion_refs TEXT);
              CREATE TABLE IF NOT EXISTS idle_activity_steps(
              session_id TEXT NOT NULL,ordinal INTEGER NOT NULL,observation_id TEXT,
              execution_status TEXT,tool TEXT,progress INTEGER NOT NULL,receipt_ref TEXT,
              PRIMARY KEY(session_id,ordinal));''')
            db.execute("UPDATE idle_activity_sessions SET status='stopped',stop_reason='process_restarted_no_automatic_resume',finished=? WHERE status='running'",(time.time(),))

    def last(self):
        with self.store.db() as db:row=db.execute('SELECT * FROM idle_activity_sessions ORDER BY started DESC LIMIT 1').fetchone()
        return dict(row) if row else None

    def view(self):
        s=self.current
        if not s:return None
        return {'id':s['id'],'goal':s['goal'] or '自主选择一件有用的小事，选定后围绕同一目标继续',
                'decisions_completed':s['decisions'],'actions_completed':s['actions'],
                'remaining_seconds':max(0,round(s['deadline']-self.clock(),1)),
                'consecutive_no_progress':s['no_progress'],'stop_after_no_progress':s['max_no_progress'],
                'last_step':s.get('last_step'),'successful_receipt_refs':s['refs'][-4:],
                'verified_actions':s['executed'][-8:],
                'verified_tool_counts':{tool:sum(x['tool']==tool for x in s['executed']) for tool in self.aut.allowed(self.store.task(s['task_id']))},
                'rule':'依据最新真实回执决定下一步；完成用done并引用本活动回执；无事可做用rest。不能把重复读取或笔记当功能验收。'}

    async def guard(self):
        s=self.current
        if s and self.clock()>=s['deadline']:return 'time_budget'
        fresh=await asyncio.to_thread(self.gate.activity_reader,self.gate.options['idle_databases'])
        if fresh['foreground_jobs'] or (s and fresh['episode_key']!=s['external_key']):return 'preempted_by_foreground_activity'
        from training_priority import status
        config=json.loads((self.gate.auto/'config.json').read_text(encoding='utf-8'))
        priority=await asyncio.to_thread(status,self.gate.auto,config)
        if priority['hold_background_inference']:return 'yielding_to_lora'
        return None

    def persist(self,reason=None):
        s=self.current
        with self.store.db(True) as db:
            db.execute('''UPDATE idle_activity_sessions SET status=?,finished=?,goal=?,stop_reason=?,decisions=?,
              actions=?,no_progress=?,last_observation_id=?,completion_refs=? WHERE id=?''',
              ('stopped' if reason else 'running',time.time() if reason else None,s['goal'],reason,s['decisions'],
               s['actions'],s['no_progress'],s.get('last_observation'),dump(s.get('completion_refs',[])),s['id']))

    def observe(self,result):
        s=self.current;s['decisions']+=1;s['last_observation']=result['id']
        selection=result['model_output']['initiative'];execution=result.get('initiative_execution',{})
        saved=self.aut.record(result['id']) or {};receipt=saved.get('receipt') or execution.get('receipt')
        if selection['kind']=='tool' and not s['goal']:s['goal']=selection['goal']
        progressed=False;ref=None
        if execution.get('status')=='executed':s['actions']+=1
        if execution.get('status')=='executed' and receipt and receipt.get('ok') and not execution.get('deduplicated') and not receipt.get('deduplicated') and not (selection['tool'] in ('dev.test','dev.run') and (receipt.get('value') or {}).get('passed') is not True):
            ref=receipt.get('evidence_ref') or receipt.get('initiative_event_ref')
            value=receipt.get('value')
            if selection['tool']=='workspace.write':value={'text':selection['arguments'].get('text'),'receipt':semantic(value)}
            fingerprint=hashlib.sha256(dump([selection['tool'],semantic(value)]).encode()).hexdigest()
            progressed=fingerprint not in s['seen'];s['seen'].add(fingerprint)
            if ref and ref not in s['refs']:s['refs'].append(ref)
            from idle_development import argument_view
            s['executed'].append({'tool':selection['tool'],'arguments':argument_view(selection['tool'],selection['arguments']),'receipt_ref':ref})
        if selection['kind']=='tool':s['no_progress']=0 if progressed else s['no_progress']+1
        s['last_step']={'tool':selection['tool'],'status':execution.get('status'),'new_evidence':progressed,
                        'receipt_ref':ref,'error':(receipt or {}).get('error')}
        with self.store.db(True) as db:
            db.execute('INSERT INTO idle_activity_steps VALUES(?,?,?,?,?,?,?)',
              (s['id'],s['decisions'],result['id'],execution.get('status'),selection['tool'],int(progressed),ref))
        if selection['kind']=='done':
            refs=selection['arguments']['evidence_refs']
            if refs and all(ref in s['refs'] for ref in refs):
                s['completion_refs']=refs;return 'goal_complete'
            s['no_progress']+=1;s['last_step']['error']='完成引用必须来自本活动已成功执行的真实回执；当前引用无效。'
        if selection['kind']=='rest':return 'model_rest'
        if execution.get('status') in ('yielding_to_lora','preempted_by_foreground_activity','time_budget','memory_capacity_limit','disabled'):
            return execution['status']
        if s['no_progress']>=s['max_no_progress']:return 'consecutive_no_progress'
        return None

    async def deliver(self,automatic=False):
        if self.lock.locked():return {'status':'state_delivery_in_progress','activity':self.last()}
        async with self.lock:
            if not self.gate.options.get('enabled'):return {'status':'disabled'}
            state=await asyncio.to_thread(self.gate.activity_reader,self.gate.options['idle_databases'])
            reason=await self.guard()
            if reason:return {'status':reason}
            if automatic and state['idle_seconds']<self.gate.options.get('min_idle_seconds',60):return {'status':'waiting_for_idle'}
            config=json.loads((self.gate.auto/'config.json').read_text(encoding='utf-8'))
            if config.get('memory_capacity',{}).get('enabled'):
                from memory_capacity import capacity
                if not capacity(config['memory_capacity'])['optional_write_allowed']:return {'status':'memory_capacity_limit'}
            tid=self.gate.shared_context.target()
            if not tid:return {'status':'waiting_for_chat_context'}
            options=config.get('idle_autonomy',{})
            budget=float(options.get('activity_budget_seconds',300));limit=int(options.get('max_consecutive_no_progress',2))
            if not 0<budget<=1800 or not 1<=limit<=5:raise ValueError('Invalid idle activity budgets')
            start=self.clock();s={'id':'idle_'+uuid.uuid4().hex[:16],'task_id':tid,'goal':'','decisions':0,'actions':0,
                'no_progress':0,'max_no_progress':limit,'deadline':start+budget,'external_key':state['episode_key'],'refs':[],'seen':set(),'executed':[]}
            self.current=s
            with self.store.db(True) as db:
                db.execute('INSERT INTO idle_activity_sessions(id,task_id,status,started,goal,budget_seconds) VALUES(?,?,?,?,?,?)',
                           (s['id'],tid,'running',time.time(),'',budget))
            self.aut.activity_view=self.view;self.aut.activity_guard=self.guard;self.gate.activity_guard=self.guard
            result={};reason=None
            try:
                while not reason:
                    reason=await self.guard()
                    if reason:break
                    result=await self.gate._deliver_step(automatic=automatic,continuation=s['decisions']>0)
                    if result.get('status')!='completed':reason=result.get('status','step_failed');break
                    reason=self.observe(result);self.persist()
                    # The previous step has released its task lock and learner lease.
                    # Immediately re-check higher-priority work before another decision.
                    await asyncio.sleep(0)
            except asyncio.CancelledError:
                reason='caller_cancelled_no_automatic_resume';raise
            except Exception:
                reason='activity_error';raise
            finally:
                reason=reason or 'activity_interrupted';self.persist(reason)
                self.aut.activity_view=None;self.aut.activity_guard=None;self.gate.activity_guard=None;self.current=None
            activity=self.last();activity['elapsed_seconds']=round(self.clock()-start,3)
            activity['completion_is_model_judgment']=reason=='goal_complete'
            activity['independent_goal_acceptance']=False
            return {**result,'status':result.get('status','stopped'),'activity':activity}
