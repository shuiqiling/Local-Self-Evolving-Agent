"""An internal input route sharing the current chat archive and real windows."""
from datetime import datetime,timezone
from pathlib import Path
import asyncio
import hashlib
import json
import sqlite3
import time
import uuid
from machine_state import collect

def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def sha(value):return hashlib.sha256(value if isinstance(value,bytes) else value.encode()).hexdigest()

SYSTEM=('你是水麒麟。当前通道是internal_state_input，不是用户聊天或工具指令。'
    '输入是程序刚采集的机器状态。请简短理解状态，指出需要关注的异常并给出建议；不要声称已执行操作。'
    '网速是采样期实际收发流量，不是可用带宽；温度为null表示未测到，不可编造。'
    '只输出JSON：summary（100字内）、attention_needed（布尔值）、suggestion（80字内）。')


def output_schema():
    return {'type':'object','properties':{'summary':{'type':'string','maxLength':100},
        'attention_needed':{'type':'boolean'},'suggestion':{'type':'string','maxLength':80}},
        'required':['summary','attention_needed','suggestion'],'additionalProperties':False}


def external_marker(db):
    columns={r[1] for r in db.execute('PRAGMA table_info(events)')}
    if not {'role','kind','body'} <= columns:
        return db.execute('SELECT MAX(seq),MAX(created) FROM events').fetchone()
    return db.execute('''SELECT MAX(seq),MAX(created) FROM events
        WHERE role!='internal_state' AND kind IN
          ('task_created','message','assistant_reply','constraint_update','tool_result',
           'retrieval_result','job_error','execution_rejected','job_canceled',
           'owner_acceptance_frozen','handoff_completed')
        AND NOT(kind='handoff_completed' AND
            COALESCE(json_extract(body,'$.reason'),'')='internal_state_context_limit')''').fetchone()


def activity(paths):
    busy=0;latest=0;identities=[];live_seq=0
    for i,path in enumerate(paths):
        if not Path(path).exists():continue
        with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) as db:
            busy+=db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            seq,created=external_marker(db)
            if i==0:
                live_seq=db.execute('SELECT MAX(seq) FROM events').fetchone()[0] or 0
                external_seq=seq or 0
            identities.append([str(path),seq or 0,created])
            if created:latest=max(latest,datetime.fromisoformat(created).timestamp())
    return {'foreground_jobs':busy,'latest_activity_at':latest,'idle_seconds':max(0,time.time()-latest) if latest else 0,
            'episode_key':sha(canonical(identities)),'source_cutoff_seq':live_seq,
            'live_external_seq':external_seq if paths and 'external_seq' in locals() else 0}


def learner_lease(path):
    import fcntl
    f=Path(path).open('a')
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:f.close();return None
    return f


class StateInput:
    def __init__(self,db,root,auto,adapter,http,options,*,activity_reader=activity,sampler=collect,lease_acquirer=learner_lease,shared_context=None):
        self.db,self.root,self.auto=Path(db),Path(root),Path(auto)
        self.adapter,self.http,self.options=adapter,http,options
        self.activity_reader,self.sampler,self.lease_acquirer=activity_reader,sampler,lease_acquirer
        self.shared_context=shared_context
        self.lock=asyncio.Lock()
        with self.connect() as c:
            c.execute('''CREATE TABLE IF NOT EXISTS internal_state_observations(
                id TEXT PRIMARY KEY,episode_key TEXT UNIQUE NOT NULL,source_cutoff_seq INTEGER NOT NULL,
                observed_at TEXT NOT NULL,snapshot TEXT,model_input TEXT,model_output TEXT,status TEXT NOT NULL,
                error TEXT,model_seconds REAL,adapter_version TEXT)''')
            columns={r[1] for r in c.execute('PRAGMA table_info(internal_state_observations)')}
            for name in ('context_task_id','context_window_id','input_event_ref','result_event_ref'):
                if name not in columns:c.execute('ALTER TABLE internal_state_observations ADD COLUMN '+name+' TEXT')
            c.execute("UPDATE internal_state_observations SET status='interrupted',error='Controller restarted during internal state delivery' WHERE status='running'")

    def connect(self):
        db=sqlite3.connect(self.db,timeout=10);db.row_factory=sqlite3.Row;return db

    def last(self):
        with self.connect() as c:row=c.execute('SELECT * FROM internal_state_observations ORDER BY observed_at DESC LIMIT 1').fetchone()
        if not row:return None
        result=dict(row)
        for key in ('snapshot','model_input','model_output'):
            if result[key]:result[key]=json.loads(result[key])
        return result

    def status(self):
        from training_priority import status as training_priority
        settings=json.loads((self.auto/'config.json').read_text(encoding='utf-8'))
        return {'channel':'internal_state_input','enabled':self.options.get('enabled',False),
            'trigger':'fresh_per_request_and_periodic_when_idle','options':self.options,'last':self.last(),
            'context':self.shared_context.status() if self.shared_context else {'mode':'standalone'},
            'training_priority':training_priority(self.auto,settings)}

    async def post_while_idle(self,body,episode_key):
        return await self.while_idle(self.http.post('/v1/chat/completions',json=body,timeout=180),episode_key)

    async def while_idle(self,awaitable,episode_key):
        task=asyncio.create_task(awaitable)
        try:
            while True:
                done,_=await asyncio.wait([task],timeout=.5)
                if done:return task.result()
                fresh=await asyncio.to_thread(self.activity_reader,self.options['idle_databases'])
                if fresh['foreground_jobs'] or fresh['episode_key']!=episode_key:
                    task.cancel();await asyncio.gather(task,return_exceptions=True)
                    return None
        finally:
            if not task.done():task.cancel();await asyncio.gather(task,return_exceptions=True)

    async def deliver(self,automatic=False):
        if not self.options.get('enabled'):return {'status':'disabled'}
        if self.lock.locked():return {'status':'state_delivery_in_progress'}
        async with self.lock:
            settings=json.loads((self.auto/'config.json').read_text(encoding='utf-8'))
            from training_priority import status as training_priority
            priority=training_priority(self.auto,settings)
            if priority['hold_background_inference']:
                return {'status':'yielding_to_lora','training_priority':priority}
            if settings.get('memory_capacity',{}).get('enabled'):
                from memory_capacity import capacity
                storage=capacity(settings['memory_capacity'])
                if not storage['write_allowed'] or not storage['optional_write_allowed']:
                    return {'status':'memory_capacity_limit','capacity':storage}
            state=await asyncio.to_thread(self.activity_reader,self.options['idle_databases'])
            if state['foreground_jobs'] or (automatic and state['idle_seconds']<self.options.get('min_idle_seconds',60)):
                return {'status':'waiting_for_idle','idle_seconds':state['idle_seconds']}
            # Preserve the existing unique column without migrating historical
            # rows, but use a fresh delivery key for every accepted request.
            episode_key=sha(canonical([state['episode_key'],uuid.uuid4().hex]))
            if automatic:
                with self.connect() as c:recent=c.execute('SELECT observed_at FROM internal_state_observations ORDER BY observed_at DESC LIMIT 1').fetchone()
                interval=self.options.get('automatic_interval_seconds',60)
                if recent and time.time()-datetime.fromisoformat(recent['observed_at']).timestamp()<interval:
                    return {'status':'automatic_interval_wait','interval_seconds':interval}
            lease=await asyncio.to_thread(self.lease_acquirer,self.auto/'learner.lock')
            if lease is None:return {'status':'learning_in_progress'}
            priority=training_priority(self.auto,settings)
            if priority['hold_background_inference']:
                lease.close();return {'status':'yielding_to_lora','training_priority':priority}
            if not priority['enabled'] and (self.auto/'pending-window-batch.json').exists():lease.close();return {'status':'learning_in_progress'}
            if self.http.generation_lock.locked():lease.close();return {'status':'model_busy'}
            identifier='state_'+uuid.uuid4().hex[:16];observed=datetime.now(timezone.utc).isoformat()
            try:
                with self.connect() as c:
                    c.execute('INSERT INTO internal_state_observations(id,episode_key,source_cutoff_seq,observed_at,status,adapter_version) VALUES(?,?,?,?,?,?)',
                        (identifier,episode_key,state['source_cutoff_seq'],observed,'running',self.adapter['version'] if self.adapter else None))
                snapshot=await asyncio.to_thread(self.sampler,self.root)
                if settings.get('memory_capacity',{}).get('enabled'):
                    snapshot['memory_storage']={key:storage[key] for key in ('limit_gib','used_percent','available_bytes','status','write_allowed','optional_write_allowed') if key in storage}
                config=json.loads((self.auto/'config.json').read_text(encoding='utf-8'))
                worker=json.loads((self.auto/'worker-status.json').read_text(encoding='utf-8')) if (self.auto/'worker-status.json').exists() else {}
                snapshot['agent']={'name':'水麒麟','model':'Qwen3-30B-A3B-Q4_K_M','adapter':self.adapter['version'] if self.adapter else None,
                    'automatic_lora_phase':worker.get('phase'),'new_windows':worker.get('new_handoffs'),
                    'windows_per_update':config.get('min_new_handoffs'),'memory_teacher_enabled':config.get('memory_teacher',{}).get('enabled',False),
                    'foreground_jobs':state['foreground_jobs'],'idle_seconds':round(state['idle_seconds'],1),
                    'state_input_trigger':'automatic_idle' if automatic else 'explicit_request'}
                after=await asyncio.to_thread(self.activity_reader,self.options['idle_databases'])
                if after['foreground_jobs'] or after['episode_key']!=state['episode_key']:
                    with self.connect() as c:c.execute("UPDATE internal_state_observations SET snapshot=?,status='superseded' WHERE id=?",(canonical(snapshot),identifier))
                    return {'status':'superseded_by_foreground_activity','id':identifier}
                with self.connect() as c:c.execute('UPDATE internal_state_observations SET snapshot=? WHERE id=?',(canonical(snapshot),identifier))
                if self.options.get('context_mode')=='shared_chat' and self.shared_context is None:
                    raise RuntimeError('Shared chat context is required but not attached')
                if self.shared_context is not None:
                    def on_request(body):
                        with self.connect() as c:c.execute('UPDATE internal_state_observations SET model_input=? WHERE id=?',(canonical(body),identifier))
                    result=await self.while_idle(self.shared_context.deliver(snapshot,identifier,SYSTEM,
                        output_schema(),self.options.get('max_output_tokens',160),on_request,parse_output,
                        state['live_external_seq']),state['episode_key'])
                    if result is None or result['status']!='completed':
                        status=(result or {}).get('status','preempted_by_foreground_activity')
                        with self.connect() as c:c.execute('UPDATE internal_state_observations SET snapshot=?,status=? WHERE id=?',(canonical(snapshot),status,identifier))
                        return {'status':status,'id':identifier}
                    return {**result,'id':identifier,'channel':'internal_state_input','snapshot':snapshot}
                body={'model':'Qwen3-30B-A3B-Q4_K_M','messages':[{'role':'system','content':SYSTEM},
                    {'role':'user','content':canonical({'channel':'internal_state_input','snapshot':snapshot})}],
                    'temperature':0,'cache_prompt':False,'max_tokens':self.options.get('max_output_tokens',160),
                    'response_format':{'type':'json_schema','json_schema':{'name':'InternalState','schema':output_schema(),'strict':True}}}
                if self.adapter:body['lora']=[{'id':self.adapter.get('id',0),'scale':1}]
                with self.connect() as c:c.execute('UPDATE internal_state_observations SET snapshot=?,model_input=? WHERE id=?',(canonical(snapshot),canonical(body),identifier))
                tick=time.monotonic();response=await self.post_while_idle(body,state['episode_key'])
                if response is None:
                    with self.connect() as c:c.execute("UPDATE internal_state_observations SET status='preempted',error='Foreground activity arrived; closed internal model request' WHERE id=?",(identifier,))
                    return {'status':'preempted_by_foreground_activity','id':identifier}
                response.raise_for_status();result=response.json();choice=result['choices'][0]
                assert choice.get('finish_reason')!='length','State response truncated'
                output=parse_output(choice['message']['content'])
                elapsed=round(time.monotonic()-tick,3)
                with self.connect() as c:c.execute("UPDATE internal_state_observations SET model_output=?,status='completed',model_seconds=? WHERE id=?",(canonical(output),elapsed,identifier))
                return {'status':'completed','id':identifier,'channel':'internal_state_input','snapshot':snapshot,'model_output':output,'model_seconds':elapsed}
            except Exception as error:
                with self.connect() as c:c.execute("UPDATE internal_state_observations SET status='failed',error=? WHERE id=?",(str(error),identifier))
                return {'status':'failed','id':identifier,'error':str(error)}
            finally:lease.close()


def parse_output(content):
    output=json.loads(content)
    assert set(output)=={'summary','attention_needed','suggestion'} and type(output['attention_needed']) is bool
    assert isinstance(output['summary'],str) and len(output['summary'])<=100
    assert isinstance(output['suggestion'],str) and len(output['suggestion'])<=80
    return output
