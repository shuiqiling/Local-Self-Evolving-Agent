"""Durable bounded execution slices, explicit pauses and native-safe context headroom."""
import hashlib,json,time,re
import httpx

VERSION='resumable-execution-v3'
MAX_STEPS=64
MAX_OUTPUT=1536
MAX_INPUT=6144
OUTPUT_FEEDBACK=('运行时反馈：上次输出被截断，任何其中的工具都没有执行。'
                 '请根据最新真实回执，只给出一个完整、简短的下一步调用。'
                 '不要生成连续重复的字符或换行；不要通过填充大段文本制造样例。'
                 '长内容应分步处理，已有文件需要查看时读取文件。')

def with_feedback(prompt,feedback):
    if any(m.get('role')=='system' and m.get('content')==feedback for m in prompt):return prompt
    return [*prompt[:-1],{'role':'system','content':feedback},prompt[-1]]

def encoded(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))

class RecoverySignal(Exception):
    def __init__(self,status,reason,state=None):
        super().__init__(reason);self.status=status
        self.result={'status':status,'reason':reason,'progress_saved':True,'recovery_version':VERSION,
                     'steps_done':(state or {}).get('steps_done',0),'automatic_resume':status=='queued'}

class Checkpoint:
    def __init__(self,store,job):
        self.store,self.job=store,job
        self.request_sha=hashlib.sha256(encoded(job['body']).encode()).hexdigest()
        with store.db() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS execution_checkpoints(
              job_id TEXT PRIMARY KEY,task_id TEXT NOT NULL,request_sha TEXT NOT NULL,state TEXT NOT NULL,updated REAL NOT NULL)''')
            row=db.execute('SELECT * FROM execution_checkpoints WHERE job_id=?',(job['id'],)).fetchone()
        if row:
            assert row['task_id']==job['task_id'] and row['request_sha']==self.request_sha,'Checkpoint scope mismatch'
            self.state=json.loads(row['state'])
        else:self.state={'steps_done':0,'initialized':False,'rounds':0}
        self.state.setdefault('progress_mark',self.progress())

    def progress(self):
        items=[(o['id'],o['status']) for o in self.store.operations(self.job['task_id'])
               if o.get('user_event')==self.job['body'].get('user_event')]
        return hashlib.sha256(encoded(items).encode()).hexdigest()

    def save(self,**changes):
        self.state.update(changes)
        with self.store.db() as db:
            db.execute('''INSERT INTO execution_checkpoints VALUES(?,?,?,?,?) ON CONFLICT(job_id)
              DO UPDATE SET state=excluded.state,updated=excluded.updated''',
              (self.job['id'],self.job['task_id'],self.request_sha,encoded(self.state),time.time()))

    def pause(self,reason):
        self.save(pause_reason=reason)
        raise RecoverySignal('paused',reason,self.state)

    def decision(self,decision):
        mark=self.progress();sig=hashlib.sha256(encoded(decision.model_dump()).encode()).hexdigest()
        same=sig==self.state.get('decision_signature') and mark==self.state.get('decision_progress')
        repetitions=self.state.get('repetitions',0)+1 if same else 1
        self.save(decision_signature=sig,decision_progress=mark,repetitions=repetitions)
        if repetitions>=3:self.pause('同一决策连续重复且没有新增执行进展，已保存进度；需要调整方法后继续。')

    def finish_slice(self):
        mark=self.progress();progressed=mark!=self.state.get('progress_mark')
        if self.state['steps_done']>=MAX_STEPS:self.pause('本次执行已达64步保护上限，已保存断点；可显式续跑，不会无限循环。')
        if not progressed:self.pause('本段未产生新增执行进展，已保存断点，避免反复空转。')
        self.save(progress_mark=mark,rounds=self.state['rounds']+1)
        self.store.append(self.job['task_id'],'runtime','execution_slice_saved',
                          {'job_id':self.job['id'],'steps_done':self.state['steps_done'],'automatic_resume':True})
        raise RecoverySignal('queued','执行分段完成，继续同一作业的下一段。',self.state)

    def status(self,runtime,status,result):
        # A user cancellation wins even if it arrives at the yield boundary.
        with self.store.db(True) as db:
            row=db.execute('SELECT status FROM jobs WHERE id=?',(self.job['id'],)).fetchone()
            if not row or row[0]=='canceled':return
            db.execute('UPDATE jobs SET status=?,result=?,error=NULL WHERE id=?',(status,encoded(result),self.job['id']))
        self.store.append(self.job['task_id'],'runtime','execution_recovery',{'job_id':self.job['id'],**result})

async def input_budget(runtime,prompt,checkpoint,output_limit=None):
    count=await runtime.model.tokens(prompt);normal=runtime.settings.working_tokens
    reasoning=getattr(runtime.model,'reasoning',{})
    reserve=(int(reasoning.get('reasoning_budget_tokens',256))+16) if reasoning.get('enabled') else 0
    output=output_limit or runtime.settings.primary_output_tokens
    hard=runtime.settings.backend_context-output-reserve-128
    if count<=min(normal,hard):return normal
    ceiling=min(MAX_INPUT,hard)
    if count<=ceiling:
        runtime.store.append(checkpoint.job['task_id'],'runtime','context_budget_recovery',
             {'job_id':checkpoint.job['id'],'input_tokens':count,'normal_budget':normal,'temporary_budget':ceiling,
              'reasoning_reserve':reserve,'output_reserve':output,'backend_context':runtime.settings.backend_context,
              'policy':'single_call_headroom_not_global_window_change'})
        return ceiling
    checkpoint.pause(f'必要输入为{count} tokens，超过本次可用预算{ceiling}；原文与进度已保存，请缩小当前步骤或分页读取。')

async def call(runtime,job,prompt,schema,checkpoint):
    limit=runtime.settings.primary_output_tokens
    feedback=checkpoint.state.get('output_recovery_feedback')
    if feedback:prompt=with_feedback(prompt,feedback)
    for attempt in range(2):
        budget=await input_budget(runtime,prompt,checkpoint,limit)
        try:
            decision=await runtime.model.call(job['task_id'],'primary',prompt,schema,limit=limit,budget=budget)
            if checkpoint.state.get('output_recovery_feedback'):checkpoint.save(output_recovery_feedback=None)
            return decision
        except (TimeoutError,httpx.TimeoutException):
            checkpoint.pause('模型调用超过等待上限，未执行未完成的输出；进度已保存，服务正常后可继续。')
        except ValueError as error:
            if 'response exceeded its reserved output budget' not in str(error):raise
            text=getattr(error,'truncated_output','')
            repetitive=bool(re.search(r'(.{1,24})\1{31,}',text[-8192:],re.DOTALL))
            feedback=OUTPUT_FEEDBACK
            checkpoint.save(output_recovery_feedback=feedback)
            if attempt or limit>=MAX_OUTPUT:checkpoint.pause('结构化输出重试后仍被截断；没有执行残缺内容，进度和纠错反馈已保存。')
            # Identical greedy input would reproduce the same failure. Add feedback
            # before counting the next prompt; do not expand obvious repetition.
            prompt=with_feedback(prompt,feedback)
            if not repetitive:limit=min(MAX_OUTPUT,limit*2)
            runtime.store.append(job['task_id'],'runtime','output_budget_recovery',
                                 {'job_id':job['id'],'visible_output_limit':limit,'attempt':2,'tool_execution_retried':False,
                                  'repetitive_output_detected':repetitive,'corrective_feedback_added':True})

def resume(store,job_id):
    job=store.job(job_id)
    if job['kind']!='message' or job['status']!='paused':raise ValueError('Only paused execution jobs can resume from a checkpoint')
    cp=Checkpoint(store,job)
    if (cp.state.get('pause_reason') or '').startswith('结构化输出'):
        cp.save(output_recovery_feedback=OUTPUT_FEEDBACK)
    # Reset the bounded allowance, not the source, last receipt or deduplication identity.
    cp.save(steps_done=0,rounds=0,repetitions=0,pause_reason=None,progress_mark=cp.progress())
    with store.db(True) as db:
        changed=db.execute("UPDATE jobs SET status='queued',error=NULL,result=NULL WHERE id=? AND status='paused'",(job_id,)).rowcount
        if not changed:raise ValueError('Job changed while resuming')
        store._event(db,store._task(db,job['task_id']),'user','job_resumed',{'job_id':job_id,'checkpoint_preserved':True})
    return store.job(job_id)
