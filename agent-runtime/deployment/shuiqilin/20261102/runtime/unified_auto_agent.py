"""The same unified Agent, with a validated optional30B adapter and learner status."""
from dataclasses import replace
from pathlib import Path
import asyncio
import hashlib
import json
import os
import time

import lea.runtime as runtime
from lea.store import Store
from lea.tools import TOOL_HELP
from unified_agent_v11 import UnifiedModel,UnifiedRuntime,UNIFIED_PRIMARY,app_factory,settings,ReadyMemory
from unified_tools import UnifiedTools,EXTRA_HELP
from compact_primary import COMPACT_PRIMARY,PROMPT_VERSION

R=Path('/home/ubuntu/AI-Native/embodied-agent')
AUTO=Path(os.environ.get('SQ_AUTOLORA_ROOT',str(R/'auto-lora-shuiqilin')))

def active_adapter():
    path=Path(os.environ.get('SQ_ADAPTER_MANIFEST',str(AUTO/'active.json')))
    if not path.exists():return None
    value=json.loads(path.read_text())
    from auto_memory_policy import validate_runtime_adapter
    return validate_runtime_adapter(value,allow_candidate=os.environ.get('SQ_ALLOW_CANDIDATE')=='1')

class AdapterGuard:
    def __init__(self,inner,adapter,reasoning=None):
        self.inner,self.adapter=inner,adapter;self.last_namespace=None;self.generation_lock=asyncio.Lock()
        self.reasoning=reasoning or {'enabled':False}
    def prepare(self,body):
        from reasoning_policy import prepare_daily
        return prepare_daily(body,self.reasoning)
    async def get(self,*args,**kwargs):return await self.inner.get(*args,**kwargs)
    async def post(self,path,**kwargs):
        if path!='/v1/chat/completions':
            if path=='/apply-template' and self.reasoning.get('enabled'):
                kwargs['json']={**kwargs['json'],'chat_template_kwargs':{'enable_thinking':True},'reasoning_format':'deepseek'}
            return await self.inner.post(path,**kwargs)
        async with self.generation_lock:
            body=self.prepare(kwargs.get('json'));final_limit=body.final_tokens
            assert body['model']=='Qwen3-30B-A3B-Q4_K_M'
            if self.adapter:
                entries=body.get('lora',[])
                selected=self.adapter.get('id',0)
                assert len(entries)==1 and entries[0]['id']==selected and entries[0]['scale'] in (0,1)
                scale=float(os.environ.get('SQ_ADAPTER_SCALE','1'));assert scale in (0.,1.)
                body={**body,'lora':[{'id':i,'scale':entries[0]['scale']*scale if i==selected else 0.} for i in self.adapter.get('loaded_ids',[selected])]}
            else:assert not body.get('lora'),'Adapter requested without validated manifest'
            namespace=tuple((x['id'],x['scale']) for x in body.get('lora',[]))
            if namespace!=self.last_namespace:body={**body,'cache_prompt':False}
            kwargs['json']=dict(body)
            try:response=await self.inner.post(path,**kwargs)
            except BaseException:
                self.last_namespace=None;raise
            self.last_namespace=namespace if response.is_success else None
            if self.reasoning.get('enabled') and response.is_success:
                result=response.json();content=result['choices'][0]['message'].get('content') or ''
                counted=await self.inner.post('/tokenize',json={'content':content,'add_special':False})
                counted.raise_for_status();visible_tokens=len(counted.json()['tokens'])
                if visible_tokens>final_limit:
                    import httpx
                    result['choices'][0]['finish_reason']='length'
                    return httpx.Response(response.status_code,json=result,request=response.request)
            return response
    async def aclose(self):await self.inner.aclose()

class AwaitingReadyMemory(ReadyMemory):
    index_ready_seconds=60
    async def query(self,*args,**kwargs):
        deadline=time.monotonic()+self.index_ready_seconds
        while True:
            try:return await super().query(*args,**kwargs)
            except RuntimeError as error:
                if not str(error).startswith('Memory indexing is pending;'):raise
                if time.monotonic()>=deadline:raise RuntimeError('Memory indexing is still pending after the bounded readiness wait') from error
                await asyncio.sleep(.5)

class AutoMemoryModel(UnifiedModel):
    def __init__(self,config,store,adapter,reasoning=None):
        super().__init__(config,store)
        # Keep the existing bounded transport retry and workflow schema layer.
        original_guard=self.http.inner
        self.http.inner=AdapterGuard(original_guard.inner,adapter,reasoning)
        self.adapter=adapter
        self.reasoning=reasoning or {'enabled':False}
    async def call(self,task_id,role,prompt,schema,**kwargs):
        if self.reasoning.get('enabled'):
            self.store.append(task_id,role,'model_inference_policy',{'thinking_enabled':True,
                'profile':'light','reasoning_budget_tokens':self.reasoning.get('reasoning_budget_tokens',256),
                'visible_output_budget':min(kwargs.get('limit',640),max(self.settings.primary_output_tokens,1536)) if role=='primary' else kwargs.get('limit',640)})
        return await super().call(task_id,role,prompt,schema,**kwargs)
    async def health(self):
        value=await super().health()
        value.update(thinking_enabled=self.reasoning.get('enabled',False),reasoning_policy=self.reasoning)
        if self.adapter and value.get('ready'):
            loaded=value.get('primary_adapter') or {}
            if not loaded.get('loaded') or loaded.get('id')!=self.adapter.get('id',0) or loaded.get('path')!=self.adapter['gguf']:
                value.update(ready=False,adapter_identity_error='Loaded adapter path differs from validated manifest')
        return value

class AutoMemoryRuntime(UnifiedRuntime):
    active_memory_enabled=True
    idle_autonomy_enabled=True

    async def execute_memory_tool(self,task_id,window,version,user_event,tool,arguments):
        from active_memory import ActiveMemory
        if not hasattr(self,'memory_pages'):self.memory_pages=ActiveMemory(self)
        return await self.memory_pages.execute(task_id,window,version,user_event,tool,arguments)

    def receipt_reply(self, job, last_result):
        from write_receipts import verified_reply
        return verified_reply(self, job, last_result)

    async def handoff(self,task_id,reason):
        if getattr(self.model,'native_kv',None):return await super().handoff(task_id,reason)
        from continuity_state import rollover
        return await rollover(self,task_id,reason)
    def prepare_primary_payload(self,task,payload,cutoff,current_ref):
        from continuity_state import prepare_primary
        return prepare_primary(self,task,payload,cutoff,current_ref)
    def prepare_primary_prompt(self,prompt):
        from continuity_state import order_prompt
        return order_prompt(prompt)
    async def fit_primary_prompt(self,task,prompt,cutoff,current_ref):
        from continuity_state import fit_prompt
        return await fit_prompt(self,task,prompt,cutoff,current_ref)
    def validate_turn_tool(self,payload,tool):
        from continuity_state import validate_tool
        return validate_tool(payload,tool)
    def prepare_internal_payload(self,task,payload,cutoff):
        from continuity_state import prepare_internal
        return prepare_internal(self,task,payload,cutoff)
    async def fit_internal_prompt(self,task,prompt,cutoff):
        from continuity_state import fit_prompt
        return await fit_prompt(self,task,prompt,cutoff,None)
    def read_context_source(self,task_id,window,version,user_event,arguments):
        from continuity_state import read_source
        return read_source(self,task_id,window,version,user_event,arguments)

    async def primary(self,job):
        from unified_schema import development_request
        guide=AUTO/'policy-memory.json'
        task=self.store.task(job['task_id'])
        if guide.exists() and any(t.startswith('dev.') for t in task['allowed_tools']) and development_request(job['body']['text']):
            from auto_memory_policy import ensure_procedure
            ensure_procedure(self.store,job['task_id'],json.loads(guide.read_text()))
        return await super().primary(job)

def create_app():
    adapter=active_adapter()
    learner_config=json.loads((AUTO/'config.json').read_text()) if (AUTO/'config.json').exists() else {}
    helper_options=learner_config.get('memory_teacher',{})
    if adapter and helper_options.get('projection_version')=='seven-field-experiences-v1':
        adapter={**adapter,'loaded_ids':sorted(set(adapter.get('loaded_ids',[adapter.get('id',0)])+[helper_options['adapter_id']]))}
    daily_reasoning=learner_config.get('reasoning_profiles',{}).get('daily',{'enabled':False})
    config=replace(settings(),model_url=os.environ.get('SQ_MODEL_URL','http://127.0.0.1:18813'),
        primary_output_tokens=768,max_steps=16,primary_adapter_id=adapter.get('id',0) if adapter else -1,
        primary_adapter_version=adapter['version'] if adapter else '')
    from navigation_runtime import memory_type,install as install_navigation
    runtime.Tools=UnifiedTools;runtime.EvidenceMemory=memory_type(AwaitingReadyMemory) if learner_config.get("memory_navigation",{}).get("enabled") else AwaitingReadyMemory;runtime.PRIMARY=COMPACT_PRIMARY;TOOL_HELP.update(EXTRA_HELP)
    import lea.app as module
    module.Runtime=AutoMemoryRuntime
    model=AutoMemoryModel(config,Store(config.data_dir),adapter,daily_reasoning)
    app=app_factory(settings=config,model=model)
    install_navigation(app,model.http.inner,AUTO)
    if helper_options.get('projection_version')=='seven-field-experiences-v1':
        from helper_off_routes import install as install_helper
        install_helper(app,model.http.inner,AUTO)
    learner_config=json.loads((AUTO/'config.json').read_text()) if (AUTO/'config.json').exists() else {}
    state_options=learner_config.get('idle_state_input',{'enabled':False})
    capacity_options=learner_config.get('memory_capacity',{})
    from memory_capacity import capacity
    @app.get('/api/memory-capacity')
    def memory_capacity_status():return capacity(capacity_options)
    @app.middleware('http')
    async def enforce_memory_capacity(request,call_next):
        # Existing reads stay available. New persistent foreground mutations
        # receive an explicit storage-full response before their transaction.
        if request.method not in ('GET','HEAD','OPTIONS') and not request.url.path.startswith('/api/internal/') and not request.url.path.endswith('/cancel'):
            current=capacity(capacity_options)
            if not current['write_allowed']:
                from fastapi.responses import JSONResponse
                return JSONResponse({'error':'记忆容量已满或存储卷不可用，已停止新增写入；原有记忆保留。','capacity':current},status_code=507)
        return await call_next(request)
    from idle_state_input import StateInput
    from shared_context import SharedChatContext
    autonomy=None
    if learner_config.get('idle_autonomy',{}).get('enabled'):
        from idle_autonomy import IdleAutonomy
        autonomy=IdleAutonomy(app.state.runtime,AUTO,learner_config['idle_autonomy'])
    shared_context=SharedChatContext(app.state.runtime,model.http.inner,autonomy=autonomy)
    state_input=StateInput(Path(config.data_dir)/'memory.sqlite3',R,AUTO,adapter,model.http.inner,state_options,
        shared_context=shared_context)
    app.state.internal_state_input=state_input
    @app.post('/api/internal/state-input')
    async def deliver_internal_state(automatic:bool=False):
        # Machine state is collected here; request bodies cannot inject state.
        return await state_input.deliver(automatic=automatic)
    @app.get('/api/internal/state-input')
    def internal_state_status():return state_input.status()
    @app.post('/api/tasks/{task_id}/tool-acceptance')
    def fixed_owner_contract(task_id:str,body:dict):
        # This is a human/operator API, deliberately absent from model tools.
        from fastapi import HTTPException
        from auto_memory_policy import validate_contract
        from lea.store import dump
        try:cases=validate_contract(body.get('name'),body.get('cases'))
        except (ValueError,TypeError) as error:raise HTTPException(400,str(error))
        store=Store(config.data_dir);store.task(task_id)
        with store.db(True) as db:
            if db.execute('SELECT 1 FROM unified_artifacts WHERE task_id=? AND name=?',(task_id,body['name'])).fetchone() or db.execute('SELECT 1 FROM unified_contracts WHERE task_id=? AND name=?',(task_id,body['name'])).fetchone():
                raise HTTPException(409,'Freeze owner cases before the first implementation; existing cases cannot be replaced')
            db.execute('INSERT INTO unified_contracts VALUES(?,?,?)',(task_id,body['name'],dump(cases)))
        store.append(task_id,'user','owner_acceptance_frozen',{'name':body['name'],'case_count':len(cases),'source':'human_operator_api','exposed_as_model_tool':False})
        return {'frozen':True,'name':body['name'],'case_count':len(cases)}
    @app.get('/api/model-profile')
    def profile():
        state=json.loads((AUTO/'state.json').read_text()) if (AUTO/'state.json').exists() else {'phase':'waiting_for_evidence'}
        learner_config=json.loads((AUTO/'config.json').read_text()) if (AUTO/'config.json').exists() else {}
        path=Path(state.get('training_report','/nonexistent'))
        if path.is_file() and path.resolve().is_relative_to(R/'experiments'):
            training=json.loads(path.read_text());state={**state,'training_progress':{'phase':training.get('phase'),'steps':len(training.get('steps',[])),'total':training.get('max_steps'),'resources':training.get('resources')}}
        return {'name':'水麒麟','model':config.model_name,'backend':config.model_url,'decision_agents':1,
            'development_executor':'direct pure-Python tools; no developer Agent call','completion_guard':True,
            'memory':'shared chat archive and retrieval','state_dir':str(config.data_dir),
            'memory_lookup':{'version':'active-memory-pages-v1','automatic_query':False,'page_size':3,'tools':['memory.search','memory.next','memory.read']},
            'native_kv_compression':False,'window_handoff':{'mode':'source-checkpoint-v1','generation_calls':0,'source_recovery':'context.read'},'primary_adapter':adapter['version'] if adapter else None,
            'adapter_selection':adapter.get('purpose','quality_accepted') if adapter else 'base',
            'adapter_quality_gate_passed':adapter.get('accepted',False) if adapter and adapter.get('quality_evaluated',True) else None,
            'primary_prompt_version':PROMPT_VERSION,'primary_prompt_chars':len(COMPACT_PRIMARY),
            'primary_prompt_sha256':hashlib.sha256(COMPACT_PRIMARY.encode()).hexdigest(),
            'reasoning_profiles':learner_config.get('reasoning_profiles',{}),
            'automatic_lora':'idle_and_window_gated' if learner_config.get('data_scope')=='all_context_windows' else 'idle_and_evidence_gated','automatic_lora_state':state,
            'automatic_lora_data_scope':learner_config.get('data_scope','verified_workflow_receipts'),
            'automatic_lora_window_interval':learner_config.get('min_new_handoffs',5),
            'kv_reset_after_lora':{**learner_config.get('kv_reset_after_lora',{}),
                'last_reset':json.loads((AUTO/'kv-reset-status.json').read_text()) if (AUTO/'kv-reset-status.json').exists() else None},
            'automatic_lora_daily_limit':learner_config.get('policy',{}).get('max_updates_per_day'),
            'memory_teacher':learner_config.get('memory_teacher',{'enabled':False}),
            'memory_capacity':capacity(capacity_options),
            'idle_activity':{k:learner_config.get('idle_autonomy',{}).get(k) for k in ('version','session_enabled','activity_budget_seconds','max_consecutive_no_progress','development_enabled','development_output_tokens')},
            'internal_state_input':{'enabled':state_options.get('enabled',False),'route':'/api/internal/state-input',
                'trigger':'fresh_per_request_and_periodic_when_idle','min_idle_seconds':state_options.get('min_idle_seconds',60),
                'manual_request_cooldown_seconds':0,'automatic_interval_seconds':state_options.get('automatic_interval_seconds',60),
                'separate_from_chat':False,'separate_entry_point':True,'increments_real_window_count':True,
                'window_counting':'only_after_committed_handoff',
                'context':shared_context.status()},
            'automatic_training_mode':learner_config.get('training_mode','bounded'),
            'automatic_training_parameters':learner_config.get('light_training'),
            'lora_quality_evaluation_enabled':learner_config.get('quality_evaluation_enabled',True),
            'lora_promotion_policy':learner_config.get('promotion_policy','quality_gate'),
            'working_tokens':config.working_tokens,'primary_output_tokens':config.primary_output_tokens,
            'answer_max_chars':config.answer_max_chars,'tools':list(EXTRA_HELP)}
    return app
