"""Primary-only native KV sessions; the legacy model remains for auxiliary roles."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
import httpx

from .store import Conflict, atomic_json, dump, now, uid


class NativeSessionClient:
    def __init__(self,settings,store):
        self.settings,self.store=settings,store
        self.http=httpx.AsyncClient(base_url=settings.native_kv_url,trust_env=False,
            timeout=httpx.Timeout(settings.call_timeout,connect=5))
        self.results={}

    async def close(self):
        await self.http.aclose()

    async def health(self):
        try:
            r=await self.http.get('/health',timeout=3);r.raise_for_status()
            return {**r.json(),'ready':r.json().get('status')=='ok'}
        except (httpx.HTTPError,ValueError) as error:
            return dict(ready=False,mode='fastkvzip',error=type(error).__name__)

    async def call(self,task_id,messages,schema,output_schema,limit,queue):
        # Snapshot sequence is bookkeeping, not part of semantic request identity.
        clean=json.loads(json.dumps(messages));payload=json.loads(clean[-1]['content'])
        payload.pop('_native_through_seq',None);clean[-1]['content']=dump(payload)
        request_id=hashlib.sha256(dump([task_id,clean,output_schema,limit]).encode()).hexdigest()
        body=dict(task_id=task_id,request_id=request_id,messages=messages,output_schema=output_schema,limit=limit)
        provenance=dict(model=self.settings.model_name,precision=self.settings.precision,
            adapter=self.settings.primary_adapter_version,backend='native_kvzip')
        async with asyncio.timeout(self.settings.call_timeout):
            async with queue.slot(0):
                self.store.append(task_id,'primary','model_started',{**provenance,'input_tokens':None,
                    'input_mode':'append_delta','output_limit':limit,'native_request_id':request_id})
                started=time.monotonic()
                try:
                    for attempt in range(2):
                        try:r=await self.http.post('/complete',json=body);break
                        except (httpx.ConnectError,httpx.ReadError,httpx.RemoteProtocolError):
                            if attempt:raise
                            await asyncio.sleep(1)
                except asyncio.CancelledError:
                    try:await asyncio.shield(self.http.post('/cancel',json={'task_id':task_id,'request_id':request_id},timeout=3))
                    except Exception:pass
                    raise
                if not r.is_success:
                    try:detail=r.json().get('detail',{})
                    except ValueError:detail={}
                    if isinstance(detail,dict) and detail.get('candidate'):
                        self.store.append(task_id,'primary','model_result',{**provenance,'text':detail['candidate'],
                            'validation_failed':True,'seconds':round(time.monotonic()-started,3)})
                    r.raise_for_status()
                result=r.json();content=result['content']
                self.store.append(task_id,'primary','model_result',{**provenance,'text':content,
                    'seconds':round(time.monotonic()-started,3),'usage':{
                        'prompt_tokens':result['input_tokens'],'completion_tokens':result['output_tokens']},
                    'timings':{k:result.get(k) for k in ['prefill_ms','first_token_ms','total_ms']}})
                decision=schema.model_validate_json(content)
                metadata={k:result.get(k) for k in ['cache_epoch','physical_tokens','next_absolute_position',
                    'total_cuts','cuts','bootstrap','replayed','input_tokens','output_tokens']}
                metadata['through_seq']=result.get('client_state',{}).get('through_seq',0)
                self.results[task_id]=metadata
                self.store.append(task_id,'runtime','native_kv_checkpoint',metadata)
                return decision

    def take_result(self,task_id):
        return self.results.pop(task_id,None)

    async def compress(self,task_id,queue):
        async with asyncio.timeout(self.settings.call_timeout):
            async with queue.slot(1):
                request_id=hashlib.sha256(uuid.uuid4().bytes).hexdigest()
                r=await self.http.post('/compress',json=dict(task_id=task_id,request_id=request_id))
                if r.status_code==409:return None
                r.raise_for_status();result=r.json()
                metadata={k:result.get(k) for k in ['cache_epoch','physical_tokens','next_absolute_position','total_cuts','cuts']}
                metadata['through_seq']=result.get('client_state',{}).get('through_seq',0)
                self.store.append(task_id,'runtime','native_kv_checkpoint',metadata)
                return metadata


def reconcile_windows(store,task_id,expected_window,expected_constraints,metadata):
    """Reflect native cuts in durable window ownership without generating text."""
    if not metadata:return store.task(task_id)
    through=int(metadata['through_seq']);total=int(metadata['total_cuts']);epoch=metadata['cache_epoch']
    with store.db(True) as db:
        task=store._task(db,task_id)
        if task['window_id']!=expected_window or task['constraint_version']!=expected_constraints:
            raise Conflict('Task/constraints changed during native inference')
        prior=task['state'].get('native_kv',{})
        recorded=prior.get('total_cuts',0) if prior.get('cache_epoch')==epoch else 0
        if total<recorded:raise Conflict('Native cache cut counter went backwards')
        if total==recorded:return task
        legacy_origin=task['state'].get('legacy_state_before_native')
        if legacy_origin is None:
            legacy_origin={k:v for k,v in task['state'].items() if k not in ('native_kv','legacy_state_before_native')}
        if through and not db.execute('SELECT 1 FROM events WHERE task_id=? AND seq=?',(task_id,through)).fetchone():
            raise ValueError('Native handoff cutoff is outside this task')
        for number in range(recorded+1,total+1):
            old=task['window_id'];new=uid('w');stamp=now()
            evidence=next((x for x in metadata.get('cuts') or [] if x['cut']==number),None)
            handoff=dict(schema_version=2,mode='native_kvzip',task_id=task_id,window_id=old,
                through_seq=through,cache_epoch=epoch,native_cut=number,cut=evidence,
                reconstructed_from_checkpoint=evidence is None,
                history_index=dict(topics=[],entities=[],entries=[]))
            atomic_json(store.root/'tasks'/task_id/'windows'/old/'handoff.json',handoff)
            store._event(db,task,'runtime','handoff_committed',dict(next_window=new,through_seq=through,
                mode='native_kvzip',native_cut=number))
            db.execute("UPDATE windows SET status='closed',closed=?,through_seq=?,handoff=? WHERE id=?",
                (stamp,through,dump(handoff),old))
            db.execute('INSERT INTO windows(id,task_id,status,created) VALUES(?,?,?,?)',(new,task_id,'active',stamp))
            state=dict(summary='',completed=[],next_steps=[],recorded_through=through,previous_window=old,
                legacy_state_before_native=legacy_origin,
                native_kv={**metadata,'total_cuts':number,'mode':'fastkvzip','summary_generated':False})
            version=task['version']+1
            db.execute('UPDATE tasks SET window_id=?,version=?,state=? WHERE id=?',(new,version,dump(state),task_id))
            task=store._task(db,task_id)
            db.execute('INSERT INTO state_versions VALUES(?,?,?)',(task_id,version,dump(task)))
            store._event(db,task,'runtime','handoff_completed',dict(from_window=old,to_window=new,
                through_seq=through,mode='native_kvzip',summary_generated=False,
                seconds=evidence.get('seconds') if evidence else None,native_cut=number))
    store.export(task_id)
    return store.task(task_id)


async def native_handoff(runtime,task_id,reason):
    task=runtime.store.task(task_id)
    metadata=await runtime.model.native_kv.compress(task_id,runtime.model.queue)
    if not metadata:
        return dict(window_id=task['window_id'],mode='native_kvzip',skipped=True,
                    summary='尚无原生KV会话，先运行一轮任务。')
    current=reconcile_windows(runtime.store,task_id,task['window_id'],task['constraint_version'],metadata)
    return dict(window_id=current['window_id'],version=current['version'],mode='native_kvzip',
                summary='KV缓存已直接继承。' if metadata.get('cuts') else '缓存未超过保留预算，无需切窗。',physical_tokens=metadata['physical_tokens'],
                total_cuts=metadata['total_cuts'],reason=reason)
