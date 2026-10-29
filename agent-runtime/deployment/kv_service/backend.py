"""Loopback-only transactional KV sessions for the primary Agent."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from lea.schemas import Decision

RELEASE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('LEA_ROOT', '/home/ubuntu/AI-Native/embodied-agent'))
STATE = Path(os.environ['LEA_KV_STATE_DIR'])
WORK = STATE/'worker'
SOCKET = WORK/'gate.sock'
IDENTITY = json.loads((RELEASE/'release.json').read_text())['identity']

def tool_arguments(name):
    # Match the actual execution tools; permissions are still checked by Tools.
    path={'type':'string','minLength':1,'maxLength':300}
    text={'type':'string','maxLength':12000}
    version={'type':'integer'}
    fields={
        'workspace.list':{'path':path},'workspace.read':{'path':path},
        'workspace.write':{'path':path,'text':text},
        'calculator':{'expression':{'type':'string','minLength':1,'maxLength':200}},
        'world.observe':{},'world.open_door':{'expected_version':version},
        'world.move':{'room':{'type':'string','enum':['A','B']},'expected_version':version},
        'world.pick':{'expected_version':version}}[name]
    return dict(type='object',properties=fields,required=list(fields),additionalProperties=False)


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True)


def atomic_json(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,separators=(',', ':'));f.flush();os.fsync(f.fileno())
    temp.replace(path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


class Worker:
    def __init__(self):
        self.lock=asyncio.Lock();self.native=None;self.gate=None;self.active=None;self.ready=False
        self.logs=[];self.last_error=None
        self.monitor=None

    async def start(self):
        STATE.mkdir(parents=True,exist_ok=True,mode=0o700);WORK.mkdir(exist_ok=True,mode=0o700)
        for name in ['gate_server.py','gate.pt','official_gate.py']:
            target=WORK/name
            if name=='gate_server.py':shutil.copy2(RELEASE/name,target)
            elif not target.exists():target.symlink_to(RELEASE/name)
        # A systemd control group owns all children; no prior worker is retained.
        if SOCKET.exists():
            if not SOCKET.is_socket():raise RuntimeError('Unexpected object at gate socket')
            SOCKET.unlink()
        (WORK/'gate-ready.json').unlink(missing_ok=True)
        gate_log=(STATE/'gate.log').open('ab');native_log=(STATE/'native.log').open('ab');self.logs=[gate_log,native_log]
        env={**os.environ,'LD_LIBRARY_PATH':str(ROOT/'runtime/q4-vnni-20261029-r1'),
             'GGML_BACKEND_PATH':str(ROOT/'runtime/q4-vnni-20261029-r1'),
             'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
        env.pop('LD_PRELOAD',None)
        self.gate=await asyncio.create_subprocess_exec(str(ROOT/'training/concise-v1/venv/bin/python'),'-u',str(WORK/'gate_server.py'),stdout=gate_log,stderr=gate_log,env=env)
        for _ in range(180):
            if (WORK/'gate-ready.json').exists():break
            if self.gate.returncode is not None:raise RuntimeError('Gate failed to load')
            await asyncio.sleep(.5)
        else:raise RuntimeError('Gate startup deadline')
        self.native=await asyncio.create_subprocess_exec(str(RELEASE/'native-worker'),str(ROOT),str(SOCKET),
            stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=native_log,env=env,limit=4*1024*1024)
        initial=await asyncio.wait_for(self.receive(),90)
        if not initial.get('ready'):raise RuntimeError('Native worker not ready')
        self.ready=True
        self.monitor=asyncio.create_task(self.watch_children())

    async def watch_children(self):
        while True:
            await asyncio.sleep(1)
            if any(child.returncode is not None for child in [self.native,self.gate]):
                self.ready=False
                os._exit(1) # systemd restarts the whole group from durable CURRENT

    async def receive(self):
        while True:
            line=await self.native.stdout.readline()
            if not line:raise RuntimeError('Native worker exited before a reply')
            try:value=json.loads(line)
            except ValueError:continue
            if not value.get('ok'):raise RuntimeError(value.get('error','Native operation failed'))
            return value

    async def rpc(self,**command):
        if not self.native or self.native.returncode is not None:raise RuntimeError('Native worker unavailable')
        self.native.stdin.write((canonical(command)+'\n').encode());await self.native.stdin.drain()
        result=await self.receive()
        return result.get('result',result)

    def folder(self,task_id):
        return STATE/'sessions'/hashlib.sha256(task_id.encode()).hexdigest()

    def manifest(self,task_id):
        folder=self.folder(task_id);path=folder/'CURRENT.json'
        if not path.exists():return None
        m=json.loads(path.read_text())
        if m['task_id']!=task_id or m['identity']!=IDENTITY:raise RuntimeError('KV checkpoint identity mismatch; explicit migration required')
        for kind in ['kv','meta']:
            name=m[kind]['file']
            if not re.fullmatch(r'[a-f0-9]{32}\.(bin|json)',name):raise RuntimeError('Invalid checkpoint filename')
        return m

    async def activate(self,task_id):
        m=self.manifest(task_id)
        if self.active!=task_id:
            await self.rpc(op='rollback')
            if m:
                folder=self.folder(task_id)
                for kind in ['kv','meta']:
                    if digest(folder/m[kind]['file'])!=m[kind]['sha256']:raise RuntimeError('KV checkpoint checksum mismatch')
                await self.rpc(op='load',kv_path=str(folder/m['kv']['file']),meta_path=str(folder/m['meta']['file']))
            else:await self.rpc(op='reset')
            self.active=task_id
        return m

    async def publish(self,task_id,old,result,client_state,request_id):
        folder=self.folder(task_id);folder.mkdir(parents=True,exist_ok=True,mode=0o700)
        if shutil.disk_usage(STATE).free<2*1024**3:raise RuntimeError('Insufficient free disk for atomic KV checkpoint')
        generation=uuid.uuid4().hex;kv=folder/(generation+'.bin');meta=folder/(generation+'.json')
        await self.rpc(op='export',kv_path=str(kv),meta_path=str(meta))
        for path in [kv,meta]:
            fd=os.open(path,os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
        m=dict(task_id=task_id,identity=IDENTITY,epoch=old['epoch'] if old else uuid.uuid4().hex,
               kv=dict(file=kv.name,sha256=digest(kv)),meta=dict(file=meta.name,sha256=digest(meta)),
               client_state=client_state,request_id=request_id,result=result,committed_at=time.time())
        # The CURRENT pointer is the commit point. Before it, rollback is safe;
        # after it, a caller retry receives the exact committed response.
        atomic_json(folder/'CURRENT.json',m)
        await self.rpc(op='commit')
        # Keep the previous generation as well as current, and only remove our
        # own generated checkpoint files from this hashed private directory.
        retain={kv.name,meta.name}
        if old:retain.update(old[k]['file'] for k in ['kv','meta'])
        for path in folder.iterdir():
            if re.fullmatch(r'[a-f0-9]{32}\.(bin|json)',path.name) and path.name not in retain:
                try:path.unlink()
                except OSError:pass # cache cleanup cannot invalidate a commit
        return m

    async def close(self):
        self.ready=False
        if self.monitor:self.monitor.cancel()
        if self.native and self.native.returncode is None:
            try:self.native.stdin.write(b'{"op":"shutdown"}\n');await self.native.stdin.drain();await asyncio.wait_for(self.native.wait(),8)
            except Exception:self.native.terminate()
        for child in [self.native,self.gate]:
            if child and child.returncode is None:
                child.terminate()
                try:await asyncio.wait_for(child.wait(),5)
                except asyncio.TimeoutError:child.kill();await child.wait()
        for log in self.logs:log.close()


def prompt_delta(messages,previous):
    """Append new records once; refresh only the current authoritative state."""
    system=messages[0]['content'];payload=json.loads(messages[-1]['content'])
    state=dict(previous or {});seen=set(state.get('seen_refs',[]));records=[]
    for record in payload.get('records',[]):
        if record.get('ref') in seen:continue
        seen.add(record['ref'])
        # These exact decisions already exist as native assistant tokens.
        if record.get('kind') not in ('primary_decision','assistant_reply'):records.append(record)
    stable={k:payload[k] for k in ['tools','goal','completion_criteria'] if k in payload}
    stable_hash=hashlib.sha256(canonical([system,stable]).encode()).hexdigest()
    bootstrap=not previous;prefix=''
    if state.get('stable_hash')!=stable_hash:
        prefix='<|im_start|>system\n'+system+'<|im_end|>\n'
    snapshot={k:v for k,v in payload.items() if k not in ('records','handoff_state') and not k.startswith('_native_')}
    # On first migration only, preserve the task's already-committed legacy
    # state. Subsequent windows carry real KV, never a generated summary.
    if bootstrap:snapshot['bootstrap_state']=payload.get('handoff_state',{})
    current=snapshot.pop('current_request',None)
    if current is not None:snapshot['current_request']=current
    # Keep live task/permission state and the current request AFTER new archive
    # records. A long record must not push the active instruction out of tail.
    body=json.dumps({'new_records':records,'current_authoritative_state':snapshot},ensure_ascii=False,separators=(',', ':'))
    control='current_request_operations是本轮执行账本，operation_status含历史。只把本轮completed视为本轮已执行，失败不算成功；已完成的调用不要重复。需要调用工具时action必须是tool，tool用授权完整名称；调用完成后根据last_result回答，action为respond或finish且response非空，tool和arguments置空。只输出JSON。'
    prompt=prefix+'<|im_start|>user\n'+body+'<|im_end|>\n<|im_start|>system\n'+control+'<|im_end|>\n<|im_start|>assistant\n'
    state.update(seen_refs=sorted(seen),stable_hash=stable_hash,through_seq=payload.get('_native_through_seq',0))
    return prompt,state,bootstrap


class Completion(BaseModel):
    task_id:str=Field(min_length=1,max_length=100)
    request_id:str=Field(pattern=r'^[a-f0-9]{64}$')
    messages:list[dict]
    output_schema:dict
    limit:int=Field(default=384,ge=1,le=640)


worker=Worker()
cancelled_requests=set()

@asynccontextmanager
async def lifespan(app):
    await worker.start()
    try:yield
    finally:await worker.close()


app=FastAPI(lifespan=lifespan)

@app.middleware('http')
async def reject_browser_origin(request:Request,call_next):
    if request.headers.get('origin'):return JSONResponse({'error':'Internal loopback API'},status_code=403)
    return await call_next(request)

@app.get('/health')
async def health():
    ready=worker.ready and worker.native is not None and worker.native.returncode is None and worker.gate.returncode is None
    return JSONResponse(dict(status='ok' if ready else 'unavailable',mode='fastkvzip',identity=IDENTITY,
        window_budget=3072,retained_per_head=614,protected_prefix=64,protected_suffix=128,
        persistence='atomic KV+per-head positions+cached scores; SHA256 verified on restore',
        reliability_test='core answers 34/50; this is an explicitly enabled experimental configuration'),status_code=200 if ready else 503)

async def complete_transaction(body:Completion):
    async with worker.lock:
        committed=False
        try:
            old=await worker.activate(body.task_id)
            if old and old.get('request_id')==body.request_id:
                return {**old['result'],'cache_epoch':old['epoch'],'client_state':old['client_state'],'replayed':True}
            prompt,client_state,bootstrap=prompt_delta(body.messages,(old or {}).get('client_state'))
            # Serialize the schema as a string so the worker's internal JSON
            # map cannot alphabetize action/response/tool/arguments properties.
            allowed_tools=list(json.loads(body.messages[-1]['content']).get('tools',{}))
            props=body.output_schema['properties'];branches=[]
            for action in props['action']['enum']:
                if action=='tool' and not allowed_tools:continue
                final=action in ('respond','finish')
                fields={
                    'action':{'type':'string','const':action},
                    'response':{**props['response'],'minLength':1} if final else {'type':'string','const':''},
                    'tool':{'type':'string','enum':allowed_tools} if action=='tool' else {'type':'string','const':''},
                    'arguments':{'type':'object','properties':{},'additionalProperties':False} if final else props['arguments']}
                if action=='tool':
                    for tool in allowed_tools:
                        tool_fields={**fields,'tool':{'type':'string','const':tool},'arguments':tool_arguments(tool)}
                        branches.append(dict(type='object',properties=tool_fields,required=list(tool_fields),additionalProperties=False))
                else:branches.append(dict(type='object',properties=fields,required=list(fields),additionalProperties=False))
            schema={'oneOf':branches}
            candidate=await worker.rpc(op='infer',prompt=prompt,schema_json=json.dumps(schema,ensure_ascii=False),limit=body.limit)
            if candidate['finish_reason']=='length':raise ValueError('Primary response reached its token limit')
            decision=Decision.model_validate_json(candidate['content'])
            props=body.output_schema['properties']
            if decision.action not in props['action']['enum']:raise ValueError('Action outside this request budget')
            if decision.action=='tool' and decision.tool not in allowed_tools:raise ValueError('Tool outside this task authorization')
            if decision.action=='tool':
                required=tool_arguments(decision.tool)['required']
                if set(required)-decision.arguments.keys():raise ValueError('Required tool arguments missing')
            if decision.action in ('respond','finish') and not decision.response.strip():raise ValueError('Final response must be nonempty')
            if len(decision.response)>props['response'].get('maxLength',200):raise ValueError('Response length exceeded')
            if (body.task_id,body.request_id) in cancelled_requests:raise ValueError('Primary request was canceled before commit')
            candidate.update(bootstrap=bootstrap,replayed=False)
            manifest=await worker.publish(body.task_id,old,candidate,client_state,body.request_id);committed=True
            return {**candidate,'cache_epoch':manifest['epoch'],'client_state':client_state}
        except Exception as error:
            # If CURRENT was already published, it is authoritative even if an
            # acknowledgement was lost. Never roll it back behind the disk.
            latest=worker.manifest(body.task_id)
            if latest and latest.get('request_id')==body.request_id:
                worker.active=None # reload the committed checkpoint before next RPC
                return {**latest['result'],'cache_epoch':latest['epoch'],'client_state':latest['client_state'],'replayed':True}
            if not committed:
                try:await worker.rpc(op='rollback')
                except Exception:worker.ready=False
            raise HTTPException(status_code=502,detail=dict(error=str(error),candidate=locals().get('candidate',{}).get('content',''))) from error

@app.post('/complete')
async def complete(body:Completion):
    return await asyncio.shield(complete_transaction(body))

@app.get('/sessions/{task_id}')
async def session(task_id:str):
    m=worker.manifest(task_id)
    if not m:raise HTTPException(status_code=404,detail='No native session')
    result={k:v for k,v in m['result'].items() if k not in ('content',)}
    return dict(cache_epoch=m['epoch'],identity=m['identity'],committed_at=m['committed_at'],
                stats=result,through_seq=m['client_state'].get('through_seq',0),checkpoint_sha256=m['kv']['sha256'])

class Handoff(BaseModel):
    task_id:str=Field(min_length=1,max_length=100)
    request_id:str=Field(pattern=r'^[a-f0-9]{64}$')

@app.post('/cancel')
async def cancel(body:Handoff):
    m=worker.manifest(body.task_id)
    if m and m.get('request_id')==body.request_id:return dict(already_committed=True)
    cancelled_requests.add((body.task_id,body.request_id))
    return dict(cancel_requested=True)

async def compress_transaction(body:Handoff):
    async with worker.lock:
        old=await worker.activate(body.task_id)
        if not old:raise HTTPException(status_code=409,detail='No native KV yet; run a primary turn first')
        if old.get('request_id')==body.request_id:return {**old['result'],'cache_epoch':old['epoch'],'client_state':old['client_state'],'replayed':True}
        try:
            result=await worker.rpc(op='compress')
            result.update(content='',finish_reason='handoff',input_tokens=0,output_tokens=0,bootstrap=False,replayed=False)
            m=await worker.publish(body.task_id,old,result,old['client_state'],body.request_id)
            return {**result,'cache_epoch':m['epoch'],'client_state':m['client_state']}
        except Exception as error:
            latest=worker.manifest(body.task_id)
            if latest and latest.get('request_id')==body.request_id:
                worker.active=None
                return {**latest['result'],'cache_epoch':latest['epoch'],'client_state':latest['client_state'],'replayed':True}
            await worker.rpc(op='rollback')
            raise HTTPException(status_code=502,detail=str(error)) from error

@app.post('/compress')
async def compress(body:Handoff):
    return await asyncio.shield(compress_transaction(body))
