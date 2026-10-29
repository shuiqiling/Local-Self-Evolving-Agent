"""Paired actual scoped-memory Runtime pilot against a frozen synthetic archive."""
import argparse
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
root=args.root.resolve();work=root/'training/memory-v1';sys.path.insert(0,str(work/'app'))
from lea.config import Settings
from lea.runtime import Runtime

source=root/'data/scoped-memory-pilot/20261028T153447Z/report.json'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='e31c6b0966d20cf07dfd8de4d45560366c9ffbd49807eae2d2116862f9abde60'
frozen=json.loads(source.read_text());folder=work/'runtime-evaluation';folder.mkdir(exist_ok=False)
report={'scope':'Four existing synthetic archive questions never used for training. Both variants use scoped-v2 retrieval, identical raw records and frozen summary, distinct adapters, same generation caps. Small descriptive pilot.',
    'started_at':datetime.now(timezone.utc).isoformat(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'samples':[],'completed':False}


def save():
    tmp=folder/'report.tmp';tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2));tmp.replace(folder/'report.json')


async def main():
    runtimes={};tasks={}
    try:
        for variant,identifier in [('concise-v2',0),('memory-v1',1)]:
            settings=replace(Settings(),data_dir=folder/variant,memory_backend='scoped',
                model_url='http://127.0.0.1:18805',precision='Q4_K_M',model_name='Qwen3-4B-Instruct-2507-Q4_K_M',
                primary_adapter_id=identifier,primary_adapter_version=variant)
            runtime=Runtime(settings);runtimes[variant]=runtime
            task=runtime.store.create_task('回答当前合成记忆问题',['依据原始记录回答，不编造'],[],[])
            tasks[variant]=task['id']
            for text in frozen['records']:runtime.store.append(task['id'],'user','message',{'text':text})
            runtime.memory.index.sync(task['id'])
            while rows:=runtime.memory.index.pending():await runtime.memory.index.encode_pending(rows)
            runtime.memory.navigation.sync(task['id'])
            while units:=runtime.memory.navigation.pending_summaries():
                runtime.memory.navigation.complete_summary(units[0],frozen['setup']['precomputed_summaries'][0]['summary'])
        for i,case in enumerate(frozen['cases']):
            for variant in (('concise-v2','memory-v1') if i%2==0 else ('memory-v1','concise-v2')):
                runtime=runtimes[variant];task_id=tasks[variant];state=runtime.store.task(task_id)
                through=runtime.store.events(task_id,limit=1,tail=True)[-1]['seq']
                runtime.store.commit_handoff(task_id,state['window_id'],through,{'summary':'合成记忆已存档。','completed':[],
                    'decisions':[],'next_steps':[],'open_questions':[],'topics':[],'entities':[],'history_entries':[]})
                job=runtime.store.add_job(task_id,'message',{'text':case['question']});tick=time.monotonic()
                try:result=await runtime.primary(job);status='completed'
                except Exception as exc:result={'error':type(exc).__name__+': '+str(exc)};status='error'
                events=runtime.store.events(task_id,after=through)
                starts=[e['body'] for e in events if e['kind']=='model_started' and e['role']=='primary']
                row={'case_id':case['id'],'question':case['question'],'variant':variant,'status':status,
                    'seconds':round(time.monotonic()-tick,3),'result':result,'primary_calls':len(starts),
                    'input_tokens':[e['input_tokens'] for e in starts],
                    'events':[e for e in events if e['kind'] in ('model_started','model_result','primary_decision','retrieval_result','archive_read_scope','archive_read_result')]}
                report['samples'].append(row);save()
                print(json.dumps({k:row[k] for k in ('case_id','variant','seconds','primary_calls','result')},ensure_ascii=False),flush=True)
        report['summary']={}
        for variant in runtimes:
            rows=[r for r in report['samples'] if r['variant']==variant]
            report['summary'][variant]={'n':len(rows),'completed':sum(r['status']=='completed' for r in rows),
                'median_seconds':statistics.median(r['seconds'] for r in rows),'total_primary_calls':sum(r['primary_calls'] for r in rows),
                'one_call_cases':sum(r['primary_calls']==1 for r in rows)}
        report['completed']=True;save();print(json.dumps({'summary':report['summary']}),flush=True)
    finally:
        for runtime in runtimes.values():await runtime.memory.close();await runtime.model.close()


asyncio.run(main())
