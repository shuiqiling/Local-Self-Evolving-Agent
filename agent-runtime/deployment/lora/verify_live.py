"""One labelled synthetic question through the deployed production HTTP service."""
from pathlib import Path
import json
import sys
import time
import urllib.request

root=Path(sys.argv[1]).resolve();version=sys.argv[2];sys.path.insert(0,str(root/'app'))
from lea.store import Store

opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def api(path,body=None):
    request=urllib.request.Request('http://127.0.0.1:18800/api'+path,
        data=None if body is None else json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
    with opener.open(request,timeout=60) as f:return json.load(f)

health=api('/health');assert health['model']['primary_adapter']['version']==version
store=Store(root/'data');task=api('/tasks',{'goal':'LoRA 上线验收（合成数据）','constraints':['仅核对虚构记忆，不执行外部动作'],'allowed_tools':[]})
identifier=task['id'];event=store.append(identifier,'user','message',{'text':'栖云设备维护安排在周五上午。','synthetic_fixture':True})
current=store.task(identifier)
book={'summary':'虚构上线验收资料已归档。','completed':[],'decisions':[],'next_steps':[],'open_questions':[],'topics':['合成上线验收'],'entities':[],'history_entries':[]}
store.commit_handoff(identifier,current['window_id'],event['seq'],book)
for _ in range(300):
    index=api('/health')['embedding']
    with store.db() as db:n=db.execute('SELECT COUNT(*) FROM archive_units WHERE task_id=?',(identifier,)).fetchone()[0]
    if n>0 and index['pending_vectors']==0:break
    time.sleep(.2)
else:raise TimeoutError('Live source was not indexed')
start=time.monotonic();job=api('/tasks/'+identifier+'/messages',{'text':'栖云设备的维护时间是什么？','request_id':'live-'+version})
for _ in range(2000):
    job=api('/jobs/'+job['id'])
    if job['status'] in ['completed','error','canceled']:break
    time.sleep(.2)
events=store.events(identifier);models=[e['body'] for e in events if e['kind']=='model_result']
answer=(job.get('result') or {}).get('response','')
report={'task_id':identifier,'health':health,'job':job,'answer':answer,'seconds':round(time.monotonic()-start,3),'model_results':models,
    'generation_calls':len(models),'archive_reads':(job.get('result') or {}).get('archive_reads'),'source_ref':event['ref']}
report['passed']=(job['status']=='completed' and '周五' in answer and '上午' in answer and len(models)==1
    and models[0].get('adapter')==version and report['archive_reads']==0 and not job['result']['answer_truncated'])
(root/'training'/version/'live-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:report[k] for k in ['task_id','answer','seconds','generation_calls','archive_reads','passed']},ensure_ascii=False),flush=True)
assert report['passed'],'Live adapter deployment check failed'
