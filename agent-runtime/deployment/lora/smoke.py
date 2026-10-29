"""Real model + embedding + runtime checks in isolated synthetic workspaces."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

root=Path(sys.argv[1]).resolve();version=sys.argv[2];sys.path.insert(0,str(root/'app'))
from fastapi.testclient import TestClient
from lea.app import create_app
from lea.config import Settings

work=root/'training'/version;report={'scope':'Isolated synthetic runtime checks, real Q4 model and real embedding; no production task or external side effects.', 'variants':{}}
for variant in ['base','lora']:
    data=work/'smoke-data'/variant;assert not data.exists(),'Keep each smoke run fresh'
    settings=replace(Settings(),data_dir=data,precision='Q4_K_M',model_name='Qwen3-4B-Instruct-2507-Q4_K_M',
        primary_adapter_id=0 if variant=='lora' else -1,primary_adapter_version=version if variant=='lora' else '')
    app=create_app(settings);store=app.state.runtime.store;results=[]
    with TestClient(app) as client:
        def api(path,body=None):
            r=client.get('/api'+path) if body is None else client.post('/api'+path,json=body)
            r.raise_for_status();return r.json()
        for name,question,tools in [('memory','芦苇仓库什么时候允许送电？',[]),('calculator','用计算器计算19×27，只告诉我结果。',['calculator'])]:
            task=api('/tasks',{'goal':'合成 LoRA 端到端验收 '+name,'constraints':['只做当前合成测试，不操作外部系统'], 'allowed_tools':tools})
            identifier=task['id']
            if name=='memory':
                event=store.append(identifier,'user','message',{'text':'芦苇仓库只有绝缘检查通过后才能送电；未通过时禁止送电。','synthetic_fixture':True})
                state=store.task(identifier)
                book={'summary':'合成条件记录已归档。','completed':[],'decisions':[],'next_steps':[],'open_questions':[],'topics':['合成测试'],'entities':[],'history_entries':[]}
                store.commit_handoff(identifier,state['window_id'],event['seq'],book)
                for _ in range(450):
                    health=api('/health');index=health['embedding']
                    if index['total_cards']>0 and index['pending_vectors']==0:break
                    time.sleep(.2)
            start=time.monotonic();job=api('/tasks/'+identifier+'/messages',{'text':question,'request_id':'lora-smoke-'+variant+'-'+name})
            for _ in range(2000):
                job=api('/jobs/'+job['id'])
                if job['status'] in ['completed','error','canceled']:break
                time.sleep(.2)
            events=store.events(identifier);model=[e['body'] for e in events if e['kind']=='model_result'];tool_results=[e['body'] for e in events if e['kind']=='tool_result']
            answer=(job.get('result') or {}).get('response','')
            okay=job['status']=='completed' and len(answer)<=200
            if name=='memory':okay=okay and '绝缘检查' in answer and '通过' in answer
            else:okay=okay and '513' in answer and any(t.get('tool')=='calculator' and t.get('ok') and t.get('value')==513 for t in tool_results)
            if variant=='lora':okay=okay and all(m.get('adapter')==version for m in model)
            row={'name':name,'job':job,'answer':answer,'seconds':round(time.monotonic()-start,3),'model_results':model,'tool_results':tool_results,'passed':okay}
            results.append(row);print(json.dumps({'variant':variant,**{k:row[k] for k in ['name','answer','seconds','passed']}},ensure_ascii=False),flush=True)
    report['variants'][variant]=results
    (work/'smoke-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
report['completed']=True;report['passed']=all(r['passed'] for rows in report['variants'].values() for r in rows)
(work/'smoke-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
assert report['passed'],'Runtime smoke regression failed'
