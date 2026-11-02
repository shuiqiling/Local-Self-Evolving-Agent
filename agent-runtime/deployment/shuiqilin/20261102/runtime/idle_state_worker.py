"""One bounded activity per request, then wait ten seconds after it stops."""
from pathlib import Path
import json,time,urllib.request
A=Path('/home/ubuntu/AI-Native/embodied-agent/auto-lora-shuiqilin')

def run():
    while True:
        try:
            config=json.loads((A/'config.json').read_text())
            budget=float(config.get('idle_autonomy',{}).get('activity_budget_seconds',300))
            request=urllib.request.Request('http://127.0.0.1:18815/api/internal/state-input?automatic=true',data=b'{}',headers={'Content-Type':'application/json'})
            # Time for the activity plus receipt finalization of an admitted tool.
            with urllib.request.urlopen(request,timeout=budget+180) as response:result=json.load(response)
            result={k:result[k] for k in ('status','id','channel','model_seconds','error','activity') if k in result}
        except Exception as error:result={'status':'route_unavailable','error':str(error)}
        result['checked_at']=time.time();p=A/'idle-state-worker.json';tmp=p.with_suffix('.tmp')
        tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
        time.sleep(10)

if __name__=='__main__':run()
