"""Poll automatic idle delivery; the route enforces an interval, not once-only."""
from pathlib import Path
import json
import time
import urllib.request

A=Path('/home/ubuntu/AI-Native/embodied-agent/auto-lora-shuiqilin')

def run():
    while True:
        try:
            request=urllib.request.Request('http://127.0.0.1:18815/api/internal/state-input?automatic=true',data=b'{}',headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=200) as response:result=json.load(response)
            result={k:result[k] for k in ('status','id','channel','model_seconds','error') if k in result}
        except Exception as error:result={'status':'route_unavailable','error':str(error)}
        result['checked_at']=time.time();p=A/'idle-state-worker.json';tmp=p.with_suffix('.tmp')
        tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
        time.sleep(10)

if __name__=='__main__':run()
