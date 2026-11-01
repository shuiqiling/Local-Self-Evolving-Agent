"""Unforced paired decisions on the real GGUF inference engine."""
from pathlib import Path
import json
import os
import subprocess
import time
import urllib.request

from auto_memory_data import judge,save,canonical,sha

def request(base,path,data=None,timeout=180):
    q=urllib.request.Request(base+path,data=json.dumps(data).encode() if data is not None else None,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(q,timeout=timeout) as response:return json.load(response)

def decision_schema():
    return {'type':'object','properties':{'action':{'type':'string','enum':['tool','read_memory','recall','handoff','finish','respond']},
        'tool':{'type':'string'},'arguments':{'type':'object','additionalProperties':True},'response':{'type':'string','maxLength':200}},
        'required':['action','tool','arguments','response'],'additionalProperties':False}

def run_code(code,cases):
    r=Path('/home/ubuntu/AI-Native/embodied-agent/experiments/shuiqilin-unified-20261030-r1/app/lea/selfdev_runner.py')
    p=subprocess.run(['/usr/bin/python3','-I','-S',str(r)],input=json.dumps({'code':code,'cases':cases}),capture_output=True,text=True,timeout=8)
    return json.loads(p.stdout) if p.returncode==0 else {'passed':False,'error':'runner failed'}

def evaluate(work,base,scale,label,adapter_id=0,adapter_count=1,cases_override=None,before_case=None):
    work=Path(work);out=work/(label+'.json');cases=cases_override if cases_override is not None else json.loads((work/'data/evaluation.json').read_text())
    report=json.loads(out.read_text()) if out.exists() else {'completed':False,'label':label,'base':base,'scale':scale,'rows':[],
        'scope':'Frozen held-out decisions without runtime workflow forcing; repair outputs executed in restricted runner, plus4 retention checks','cases_sha256':sha(canonical(cases))}
    assert report.get('cases_sha256',sha(canonical(cases)))==sha(canonical(cases)),'Evaluation cases changed'
    if report['completed']:return report
    tick=time.monotonic()
    if scale is None:assert not request(base,'/lora-adapters'), 'Baseline endpoint unexpectedly has adapters loaded'
    for case in cases:
        if any(row['id']==case['id'] for row in report['rows']):continue
        if before_case:before_case()
        body={'model':'Qwen3-30B-A3B-Q4_K_M','messages':case['messages'],'temperature':0,'presence_penalty':0,
            'cache_prompt':False,'max_tokens':512 if case['kind']=='repair' else 256 if case['kind']!='retention' else 96,
            'chat_template_kwargs':{'enable_thinking':False},'seed':20261030}
        if case['kind']!='retention':body['response_format']={'type':'json_schema','json_schema':{'name':'Decision','schema':decision_schema(),'strict':True}}
        if scale is not None:body['lora']=[{'id':i,'scale':scale if i==adapter_id else 0.} for i in range(adapter_count)]
        started=time.monotonic();result=request(base,'/v1/chat/completions',body)
        text=result['choices'][0]['message']['content'];passed=judge(case,text,run_code) and result['choices'][0].get('finish_reason')!='length'
        row={'id':case['id'],'kind':case['kind'],'passed':passed,'answer':text,'seconds':round(time.monotonic()-started,3),
            'timings':result.get('timings'),'usage':result.get('usage'),'finish_reason':result['choices'][0].get('finish_reason')}
        report['rows'].append(row);save(out,report);print(json.dumps({k:row[k] for k in ('id','passed','seconds')},ensure_ascii=False),flush=True)
    groups={g:[r for r in report['rows'] if (r['kind']==g if g!='workflow' else r['kind'] not in ('repair','retention'))] for g in ('workflow','repair','retention')}
    report.update(completed=True,seconds=round(time.monotonic()-tick,3),scores={g:{'passed':sum(r['passed'] for r in rows),'total':len(rows)} for g,rows in groups.items()})
    save(out,report);return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:18813');p.add_argument('--scale',type=float);p.add_argument('--label',required=True);a=p.parse_args()
    print(json.dumps(evaluate(Path(__file__).resolve().parent,a.base,a.scale,a.label)['scores']))
