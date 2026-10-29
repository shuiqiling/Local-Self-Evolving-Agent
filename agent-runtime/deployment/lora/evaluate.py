"""Paired llama.cpp baseline/LoRA eval with frozen prompts and completion caps."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import urllib.request

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--split',choices=['dev','test'],default='dev')
p.add_argument('--version',default='concise-v1')
args=p.parse_args();root=args.root.resolve();sys.path.insert(0,str(root/'app'))
from lea.schemas import Decision
from lea.runtime import PRIMARY
from lea.evidence import EvidenceMemory

assert args.version.startswith('concise-v') and args.version.removeprefix('concise-v').isdigit()
work=root/'training'/args.version;folder=work/('evaluation-'+args.split);folder.mkdir(exist_ok=True)
rows=[json.loads(s) for s in (work/f'data/{args.split}.jsonl').read_text().splitlines()]
if args.split=='test':
    # Dev already covers template/control cases; spend the final budget on novel
    # wording plus the prior fixed evidence regression, rather than duplicates.
    rows=[row for row in rows if row['category']=='novel_wording']
    # Reuse frozen evidence, not a newly tuned retrieval run. Never training data.
    old=json.loads((root/'data/hybrid-accuracy-latest.json').read_text())
    for a in old['answers']:
        query=a['prepared_context'][0]
        payload={'current_request':a['question'],'goal':'回答当前记忆问题','constraints':['简短回答，不虚构'],
                 'last_result':EvidenceMemory.for_primary(query),'tools':{},'environment':{},'records':[],
                 'answer_max_chars':200,'memory_queries_remaining':0,'memory_reads_remaining':1}
        rows.append({'id':'previous-'+a['id'],'category':'previous_retrieval_regression','question':a['question'],
            'messages':[{'role':'system','content':PRIMARY},{'role':'user','content':json.dumps(payload,ensure_ascii=False,separators=(',',':'))}],
            'expected':{'action':'finish','tool':'','arguments':{}},'required':[],'forbidden':[],
            'answer_groups':a['answer_groups'],'no_answer':a['no_answer'],'manual_review_required':True})
    probes=work/'data/probes.jsonl'
    if probes.exists():rows.extend(json.loads(line) for line in probes.read_text().splitlines())

def api(path,body=None):
    data=None if body is None else json.dumps(body,ensure_ascii=False).encode()
    request=urllib.request.Request('http://127.0.0.1:18801'+path,data=data,headers={'Content-Type':'application/json'})
    # Never inherit host proxy settings for localhost.
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=300) as f:return json.load(f)

adapters=api('/lora-adapters');assert len(adapters)==1,adapters
report={'started_at':datetime.now(timezone.utc).isoformat(),'split':args.split,'adapters_loaded':adapters,
        'adapter_version':args.version,
        'scope':'Same Q4 base, frozen prompts/context, schema and caps; paired baseline scale 0 and style LoRA scale 1.',
        'max_tokens':384,'max_response_chars':200,'temperature':0,'answers':[],
        'dataset_sha256':hashlib.sha256((work/f'data/{args.split}.jsonl').read_bytes()).hexdigest(),
        'evaluated_case_ids':[row['id'] for row in rows],
        'grader':'Deterministic checks are preliminary. Entire answers need source-based manual review.'}
if args.split=='test' and (work/'data/probes.jsonl').exists():
    report['counterfactual_probe_sha256']=hashlib.sha256((work/'data/probes.jsonl').read_bytes()).hexdigest()
def save():
    path=folder/'report.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2));tmp.replace(path)
save()

for index,row in enumerate(rows):
    messages=row['messages'][:2]
    schema=Decision.model_json_schema();schema['properties']['response']['maxLength']=200
    payload=json.loads(messages[-1]['content']);actions=schema['properties']['action']['enum']
    if payload.get('memory_queries_remaining',1)<=0:actions.remove('recall')
    if not payload.get('last_result',{}).get('memory_candidates') or payload.get('memory_reads_remaining',0)<=0:actions.remove('read_memory')
    for name,scale in ([('base',0.),('lora',1.)] if index%2==0 else [('lora',1.),('base',0.)]):
        body={'messages':messages,'temperature':0,'seed':20261028,'max_tokens':384,'stream':False,
              'cache_prompt':False,'lora':[{'id':adapters[0]['id'],'scale':scale}],
              'response_format':{'type':'json_schema','json_schema':{'name':'Decision','schema':schema,'strict':True}}}
        started=time.monotonic();value=api('/v1/chat/completions',body);elapsed=time.monotonic()-started
        text=value['choices'][0]['message']['content'];finish=value['choices'][0].get('finish_reason')
        checks=[]
        try:
            result=Decision.model_validate_json(text).model_dump();expected=row['expected']
            checks.append(result['action'] in ['finish','respond'] if expected['action']=='finish' else result['action']==expected['action'])
            checks.append(result['tool']==expected['tool']);checks.append(result['arguments']==expected['arguments'])
            checks.extend(word in result['response'] for word in row.get('required',[]))
            checks.extend(word not in result['response'] for word in row.get('forbidden',[]))
            checks.append(finish!='length' and len(result['response'])<=200)
        except Exception as exc:result={'response':''};checks=[False]
        entry={'id':row['id'],'variant':name,'category':row['category'],'seconds':round(elapsed,3),
               'response_chars':len(result['response']),'result':result,'raw_text':text,'finish_reason':finish,
               'usage':value.get('usage',{}),'timings':value.get('timings',{}),'preliminary_checks_passed':all(checks),
               'source_case':row}
        report['answers'].append(entry);save()
        print(json.dumps({k:entry[k] for k in ['id','variant','seconds','response_chars','preliminary_checks_passed','result']},ensure_ascii=False),flush=True)

summary={}
for name in ['base','lora']:
    items=[r for r in report['answers'] if r['variant']==name]
    answers=[r for r in items if r['source_case']['expected']['action']=='finish']
    summary[name]={'n':len(items),'preliminary_checks_passed':sum(r['preliminary_checks_passed'] for r in items),
                   'mean_response_chars':statistics.mean(r['response_chars'] for r in answers),
                   'median_response_chars':statistics.median(r['response_chars'] for r in answers),
                   'median_seconds':statistics.median(r['seconds'] for r in items),
                   'mean_completion_tokens':statistics.mean(r['usage'].get('completion_tokens',0) for r in items)}
report['summary']=summary;report['completed']=True;save()
print(json.dumps({'summary':summary},ensure_ascii=False),flush=True)
