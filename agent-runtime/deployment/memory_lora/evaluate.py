"""Frozen full-prompt memory-policy comparison: current concise-v2 versus continued memory-v1."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import urllib.request

p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
p.add_argument('--split',choices=['dev','test'],required=True); args=p.parse_args()
root=args.root.resolve(); work=root/'training/memory-v1'; sys.path.insert(0,str(work/'app'))
from lea.schemas import Decision

path=work/'data'/f'{args.split}.jsonl'
manifest=json.loads((work/'data/manifest.json').read_text())
dataset_sha=hashlib.sha256(path.read_bytes()).hexdigest()
assert dataset_sha==manifest['splits'][args.split]['sha256']
rows=[json.loads(line) for line in path.read_text().splitlines()]
folder=work/f'evaluation-{args.split}';folder.mkdir(exist_ok=False)
report={'split':args.split,'started_at':datetime.now(timezone.utc).isoformat(),'dataset_sha256':dataset_sha,
    'scope':'Authored synthetic frozen full-prompt decision tests; not real-use accuracy. Paired current concise-v2 and continued memory-v1, same Q4 base and caps.',
    'grader':'Action/IDs/budgets and keyword checks are preliminary; answer facts and question intent need manual review.',
    'answers':[],'completed':False}
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))


def api(path,body=None):
    request=urllib.request.Request('http://127.0.0.1:18805'+path,
        data=None if body is None else json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
    with opener.open(request,timeout=300) as response:return json.load(response)


def save():
    temp=folder/'report.tmp';temp.write_text(json.dumps(report,ensure_ascii=False,indent=2));temp.replace(folder/'report.json')


adapters=api('/lora-adapters');assert len(adapters)==2 and all(a['scale']==0 for a in adapters)
report['adapters']=adapters;save()
for i,row in enumerate(rows):
    payload=json.loads(row['messages'][1]['content']);schema=Decision.model_json_schema()
    schema['properties']['response']['maxLength']=200
    actions=schema['properties']['action']['enum']
    if payload.get('memory_queries_remaining',0)<=0:actions.remove('recall')
    if not payload['last_result'].get('memory_candidates') or payload.get('memory_reads_remaining',0)<=0:actions.remove('read_memory')
    for variant,identifier in ([('concise-v2',0),('memory-v1',1)] if i%2==0 else [('memory-v1',1),('concise-v2',0)]):
        body={'messages':row['messages'][:2],'temperature':0,'seed':20261029,'max_tokens':384,'stream':False,
            'cache_prompt':False,'lora':[{'id':0,'scale':float(identifier==0)},{'id':1,'scale':float(identifier==1)}],
            'response_format':{'type':'json_schema','json_schema':{'name':'Decision','schema':schema,'strict':True}}}
        tick=time.monotonic();value=api('/v1/chat/completions',body);seconds=time.monotonic()-tick
        raw=value['choices'][0]['message']['content'];finish=value['choices'][0].get('finish_reason')
        try:
            actual=Decision.model_validate_json(raw).model_dump();expected=row['expected']
            action_ok=actual['action'] in ('finish','respond') if expected['action']=='finish' else actual['action']==expected['action']
            selection_ok=True
            if expected['action']=='read_memory':
                selection_ok=(actual['arguments'].get('request_id')==expected['arguments']['request_id'] and
                    set(actual['arguments'].get('card_ids',[]))==set(expected['arguments']['card_ids']) and
                    isinstance(actual['arguments'].get('question'),str) and bool(actual['arguments']['question'].strip()))
            elif expected['action']=='recall':
                selection_ok=isinstance(actual['arguments'].get('question'),str) and bool(actual['arguments']['question'].strip())
            checks={'action':action_ok,'selection':selection_ok,'tool':actual['tool'] in ('','none'),
                'required':all(s in actual['response'] for s in row['required']),
                'forbidden':all(s not in actual['response'] for s in row['forbidden']),
                'bounded':finish!='length' and len(actual['response'])<=200}
        except Exception as exc:actual={'parse_error':str(exc)};checks={'parse':False}
        entry={'id':row['id'],'category':row['category'],'variant':variant,'result':actual,'raw':raw,
            'seconds':round(seconds,3),'usage':value.get('usage',{}),'timings':value.get('timings',{}),
            'checks':checks,'preliminary_pass':all(checks.values()),'source_case':row}
        report['answers'].append(entry);save()
        print(json.dumps({k:entry[k] for k in ('id','variant','seconds','result','checks')},ensure_ascii=False),flush=True)
summary={}
for variant in ('concise-v2','memory-v1'):
    items=[e for e in report['answers'] if e['variant']==variant]
    direct=[e for e in items if e['source_case']['expected']['action']=='finish']
    reads=[e for e in items if e['source_case']['expected']['action']=='read_memory']
    summary[variant]={'n':len(items),'preliminary_pass':sum(e['preliminary_pass'] for e in items),
        'correct_action':sum(e['checks'].get('action',False) for e in items),
        'unnecessary_read_calls':sum(e['result'].get('action')=='read_memory' for e in direct),
        'missed_required_reads':sum(e['result'].get('action')!='read_memory' for e in reads),
        'correct_read_selection':sum(e['checks'].get('action',False) and e['checks'].get('selection',False) for e in reads),
        'median_seconds':statistics.median(e['seconds'] for e in items)}
report.update(summary=summary,completed=True);save();print(json.dumps({'summary':summary}),flush=True)
