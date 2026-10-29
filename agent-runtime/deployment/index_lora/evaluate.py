"""Preliminary indexing checks; literal/coverage checks do not establish factual fidelity."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import time
import urllib.request

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--split',choices=['dev','test'],required=True);args=p.parse_args()
root=args.root.resolve();work=root/'training/index-v1';sys.path.insert(0,str(work/'app'))
from lea.schemas import CatalogSummary
path=work/f'data/{args.split}.jsonl';m=json.loads((work/'data/manifest.json').read_text());assert hashlib.sha256(path.read_bytes()).hexdigest()==m['splits'][args.split]['sha256']
rows=[json.loads(x) for x in path.read_text().splitlines()];folder=work/f'evaluation-{args.split}';folder.mkdir(exist_ok=False)
report={'split':args.split,'started_at':datetime.now(timezone.utc).isoformat(),'dataset_sha256':m['splits'][args.split]['sha256'],
    'scope':'Same index-writing prompt, Q4 base and 200-token/100-character caps. Base (current indexer) versus index-only candidate. Not a general accuracy estimate.',
    'grader':'Preliminary schema, literal existence, topic coverage and completion checks only; source-based semantic review required.', 'answers':[],'completed':False}
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))


def save():
    p=folder/'report.tmp';p.write_text(json.dumps(report,ensure_ascii=False,indent=2));p.replace(folder/'report.json')


for i,row in enumerate(rows):
    for variant,scale in ([('base',0.),('index-v1',1.)] if i%2==0 else [('index-v1',1.),('base',0.)]):
        body={'messages':row['messages'][:2],'temperature':0,'seed':20261029,'max_tokens':200,'stream':False,'cache_prompt':False,
            'lora':[{'id':0,'scale':scale}],'response_format':{'type':'json_schema','json_schema':{'name':'CatalogSummary','schema':CatalogSummary.model_json_schema(),'strict':True}}}
        request=urllib.request.Request('http://127.0.0.1:18806/v1/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
        tick=time.monotonic()
        with opener.open(request,timeout=300) as f:value=json.load(f)
        raw=value['choices'][0]['message']['content'];finish=value['choices'][0].get('finish_reason')
        try:
            output=CatalogSummary.model_validate_json(raw).summary
            literals=re.findall(r'[A-Za-z]+[-_/][A-Za-z0-9_.-]*\d[A-Za-z0-9_.-]*|[-+]?\d+(?:\.\d+)?',output)
            unsupported=[x for x in literals if x.lower() not in row['source']['archive_text'].lower()]
            missing=[g for g in row['coverage_groups'] if not any(x in output for x in g)]
            checks={'schema':True,'length':len(output)<=100,'literal_supported':not unsupported,'coverage':not missing,
                    'identity':all(x in output for x in row['required_identifiers']+row['required_names']),
                    'finished':finish!='length' and not output.endswith((',', '，', ':', '：', '/', '-', '、'))}
        except Exception as exc:output='';unsupported=[];missing=[];checks={'schema':False,'error':str(exc)}
        entry={'id':row['id'],'variant':variant,'summary':output,'raw':raw,'seconds':round(time.monotonic()-tick,3),
            'usage':value.get('usage',{}),'checks':checks,'unsupported_literals':unsupported,'missing_coverage':missing,'source_case':row}
        report['answers'].append(entry);save();print(json.dumps({k:entry[k] for k in ('id','variant','summary','seconds','checks')},ensure_ascii=False),flush=True)
report['summary']={}
for variant in ('base','index-v1'):
    items=[a for a in report['answers'] if a['variant']==variant]
    report['summary'][variant]={'n':len(items),'schema_ok':sum(a['checks'].get('schema',False) for a in items),
        'literal_supported':sum(a['checks'].get('literal_supported',False) for a in items),
        'coverage_ok':sum(a['checks'].get('coverage',False) for a in items),
        'identity_ok':sum(a['checks'].get('identity',False) for a in items),
        'finished':sum(a['checks'].get('finished',False) for a in items),
        'mean_chars':statistics.mean(len(a['summary']) for a in items),'median_seconds':statistics.median(a['seconds'] for a in items)}
report['completed']=True;save();print(json.dumps({'summary':report['summary']}),flush=True)
