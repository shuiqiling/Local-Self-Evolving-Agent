"""Frozen old/new adapter comparison; raw summaries retained without trimming or repair."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import urllib.request
from quality import check

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--split',choices=['dev','test'],required=True);args=p.parse_args()
root=args.root.resolve();work=root/'training/index-v2';sys.path.insert(0,str(work/'app'))
from lea.schemas import CatalogSummary
path=work/f'data/{args.split}.jsonl';manifest=json.loads((work/'data/manifest.json').read_text())
assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['splits'][args.split]['sha256']
rows=[json.loads(line) for line in path.read_text().splitlines()];folder=work/f'evaluation-{args.split}';folder.mkdir(exist_ok=False)
report={'version':'index-v2','split':args.split,'started_at':datetime.now(timezone.utc).isoformat(),
    'dataset_sha256':manifest['splits'][args.split]['sha256'],'scope':'Same original records, exact v1 prompt, 100-character schema, 200 output tokens and Q4 base. Only adapter differs. Raw generation, no repair or truncation.',
    'grader':'Whole literal / signed-number and exact condition checks are diagnostics. Binding correctness, condition equivalence, and true half-sentence errors require source review.',
    'answers':[],'completed':False}
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))


def save():
    temp=folder/'report.tmp';temp.write_text(json.dumps(report,ensure_ascii=False,indent=2));temp.replace(folder/'report.json')


for i,row in enumerate(rows):
    for variant,adapter in ([('index-v1',0),('index-v2',1)] if i%2==0 else [('index-v2',1),('index-v1',0)]):
        body={'messages':row['messages'][:2],'temperature':0,'seed':20261030,'max_tokens':200,'stream':False,'cache_prompt':False,
            'lora':[{'id':0,'scale':float(adapter==0)},{'id':1,'scale':float(adapter==1)}],
            'response_format':{'type':'json_schema','json_schema':{'name':'CatalogSummary','schema':CatalogSummary.model_json_schema(),'strict':True}}}
        req=urllib.request.Request('http://127.0.0.1:18807/v1/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
        started=time.monotonic()
        with opener.open(req,timeout=300) as response:value=json.load(response)
        raw=value['choices'][0]['message']['content'];reason=value['choices'][0].get('finish_reason')
        try:
            summary=CatalogSummary.model_validate_json(raw).summary
            checks={'schema':True,'not_output_token_capped':reason!='length',**check(row,summary)}
        except Exception as exc:summary='';checks={'schema':False,'error':str(exc)}
        entry={'id':row['id'],'category':row['category'],'variant':variant,'summary':summary,'raw':raw,
            'seconds':round(time.monotonic()-started,3),'finish_reason':reason,'usage':value.get('usage',{}),
            'checks':checks,'source_case':row}
        report['answers'].append(entry);save();print(json.dumps({k:entry[k] for k in ('id','variant','summary','seconds','checks')},ensure_ascii=False),flush=True)
report['summary']={}
for variant in ('index-v1','index-v2'):
    items=[x for x in report['answers'] if x['variant']==variant]
    numeric=[x for x in items if x['source_case']['key_bindings']]
    report['summary'][variant]={'n':len(items),'schema_ok':sum(x['checks'].get('schema',False) for x in items),
        'identity_ok':sum(x['checks'].get('identity',False) for x in items),
        'literal_supported':sum(x['checks'].get('literal_supported',False) for x in items),
        'critical_exact':sum(x['checks'].get('critical_exact',False) for x in items),
        'numeric_required_cases':len(numeric),'key_values_present':sum(x['checks'].get('key_values_present',False) for x in numeric),
        'normal_end':sum(x['checks'].get('normal_end',False) for x in items),
        'at_character_cap':sum(x['checks'].get('at_character_cap',False) for x in items),
        'mean_chars':statistics.mean(len(x['summary']) for x in items),'median_seconds':statistics.median(x['seconds'] for x in items)}
report['completed']=True;save();print(json.dumps({'summary':report['summary']}),flush=True)
