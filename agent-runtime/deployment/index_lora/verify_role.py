"""Exercise the actual client role switch without enabling any production adapter."""
import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
root=args.root.resolve();work=root/'training/index-v1';sys.path.insert(0,str(work/'app'))
from lea.config import Settings
from lea.indexing import INDEX_SUMMARY_PROMPT
from lea.model import ModelClient,messages
from lea.schemas import CatalogSummary
from lea.store import Store


async def main():
    store=Store(work/'role-integration');task=store.create_task('索引角色接线验证',[],[],[])
    settings=replace(Settings(data_dir=store.root),model_url='http://127.0.0.1:18806',precision='Q4_K_M',
        model_name='Qwen3-4B-Instruct-2507-Q4_K_M',primary_adapter_id=-1,indexer_adapter_id=0,indexer_adapter_version='index-v1')
    model=ModelClient(settings,store)
    try:
        row=json.loads((work/'data/dev.jsonl').read_text().splitlines()[0]);outputs={}
        for role in ('indexer','handoff'):
            result=await model.call(task['id'],role,messages(INDEX_SUMMARY_PROMPT,row['source']),CatalogSummary,limit=200)
            outputs[role]=result.model_dump()
        results=[e['body'] for e in store.events(task['id']) if e['kind']=='model_result']
        assert [r['adapter'] for r in results]==['index-v1','base']
        health=await model.health();assert health['ready'] and health['indexer_adapter']['loaded']
        response=await model.http.get('/lora-adapters');response.raise_for_status();adapters=response.json()
        assert len(adapters)==1 and adapters[0]['scale']==0
        (work/'role-smoke-report.json').write_text(json.dumps({'passed':True,'outputs':outputs,'health':health,
            'adapter_provenance':[r['adapter'] for r in results],'global_adapter_state':adapters},ensure_ascii=False,indent=2))
    finally:await model.close()


asyncio.run(main())
