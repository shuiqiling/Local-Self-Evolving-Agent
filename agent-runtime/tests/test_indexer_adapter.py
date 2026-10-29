import asyncio
from dataclasses import replace
import json

import httpx

from lea.config import Settings
from lea.model import ModelClient, messages
from lea.schemas import CatalogSummary, Decision
from lea.store import Store


def test_indexer_adapter_is_applied_only_to_indexer_and_other_roles_are_explicit(tmp_path):
    async def run():
        store=Store(tmp_path);task=store.create_task('角色隔离',[],[],[]);seen=[]
        settings=replace(Settings(data_dir=tmp_path),primary_adapter_id=0,primary_adapter_version='concise-v2',
                         indexer_adapter_id=1,indexer_adapter_version='index-v1')
        def handle(request):
            if request.url.path=='/health':return httpx.Response(200,json={'status':'ok'})
            if request.url.path=='/lora-adapters':return httpx.Response(200,json=[{'id':0,'scale':0},{'id':1,'scale':0}])
            if request.url.path=='/apply-template':return httpx.Response(200,json={'prompt':'test'})
            if request.url.path=='/tokenize':return httpx.Response(200,json={'tokens':[1]})
            body=json.loads(request.content);seen.append(body)
            output={'summary':'设备索引。'} if body['response_format']['json_schema']['name']=='CatalogSummary' else {'action':'finish','response':'已答复','tool':'','arguments':{}}
            return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(output,ensure_ascii=False)},'finish_reason':'stop'}]})
        model=ModelClient(settings,store);await model.http.aclose()
        model.http=httpx.AsyncClient(base_url='http://test',transport=httpx.MockTransport(handle))
        payload={'memory_queries_remaining':0,'memory_reads_remaining':0,'last_result':{}}
        await model.call(task['id'],'primary',messages('test',payload),Decision)
        await model.call(task['id'],'indexer',messages('test',{'archive_text':'设备资料。'}),CatalogSummary,limit=200)
        await model.call(task['id'],'handoff',messages('test',payload),Decision)
        assert [r['lora'] for r in seen]==[
            [{'id':0,'scale':1.0},{'id':1,'scale':0.0}],
            [{'id':0,'scale':0.0},{'id':1,'scale':1.0}],
            [{'id':0,'scale':0.0},{'id':1,'scale':0.0}]]
        results=[e['body']['adapter'] for e in store.events(task['id']) if e['kind']=='model_result']
        assert results==['concise-v2','index-v1','base']
        health=await model.health();assert health['ready'] and health['indexer_adapter']['loaded']
        await model.close()
    asyncio.run(run())
