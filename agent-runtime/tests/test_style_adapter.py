import asyncio
from dataclasses import replace
import json

import httpx

from lea.config import Settings
from lea.model import ModelClient
from lea.schemas import Decision
from lea.store import Store


def test_style_adapter_only_applies_to_primary_and_is_logged(tmp_path):
    async def check():
        store=Store(tmp_path);task=store.create_task('test',[],[],[])
        settings=replace(Settings(),primary_adapter_id=0,primary_adapter_version='concise-v1')
        calls=[]
        def handle(request):
            if request.url.path=='/apply-template':return httpx.Response(200,json={'prompt':'test'})
            if request.url.path=='/tokenize':return httpx.Response(200,json={'tokens':[1]})
            if request.url.path=='/health':return httpx.Response(200,json={'status':'ok'})
            if request.url.path=='/lora-adapters':return httpx.Response(200,json=[{'id':0,'scale':0}])
            body=json.loads(request.content);calls.append(body)
            return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps({'action':'finish','response':'完成','tool':'','arguments':{}})},'finish_reason':'stop'}]})
        client=ModelClient(settings,store);await client.http.aclose()
        client.http=httpx.AsyncClient(base_url='http://test',transport=httpx.MockTransport(handle))
        message=[{'role':'user','content':json.dumps({'memory_queries_remaining':0,'memory_reads_remaining':0})}]
        for role in ['primary','handoff']:
            await client.call(task['id'],role,message,Decision)
        assert calls[0]['lora']==[{'id':0,'scale':1.0}]
        assert calls[1]['lora']==[{'id':0,'scale':0.0}]
        assert calls[0]['max_tokens']==settings.primary_output_tokens
        assert calls[0]['response_format']['json_schema']['schema']['properties']['response']['maxLength']==200
        events=store.events(task['id']);results=[e['body'] for e in events if e['kind']=='model_result']
        assert [r['adapter'] for r in results]==['concise-v1','base']
        health=await client.health();assert health['ready'] and health['primary_adapter']['scope']=='primary_only'
        await client.close()
    asyncio.run(check())
