"""Isolated30B memory-agent integration, with explicit model/KV boundaries."""
from dataclasses import replace
import httpx
from lea.config import Settings
from lea.model import ModelClient
from lea.app import create_app as base_app

ALIAS='Qwen3-30B-A3B-Q4_K_M';URL='http://127.0.0.1:18813'

def settings():
    return replace(Settings(),model_url=URL,model_name=ALIAS,precision='Q4_K_M',
        native_kv_url='',primary_adapter_id=-1,primary_adapter_version='',model_status_file='',
        primary_cache_slot=-1,auxiliary_cache_slot=-1,window_prompt_layout='stable_v1',
        working_tokens=4096,primary_output_tokens=384,answer_max_chars=200)

class GuardedHTTP:
    def __init__(self,inner):self.inner=inner
    async def get(self,*args,**kwargs):return await self.inner.get(*args,**kwargs)
    async def post(self,path,**kwargs):
        body=kwargs.get('json')
        if path=='/v1/chat/completions':
            if body.get('model')!=ALIAS or body.get('lora'):
                raise ValueError('Memory pilot requires30B base and no4B adapters')
            kwargs['json']={**body,'presence_penalty':0,'chat_template_kwargs':{'enable_thinking':False}}
        return await self.inner.post(path,**kwargs)
    async def aclose(self):await self.inner.aclose()

class MemoryModel(ModelClient):
    def __init__(self,config,store):
        super().__init__(config,store);assert self.native_kv is None
        self.http=GuardedHTTP(self.http)
    async def health(self):
        value=await super().health()
        if not value.get('ready'):return value
        try:
            response=await self.http.get('/v1/models',timeout=3);response.raise_for_status()
            identity=next((m for m in response.json()['data'] if m['id']==ALIAS),None)
            assert identity and identity['meta']['n_params']>30_000_000_000,'Unexpected model at30B endpoint'
            assert identity['meta']['n_ctx']>=self.settings.backend_context,'Backend context too small'
            value.update(active_primary_backend='llama_http_30b',native_kv_reused=False,thinking_enabled=False)
        except (httpx.HTTPError,ValueError,AssertionError,KeyError) as error:
            value.update(ready=False,identity_error=str(error))
        return value

def create_app():
    from lea.store import Store
    config=settings();model=MemoryModel(config,Store(config.data_dir))
    app=base_app(settings=config,model=model)
    @app.get('/api/model-profile')
    def profile():
        return {'model':ALIAS,'backend':URL,'primary_adapter':None,'native_kv_reused':False,
            'native_kv_compression':False,'handoff':'text summary with verified source anchors',
            'state_dir':str(config.data_dir),'primary_chat_18800_changed':False,
            'memory':'deployed hybrid_rerank original-record retrieval, optional archive reads',
            'corpus':'production snapshot; later writes isolated','working_tokens':config.working_tokens,
            'automatic_lora':'not_enabled','experimental':True}
    return app
