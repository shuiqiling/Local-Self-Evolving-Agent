"""Local auxiliary extraction API sharing one backend with foreground priority."""
from pathlib import Path
import asyncio,json,time,uuid
import httpx
import helper_off_core as H

def install(app,guard,auto_root):
    from fastapi import Request,HTTPException
    auto_root=Path(auto_root)
    def configuration():
        config=json.loads((auto_root/'config.json').read_text());options=config.get('memory_teacher',{})
        if options.get('projection_version')!=H.VERSION or not options.get('enabled'):
            raise HTTPException(503,'辅助经历提炼尚未启用')
        path=Path(options['runtime_manifest'])
        if H.sha(path.read_bytes())!=options['runtime_manifest_sha256']:raise HTTPException(503,'辅助运行配置哈希不匹配')
        return config,H.manifest(path)
    @app.get('/api/helper-profile')
    def profile():
        config,meta=configuration()
        return {'enabled':True,'version':H.VERSION,'model':meta['model'],'weight_origin':meta['weight_origin'],
          'adapter_id':meta['adapter_id'],'adapter_sha256':meta['gguf_sha256'],'thinking_enabled':False,
          'max_attempts':1,'context_tokens':meta['context_tokens'],'output':'source-selected seven-field records; impact unknown unless separately verified',
          'extraction_mode':meta.get('extraction_mode','legacy'),
          'requires_source_review':True,'automatic_training_enabled':bool(config.get('enabled')),
          'main_behavior_targets_generated':False,'endpoint':'/api/internal/helper-extract'}
    @app.post('/api/internal/helper-extract')
    async def extract(request:Request):
        config,meta=configuration();raw_input=await request.body()
        if len(raw_input)>1024*1024:raise HTTPException(413,'提炼输入超过1MiB，请按完整任务拆分')
        try:
            body=json.loads(raw_input);payload=body['input'];call_body=H.request_for(payload,meta)
        except (ValueError,KeyError,TypeError) as error:raise HTTPException(422,str(error)) from error
        from memory_capacity import capacity
        if not capacity(config.get('memory_capacity',{}))['write_allowed']:raise HTTPException(507,'记忆存储已满，暂停新增提炼记录')
        from reasoning_policy import busy,post_while_idle,ForegroundYield
        paths=config['memory_teacher'].get('idle_databases',[])
        if await asyncio.to_thread(busy,paths):raise HTTPException(409,'前台任务运行中，辅助提炼稍后重试')
        async with guard.generation_lock:
            if await asyncio.to_thread(busy,paths):raise HTTPException(409,'前台任务优先')
            guard.last_namespace=None
            async with httpx.AsyncClient(trust_env=False) as client:
                tick=time.monotonic()
                try:
                    response=await client.get(meta['backend']+'/lora-adapters',timeout=15);response.raise_for_status()
                    loaded={a['id']:a['path'] for a in response.json()}
                    assert loaded.get(meta['adapter_id'])==meta['gguf'],'辅助LoRA未加载或标识不匹配'
                    assert loaded.get(meta['chat_adapter_id'])==meta['chat_gguf'],'主聊天LoRA标识不匹配'
                    response=await client.post(meta['backend']+'/apply-template',json=call_body,timeout=15);response.raise_for_status()
                    response=await client.post(meta['backend']+'/tokenize',json={'content':response.json()['prompt'],'add_special':False,'parse_special':True},timeout=15);response.raise_for_status()
                    input_tokens=len(response.json()['tokens'])
                    if input_tokens+meta['max_output_tokens']+32>meta['context_tokens']:
                        raise HTTPException(413,'完整原文超出当前提炼窗口，请按任务及操作回执拆分；未截断原文')
                    result=await post_while_idle(meta['backend'],call_body,paths,meta['request_timeout_seconds'],client=client)
                    envelope=H.finish(payload,result,meta.get('extraction_mode'))
                except ForegroundYield as error:raise HTTPException(409,'已让出推理资源给前台任务，本次提炼未完成') from error
                except (httpx.HTTPError,AssertionError) as error:raise HTTPException(503,str(error)) from error
                finally:guard.last_namespace=None
        envelope.update(seconds=round(time.monotonic()-tick,3),input_tokens=input_tokens,
          helper_version=H.VERSION,helper_adapter_sha256=meta['gguf_sha256'])
        identifier='experience-'+uuid.uuid4().hex
        audit={'id':identifier,'request':call_body,'input':payload,'response':result,'result':envelope}
        directory=Path(meta['audit_dir']);directory.mkdir(parents=True,exist_ok=True)
        target=directory/(identifier+'.json');temporary=target.with_suffix('.tmp')
        try:temporary.write_text(json.dumps(audit,ensure_ascii=False,indent=2));temporary.replace(target)
        except OSError as error:raise HTTPException(507,'提炼已停止：无法保存来源审计记录') from error
        envelope.update(id=identifier,audit_path=str(target))
        return envelope
