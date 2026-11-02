"""Background auxiliary indexing, independent of the automatic training gate."""
import asyncio
import json
import time
from pathlib import Path
import httpx
import navigation_contract as C


class Navigator:
    def __init__(self, memory, guard, auto_root):
        self.memory, self.store, self.guard = memory, memory.store, guard
        self.auto_root = Path(auto_root)
        self.task = None
        self.last_error = None
        self.calls = 0
        with self.store.db() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS memory_navigation(
              event_ref TEXT NOT NULL, source_sha TEXT NOT NULL, version TEXT NOT NULL,
              status TEXT NOT NULL, record TEXT, attempts INTEGER NOT NULL DEFAULT 0,
              updated REAL NOT NULL, PRIMARY KEY(event_ref,source_sha,version))''')

    def config(self):
        return json.loads((self.auto_root/'config.json').read_text())

    def get(self, event):
        with self.store.db() as db:
            row = db.execute('SELECT record,status FROM memory_navigation WHERE event_ref=? AND source_sha=? AND version=?',
                             (event['ref'], C.sha(event['text']), C.VERSION)).fetchone()
        if row and row['status'] == 'ready':
            value=json.loads(row['record']); C.validate(event['text'],value); return value
        return C.direct(event['text'])

    def save(self, event, record=None, status='ready'):
        if record: C.validate(event['text'],record)
        with self.store.db() as db:
            db.execute('''INSERT INTO memory_navigation VALUES(?,?,?,?,?,1,?)
              ON CONFLICT(event_ref,source_sha,version) DO UPDATE SET status=excluded.status,
              record=excluded.record,attempts=memory_navigation.attempts+1,updated=excluded.updated''',
              (event['ref'],C.sha(event['text']),C.VERSION,status,
               json.dumps(record,ensure_ascii=False) if record else None,time.time()))

    def pending(self):
        # Source selection is not gated by teacher training eligibility or ten windows.
        with self.store.db() as db:
            rows=db.execute('''SELECT DISTINCT e.* FROM archive_units u JOIN events e ON e.id=u.event_id AND e.task_id=u.task_id
              LEFT JOIN memory_navigation n ON n.event_ref=u.event_ref AND n.version=?
              WHERE u.version='archive-window-v1' AND (n.event_ref IS NULL OR (n.status='retry' AND n.attempts<2))
              ORDER BY e.seq DESC LIMIT 16''',(C.VERSION,)).fetchall()
        return [self.store._decode_event(row) for row in rows]

    def stats(self):
        with self.store.db() as db:
            groups={r['status']:r['n'] for r in db.execute('SELECT status,count(*) n FROM memory_navigation WHERE version=? GROUP BY status',(C.VERSION,))}
            methods={r['method']:r['n'] for r in db.execute("SELECT json_extract(record,'$.method') method,count(*) n FROM memory_navigation WHERE version=? AND status='ready' GROUP BY method",(C.VERSION,))}
            total=db.execute("SELECT count(DISTINCT event_ref) FROM archive_units WHERE version='archive-window-v1'").fetchone()[0]
        return {'version':C.VERSION,'enabled':self.config().get('memory_navigation',{}).get('enabled',False),
                'states':groups,'methods':methods,'eligible_source_records':total,'not_processed_records':max(0,total-sum(groups.values())),
                'calls_since_start':self.calls,'last_error':self.last_error,
                'helper_adapter':'dedicated_auxiliary','thinking_enabled':False,'index_trigger':'new_original_records_when_idle',
                'training_gate_required':False,'original_text_embeddings':True,'automatic_raw_evidence':False,
                'short_source_chars':C.TARGET,'display_budget_chars_per_card':240,'fallback':'explicit_incomplete_original_hit'}

    async def infer(self, text, source_type):
        from helper_off_core import manifest,sha
        from reasoning_policy import busy,post_while_idle,ForegroundYield
        config=self.config();options=config['memory_teacher'];path=Path(options['runtime_manifest'])
        from training_priority import status as training_status
        if (await asyncio.to_thread(training_status,self.auto_root,config))['hold_background_inference']:
            raise ForegroundYield()
        if sha(path.read_bytes())!=options['runtime_manifest_sha256']:raise ValueError('helper_manifest_changed')
        meta=await asyncio.to_thread(manifest,path)
        body=C.request_for(text,source_type,meta)
        paths=options.get('idle_databases',[])
        if await asyncio.to_thread(busy,paths):raise ForegroundYield()
        async with self.guard.generation_lock:
            if await asyncio.to_thread(busy,paths):raise ForegroundYield()
            self.guard.last_namespace=None
            started=time.monotonic()
            try:
                async with httpx.AsyncClient(trust_env=False) as client:
                    loaded=await client.get(meta['backend']+'/lora-adapters',timeout=10);loaded.raise_for_status()
                    identities={x['id']:x['path'] for x in loaded.json()}
                    if identities.get(meta['adapter_id'])!=meta['gguf'] or identities.get(meta['chat_adapter_id'])!=meta['chat_gguf']:
                        raise ValueError('adapter_identity_mismatch')
                    applied=await client.post(meta['backend']+'/apply-template',json=body,timeout=15);applied.raise_for_status()
                    counted=await client.post(meta['backend']+'/tokenize',json={'content':applied.json()['prompt'],'add_special':False,'parse_special':True},timeout=15);counted.raise_for_status()
                    tokens=len(counted.json()['tokens'])
                    if tokens+body['max_tokens']+32>meta['context_tokens']:raise ValueError('source_exceeds_token_budget')
                    response=await post_while_idle(meta['backend'],body,paths,120,client=client)
                record=C.finish(text,response)
                record.update(helper_adapter_sha256=meta['gguf_sha256'],seconds=round(time.monotonic()-started,3),
                              input_tokens=tokens,usage=response.get('usage'),selection_response=response['choices'][0]['message']['content'])
                self.calls+=1
                return record
            finally:self.guard.last_namespace=None

    async def loop(self):
        from lea.embedding import source_type
        from reasoning_policy import ForegroundYield
        while True:
            try:
                config=self.config()
                if not config.get('memory_navigation',{}).get('enabled',False):await asyncio.sleep(2);continue
                if config.get('memory_navigation',{}).get('historical_backfill_active',False):await asyncio.sleep(2);continue
                from training_priority import status as training_status
                priority=await asyncio.to_thread(training_status,self.auto_root,config)
                if priority['hold_background_inference']:await asyncio.sleep(5);continue
                from memory_capacity import capacity
                if not capacity(config.get('memory_capacity',{}))['optional_write_allowed']:await asyncio.sleep(10);continue
                batch=await asyncio.to_thread(self.pending)
                for event in batch:
                    text=event['text'];record=C.direct(text)
                    if not record and (len(text)>C.MAX_INPUT or len(C.units(text))>96):
                        await asyncio.to_thread(self.save,event,None,'oversize');continue
                    # Large structured tool payloads remain fully accessible by read_memory;
                    # never turn expected test values or code strings into factual summaries.
                    if not record and text.lstrip().startswith(('{','[')):
                        await asyncio.to_thread(self.save,event,None,'structured_source');continue
                    if not record:
                        record=await self.infer(text,source_type(event))
                    await asyncio.to_thread(self.save,event,record)
                    self.last_error=None
                    if record['method']=='helper_selected':break
                await asyncio.sleep(2)
            except asyncio.CancelledError:raise
            except ForegroundYield:await asyncio.sleep(3)
            except Exception as error:
                self.last_error=type(error).__name__+': '+str(error)[:200]
                if 'event' in locals():await asyncio.to_thread(self.save,event,None,'retry')
                await asyncio.sleep(5)


def memory_type(base):
    class NavigationMemory(base):
        async def start(self):
            await super().start()
            self.navigator.task=asyncio.create_task(self.navigator.loop())

        async def close(self):
            if self.navigator.task:
                self.navigator.task.cancel();await asyncio.gather(self.navigator.task,return_exceptions=True)
            await super().close()

        async def health(self):
            value=await super().health();value['navigation']=self.navigator.stats();return value

        def _prepare_evidence(self,task_id,result):
            for card in self.selected_cards(result):
                pointer=card['_sources'][0];raw=self.store.resolve(task_id,pointer['event_ref'])
                if C.sha(raw['text'][pointer['start']:pointer['end']])!=pointer['sha256']:
                    raise ValueError('original_hit_changed')
                card.update(C.display(raw['text'],self.navigator.get(raw),pointer['start'],pointer['end']))
                pointer.update(self.index_pointer(raw,0,len(raw['text'])))
            result.update(evidence=[],evidence_refs=[],inspected_scope=[],automatic_context_prepared=False,
                          archive_read=False,context_budget_chars=0)

        def finalize_navigation(self,result):
            result.update(pipeline=C.VERSION,answer='找到导航索引；缺少细节或索引不完整时用read_memory读取对应原文。'
                          if result.get('memory_candidates') else '没有达到匹配门槛的候选。',
                          navigation_version=C.VERSION)

        @staticmethod
        def for_primary(result):
            from lea.retrieval import HybridMemory
            brief=HybridMemory.for_primary(result)
            compact=[];seen=set()
            for shown,card in zip(brief['memory_candidates'],HybridMemory.selected_cards(result)):
                identity=(tuple(card.get('event_refs',[])),shown['summary'])
                if identity in seen:continue
                seen.add(identity)
                value={k:shown[k] for k in ('card_id','summary','match_score','source_types','recorded_at')}
                value['details_required']=card.get('details_required',True)
                if card.get('preview_only'):value['preview_only']=True
                # Keep the original M ordinal: runtime aliases still map to the
                # original ranked source list even when duplicate displays vanish.
                compact.append(value)
            brief['memory_candidates']=compact
            brief.pop('answer',None);brief.pop('semantic_verification',None)
            brief.update(context_kind='explicit_archive_read' if brief['evidence'] else 'candidate_navigation_only',
                         navigation_policy='已有evidence足够就回答；否则details_required的索引需read_memory核对细节。')
            return brief
    return NavigationMemory


def install(app,guard,auto_root):
    from fastapi import Request,HTTPException
    from reasoning_policy import ForegroundYield
    navigator=Navigator(app.state.runtime.memory,guard,auto_root)
    app.state.runtime.memory.navigator=navigator

    @app.get('/api/memory-index-profile')
    def profile():return navigator.stats()

    @app.post('/api/internal/memory-index-preview')
    async def preview(request:Request):
        raw=await request.body()
        if len(raw)>65536:raise HTTPException(413,'source too large')
        try:
            body=json.loads(raw);text=body['text']
            if not isinstance(text,str) or not text.strip():raise ValueError('invalid source')
            if body.get('force_helper') is True:
                return await navigator.infer(text,body.get('source_type','user_statement'))
            return C.direct(text) or await navigator.infer(text,body.get('source_type','user_statement'))
        except ForegroundYield:raise HTTPException(409,'foreground priority')
        except (ValueError,KeyError,TypeError) as error:raise HTTPException(422,str(error))
