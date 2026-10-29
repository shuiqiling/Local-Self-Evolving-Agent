import asyncio
from dataclasses import replace
import json
import threading
import numpy as np
import pytest
import httpx
from lea.config import Settings
from lea.catalog import CATALOG_VERSION
from lea.retrieval import HybridMemory
from lea.runtime import Runtime, short_answer
from lea.model import ModelClient, messages
from lea.schemas import Decision, CatalogSummary
from lea.store import Store
from test_runtime import FakeModel, handbook


class FakeEncoder:
    model_key='test-encoder-v1'
    def __init__(self):self.inputs=[]
    async def health(self):return {'ready':True,'model':'test'}
    async def close(self):pass
    async def encode(self,texts,query=False):
        self.inputs.append((list(texts),query));result=np.zeros((len(texts),1024),dtype=np.float32)
        for i,text in enumerate(texts):
            for j,words in enumerate((('补给','加水','蓝港'),('温度','摄氏度'),('通行','门'),('代号','测试鸟'))):
                if any(word in text for word in words):result[i,j]=1
            if not result[i].any():result[i,20]=1
        return result/np.linalg.norm(result,axis=1,keepdims=True)


def setup(tmp_path,encoder=None):
    store=Store(tmp_path);task=store.create_task('检索验证',[],[],[]);model=FakeModel()
    memory=HybridMemory(store,model,Settings(data_dir=tmp_path),encoder or FakeEncoder())
    return store,task,model,memory


def warm(memory,task_id):
    memory.index.sync(task_id)
    async def run():
        while rows:=memory.index.pending():await memory.index.encode_pending(rows)
    asyncio.run(run())


def test_default_search_returns_summary_not_archive_or_proven_facts(tmp_path,monkeypatch):
    store,task,model,memory=setup(tmp_path)
    original=store.append(task['id'],'user','message',{'text':'通行条件：门未经现场确认打开之前，不可通行。'})
    warm(memory,task['id'])
    def forbidden(*a):raise AssertionError('default retrieval must not open archive text')
    monkeypatch.setattr(store,'resolve',forbidden)
    result=asyncio.run(memory.query(task['id'],'通行条件'))
    assert result['answer_status']=='index_matches' and result['memory_candidates'][0]['event_refs']==[original['ref']]
    assert result['evidence']==[] and result['supported_facts']==[] and not result['archive_read']
    assert not model.roles and result['generation_calls']==0
    brief=memory.for_primary(result)
    assert '_sources' not in json.dumps(brief) and not brief['evidence']


def test_keyword_channel_is_independent_and_both_low_returns_empty(tmp_path):
    class Encoder(FakeEncoder):
        async def encode(self,texts,query=False):
            result=np.zeros((len(texts),1024),dtype=np.float32)
            result[:,0]=.6 if query else 1
            if query:result[:,1]=.8
            return result
    store,task,model,memory=setup(tmp_path,Encoder())
    store.append(task['id'],'user','message',{'text':'设备 HK-88 的温度上限是35摄氏度。'})
    warm(memory,task['id'])
    result=asyncio.run(memory.query(task['id'],'HK-88 的温度上限'))
    card=result['memory_candidates'][0]
    assert card['semantic_score']==.6 and card['keyword_score']==1.0 and card['retrieval_channels']==['keyword']
    assert not result['evidence']
    result=asyncio.run(memory.query(task['id'],'完全无关的主题'))
    assert not result['memory_candidates']
    result=asyncio.run(memory.query(task['id'],'完全无关的主题',min_score=.5))
    assert result['memory_candidates'][0]['match_score']==.6


def test_embedding_only_sees_precomputed_summary_and_never_long_raw_text(tmp_path):
    store,task,model,memory=setup(tmp_path)
    original='背景描述。'*45+'通行前必须确认门已打开，未确认时禁止通行。'
    store.append(task['id'],'user','message',{'text':original});memory.index.sync(task['id'])
    units=memory.index.pending_summaries();assert units and not memory.index.pending()
    memory.index.complete_summary(units[0],'通行条件：未确认门打开时禁止通行。')
    warm(memory,task['id'])
    ingested=[text for batch,query in memory.embedder.inputs if not query for text in batch]
    assert original not in ingested and ingested==['通行条件：未确认门打开时禁止通行。']


def test_explicit_read_is_bounded_and_uses_exact_authorized_original(tmp_path):
    store,task,model,memory=setup(tmp_path)
    event=store.append(task['id'],'user','message',{'text':'通行前必须确认门已打开；未确认时禁止通行。'})
    warm(memory,task['id']);result=asyncio.run(memory.query(task['id'],'通行条件'))
    read=memory.read(task['id'],result['request_id'],[result['memory_candidates'][0]['card_id']],'通行条件')
    assert read['archive_read'] and read['evidence'][0]['event_ref']==event['ref']
    assert '禁止' in read['evidence'][0]['quote'] and len(''.join(x['quote'] for x in read['evidence']))<=400
    assert not model.roles
    other=store.create_task('其他',[],[],[])
    with pytest.raises(ValueError):memory.read(other['id'],result['request_id'],[result['memory_candidates'][0]['card_id']])
    with pytest.raises(ValueError):memory.read(task['id'],result['request_id'],['not-selected'])


def test_edited_query_cache_cannot_expand_archive_read_scope(tmp_path):
    store,task,_,memory=setup(tmp_path)
    event=store.append(task['id'],'user','message',{'text':'通行必须先开门。'});warm(memory,task['id'])
    result=asyncio.run(memory.query(task['id'],'通行'))
    with store.db(True) as db:db.execute("UPDATE queries SET result='{}'")
    read=memory.read(task['id'],result['request_id'],[result['memory_candidates'][0]['card_id']])
    assert read['evidence'][0]['quote']==event['body']['text']


def test_summary_corruption_is_detected(tmp_path):
    store,task,_,memory=setup(tmp_path)
    store.append(task['id'],'user','message',{'text':'通行必须先开门。'});warm(memory,task['id'])
    with store.db(True) as db:db.execute("UPDATE memory_catalog SET summary='通行无需开门。'")
    result=asyncio.run(memory.query(task['id'],'通行'))
    assert not result['memory_candidates'] and 'integrity' in result['error']


def test_vector_fingerprint_change_does_not_reuse_prior_vectors(tmp_path):
    store,task,model,memory=setup(tmp_path);store.append(task['id'],'user','message',{'text':'温度低于20摄氏度。'});warm(memory,task['id'])
    encoder=FakeEncoder();encoder.model_key='changed'
    second=HybridMemory(store,model,Settings(data_dir=tmp_path),encoder)
    result=asyncio.run(second.query(task['id'],'温度限制'))
    assert not result['memory_candidates'] and result['index']['embedded_cards']==0


def test_old_snapshot_does_not_regress_catalog_watermark(tmp_path):
    store,task,_,memory=setup(tmp_path)
    first=store.append(task['id'],'user','message',{'text':'温度低于20摄氏度。'})
    later=store.append(task['id'],'user','message',{'text':'更正：温度低于18摄氏度。'})
    memory.index.sync(task['id'],later['seq']);memory.index.sync(task['id'],first['seq'])
    with store.db() as db:assert db.execute('SELECT through_seq FROM catalog_watermarks WHERE task_id=? AND version=?',(task['id'],CATALOG_VERSION)).fetchone()[0]==later['seq']
    rows,_,_=memory.index.candidates(task['id'],'温度',first['seq'],100,None)
    assert all(row['seq']<=first['seq'] for row in rows)


def test_vector_outage_still_uses_summary_keywords_without_archive(tmp_path):
    class Down(FakeEncoder):
        async def encode(self,*a,**k):raise ConnectionError('offline')
    store,task,model,memory=setup(tmp_path);store.append(task['id'],'user','message',{'text':'通行条件是先开门。'});warm(memory,task['id'])
    memory.embedder=Down();result=asyncio.run(memory.query(task['id'],'通行条件'))
    assert result['memory_candidates'] and result['warnings'] and not result['archive_read']
    assert result['memory_candidates'][0]['retrieval_channels']==['keyword']
    rescue=asyncio.run(memory.query(task['id'],'通行条件',search_archive=True))
    assert rescue['score_type']=='keyword_coverage' and rescue['memory_candidates'] and not rescue['evidence']
    assert not model.roles


def test_handoff_index_is_ingested_with_stable_archive_links(tmp_path):
    store,task,_,memory=setup(tmp_path)
    event=store.append(task['id'],'user','message',{'text':'补给站在蓝港。'})
    book=handbook();book['history_entries']=[{'statement':'蓝港补给站位置','evidence_refs':[event['ref']]}]
    store.commit_handoff(task['id'],task['window_id'],event['seq'],book);warm(memory,task['id'])
    result=asyncio.run(memory.query(task['id'],'在哪里加水？'))
    assert any(x['summary_method']=='handoff_model' and event['ref'] in x['event_refs'] for x in result['memory_candidates'])


def test_optional_verification_uses_one_call_and_canonical_refs(tmp_path):
    store,task,model,memory=setup(tmp_path)
    event=store.append(task['id'],'user','message',{'text':'通行前必须确认门已打开。'});warm(memory,task['id'])
    result=asyncio.run(memory.query(task['id'],'通行条件',verify=True))
    assert model.roles==['coordinator'] and result['generation_calls']==1
    assert result['answer_status']=='model_reviewed' and result['verification_scope'][0]['event_ref']==event['ref']


def test_primary_uses_index_then_explicit_read_and_respects_answer_limit(tmp_path):
    class Agent(FakeModel):
        async def call(self,task_id,role,prompt,schema,**kwargs):
            self.roles.append(role);payload=json.loads(prompt[1]['content']);last=payload['last_result']
            assert kwargs['limit']==384
            if last is None:return Decision(action='recall',response='',tool='',arguments={'question':'通行条件'})
            if not last['archive_read']:return Decision(action='read_memory',response='',tool='',arguments={'request_id':last['request_id'],'card_ids':[last['memory_candidates'][0]['card_id']]})
            assert last['evidence'][0]['quote']=='未确认门打开时禁止通行。'
            return Decision(action='finish',response='未确认门打开时禁止通行。'+'附加说明。'*80,tool='',arguments={})
    async def run():
        runtime=Runtime(Settings(data_dir=tmp_path),model=Agent(),embedder=FakeEncoder());task=runtime.store.create_task('验证',[],[],[])
        runtime.store.append(task['id'],'user','message',{'text':'未确认门打开时禁止通行。'})
        runtime.memory.index.sync(task['id']);await runtime.memory.index.encode_pending(runtime.memory.index.pending())
        job=runtime.store.add_job(task['id'],'message',{'text':'请查证通行条件'})
        result=await runtime.primary(job)
        assert len(result['response'])<=200 and result['answer_truncated'] and '已截短' in result['response']
        assert runtime.model.roles==['primary']*2
        await runtime.memory.close()
    asyncio.run(run())


def test_short_answer_marks_truncation():
    answer,truncated=short_answer('甲。'*300,200)
    assert len(answer)<=200 and truncated and answer.endswith('…（已截短）')


def test_prepared_context_primary_answers_in_one_call_without_archive_tool(tmp_path):
    class Agent(FakeModel):
        async def call(self,task_id,role,prompt,schema,**kwargs):
            self.roles.append(role);payload=json.loads(prompt[1]['content'])
            assert payload['memory_queries_remaining']==0 and not payload['last_result']['archive_read']
            assert payload['last_result']['memory_candidates'][0]['summary']=='通行前必须现场确认开门。'
            return Decision(action='finish',response='据记忆摘要，通行前必须现场确认开门。',tool='',arguments={})
    async def run():
        runtime=Runtime(Settings(data_dir=tmp_path),model=Agent(),embedder=FakeEncoder());task=runtime.store.create_task('验证',[],[],[])
        event=runtime.store.append(task['id'],'user','message',{'text':'通行前必须现场确认开门。'})
        runtime.store.commit_handoff(task['id'],task['window_id'],event['seq'],handbook())
        runtime.memory.index.sync(task['id']);await runtime.memory.index.encode_pending(runtime.memory.index.pending())
        job=runtime.store.add_job(task['id'],'message',{'text':'通行条件是什么？'})
        result=await runtime.primary(job)
        assert runtime.model.roles==['primary'] and result['archive_reads']==0 and result['memory_basis']=='retrieved_context'
        assert not any(e['kind']=='archive_read_scope' for e in runtime.store.events(task['id']))
        await runtime.memory.close()
    asyncio.run(run())


def test_vector_encoding_and_keyword_lookup_overlap(tmp_path,monkeypatch):
    vector_started=threading.Event();keyword_started=threading.Event()
    class Encoder(FakeEncoder):
        async def encode(self,texts,query=False):
            if query:
                vector_started.set()
                assert await asyncio.to_thread(keyword_started.wait,1)
            return await super().encode(texts,query=query)
    store,task,_,memory=setup(tmp_path,Encoder());store.append(task['id'],'user','message',{'text':'通行前必须开门。'});warm(memory,task['id'])
    real=memory.index.keyword_candidates
    def lookup(*args):
        keyword_started.set();assert vector_started.wait(1)
        return real(*args)
    monkeypatch.setattr(memory.index,'keyword_candidates',lookup)
    result=asyncio.run(memory.query(task['id'],'通行条件'))
    assert result['memory_candidates'] and 'error' not in result


def test_related_summary_completes_identifier_context_without_archive(tmp_path,monkeypatch):
    store,task,model,memory=setup(tmp_path)
    first=store.append(task['id'],'user','message',{'text':'设备 DV-73 对应工单 TC-804。'})
    second=store.append(task['id'],'user','message',{'text':'工单 TC-804 的联络人是段清。'})
    warm(memory,task['id'])
    def forbidden(*a):raise AssertionError('context completion must use the summary catalog')
    monkeypatch.setattr(store,'resolve',forbidden)
    result=asyncio.run(memory.query(task['id'],'DV-73 的联络人是谁？'))
    assert first['ref'] in result['memory_candidates'][0]['event_refs']
    assert second['ref'] in result['context_supplements'][0]['event_refs']
    assert len(memory.for_primary(result)['memory_candidates'])==2 and not result['evidence'] and not model.roles


def test_backend_schema_allows_answer_or_optional_read_and_caps_answer(tmp_path):
    async def run():
        store=Store(tmp_path);task=store.create_task('语法约束',[],[],[]);seen=[]
        def handler(request):
            if request.url.path=='/apply-template':return httpx.Response(200,json={'prompt':'test'})
            if request.url.path=='/tokenize':return httpx.Response(200,json={'tokens':[1,2]})
            data=json.loads(request.content);seen.append(data)
            return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps({'action':'read_memory','response':'','tool':'','arguments':{'request_id':'q1','card_ids':['M1']}})},'finish_reason':'stop'}]})
        model=ModelClient(Settings(data_dir=tmp_path),store);await model.http.aclose()
        model.http=httpx.AsyncClient(base_url='http://test',transport=httpx.MockTransport(handler))
        payload={'memory_queries_remaining':0,'memory_reads_remaining':1,'last_result':{'request_id':'q1','memory_candidates':[{'card_id':'M1'}],'archive_read':False}}
        await model.call(task['id'],'primary',messages('test',payload),Decision,limit=900)
        schema=seen[0]['response_format']['json_schema']['schema']
        assert seen[0]['max_tokens']==384
        assert {'read_memory','respond','finish'}<=set(schema['properties']['action']['enum'])
        assert 'recall' not in schema['properties']['action']['enum']
        assert schema['properties']['response']['maxLength']==200
        await model.close()
    asyncio.run(run())


def test_explicit_cancel_is_not_resurrected_during_shutdown(tmp_path):
    async def run():
        model=FakeModel(delay=30);runtime=Runtime(Settings(data_dir=tmp_path),model=model,embedder=FakeEncoder())
        task=runtime.store.create_task('取消验证',[],[],[]);job=runtime.store.add_job(task['id'],'message',{'text':'你好'})
        await runtime.start()
        for _ in range(100):
            if model.started.is_set():break
            await asyncio.sleep(.01)
        assert model.started.is_set();runtime.cancel(job['id']);await runtime.close();runtime.store.recover()
        assert runtime.store.job(job['id'])['status']=='canceled'
    asyncio.run(run())


def test_direct_query_does_not_wait_for_execution_lock_or_generator(tmp_path):
    async def run():
        class Down(FakeModel):
            async def health(self):return {'ready':False}
        runtime=Runtime(Settings(data_dir=tmp_path),model=Down(),embedder=FakeEncoder());task=runtime.store.create_task('并发',[],[],[])
        job=runtime.store.add_job(task['id'],'query',{'question':'通行条件'})
        async with runtime.locks[task['id']]:result=await asyncio.wait_for(runtime.dispatch(job),timeout=2)
        assert result['generation_calls']==0 and not runtime.model.roles
        await runtime.memory.close()
    asyncio.run(run())
