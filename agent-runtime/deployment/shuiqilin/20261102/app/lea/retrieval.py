"""Summary-first retrieval, hard score gate, explicit bounded archive reads."""
from __future__ import annotations
import asyncio
import hashlib
import json
import re
import time
import numpy as np
import httpx
from .catalog import SummaryCatalog, SUMMARY_PROMPT
from .embedding import EmbeddingClient, source_type, tokens
from .memory import Memory, SYNTHESIZER
from .model import messages
from .schemas import CatalogSummary, Synthesis
from .store import dump, now, uid


class HybridMemory:
    rule_rerank=True
    strategy='summary_hybrid'
    def __init__(self,store,model,settings,embedder=None):
        self.store,self.model,self.settings=store,model,settings
        self.embedder=embedder or EmbeddingClient(settings)
        self.index=SummaryCatalog(store,self.embedder)
        self.worker=None;self.summarizer=None;self.last_index_error=None;self.last_query_at=time.monotonic()

    async def start(self):
        self.worker=asyncio.create_task(self._index_worker())
        self.summarizer=asyncio.create_task(self._summary_worker())

    async def close(self):
        workers=[w for w in (self.worker,self.summarizer) if w]
        for worker in workers:worker.cancel()
        await asyncio.gather(*workers,return_exceptions=True)
        await self.embedder.close()

    async def _index_worker(self):
        while True:
            try:
                if getattr(getattr(self.model,'queue',None),'busy',False):
                    await asyncio.sleep(.5);continue
                for task in self.store.tasks():await asyncio.to_thread(self.index.sync,task['id'])
                rows=self.index.pending(4)
                if rows:
                    await self.index.encode_pending(rows);self.last_index_error=None
                    await asyncio.sleep(.05)
                else:await asyncio.sleep(1)
            except asyncio.CancelledError:raise
            except Exception as exc:
                self.last_index_error=type(exc).__name__+': '+str(exc);await asyncio.sleep(3)

    async def _summary_worker(self):
        while True:
            units=self.index.pending_summaries()
            busy=getattr(getattr(self.model,'queue',None),'busy',False)
            if not units or busy or self.index.pending(1) or time.monotonic()-self.last_query_at<5 or not (await self.model.health())['ready']:
                await asyncio.sleep(2);continue
            unit=units[0]
            try:
                original=self.store.resolve(unit['task_id'],unit['event_ref'])
                text=original['text'][unit['start']:unit['end']]
                prompt=messages(SUMMARY_PROMPT,{'source_type':source_type(original),'recorded_at':original['created'],'archive_text':text})
                answer=await self.model.call(unit['task_id'],'indexer',prompt,CatalogSummary,limit=200)
                self.index.complete_summary(unit,answer.summary)
            except asyncio.CancelledError:raise
            except Exception as exc:self.index.fail_summary(unit,exc)
            await asyncio.sleep(1)

    async def health(self):
        state=await self.embedder.health();counts=self.index.counts()
        return {**state,**counts,'index_kind':'summary_catalog','indexed_chunks':counts['embedded_cards'],
                'total_chunks':counts['total_cards'],'pending_chunks':counts['pending_vectors'],
                'min_score':self.settings.retrieval_min_cosine,'score_type':'hybrid_relevance',
                'pipeline':'parallel_vector_keyword_rrf_context_v2','index_error':self.last_index_error}

    @staticmethod
    def _card(row,score,score_type,**extra):
        return {'card_id':row['id'],'summary':row['summary'],'match_score':round(score,6),'score_type':score_type,
            'source_types':row['source_types'],'recorded_at':row['recorded_at'],
            'event_refs':list(dict.fromkeys(p['event_ref'] for p in row['sources'])),
            'summary_method':row['method'],'_sources':row['sources'],'_catalog_digest':row['digest'],**extra}

    def query_policy(self,min_score):
        value=self.settings.retrieval_min_cosine if min_score is None else min_score
        return {'semantic_min':value,'keyword_min':value,'calibrated':False}

    @staticmethod
    def selected_cards(result):
        direct=result.get('memory_candidates',[]);related=result.get('context_supplements',[])
        return direct[:2]+related[:1] if related else direct[:3]

    async def _complete_context(self,task_id,question,cards,cutoff,exclude):
        # Follow at most two explicit identifier links in summaries. No archive text.
        pattern=r'(?<![a-z0-9_-])[a-z]+[-_/][a-z0-9_-]*\d[a-z0-9_-]*(?![a-z0-9_-])'
        known=set(re.findall(pattern,question.lower()));links={}
        for parent in cards[:2]:
            for identifier in re.findall(pattern,parent['summary'].lower()):
                if identifier not in known:links.setdefault(identifier,parent['card_id'])
        links=dict(list(links.items())[:2])
        if not links:return []
        values=await asyncio.gather(*(asyncio.to_thread(self.index.keyword_candidates,task_id,key,cutoff,exclude) for key in links))
        existing={c['card_id'] for c in cards};terms=set(tokens(question));ranked=[]
        for (key,parent),rows in zip(links.items(),values):
            for row in rows:
                if row['id'] in existing or key not in set(tokens(row['summary'])):continue
                score=len(terms&set(tokens(row['summary'])))+(0 if row['source_types']==['model_conclusion'] else 1)
                ranked.append((score,row,key,parent))
        if not ranked:return []
        _,row,key,parent=max(ranked,key=lambda x:(x[0],x[1]['seq']))
        return [self._card(row,1.,'linked_identifier',context_role='related_summary',linked_identifier=key,
            linked_from_card=parent,retrieval_channels=['linked_keyword'])]

    @staticmethod
    def for_primary(result):
        brief={k:result[k] for k in ('request_id','answer_status','answer','scope_complete','warnings','error','semantic_verification') if k in result}
        brief['memory_candidates']=[{'card_id':f'M{i+1}',**{k:c[k] for k in ('summary','match_score','source_types','recorded_at')},
            'context_role':c.get('context_role','direct_match'),'has_qualification':bool(re.search(r'未|不|仅|更正|低于|高于',c['summary']))}
                                    for i,c in enumerate(HybridMemory.selected_cards(result))]
        brief['evidence']=[{k:e[k] for k in ('event_ref','quote','source_type','excerpt_only')} for e in result.get('evidence',[])]
        brief['archive_read']=bool(result.get('archive_read'))
        return brief

    async def query(self,task_id,question,exhaustive=False,verify=False,exclude_event_id=None,
                    read_archive=False,min_score=None,search_archive=False,_persist=True):
        started=time.monotonic();self.last_query_at=started;task=self.store.task(task_id);query_id=uid('query')
        threshold=self.settings.retrieval_min_cosine if min_score is None else float(min_score)
        policy=self.query_policy(min_score)
        if not 0<=threshold<=1:raise ValueError('min_score must be within [0,1]')
        with self.store.db(True) as db:
            db.execute('INSERT INTO queries VALUES(?,?,?,?,?,?,?)',(query_id,task_id,task['version'],'running',dump({'question':question,'verify':verify,'read_archive':read_archive,'min_score':threshold,'search_archive':search_archive}),None,now()))
        try:
            self.index.mark_query_origin(task_id,exclude_event_id)
            cutoff=await asyncio.to_thread(self.index.sync,task_id)
            warnings=[];cards=[];embedding_ms=0.;rerank_ms=0.;max_score=None
            if search_archive:
                _,rows,total=await asyncio.to_thread(self.index.raw.candidates,task_id,question,cutoff,self.settings.retrieval_max_chunks,exclude_event_id)
                terms=set(tokens(question));ranked=[]
                for row in rows:
                    coverage=len(terms&set(tokens(row['text'])))/max(len(terms),1);max_score=max(max_score or 0,coverage)
                    if coverage<threshold:continue
                    source=self.store.resolve(task_id,row['event_ref'])
                    if source['text'][row['start']:row['end']]!=row['text']:raise ValueError('Raw search index disagrees with source')
                    ranked.append((coverage,{'card_id':'raw:'+row['id'],'summary':'原文关键词补查入口：'+row['text'][:80],
                        'match_score':round(coverage,6),'score_type':'keyword_coverage','source_types':[source_type(source)],
                        'recorded_at':source['created'],'event_refs':[source['ref']],'summary_method':'explicit_keyword_rescue',
                        '_sources':[self.index.pointer(source,row['start'],row['end'])]}))
                cards=[c for _,c in sorted(ranked,key=lambda pair:-pair[0])]
                warnings.append('显式原文关键词补查：分数是词项覆盖率，不是向量相似度。')
            else:
                tick=time.monotonic()
                dense_result,lexical,encoded=await asyncio.gather(
                    asyncio.to_thread(self.index.vector_candidates,task_id,cutoff,self.settings.retrieval_max_chunks,exclude_event_id),
                    asyncio.to_thread(self.index.keyword_candidates,task_id,question,cutoff,exclude_event_id),
                    asyncio.wait_for(self.embedder.encode([question],query=True),timeout=3),return_exceptions=True)
                for value in (dense_result,lexical):
                    if isinstance(value,BaseException):raise value
                rows,total=dense_result;all_rows={row['id']:row for row in rows}
                all_rows.update({row['id']:row for row in lexical});semantic={}
                if isinstance(encoded,BaseException):
                    if isinstance(encoded,asyncio.CancelledError):raise encoded
                    warnings.append('向量通道暂不可用，仍独立评估摘要关键词候选：'+type(encoded).__name__)
                else:
                    vector=encoded[0];indexed=[row for row in all_rows.values() if row['vector'] is not None]
                    if indexed:
                        matrix=np.stack([np.frombuffer(row['vector'],dtype='<f4') for row in indexed])
                        if matrix.shape[1]!=len(vector) or not np.isfinite(matrix).all():raise ValueError('Invalid catalog vectors')
                        semantic={row['id']:float(score) for row,score in zip(indexed,matrix@vector)}
                embedding_ms=round((time.monotonic()-tick)*1000,3)
                rerank_started=time.monotonic()
                dense_rank={identifier:i+1 for i,identifier in enumerate(sorted(semantic,key=semantic.get,reverse=True)[:40])}
                lex_rank={row['id']:i+1 for i,row in enumerate(lexical)}
                terms=set(tokens(question));identifiers=set(re.findall(r'(?<![a-z0-9_-])[a-z]+[-_/][a-z0-9_-]*\d[a-z0-9_-]*(?![a-z0-9_-])',question.lower()))
                current=bool(re.search(r'现在|目前|当前|最新|最终|current|latest',question,re.I));original=bool(re.search(r'最初|最早|原先|original|initial',question,re.I))
                ranked=[]
                for identifier in set(dense_rank)|set(lex_rank):
                    row=all_rows[identifier];words=set(tokens(row['summary']));overlap=len(terms&words)
                    if self.strategy!='sentence_threshold' and identifiers and not identifiers<=words:continue
                    lexical_score=overlap/max(len(terms),1) if identifier in lex_rank and (overlap>=2 or identifiers) else 0.
                    cosine=semantic.get(identifier);score=max(max(cosine or 0,0),lexical_score);max_score=max(max_score or 0,score)
                    if not (cosine is not None and cosine>=policy['semantic_min'] or lexical_score>=policy['keyword_min']):continue
                    channels=[]
                    if cosine is not None and cosine>=policy['semantic_min']:channels.append('vector')
                    if lexical_score>=policy['keyword_min']:channels.append('keyword')
                    rank=(1/(60+dense_rank[identifier]) if identifier in dense_rank else 0)+(1/(60+lex_rank[identifier]) if identifier in lex_rank else 0)
                    rank+=.004*score
                    if self.strategy=='sentence_threshold':rank=cosine or 0
                    if self.rule_rerank and row['source_types']==['model_conclusion']:rank*=.75
                    if self.rule_rerank and row.get('question_context'):rank*=.35
                    correction=bool(re.search(r'更正|修正|改为|更新为|作废|撤销',row['summary']))
                    if self.rule_rerank and current and not original:rank+=(.003 if correction else 0)+.001*row['seq']/max(cutoff,1)
                    if self.rule_rerank and original and correction:rank-=.003
                    ranked.append((rank,row,score,cosine,lexical_score,channels))
                seen=[]
                for rank,row,score,cosine,lexical_score,channels in sorted(ranked,key=lambda x:(-x[0],-x[1]['seq'])):
                    refs={p['event_ref'] for p in row['sources']};words=set(tokens(row['summary']))
                    if any(refs==old_refs and len(words&old_words)/max(len(words|old_words),1)>.7 for old_refs,old_words in seen):continue
                    seen.append((refs,words))
                    cards.append(self._card(row,score,'hybrid_relevance',semantic_score=round(cosine,6) if cosine is not None else None,
                        keyword_score=round(lexical_score,6),retrieval_channels=channels,rerank_score=round(rank,6)))
                rerank_ms=round((time.monotonic()-rerank_started)*1000,3)
            cards=cards[:self.settings.retrieval_top_k*(2 if exhaustive else 1)];counts=self.index.counts(task_id)
            context_started=time.monotonic()
            supplements=await self._complete_context(task_id,question,cards,cutoff,exclude_event_id) if cards and not search_archive else []
            linked_context_ms=round((time.monotonic()-context_started)*1000,3)
            if counts['pending_summaries'] or counts['pending_vectors']:warnings.append('摘要目录尚未完全就绪，不能据空结果断言历史中不存在答案。')
            result={'request_id':query_id,'task_id':task_id,'question':question,'requested_task_version':task['version'],
                'retrieval_mode':'archive_keyword' if search_archive else 'hybrid_summary','answer_status':'index_matches' if cards else 'no_matches',
                'answer':f'找到{len(cards)}条摘要候选；摘要足够时可直接回答，缺少细节时再读档。' if cards else '没有达到匹配门槛的候选。',
                'memory_candidates':cards,'context_supplements':supplements,'evidence':[],'evidence_refs':[],'supported_facts':[],'inferences':[],'conflicts':[],
                'unresolved_questions':[],'scope_complete':False,'inspected_scope':[],'source_cutoff_seq':cutoff,
                'match_threshold':threshold,'max_match_score':round(max_score,6) if max_score is not None else None,
                'score_type':'keyword_coverage' if search_archive else 'hybrid_relevance','score_is_probability':False,
                'match_rule':'at_least_one_channel_passes_then_rrf_rerank','context_completed_without_archive':True,
                'acceptance_policy':policy,
                'index':{**counts,'model_key':self.index.model_key,'scanned_candidates':total},
                'archive_read':False,'semantic_verification':'not_requested','warnings':warnings,'generation_calls':0,'pipeline':'parallel_vector_keyword_rrf_context_v2',
                'timings':{'query_embedding_ms':embedding_ms,'parallel_recall_ms':embedding_ms,'fusion_rerank_ms':rerank_ms,'linked_context_ms':linked_context_ms}}
            if (read_archive or verify) and cards:self._read_selected(task_id,result,[c['card_id'] for c in self.selected_cards(result)],question)
            if verify and result['evidence']:await self._verify(task_id,question,result)
            result['timings']['search_ms']=round((time.monotonic()-started)*1000,3)
            if _persist:
                Memory.record(self,query_id,'completed',result);self.store.append(task_id,'retriever','retrieval_result',result)
            result['timings']['service_ms']=round((time.monotonic()-started)*1000,3)
            return result
        except asyncio.CancelledError:
            Memory.record(self,query_id,'canceled');raise
        except Exception as exc:
            result={'request_id':query_id,'task_id':task_id,'answer_status':'unable_to_judge','answer':'检索未完成。',
                'memory_candidates':[],'evidence':[],'scope_complete':False,'error':str(exc),'error_type':type(exc).__name__,'warnings':[]}
            Memory.record(self,query_id,'error',result);return result

    def _read_selected(self,task_id,result,card_ids,question):
        allowed={c['card_id']:c for c in result['memory_candidates']+result.get('context_supplements',[])}
        if not card_ids or len(card_ids)>3 or set(card_ids)-set(allowed):raise ValueError('Read requires 1–3 cards selected by this query')
        terms=set(tokens(question));remaining=self.settings.memory_read_chars;evidence=[];seen=set()
        for identifier in card_ids:
            for p in allowed[identifier]['_sources']:
                event=self.store.resolve(task_id,p['event_ref'])
                if event['seq']>result['source_cutoff_seq']:raise ValueError('Archive read exceeded query snapshot')
                start,end=p['start'],p['end'];text=event['text'][start:end]
                if not 0<=start<end<=len(event['text']) or hashlib.sha256(text.encode()).hexdigest()!=p['sha256']:raise ValueError('Archive pointer disagrees with source')
                sentences=list(re.finditer(r'[^。！？!?\n]+[。！？!?\n]?',text))
                ranked=sorted(sentences,key=lambda m:-(len(terms&set(tokens(m.group())))+(.5 if re.search(r'仅|未|不|更正|低于|高于',m.group()) else 0)))
                for part in ranked[:2]:
                    lo=start+part.start();hi=start+part.end()
                    if (event['ref'],lo,hi) in seen or remaining<=0:continue
                    quote=event['text'][lo:hi];size=min(len(quote),remaining,200)
                    if not size:continue
                    evidence.append({'event_ref':event['ref'],'quote':quote[:size],'start':lo,'end':lo+size,'source_type':source_type(event),
                        'recorded_at':event['created'],'seq':event['seq'],'excerpt_only':size<len(quote),'card_id':identifier,'relevance':'按需读取原文，尚未语义核验'})
                    seen.add((event['ref'],lo,hi));remaining-=size
        result.update(evidence=evidence,evidence_refs=list(dict.fromkeys(e['event_ref'] for e in evidence)),archive_read=True,
            answer_status='retrieved' if evidence else 'no_matches',answer=f'按需读取了{len(evidence)}条原文片段，仍需核对语义适用性。',
            inspected_scope=[{k:e[k] for k in ('event_ref','start','end')} for e in evidence])
        self.store.append(task_id,'archive_reader','archive_read_scope',{'request_id':result['request_id'],'card_ids':card_ids,'inspected_scope':result['inspected_scope']})

    def read(self,task_id,request_id,card_ids,question=''):
        with self.store.db() as db:
            row=db.execute("SELECT body FROM events WHERE task_id=? AND kind='retrieval_result' AND json_extract(body,'$.request_id')=? ORDER BY seq DESC LIMIT 1",(task_id,request_id)).fetchone()
        if row is None:raise ValueError('Query scope is outside this task or unavailable')
        result=json.loads(row['body']);self._read_selected(task_id,result,card_ids,question);result['generation_calls']=0
        self.store.append(task_id,'archive_reader','archive_read_result',result)
        return result

    async def _verify(self,task_id,question,result):
        try:
            refs={f'S{i+1}':ref for i,ref in enumerate(result['evidence_refs'])};reverse={v:k for k,v in refs.items()}
            evidence=[{**item,'event_ref':reverse[item['event_ref']]} for item in result['evidence']]
            prompt=messages(SYNTHESIZER,{'question':question,'evidence':evidence,'scope_complete':False})
            while await self.model.tokens(prompt)>self.settings.working_tokens and evidence:
                evidence.pop();prompt=messages(SYNTHESIZER,{'question':question,'evidence':evidence,'scope_complete':False})
            if not evidence:raise ValueError('Verification evidence exceeds working budget')
            self.store.append(task_id,'verifier','verification_scope',{'request_id':result['request_id'],'ref_aliases':refs,
                'visible_aliases':[item['event_ref'] for item in evidence]})
            result['generation_calls']=1
            answer=await self.model.call(task_id,'coordinator',prompt,Synthesis,limit=320)
            visible={item['event_ref'] for item in evidence}
            def findings(items):
                return [{'statement':item.statement,'evidence_refs':[refs[ref] for ref in item.evidence_refs]} for item in items
                        if item.evidence_refs and set(item.evidence_refs)<=visible]
            result.update(supported_facts=findings(answer.findings),conflicts=findings(answer.conflicts),inferences=answer.inferences,
                          unresolved_questions=answer.unresolved_questions,semantic_verification='model_assisted',generation_calls=1)
            result['answer']='\n'.join(item['statement'] for item in result['supported_facts']) or result['answer']
            result['model_review_status']=answer.status
            result['answer_status']='model_reviewed'
            result['semantic_review_required']=True
            result['verification_scope']=[{'event_ref':refs[item['event_ref']],'start':item['start'],'end':item['end']} for item in evidence]
            if len(result['supported_facts'])!=len(answer.findings) or len(result['conflicts'])!=len(answer.conflicts):
                result['warnings'].append('部分模型结论引用无效，已丢弃；保留原文候选供核对。')
        except Exception as exc:
            result['semantic_verification']='failed';result['warnings'].append('语义核验未完成，原文候选保留：'+type(exc).__name__)
