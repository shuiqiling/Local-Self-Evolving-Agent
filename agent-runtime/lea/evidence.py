"""Hybrid original-record retrieval. Summaries navigate; they do not gate the archive."""
from __future__ import annotations
import asyncio
import json
import time
from pathlib import Path
from .archive import ArchiveCatalog,sentence_spans
from .catalog import SummaryCatalog
from .embedding import EmbeddingClient,source_type
from .retrieval import HybridMemory


class EvidenceMemory(HybridMemory):
    def __init__(self,store,model,settings,embedder=None,strategy=None,policy=None):
        self.store,self.model,self.settings=store,model,settings
        self.embedder=embedder or EmbeddingClient(settings)
        file=store.root/'retrieval-calibration.json';loaded={}
        if policy is None and file.exists():loaded=json.loads(file.read_text(encoding='utf-8'))
        self.policy=policy or loaded.get('policy') or {'semantic_min':.5,'keyword_min':.65,'calibrated':False,'source':'provisional_unvalidated'}
        self.strategy=strategy or loaded.get('strategy','hybrid_rerank')
        if self.strategy not in ('sentence_threshold','hybrid_window','hybrid_rerank'):raise ValueError('Unknown retrieval strategy')
        self.rule_rerank=self.strategy=='hybrid_rerank'
        self.index=ArchiveCatalog(store,self.embedder,'sentence' if self.strategy=='sentence_threshold' else 'window')
        self.navigation=SummaryCatalog(store,self.embedder)
        self.worker=None;self.summarizer=None;self.last_index_error=None
        self.last_query_at=time.monotonic()

    async def start(self):self.worker=asyncio.create_task(self._index_worker())

    def query_policy(self,min_score):
        if min_score is not None:return {'semantic_min':min_score,'keyword_min':min_score,'calibrated':False,'source':'manual_override'}
        if self.strategy=='sentence_threshold':return {'semantic_min':.65,'keyword_min':1.01,'calibrated':False,'source':'fixed_baseline'}
        return dict(self.policy)

    async def health(self):
        value=await super().health()
        value.update(index_kind='original_record_windows',strategy=self.strategy,acceptance_policy=self.query_policy(None),
            summaries_are_only_entry=False,pipeline='parallel_recall_expand_once_answer_v3',model_key=self.index.model_key,unit_version=self.index.version)
        return value

    async def _complete_context(self,*args):
        return [] if self.strategy=='sentence_threshold' else await super()._complete_context(*args)

    @staticmethod
    def for_primary(result):
        brief=HybridMemory.for_primary(result)
        for card in brief['memory_candidates']:
            card['preview_only']=len(card['summary'])>40
            card['summary']=card['summary'][:40]
        brief['evidence']=[{'event_ref':f'E{i+1}','quote':e['quote'],'source_type':e['source_type'],
            'recorded_at':e['recorded_at'],'excerpt_only':e['excerpt_only']} for i,e in enumerate(result.get('evidence',[]))]
        brief['context_kind']='original_record_excerpts' if brief['evidence'] else 'candidate_navigation_only'
        brief['may_answer_without_archive_tool']=True
        return brief

    def _prepare_evidence(self,task_id,result):
        intervals=[];cache={}
        for card in self.selected_cards(result):
            for pointer in card['_sources']:
                ref=pointer['event_ref'];raw=cache.setdefault(ref,self.store.resolve(task_id,ref))
                lo,hi=pointer['start'],pointer['end']
                import hashlib
                if hashlib.sha256(raw['text'][lo:hi].encode()).hexdigest()!=pointer['sha256']:raise ValueError('Indexed evidence disagrees with immutable source')
                if self.strategy!='sentence_threshold':
                    spans=sentence_spans(raw['text']);indices=[i for i,(a,b) in enumerate(spans) if a<hi and b>lo]
                    if indices:lo=spans[max(0,min(indices)-1)][0];hi=spans[min(len(spans)-1,max(indices)+1)][1]
                old=next((x for x in intervals if x[0]==ref and x[1]<=hi and lo<=x[2]),None)
                if old:old[1]=min(old[1],lo);old[2]=max(old[2],hi)
                else:intervals.append([ref,lo,hi,card['card_id'],pointer['start'],pointer['end']])
                # Optional model-requested read may inspect more of this same authorized event.
                pointer.update(self.index_pointer(raw,0,len(raw['text'])))
        evidence=[];remaining=700
        for ref,lo,hi,card_id,hit_lo,hit_hi in intervals:
            raw=cache[ref]
            if hi-lo>remaining:lo,hi=hit_lo,hit_hi
            if hi-lo>remaining:
                result['warnings'].append('一个完整命中片段超过本轮证据预算，未静默截断；可请求更多信息。');continue
            if not hi>lo:continue
            evidence.append({'event_ref':ref,'start':lo,'end':hi,'quote':raw['text'][lo:hi],'source_type':source_type(raw),
                'recorded_at':raw['created'],'seq':raw['seq'],'excerpt_only':lo!=0 or hi!=len(raw['text']),
                'card_id':card_id,'context_expanded':self.strategy!='sentence_threshold'})
            remaining-=hi-lo
        result.update(evidence=evidence,evidence_refs=list(dict.fromkeys(e['event_ref'] for e in evidence)),
            inspected_scope=[{k:e[k] for k in ('event_ref','start','end')} for e in evidence],
            automatic_context_prepared=True,archive_read=False,context_budget_chars=700)

    @staticmethod
    def index_pointer(raw,lo,hi):
        import hashlib
        return {'event_ref':raw['ref'],'start':lo,'end':hi,'sha256':hashlib.sha256(raw['text'][lo:hi].encode()).hexdigest()}

    async def query(self,task_id,question,exhaustive=False,verify=False,exclude_event_id=None,read_archive=False,min_score=None,search_archive=False):
        # Retrieval and context preparation use zero generative calls.
        started=time.monotonic()
        result=await super().query(task_id,question,exhaustive,False,exclude_event_id,False,min_score,search_archive,_persist=False)
        if 'error' in result:return result
        tick=time.monotonic();self._prepare_evidence(task_id,result)
        result['timings']['context_completion_ms']=round((time.monotonic()-tick)*1000+result['timings'].get('linked_context_ms',0),3)
        if read_archive and self.selected_cards(result):self._read_selected(task_id,result,[c['card_id'] for c in self.selected_cards(result)],question)
        if verify and result['evidence']:await self._verify(task_id,question,result)
        result.update(retrieval_mode=self.strategy,pipeline='parallel_recall_expand_once_answer_v3',summaries_are_only_entry=False,
            answer=f'已准备{len(result["evidence"])}段必要证据，可直接组织回答；只有缺少细节时才额外读档。' if result['evidence'] else '当前未取得足够证据，请保留不确定性。')
        # Persist the prepared scope immutably for optional later read_memory calls.
        from .memory import Memory
        Memory.record(self,result['request_id'],'completed',result)
        self.store.append(task_id,'retriever','retrieval_result',result)
        result['timings']['service_ms']=round((time.monotonic()-started)*1000,3)
        return result
