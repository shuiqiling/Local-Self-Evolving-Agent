"""Task-scoped, snapshot-stable memory pages. Indexes navigate immutable sources."""
import asyncio
import copy
from dataclasses import replace
import hashlib
import json
import time
import uuid

VERSION='active-memory-pages-v1'
HELP={
 'memory.search':'{"query":"当前需要的历史信息"} 混合检索原文，返回最多3条摘要索引；没有匹配则空。',
 'memory.next':'{"cursor":"返回的next_cursor"} 查看同一次查询的下一批最多3条索引。',
 'memory.read':'{"index_id":"返回的索引ID","query":"需要的细节"} 只在该索引所属原文中找细节。',
}
FIELDS={
 'memory.search':{'query':{'type':'string','minLength':1,'maxLength':200}},
 'memory.next':{'cursor':{'type':'string','minLength':1,'maxLength':80}},
 'memory.read':{'index_id':{'type':'string','minLength':1,'maxLength':80},'query':{'type':'string','minLength':1,'maxLength':200}},
}
def dump(v):return json.dumps(v,ensure_ascii=False,separators=(',',':'),sort_keys=True)
def uid(prefix):return prefix+uuid.uuid4().hex[:16]


class ActiveMemory:
    def __init__(self,runtime):
        self.runtime,self.store,self.memory=runtime,runtime.store,runtime.memory
        with self.store.db(True) as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS active_memory_searches(
              id TEXT PRIMARY KEY,task_id TEXT NOT NULL,query_id TEXT NOT NULL,card_ids TEXT NOT NULL,created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS active_memory_pages(
              cursor TEXT PRIMARY KEY,task_id TEXT NOT NULL,search_id TEXT NOT NULL,offset INTEGER NOT NULL,next_cursor TEXT,
              UNIQUE(search_id,offset));
            CREATE TABLE IF NOT EXISTS active_memory_grants(
              index_id TEXT PRIMARY KEY,task_id TEXT NOT NULL,search_id TEXT NOT NULL,card_id TEXT NOT NULL,UNIQUE(search_id,card_id));
            ''')

    def result(self,task_id,search_id):
        with self.store.db() as db:
            search=db.execute('SELECT * FROM active_memory_searches WHERE id=? AND task_id=?',(search_id,task_id)).fetchone()
            if not search:raise ValueError('Unknown search in this task; call memory.search')
            event=db.execute("SELECT body FROM events WHERE task_id=? AND kind='retrieval_result' AND json_extract(body,'$.request_id')=? ORDER BY seq DESC LIMIT 1",(task_id,search['query_id'])).fetchone()
        if not event:raise ValueError('Search snapshot unavailable; call memory.search')
        result=json.loads(event['body']);cards={c['card_id']:c for c in result.get('memory_candidates',[])}
        return dict(search),result,[cards[cid] for cid in json.loads(search['card_ids'])]

    async def search(self,task_id,query,user_event):
        # A private view changes only this query's result pool, never live settings.
        view=copy.copy(self.memory);view.settings=replace(self.memory.settings,retrieval_top_k=80)
        result=await view.query(task_id,query,exclude_event_id=user_event)
        if result.get('error'):return {'items':[],'next_cursor':None,'status':'unavailable','error':result['error']}
        cards=[];seen=set()
        for c in result.get('memory_candidates',[]):
            source=tuple(p['event_ref'] for p in c.get('_sources',[]))
            if not source or source in seen:continue
            seen.add(source);cards.append(c['card_id'])
        sid=uid('ms_');cursor=uid('mp_')
        with self.store.db(True) as db:
            db.execute('INSERT INTO active_memory_searches VALUES(?,?,?,?,?)',(sid,task_id,result['request_id'],dump(cards),time.time()))
            db.execute('INSERT INTO active_memory_pages VALUES(?,?,?,?,NULL)',(cursor,task_id,sid,0))
        return self.page(task_id,cursor)

    def page(self,task_id,cursor):
        with self.store.db() as db:page=db.execute('SELECT * FROM active_memory_pages WHERE cursor=? AND task_id=?',(cursor,task_id)).fetchone()
        if not page:raise ValueError('Unknown page cursor in this task; use the returned next_cursor')
        search,result,cards=self.result(task_id,page['search_id']);offset=page['offset'];items=[]
        with self.store.db(True) as db:
            for card in cards[offset:offset+3]:
                grant=db.execute('SELECT index_id FROM active_memory_grants WHERE search_id=? AND card_id=?',(search['id'],card['card_id'])).fetchone()
                index_id=grant[0] if grant else uid('mi_')
                if not grant:db.execute('INSERT INTO active_memory_grants VALUES(?,?,?,?)',(index_id,task_id,search['id'],card['card_id']))
                summary=card.get('summary','');short=summary[:100]
                items.append({'index_id':index_id,'summary':short,'summary_excerpt':len(summary)>100 or bool(card.get('preview_only')),
                              'score':card['match_score'],'source':card.get('source_types',[]),'time':card.get('recorded_at','')[:19],
                              'details_required':bool(card.get('details_required',True)) or len(summary)>100})
            next_cursor=None
            if offset+3<len(cards):
                existing=db.execute('SELECT cursor FROM active_memory_pages WHERE search_id=? AND offset=?',(search['id'],offset+3)).fetchone()
                next_cursor=existing[0] if existing else uid('mp_')
                if not existing:db.execute('INSERT INTO active_memory_pages VALUES(?,?,?,?,NULL)',(next_cursor,task_id,search['id'],offset+3))
                db.execute('UPDATE active_memory_pages SET next_cursor=? WHERE cursor=?',(next_cursor,cursor))
        return {'items':items,'next_cursor':next_cursor,'status':'matches' if items else 'no_matches',
                'score_is_probability':False,'scope':'同一任务的固定检索快照；索引不是执行证明。',
                'scope_complete':False,'warnings':result.get('warnings',[])[:2]}

    async def read(self,task_id,index_id,query):
        with self.store.db() as db:grant=db.execute('SELECT * FROM active_memory_grants WHERE index_id=? AND task_id=?',(index_id,task_id)).fetchone()
        if not grant:raise ValueError('Choose an index_id already returned in this task')
        _,snapshot,cards=self.result(task_id,grant['search_id'])
        card=next(c for c in cards if c['card_id']==grant['card_id']);pointer=card['_sources'][0]
        event=self.store.resolve(task_id,pointer['event_ref']);text=event['text']
        if event['seq']>snapshot['source_cutoff_seq'] or hashlib.sha256(text[pointer['start']:pointer['end']].encode()).hexdigest()!=pointer['sha256']:
            raise ValueError('Memory source disagrees with the authorized snapshot')
        from lea.embedding import source_type,tokens
        from lea.archive import source_units,sentence_spans
        if len(text)<=900:
            lo,hi,mode=0,len(text),'whole_selected_record'
        else:
            index=self.memory.index
            with self.store.db() as db:
                rows=[dict(r) for r in db.execute('''SELECT u.*,v.vector FROM archive_units u LEFT JOIN archive_vectors v
                  ON v.unit_id=u.id AND v.model_key=? WHERE u.task_id=? AND u.event_ref=? AND u.version=? AND u.seq<=?''',
                  (index.model_key,task_id,event['ref'],index.version,snapshot['source_cutoff_seq']))]
            if not rows:rows=[{'start':a,'end':b,'text':text[a:b],'vector':None} for a,b,_ in source_units(text,'window')]
            words=set(tokens(query));ranked=[];vector=None
            if any(r['vector'] is not None for r in rows):
                try:vector=(await asyncio.wait_for(self.memory.embedder.encode([query],query=True),3))[0]
                except Exception:pass  # Scoped keyword fallback; cancellation still propagates.
            import numpy as np
            for r in rows:
                if text[r['start']:r['end']]!=r['text']:raise ValueError('Indexed detail changed')
                lexical=len(words&set(tokens(r['text'])))/max(1,len(words));semantic=0.
                if vector is not None and r['vector'] is not None:
                    v=np.frombuffer(r['vector'],dtype='<f4')
                    if len(v)!=len(vector) or not np.isfinite(v).all():raise ValueError('Invalid scoped memory vector')
                    semantic=float(v@vector)
                ranked.append((max(lexical,semantic),r['start'],r['end']))
            score,lo,hi=max(ranked)
            if score<=0:return {'index_id':index_id,'evidence':[],'status':'no_relevant_detail','scope':'仅查询选中索引的原文'}
            spans=sentence_spans(text);neighbors=[i for i,(a,b) in enumerate(spans) if a<hi and b>lo]
            if neighbors:
                a=spans[max(0,min(neighbors)-1)][0];b=spans[min(len(spans)-1,max(neighbors)+1)][1]
                if b-a<=900:lo,hi=a,b
            mode='scoped_vector_keyword' if vector is not None else 'scoped_keyword'
        return {'index_id':index_id,'status':'read','method':mode,'evidence':[{'ref':event['ref'],'source':source_type(event),
                'time':event['created'],'text':text[lo:hi],'start':lo,'end':hi,'excerpt_only':lo!=0 or hi!=len(text),
                'source_chars':len(text)}],'scope':'只读选中索引的原文；文件内容和模型说法不证明功能已测试。'}

    async def execute(self,task_id,window,version,user_event,tool,arguments):
        fields=FIELDS.get(tool)
        if not fields or not isinstance(arguments,dict) or set(arguments)!=set(fields):raise ValueError('Invalid memory tool arguments')
        for key,spec in fields.items():
            value=arguments[key]
            if not isinstance(value,str) or not value.strip() or len(value)>spec['maxLength']:raise ValueError('Invalid '+key)
        operation,previous=self.store.begin_operation(task_id,window,version,user_event,tool,arguments)
        if previous:return {**previous,'deduplicated':True}
        try:
            if tool=='memory.search':value=await self.search(task_id,arguments['query'],user_event)
            elif tool=='memory.next':value=self.page(task_id,arguments['cursor'])
            else:value=await self.read(task_id,arguments['index_id'],arguments['query'])
            result={'ok':True,'tool':tool,'operation_id':operation,'value':value}
        except asyncio.CancelledError:
            self.store.finish_operation(task_id,operation,{'ok':False,'tool':tool,'operation_id':operation,'error':'Memory lookup interrupted; no completed result.'},failed=True)
            raise
        except Exception as error:result={'ok':False,'tool':tool,'operation_id':operation,'error':str(error)}
        event=self.store.finish_operation(task_id,operation,result,failed=not result['ok'])
        return {**result,'evidence_ref':event['ref']}
