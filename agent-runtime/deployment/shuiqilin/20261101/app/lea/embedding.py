"""Local embedding client and rebuildable, versioned indexes over immutable events."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time

import httpx
import numpy as np

from .store import dump

CHUNK_VERSION = 'sentences-600-overlap-80-v1'
QUERY_INSTRUCTION = 'Given a question, retrieve relevant conversation or tool records that answer it, including necessary conditions and source context.'
STOP = set('什么 么是 是什 多少 哪个 哪里 怎样 怎么 是否 请问 查询 记录 历史 信息 记忆 有关 一下 的是 现在 目前 当前 最新'.split())
ENGLISH_STOP = {'the','a','an','is','are','what','where','when','how','please','of','in','to','and','for','my'}


def tokens(text):
    text = text.lower()
    text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[a-z0-9])', '', text)
    result = re.findall(r'[a-z0-9][a-z0-9_./%\-]*',text)
    for segment in re.findall(r'[\u4e00-\u9fffa-z0-9]+',text):
        if re.search(r'[\u4e00-\u9fff]',segment):
            result.extend(segment[i:i+2] for i in range(len(segment)-1))
    return [word for word in result if word not in STOP and word not in ENGLISH_STOP]


def chunks(text):
    start = 0
    while start < len(text):
        end = min(start+600,len(text))
        if end < len(text):
            boundary = max((m.end() for m in re.finditer(r'[。！？!?；;\n]',text[start:end]) if m.end()>=250),default=600)
            end = start+boundary
        yield start,end,text[start:end]
        if end == len(text):break
        start = max(start+1,end-80)


def eligible(event):
    return ((event['role']=='user' and event['kind'] in ('message','constraint_update'))
            or (event['role']=='tool' and event['kind']=='tool_result')
            or (event['role']=='primary' and event['kind']=='assistant_reply'))


def source_type(event):
    return {'user':'user_statement','tool':'tool_observation'}.get(event['role'],'model_conclusion')


class EmbeddingClient:
    def __init__(self,settings):
        self.settings=settings
        self.model_key=hashlib.sha256(dump([settings.embedding_fingerprint,QUERY_INSTRUCTION,CHUNK_VERSION,'last-l2-1024']).encode()).hexdigest()
        self.http=httpx.AsyncClient(base_url=settings.embedding_url,trust_env=False,timeout=httpx.Timeout(20,connect=1))
        self.lock=asyncio.Lock()
        self.verified=False

    async def health(self):
        try:
            response=await self.http.get('/v1/models',timeout=1)
            response.raise_for_status()
            ready=any(item['id']==self.settings.embedding_model for item in response.json()['data'])
            return {'ready':ready,'model':self.settings.embedding_model}
        except (httpx.HTTPError,ValueError,KeyError):
            return {'ready':False,'model':self.settings.embedding_model}

    async def encode(self,texts,query=False):
        if not texts:return np.empty((0,1024),dtype=np.float32)
        if query:texts=[f'Instruct: {QUERY_INSTRUCTION}\nQuery:{text}' for text in texts]
        async with self.lock:
            if not self.verified:
                if not (await self.health())['ready']:raise ConnectionError('Embedding model unavailable or identity mismatch')
                self.verified=True
            response=await self.http.post('/v1/embeddings',json={'model':self.settings.embedding_model,'input':texts,'encoding_format':'float'})
            response.raise_for_status()
            values=response.json()['data']
            if {item['index'] for item in values}!=set(range(len(texts))) or len(values)!=len(texts):
                raise ValueError('Embedding response indexes do not match inputs')
            array=np.asarray([item['embedding'] for item in sorted(values,key=lambda item:item['index'])],dtype=np.float32)
            if array.shape!=(len(texts),1024) or not np.isfinite(array).all():raise ValueError('Invalid embedding dimensions or values')
            norms=np.linalg.norm(array,axis=1,keepdims=True)
            if np.any(norms<1e-8):raise ValueError('Zero embedding vector')
            return array/norms

    async def close(self):await self.http.aclose()


class MemoryIndex:
    def __init__(self,store,embedder):
        self.store,self.embedder=store,embedder
        with store.db() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS memory_chunks(
              id TEXT PRIMARY KEY,task_id TEXT NOT NULL,event_ref TEXT NOT NULL,event_id TEXT NOT NULL,
              seq INTEGER NOT NULL,version TEXT NOT NULL,start INTEGER NOT NULL,end INTEGER NOT NULL,
              text TEXT NOT NULL,source_type TEXT NOT NULL,recorded_at TEXT NOT NULL,question_context INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS memory_chunks_task ON memory_chunks(task_id,version,seq);
            CREATE TABLE IF NOT EXISTS memory_vectors(
              chunk_id TEXT NOT NULL,model_key TEXT NOT NULL,vector BLOB NOT NULL,PRIMARY KEY(chunk_id,model_key));
            CREATE TABLE IF NOT EXISTS memory_watermarks(
              task_id TEXT NOT NULL,version TEXT NOT NULL,through_seq INTEGER NOT NULL,PRIMARY KEY(task_id,version));
            CREATE TABLE IF NOT EXISTS memory_query_origins(task_id TEXT NOT NULL,event_id TEXT NOT NULL,PRIMARY KEY(task_id,event_id));
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(terms,chunk_id UNINDEXED,task_id UNINDEXED);
            ''')

    def sync(self,task_id,cutoff=None):
        self.store.task(task_id)
        with self.store.db() as db:
            if cutoff is None:cutoff=db.execute('SELECT COALESCE(MAX(seq),0) FROM events WHERE task_id=?',(task_id,)).fetchone()[0]
        with self.store.db(True) as db:
            row=db.execute('SELECT through_seq FROM memory_watermarks WHERE task_id=? AND version=?',(task_id,CHUNK_VERSION)).fetchone()
            through=row[0] if row else 0
            rows=db.execute('SELECT * FROM events WHERE task_id=? AND seq>? AND seq<=? ORDER BY seq',(task_id,through,cutoff)).fetchall()
            for row in rows:
                event=self.store._decode_event(row)
                if not eligible(event):continue
                question_context=int(bool(re.search(r'调用\s*recall|(?:查询|检索|找回).{0,8}(?:历史|记忆)|[?？]',event['text'],re.I)))
                for start,end,text in chunks(event['text']):
                    identifier=hashlib.sha256(dump([task_id,event['id'],CHUNK_VERSION,start,end]).encode()).hexdigest()
                    inserted=db.execute('INSERT OR IGNORE INTO memory_chunks VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                        (identifier,task_id,event['ref'],event['id'],event['seq'],CHUNK_VERSION,start,end,text,source_type(event),event['created'],question_context)).rowcount
                    if inserted:db.execute('INSERT INTO memory_fts VALUES(?,?,?)',(' '.join(tokens(text)),identifier,task_id))
            db.execute('INSERT OR REPLACE INTO memory_watermarks VALUES(?,?,?)',(task_id,CHUNK_VERSION,max(through,cutoff)))
        return cutoff

    def mark_query_origin(self,task_id,event_id):
        if not event_id:return
        with self.store.db(True) as db:
            if db.execute('SELECT 1 FROM events WHERE task_id=? AND id=?',(task_id,event_id)).fetchone():
                db.execute('INSERT OR IGNORE INTO memory_query_origins VALUES(?,?)',(task_id,event_id))

    def pending(self,limit=4):
        with self.store.db() as db:
            return [dict(row) for row in db.execute('''SELECT c.* FROM memory_chunks c WHERE version=?
                AND NOT EXISTS(SELECT 1 FROM memory_vectors v WHERE v.chunk_id=c.id AND v.model_key=?)
                ORDER BY c.seq DESC LIMIT ?''',(CHUNK_VERSION,self.embedder.model_key,limit))]

    async def encode_pending(self,rows):
        vectors=await self.embedder.encode([row['text'] for row in rows])
        with self.store.db(True) as db:
            for row,vector in zip(rows,vectors):
                db.execute('INSERT OR REPLACE INTO memory_vectors VALUES(?,?,?)',(row['id'],self.embedder.model_key,np.asarray(vector,dtype='<f4').tobytes()))

    def candidates(self,task_id,question,cutoff,limit,exclude):
        params=(self.embedder.model_key,task_id,CHUNK_VERSION,cutoff,exclude or '',limit)
        sql='''SELECT c.*,v.vector,EXISTS(SELECT 1 FROM memory_query_origins o WHERE o.task_id=c.task_id AND o.event_id=c.event_id) AS query_origin
               FROM memory_chunks c LEFT JOIN memory_vectors v ON v.chunk_id=c.id AND v.model_key=?
               WHERE c.task_id=? AND c.version=? AND c.seq<=? AND c.event_id!=? ORDER BY c.seq DESC LIMIT ?'''
        with self.store.db() as db:
            recent=[dict(row) for row in db.execute(sql,params)]
            total=db.execute('SELECT COUNT(*) FROM memory_chunks WHERE task_id=? AND version=? AND seq<=? AND event_id!=?',(task_id,CHUNK_VERSION,cutoff,exclude or '')).fetchone()[0]
            words=list(dict.fromkeys(tokens(question)))[:64]
            lexical=[]
            if words:
                match=' OR '.join('"'+word.replace('"','""')+'"' for word in words)
                lexical=[dict(row) for row in db.execute('''SELECT c.*,v.vector,EXISTS(SELECT 1 FROM memory_query_origins o WHERE o.task_id=c.task_id AND o.event_id=c.event_id) AS query_origin
                   FROM memory_fts f JOIN memory_chunks c ON c.id=f.chunk_id
                   LEFT JOIN memory_vectors v ON v.chunk_id=c.id AND v.model_key=?
                   WHERE memory_fts MATCH ? AND c.task_id=? AND c.version=? AND c.seq<=? AND c.event_id!=?
                   ORDER BY bm25(memory_fts) LIMIT 40''',(self.embedder.model_key,match,task_id,CHUNK_VERSION,cutoff,exclude or ''))]
        return recent,lexical,total
