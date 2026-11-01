"""Independent original-record indexes: sentence baseline and condition-preserving windows."""
from __future__ import annotations
import hashlib
import json
import re
import numpy as np
from .embedding import eligible,source_type,tokens
from .store import dump


def sentence_spans(text):
    spans=[];start=0
    for match in re.finditer(r'[。！？!?\n]+|(?<=[a-zA-Z])\.(?=\s|$)',text):
        if text[start:match.end()].strip():spans.append((start,match.end()))
        start=match.end()
    if text[start:].strip():spans.append((start,len(text)))
    return spans


def source_units(text,mode):
    spans=sentence_spans(text)
    for i,(start,end) in enumerate(spans):
        if mode=='window' and i+1<len(spans) and spans[i+1][1]-start<=400:end=spans[i+1][1]
        # Very long unpunctuated text remains accessible with a visible partial-span marker.
        if end-start>800:
            for lo in range(start,end,700):yield lo,min(lo+800,end),True
        else:yield start,end,False


class ArchiveCatalog:
    def __init__(self,store,embedder,mode='window'):
        self.store,self.embedder,self.mode=store,embedder,mode
        self.version='archive-'+mode+'-v1';self.enable_keyword=mode!='sentence'
        self.model_key=hashlib.sha256(dump([embedder.model_key,self.version]).encode()).hexdigest()
        self.raw=self
        with store.db() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS archive_units(
              id TEXT PRIMARY KEY,task_id TEXT NOT NULL,event_ref TEXT NOT NULL,event_id TEXT NOT NULL,
              seq INTEGER NOT NULL,version TEXT NOT NULL,start INTEGER NOT NULL,end INTEGER NOT NULL,
              text TEXT NOT NULL,digest TEXT NOT NULL,partial INTEGER NOT NULL,question_context INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS archive_task ON archive_units(task_id,version,seq);
            CREATE TABLE IF NOT EXISTS archive_vectors(unit_id TEXT NOT NULL,model_key TEXT NOT NULL,vector BLOB NOT NULL,PRIMARY KEY(unit_id,model_key));
            CREATE TABLE IF NOT EXISTS archive_watermarks(task_id TEXT NOT NULL,version TEXT NOT NULL,through_seq INTEGER NOT NULL,PRIMARY KEY(task_id,version));
            CREATE TABLE IF NOT EXISTS memory_query_origins(task_id TEXT NOT NULL,event_id TEXT NOT NULL,PRIMARY KEY(task_id,event_id));
            CREATE VIRTUAL TABLE IF NOT EXISTS archive_fts USING fts5(terms,unit_id UNINDEXED,task_id UNINDEXED);
            ''')

    def sync(self,task_id,cutoff=None):
        self.store.task(task_id)
        with self.store.db(True) as db:
            if cutoff is None:cutoff=db.execute('SELECT COALESCE(MAX(seq),0) FROM events WHERE task_id=?',(task_id,)).fetchone()[0]
            row=db.execute('SELECT through_seq FROM archive_watermarks WHERE task_id=? AND version=?',(task_id,self.version)).fetchone();through=row[0] if row else 0
            for row in db.execute('SELECT * FROM events WHERE task_id=? AND seq>? AND seq<=? ORDER BY seq',(task_id,through,cutoff)).fetchall():
                event=self.store._decode_event(row)
                if not eligible(event):continue
                question=int(bool(re.search(r'[?？]|调用\s*recall',event['text'],re.I)))
                for start,end,partial in source_units(event['text'],self.mode):
                    text=event['text'][start:end];key=hashlib.sha256(dump([task_id,event['ref'],self.version,start,end]).encode()).hexdigest()
                    digest=hashlib.sha256(dump([event['ref'],start,end,text]).encode()).hexdigest()
                    inserted=db.execute('INSERT OR IGNORE INTO archive_units VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                        (key,task_id,event['ref'],event['id'],event['seq'],self.version,start,end,text,digest,int(partial),question)).rowcount
                    if inserted:db.execute('INSERT INTO archive_fts VALUES(?,?,?)',(' '.join(tokens(text)),key,task_id))
            db.execute('INSERT OR REPLACE INTO archive_watermarks VALUES(?,?,?)',(task_id,self.version,max(through,cutoff)))
        return cutoff

    def mark_query_origin(self,task_id,event_id):
        if not event_id:return
        with self.store.db(True) as db:
            if db.execute('SELECT 1 FROM events WHERE task_id=? AND id=?',(task_id,event_id)).fetchone():
                db.execute('INSERT OR IGNORE INTO memory_query_origins VALUES(?,?)',(task_id,event_id))

    @staticmethod
    def pointer(event,start,end):
        return {'event_ref':event['ref'],'start':start,'end':end,'sha256':hashlib.sha256(event['text'][start:end].encode()).hexdigest()}

    def _validated(self,task_id,cutoff,exclude,rows):
        result=[]
        with self.store.db() as db:
            for row in rows:
                if row['event_id']==exclude:continue
                digest=hashlib.sha256(dump([row['event_ref'],row['start'],row['end'],row['text']]).encode()).hexdigest()
                if digest!=row['digest']:raise ValueError('Archive index integrity mismatch')
                window,event_id=row['event_ref'].split(':',1)
                source=db.execute('SELECT role,created,seq FROM events WHERE task_id=? AND window_id=? AND id=? AND seq<=?',(task_id,window,event_id,cutoff)).fetchone()
                if source is None:raise ValueError('Archive candidate outside task/snapshot')
                row.update(summary=row['text'],sources=[{'event_ref':row['event_ref'],'start':row['start'],'end':row['end'],
                    'sha256':hashlib.sha256(row['text'].encode()).hexdigest()}],method='original_'+self.mode,
                    source_types=[source_type(dict(source))],recorded_at=source['created'],seq=source['seq'])
                result.append(row)
        return result

    def vector_candidates(self,task_id,cutoff,limit,exclude=None):
        with self.store.db() as db:
            rows=[dict(r) for r in db.execute('''SELECT c.*,v.vector FROM archive_units c LEFT JOIN archive_vectors v
                ON v.unit_id=c.id AND v.model_key=? WHERE c.task_id=? AND c.version=? AND c.seq<=? ORDER BY c.seq DESC LIMIT ?''',
                (self.model_key,task_id,self.version,cutoff,limit))]
            total=db.execute('SELECT COUNT(*) FROM archive_units WHERE task_id=? AND version=? AND seq<=?',(task_id,self.version,cutoff)).fetchone()[0]
        return self._validated(task_id,cutoff,exclude,rows),total

    def keyword_candidates(self,task_id,question,cutoff,exclude=None):
        if not self.enable_keyword:return []
        words=list(dict.fromkeys(tokens(question)))[:64]
        if not words:return []
        match=' OR '.join('"'+w.replace('"','""')+'"' for w in words)
        with self.store.db() as db:
            rows=[dict(r) for r in db.execute('''SELECT c.*,v.vector,bm25(archive_fts) AS bm25_score FROM archive_fts f
                JOIN archive_units c ON c.id=f.unit_id LEFT JOIN archive_vectors v ON v.unit_id=c.id AND v.model_key=?
                WHERE archive_fts MATCH ? AND c.task_id=? AND c.version=? AND c.seq<=? ORDER BY bm25(archive_fts) LIMIT 40''',
                (self.model_key,match,task_id,self.version,cutoff))]
        return self._validated(task_id,cutoff,exclude,rows)

    def candidates(self,task_id,question,cutoff,limit,exclude):
        rows,total=self.vector_candidates(task_id,cutoff,limit,exclude);lex=self.keyword_candidates(task_id,question,cutoff,exclude)
        found={r['id'] for r in rows};rows.extend(r for r in lex if r['id'] not in found)
        return rows,lex,total

    def pending(self,limit=4):
        with self.store.db() as db:
            return [dict(r) for r in db.execute('''SELECT c.* FROM archive_units c WHERE version=? AND NOT EXISTS
                (SELECT 1 FROM archive_vectors v WHERE v.unit_id=c.id AND v.model_key=?) ORDER BY seq DESC LIMIT ?''',(self.version,self.model_key,limit))]

    async def encode_pending(self,rows):
        for offset in range(0,len(rows),2):
            batch=rows[offset:offset+2];vectors=await self.embedder.encode([r['text'] for r in batch])
            with self.store.db(True) as db:
                for row,vector in zip(batch,vectors):db.execute('INSERT OR REPLACE INTO archive_vectors VALUES(?,?,?)',(row['id'],self.model_key,np.asarray(vector,dtype='<f4').tobytes()))

    def pending_summaries(self,limit=1):return []

    def counts(self,task_id=None):
        scope=' AND c.task_id=?' if task_id else '';args=[self.version]+([task_id] if task_id else [])
        with self.store.db() as db:
            total=db.execute('SELECT COUNT(*) FROM archive_units c WHERE c.version=?'+scope,args).fetchone()[0]
            done=db.execute('SELECT COUNT(*) FROM archive_units c JOIN archive_vectors v ON v.unit_id=c.id WHERE c.version=? AND v.model_key=?'+scope,[self.version,self.model_key]+([task_id] if task_id else [])).fetchone()[0]
        return {'total_cards':total,'embedded_cards':done,'pending_vectors':total-done,'pending_summaries':0,'failed_summaries':0}
