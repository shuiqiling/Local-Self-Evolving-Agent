"""Search original vectors, expose navigation only, then search selected spans on demand."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time

import numpy as np
import httpx

from .archive import sentence_spans, source_units
from .catalog import CATALOG_VERSION, SUMMARY_PROMPT, signature
from .embedding import chunks, source_type
from .evidence import EvidenceMemory
from .memory import Memory
from .model import messages
from .indexing import INDEX_SUMMARY_PROMPT
from .retrieval import HybridMemory
from .schemas import CatalogSummary
from .store import dump


PIPELINE = 'original_vectors_navigation_scoped_read_v2'


def navigation_identity(text, summary):
    """Keep a verbatim subject; reject unsupported literal numbers/IDs, not a truth proof."""
    spans = sentence_spans(text)
    title = text[:spans[0][1] if spans else 48].strip()[:48]
    anchors = re.findall(r'[A-Za-z]+[-_/][A-Za-z0-9_.-]*\d[A-Za-z0-9_.-]*|[-+]?\d+(?:\.\d+)?', summary)
    valid = all(anchor.lower() in text.lower() for anchor in anchors)
    return title, valid


class ScopedMemory(EvidenceMemory):
    """Existing raw vectors remain the recall index; summaries never gate recall."""

    async def start(self):
        await super().start()
        self.summarizer = asyncio.create_task(self._navigation_worker())

    async def _navigation_worker(self):
        while True:
            unit = None
            try:
                for task in self.store.tasks():
                    await asyncio.to_thread(self.navigation.sync, task['id'])
                units = self.navigation.pending_summaries()
                if (not units or getattr(getattr(self.model, 'queue', None), 'busy', False) or
                        time.monotonic() - self.last_query_at < 10):
                    await asyncio.sleep(2)
                    continue
                unit = units[0]
                original = self.store.resolve(unit['task_id'], unit['event_ref'])
                prompt = messages(INDEX_SUMMARY_PROMPT, {'source_type': source_type(original),
                    'recorded_at': original['created'], 'archive_text': original['text'][unit['start']:unit['end']]})
                answer = await self.model.call(unit['task_id'], 'indexer', prompt, CatalogSummary, limit=200)
                self.navigation.complete_summary(unit, answer.summary)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if unit is not None:
                    self.navigation.fail_summary(unit, exc)
            await asyncio.sleep(2)

    async def _complete_context(self, *args):
        # No automatic identifier following or original-context expansion.
        return []

    def query_policy(self, min_score):
        policy = super().query_policy(min_score)
        return {**policy, 'keyword_min': 1.01, 'automatic_channel': 'original_vector'}

    async def health(self):
        return {**await super().health(), 'pipeline': PIPELINE,
                'automatic_payload': 'segment_navigation_only', 'scoped_read_chars': self.settings.memory_read_chars,
                'navigation': self.navigation.counts(), 'summaries_are_only_entry': False}

    def _navigation_cards(self, task_id, result):
        self.navigation.sync(task_id, result['source_cutoff_seq'])
        with self.store.db() as db:
            rows = db.execute('SELECT summary,sources,method,digest FROM memory_catalog WHERE task_id=? AND version=?',
                              (task_id, CATALOG_VERSION)).fetchall()
        summaries = {}
        for row in rows:
            sources = json.loads(row['sources'])
            if signature(row['summary'], sources, row['method']) != row['digest']:
                continue  # A damaged derived index cannot block intact original recall.
            for pointer in sources:
                summaries[(pointer['event_ref'], pointer['start'], pointer['end'], pointer['sha256'])] = (row['summary'], row['method'])
        cards = []; seen = set()
        for hit in result['memory_candidates']:
            pointer = hit['_sources'][0]
            event = self.store.resolve(task_id, pointer['event_ref'])
            lo, hi = pointer['start'], pointer['end']
            if hashlib.sha256(event['text'][lo:hi].encode()).hexdigest() != pointer['sha256']:
                raise ValueError('Original hit disagrees with immutable source')
            # Bind an index to the containing source segment, never to another event.
            ranges = list(chunks(event['text']))
            containing = next(((a, b) for a, b, _ in ranges if a <= lo and hi <= b), None)
            if containing:
                start, end = containing
            else:
                start, end = next(((a, max(b, hi)) for a, b, _ in ranges if a <= lo < b), (lo, hi))
            parent = self.index_pointer(event, start, end)
            key = (event['ref'], start, end, parent['sha256'])
            if key in seen:
                continue
            seen.add(key)
            prepared = summaries.get(key)
            title, literals_valid = navigation_identity(event['text'][start:end], prepared[0] if prepared else '')
            summary_rejected = prepared is not None and not literals_valid
            # A pending summary never blocks an original-vector hit. Mark this preview explicitly.
            summary, method = (prepared if prepared and literals_valid else
                               (title[:40], 'rejected_summary_preview' if summary_rejected else 'pending_summary_preview'))
            identifier = hashlib.sha256(dump(key).encode()).hexdigest()
            cards.append({**hit, 'card_id': identifier, 'summary': summary, 'summary_method': method,
                'summary_pending': prepared is None, '_sources': [parent], 'event_refs': [event['ref']],
                'title': title, 'summary_rejected': summary_rejected,
                'matched_original_span': {'start': lo, 'end': hi}, 'retrieval_channels': ['vector'],
                'match_score': hit['semantic_score'], 'score_type': 'cosine_similarity'})
        return cards

    async def query(self, task_id, question, exhaustive=False, verify=False, exclude_event_id=None,
                    read_archive=False, min_score=None, search_archive=False):
        if read_archive or verify or search_archive:
            raise ValueError('Scoped mode returns navigation first; use read_memory to search selected originals')
        started = time.monotonic()
        result = await HybridMemory.query(self, task_id, question, exhaustive, False, exclude_event_id,
                                         False, min_score, False, _persist=False)
        if 'error' in result:
            return result
        result['memory_candidates'] = self._navigation_cards(task_id, result)
        result.update(pipeline=PIPELINE, retrieval_mode='original_vector_segment_navigation',
            evidence=[], evidence_refs=[], inspected_scope=[], context_supplements=[],
            automatic_context_prepared=False, context_completed_without_archive=False,
            summaries_are_only_entry=False, score_type='original_vector_relevance',
            answer_status='index_matches' if result['memory_candidates'] else 'no_matches',
            answer='仅返回记忆片段索引。摘要足够就回答；需要细节时用read_memory在选中片段内搜索。')
        Memory.record(self, result['request_id'], 'completed', result)
        self.store.append(task_id, 'retriever', 'retrieval_result', result)
        result['timings']['service_ms'] = round((time.monotonic() - started) * 1000, 3)
        return result

    @staticmethod
    def for_primary(result):
        brief = EvidenceMemory.for_primary(result)
        cards = HybridMemory.selected_cards(result)
        for visible, card in zip(brief['memory_candidates'], cards):
            visible.update(summary=card['summary'], summary_pending=card.get('summary_pending', False),
                           title=card.get('title', ''), summary_rejected=card.get('summary_rejected', False),
                           preview_only=card.get('summary_pending', False) or card.get('summary_rejected', False))
        brief['context_kind'] = 'selected_original_excerpts' if result.get('archive_read') else 'segment_navigation_only'
        return brief

    async def read(self, task_id, request_id, card_ids, question=''):
        with self.store.db() as db:
            row = db.execute("SELECT body FROM events WHERE task_id=? AND kind='retrieval_result' AND json_extract(body,'$.request_id')=? ORDER BY seq DESC LIMIT 1",
                             (task_id, request_id)).fetchone()
        if row is None:
            raise ValueError('Query scope is outside this task or unavailable')
        result = json.loads(row['body'])
        if result.get('pipeline') != PIPELINE:
            raise ValueError('This query did not authorize scoped vector reads')
        allowed = {c['card_id']: c for c in self.selected_cards(result)}
        if (not isinstance(card_ids, list) or not 1 <= len(card_ids) <= 3 or
                any(not isinstance(x, str) or x not in allowed for x in card_ids)):
            raise ValueError('Read requires 1–3 visible indices from this query')
        question = question or result['question']
        if not isinstance(question, str) or not question.strip() or len(question) > 1500:
            raise ValueError('Read requires a specific question of at most 1500 characters')
        started = time.monotonic(); units = []; authorized = []
        for identifier in dict.fromkeys(card_ids):
            for pointer in allowed[identifier]['_sources']:
                event = self.store.resolve(task_id, pointer['event_ref'])
                lo, hi = pointer['start'], pointer['end']
                if (event['seq'] > result['source_cutoff_seq'] or not 0 <= lo < hi <= len(event['text']) or
                        hashlib.sha256(event['text'][lo:hi].encode()).hexdigest() != pointer['sha256']):
                    raise ValueError('Selected source is outside the snapshot or its original span')
                authorized.append({**pointer, 'card_id': identifier})
                for a, b, partial in source_units(event['text'][lo:hi], 'window'):
                    units.append({'event': event, 'start': lo+a, 'end': lo+b, 'partial': partial,
                                  'card_id': identifier, 'scope_start': lo, 'scope_end': hi})
        with self.store.db(True) as db:
            db.execute('CREATE TABLE IF NOT EXISTS scoped_memory_vectors(cache_key TEXT PRIMARY KEY, vector BLOB NOT NULL)')
        vectors = [None] * len(units); missing = []; reused_original_vectors = 0
        with self.store.db() as db:
            for i, unit in enumerate(units):
                text = unit['event']['text'][unit['start']:unit['end']]
                unit['key'] = hashlib.sha256(dump([self.embedder.model_key, unit['event']['ref'], unit['start'], unit['end'], text]).encode()).hexdigest()
                row = db.execute('SELECT vector FROM scoped_memory_vectors WHERE cache_key=?', (unit['key'],)).fetchone()
                if row is None:
                    # Reuse an existing raw vector only for this exact authorized span.
                    row = db.execute('''SELECT v.vector FROM archive_units u JOIN archive_vectors v ON v.unit_id=u.id
                        WHERE u.task_id=? AND u.event_ref=? AND u.version=? AND u.start=? AND u.end=? AND v.model_key=?''',
                        (task_id, unit['event']['ref'], self.index.version, unit['start'], unit['end'], self.index.model_key)).fetchone()
                    if row:
                        reused_original_vectors += 1
                if row:
                    vectors[i] = np.frombuffer(row[0], dtype='<f4')
                else:
                    missing.append(i)
        try:
            for offset in range(0, len(missing), 2):
                indices = missing[offset:offset+2]
                encoded = await self.embedder.encode([units[i]['event']['text'][units[i]['start']:units[i]['end']] for i in indices])
                with self.store.db(True) as db:
                    for i, vector in zip(indices, encoded):
                        vectors[i] = vector
                        db.execute('INSERT OR REPLACE INTO scoped_memory_vectors VALUES(?,?)',
                                   (units[i]['key'], np.asarray(vector, dtype='<f4').tobytes()))
            query = (await self.embedder.encode([question], query=True))[0]
        except (ConnectionError, TimeoutError, httpx.HTTPError) as exc:
            raise ValueError('Scoped vector search unavailable; no broader search was performed') from exc
        matrix = np.stack(vectors)
        if matrix.shape != (len(units), len(query)) or not np.isfinite(matrix).all():
            raise ValueError('Invalid scoped vectors')
        scores = matrix @ query
        remaining = self.settings.memory_read_chars; selected = []
        for i in np.argsort(-scores, kind='stable'):
            unit = units[i]; event = unit['event']; lo, hi = unit['start'], unit['end']
            # Expand conditions inside the selected parent only, and keep whole ranges.
            spans = sentence_spans(event['text'][unit['scope_start']:unit['scope_end']])
            indices = [j for j, (a, b) in enumerate(spans) if a+unit['scope_start'] < hi and b+unit['scope_start'] > lo]
            if indices:
                a = spans[max(0, min(indices)-1)][0] + unit['scope_start']
                b = spans[min(len(spans)-1, max(indices)+1)][1] + unit['scope_start']
                if b-a <= remaining:
                    lo, hi = a, b
            if any(x['event_ref'] == event['ref'] and x['start'] < hi and lo < x['end'] for x in selected):
                continue
            if hi-lo > remaining:
                continue
            selected.append({'event_ref': event['ref'], 'start': lo, 'end': hi, 'quote': event['text'][lo:hi],
                'source_type': source_type(event), 'recorded_at': event['created'], 'seq': event['seq'],
                'excerpt_only': lo != 0 or hi != len(event['text']), 'partial_unit': unit['partial'],
                'card_id': unit['card_id'], 'semantic_score': round(float(scores[i]), 6), 'context_expanded': True})
            remaining -= hi-lo
            if len(selected) >= 2:
                break
        result.update(evidence=selected, evidence_refs=list(dict.fromkeys(x['event_ref'] for x in selected)),
            archive_read=True, answer_status='retrieved' if selected else 'no_matches',
            answer='已在选中记忆片段内搜索；仅返回必要原文，仍需判断是否包含答案。',
            inspected_scope=[{k: x[k] for k in ('event_ref', 'start', 'end')} for x in selected],
            scoped_search={'authorized_spans': authorized, 'searched_units': len(units), 'encoded_units': len(missing),
                           'cached_units': len(units)-len(missing), 'reused_original_vectors': reused_original_vectors,
                           'question': question,
                           'seconds': round(time.monotonic()-started, 3), 'global_search': False}, generation_calls=0)
        if not selected:
            result['warnings'].append('选中片段未返回预算内的完整原文；不能据此断言没有答案。')
        self.store.append(task_id, 'archive_reader', 'archive_read_scope',
                          {'request_id': request_id, 'card_ids': card_ids, **result['scoped_search'], 'inspected_scope': result['inspected_scope']})
        self.store.append(task_id, 'archive_reader', 'archive_read_result', result)
        return result
