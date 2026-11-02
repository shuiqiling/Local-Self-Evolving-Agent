"""Trial-only index readiness and explicit on-demand archive read policy."""
import asyncio
import time

from lea.evidence import EvidenceMemory
from lea.archive import ArchiveCatalog
import lea.runtime as runtime
import memory_agent as base

READ_POLICY = """
内置读档操作可立即执行，不需要用户另外授权。判断细节是否足够，再选择动作：
1. 原文evidence已包含所需答案且用户没有要求重新读档，直接回答。
2. 用户明确要求先读档核对，或当前只有相关memory_candidates导航而没有回答所需的精确细节，
   且memory_reads_remaining>0，先输出action=read_memory，不能用“需查看原文”结束本轮。
3. read_memory的arguments必须从last_result复制真实request_id和候选card_id，question写所需细节。
   格式示例：{"action":"read_memory","response":"","tool":"","arguments":{"request_id":"本轮真实查询ID","card_ids":["M1"],"question":"核对最新更正的精确值"}}。
4. 收到读档结果后依据原文简答；读档失败、无相关入口或已无读取额度时才说明无法查证。
"""


class QuerySafeArchive(ArchiveCatalog):
    def _validated(self, task_id, cutoff, exclude, rows):
        # Match the existing summary catalog's query-origin exclusion. A user's
        # request to verify a fact is not the archived statement establishing it.
        with self.store.db() as db:
            origins = {r[0] for r in db.execute(
                'SELECT event_id FROM memory_query_origins WHERE task_id=?', (task_id,))}
        return super()._validated(task_id, cutoff, exclude,
                                  [row for row in rows if row['event_id'] not in origins])


class ReadyMemory(EvidenceMemory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = QuerySafeArchive(self.store, self.embedder,
                                      'sentence' if self.strategy == 'sentence_threshold' else 'window')

    async def query(self, task_id, question, exhaustive=False, verify=False,
                    exclude_event_id=None, read_archive=False, min_score=None,
                    search_archive=False):
        started = time.monotonic()
        self.index.mark_query_origin(task_id, exclude_event_id)
        cutoff = await asyncio.to_thread(self.index.sync, task_id)
        rows, _ = await asyncio.to_thread(
            self.index.vector_candidates, task_id, cutoff,
            self.settings.retrieval_max_chunks, exclude_event_id)
        pending = [row for row in rows if row['vector'] is None]
        if len(pending) > 8:
            raise RuntimeError('Memory indexing is pending; retry after background indexing completes')
        if pending:
            # Fail explicitly on timeout rather than reporting no relevant memory.
            await asyncio.wait_for(self.index.encode_pending(pending), timeout=30)
        self.store.append(task_id, 'retriever', 'index_readiness', {
            'encoded_units': len(pending), 'cutoff': cutoff,
            'seconds': round(time.monotonic() - started, 3),
            'scope': 'task-local candidate set', 'ready': True})
        return await super().query(task_id, question, exhaustive, verify,
                                  exclude_event_id, read_archive, min_score, search_archive)


def create_app():
    runtime.EvidenceMemory = ReadyMemory
    if READ_POLICY not in runtime.PRIMARY:
        runtime.PRIMARY += READ_POLICY
    app = base.create_app()

    @app.get('/api/index-readiness-profile')
    def profile():
        return {'enabled': True, 'max_on_demand_units': 8,
                'timeout_seconds': 30, 'failure': 'explicit job error; no false empty result',
                'query_origins_excluded': True,
                'read_policy': 'answer if evidence suffices; otherwise use relevant archive navigation'}
    return app
