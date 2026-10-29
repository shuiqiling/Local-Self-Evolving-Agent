import asyncio
from dataclasses import replace
import json

import pytest

from lea.config import Settings
from lea.runtime import Runtime
from lea.scoped_memory import PIPELINE, ScopedMemory, navigation_identity
from lea.schemas import Decision
from lea.store import Store
from test_hybrid import FakeEncoder
from test_runtime import FakeModel


def setup(tmp_path):
    store = Store(tmp_path)
    task = store.create_task('分层记忆', [], [], [])
    memory = ScopedMemory(store, FakeModel(), Settings(data_dir=tmp_path), FakeEncoder())
    return store, task, memory


async def warm(memory, task_id):
    memory.index.sync(task_id)
    while rows := memory.index.pending():
        await memory.index.encode_pending(rows)


def test_raw_vector_hit_survives_summary_omission_and_returns_only_navigation(tmp_path):
    async def run():
        store, task, memory = setup(tmp_path)
        text = '设备操作说明。温度低于20摄氏度才可使用。未完成检查时禁止操作。' + '设备有独立说明。'*25
        event = store.append(task['id'], 'user', 'message', {'text': text})
        event = store.resolve(task['id'], event['ref'])
        with store.db(True) as db:
            memory.navigation._add(db, task['id'], event['seq'], '设备操作说明索引',
                                   [memory.index_pointer(event, 0, len(text))], 'test_precomputed')
        await warm(memory, task['id'])
        result = await memory.query(task['id'], '温度限制是什么？')
        brief = memory.for_primary(result)
        assert result['pipeline'] == PIPELINE and result['memory_candidates']
        assert result['memory_candidates'][0]['summary'] == '设备操作说明索引'
        assert brief['evidence'] == [] and not brief['archive_read']
        assert '20摄氏度' not in json.dumps(brief, ensure_ascii=False)
        assert any('20摄氏度' in text for texts, query in memory.embedder.inputs if not query for text in texts)
        assert not any('设备操作说明索引' == text for texts, query in memory.embedder.inputs if not query for text in texts)
        assert not memory.model.roles
    asyncio.run(run())


def test_tool_search_is_restricted_to_selected_segment_and_reuses_vectors(tmp_path):
    async def run():
        store, task, memory = setup(tmp_path)
        text = '温度低于20摄氏度才可使用。未完成检查时禁止操作。' + '背景说明。'*170 + '另一段秘密口令是EXTERNAL-SECRET。'
        event = store.append(task['id'], 'user', 'message', {'text': text})
        other = store.append(task['id'], 'user', 'message', {'text': '其他设备温度低于99摄氏度，另一个口令OUTSIDE。'})
        await warm(memory, task['id'])
        result = await memory.query(task['id'], '温度限制是什么？')
        card = next(c for c in memory.selected_cards(result) if c['event_refs'] == [event['ref']])
        # Exercise cold on-demand encoding even when a first-stage index once existed.
        with store.db(True) as db:
            db.execute('DELETE FROM archive_vectors WHERE unit_id IN (SELECT id FROM archive_units WHERE event_ref=?)', (event['ref'],))
        memory.embedder.inputs.clear()
        read = await memory.read(task['id'], result['request_id'], [card['card_id']], '温度限制及操作前提')
        scope = read['scoped_search']['authorized_spans'][0]
        assert scope['end'] < len(text) and read['scoped_search']['global_search'] is False
        assert read['evidence'] and '20摄氏度' in read['evidence'][0]['quote'] and '禁止' in read['evidence'][0]['quote']
        assert sum(len(e['quote']) for e in read['evidence']) <= 400
        for evidence in read['evidence']:
            assert evidence['event_ref'] == event['ref'] != other['ref']
            assert scope['start'] <= evidence['start'] < evidence['end'] <= scope['end']
            assert evidence['quote'] == text[evidence['start']:evidence['end']]
        encoded = [t for texts, query in memory.embedder.inputs if not query for t in texts]
        assert encoded and not any('EXTERNAL-SECRET' in t or 'OUTSIDE' in t for t in encoded)
        memory.embedder.inputs.clear()
        again = await memory.read(task['id'], result['request_id'], [card['card_id']], '操作前提')
        assert again['scoped_search']['encoded_units'] == 0
        assert not any(not query for _, query in memory.embedder.inputs)
        assert not memory.model.roles
    asyncio.run(run())


def test_no_cross_task_unshown_indices_or_mutable_query_scope(tmp_path):
    async def run():
        store, task, memory = setup(tmp_path)
        event = store.append(task['id'], 'user', 'message', {'text': '通行前必须开门。未确认时禁止通行。'})
        await warm(memory, task['id'])
        result = await memory.query(task['id'], '通行条件')
        card = result['memory_candidates'][0]['card_id']
        other = store.create_task('其他任务', [], [], [])
        with pytest.raises(ValueError):
            await memory.read(other['id'], result['request_id'], [card])
        with pytest.raises(ValueError):
            await memory.read(task['id'], result['request_id'], ['forged'])
        with store.db(True) as db:
            db.execute("UPDATE queries SET result='{}'")
        later = store.append(task['id'], 'user', 'message', {'text': '通行无需开门，这是后续记录。'})
        read = await memory.read(task['id'], result['request_id'], [card])
        assert read['evidence'][0]['event_ref'] == event['ref'] != later['ref']
    asyncio.run(run())


def test_first_generation_has_no_historical_originals_and_tool_is_optional(tmp_path):
    class Agent(FakeModel):
        async def call(self, task_id, role, prompt, schema, **kwargs):
            self.roles.append(role)
            payload = json.loads(prompt[1]['content']); last = payload['last_result']
            if not last['archive_read']:
                assert '20摄氏度' not in json.dumps(payload, ensure_ascii=False)
                assert last['context_kind'] == 'segment_navigation_only'
                return Decision(action='read_memory', response='', tool='', arguments={
                    'request_id': last['request_id'], 'card_ids': ['M1'], 'question': '温度限制与使用条件'})
            assert any('20摄氏度' in e['quote'] for e in last['evidence'])
            return Decision(action='finish', response='温度低于20摄氏度且完成检查后才可使用。', tool='', arguments={})

    async def run():
        runtime = Runtime(replace(Settings(data_dir=tmp_path), memory_backend='scoped'), model=Agent(), embedder=FakeEncoder())
        task = runtime.store.create_task('分层记忆测试', [], [], [])
        text = '设备操作条件。温度低于20摄氏度才可使用。未完成检查时禁止操作。' + '设备背景说明。'*20
        event = runtime.store.append(task['id'], 'user', 'message', {'text': text})
        event = runtime.store.resolve(task['id'], event['ref'])
        with runtime.store.db(True) as db:
            runtime.memory.navigation._add(db, task['id'], event['seq'], '设备操作条件索引',
                [runtime.memory.index_pointer(event, 0, len(text))], 'test_precomputed')
        await warm(runtime.memory, task['id'])
        job = runtime.store.add_job(task['id'], 'message', {'text': '温度限制是什么？'})
        result = await runtime.primary(job)
        assert result['archive_reads'] == 1 and result['steps'] == 2
        assert runtime.model.roles == ['primary', 'primary']
        assert '20摄氏度' in result['response']
        await runtime.memory.close()
    asyncio.run(run())


def test_index_preserves_verbatim_subject_and_rejects_invented_identifier(tmp_path):
    async def run():
        store, task, memory = setup(tmp_path)
        text = '晨岚设备 CL-427 的温度档案。工单是RP-618。' + '设备背景说明。'*20
        event = store.append(task['id'], 'user', 'message', {'text': text})
        event = store.resolve(task['id'], event['ref'])
        with store.db(True) as db:
            memory.navigation._add(db, task['id'], event['seq'], '设备工单RP-617，温度档案',
                [memory.index_pointer(event, 0, len(text))], 'test_bad_summary')
        await warm(memory, task['id'])
        result = await memory.query(task['id'], '温度档案')
        card = memory.for_primary(result)['memory_candidates'][0]
        assert card['title'] == '晨岚设备 CL-427 的温度档案。'
        assert card['summary_rejected'] and card['preview_only']
        assert 'RP-617' not in card['summary'] and not result['evidence']
        assert navigation_identity('温度系数1.037，偏移量负0.8。', '系数1.037，偏移0.8')[1]
    asyncio.run(run())


def test_summary_sufficient_does_not_force_tool_and_pending_summary_does_not_block_hit(tmp_path):
    async def run():
        store, task, memory = setup(tmp_path)
        store.append(task['id'], 'user', 'message', {'text': '补给站在蓝港。'})
        await warm(memory, task['id'])
        result = await memory.query(task['id'], '补给在哪里？')
        assert result['memory_candidates'] and not result['memory_candidates'][0]['summary_pending']
        assert memory.for_primary(result)['memory_candidates'][0]['summary'] == '补给站在蓝港。'
        assert not result['archive_read']
        event = store.append(task['id'], 'user', 'message', {'text': '温度限制与设备说明。'*30})
        await warm(memory, task['id'])
        result = await memory.query(task['id'], '温度限制')
        card = next(c for c in result['memory_candidates'] if event['ref'] in c['event_refs'])
        assert card['summary_pending'] and len(card['summary']) <= 40
    asyncio.run(run())
