import asyncio
from dataclasses import replace
import json

import httpx
import pytest

from lea.config import Settings
from lea.model import ModelClient, messages
from lea.runtime import Runtime
from lea.schemas import Decision, Handoff
from lea.store import Store


def book():
    return dict(summary='设备压力已更正，复测待完成', completed=[], decisions=[],
                next_steps=['复测'], open_questions=[], topics=['设备'], entities=['KX-6186'],
                history_entries=[])


class SummaryModel:
    async def tokens(self, prompt):
        return 100

    async def call(self, task_id, role, prompt, schema, **kwargs):
        return Handoff(**book())


def test_layout_keeps_data_and_fixed_prefix_when_window_changes():
    first = dict(current_request='查询压力', tools={'calculator': '计算'}, goal='检验设备',
                 constraints=['温度必须低于33℃'], constraint_version=1,
                 handoff_state={'previous_window': 'w_old', 'summary': '压力31kPa'},
                 last_result={'evidence': '旧值24kPa'}, latest_user_update={'text': '更正为31kPa'},
                 task_version=2, records=[{'ref': 'w_old:e_1', 'text': '事实'}])
    second = {**first, 'task_version': 3,
              'handoff_state': {'previous_window': 'w_new', 'summary': '等待复测'}}
    a = messages('固定规则', first, layout='stable_v1')
    b = messages('固定规则', second, layout='stable_v1')
    assert json.loads(a[1]['content']) == first
    assert json.loads(b[1]['content']) == second
    assert a[1]['content'].split('"handoff_state":')[0] == b[1]['content'].split('"handoff_state":')[0]
    assert a[1]['content'].index('"tools":') < a[1]['content'].index('"handoff_state":')
    assert a[1]['content'].index('"last_result":') < a[1]['content'].index('"latest_user_update":') < a[1]['content'].index('"current_request":')
    # Latest effective constraints remain distinct from historical handbook content.
    corrected = messages('固定规则', {**second, 'constraints': ['温度必须低于30℃'],
                                    'constraint_version': 2}, layout='stable_v1')
    assert json.loads(corrected[1]['content'])['constraints'] == ['温度必须低于30℃']
    assert corrected[1]['content'] != b[1]['content']


def test_message_arriving_between_snapshot_and_marker_survives_switch(tmp_path):
    runtime = Runtime(Settings(data_dir=tmp_path, memory_backend='llm',
                               window_prompt_layout='stable_v1'), model=SummaryModel())
    task = runtime.store.create_task('继续检验', ['未经复测禁止启动'], [], [])
    fact = runtime.store.append(task['id'], 'user', 'message', {'text': 'KX-6186压力更正为31kPa。'})
    append = runtime.store.append
    late = []

    def append_with_late_message(task_id, role, kind, body):
        if kind == 'handoff_requested':
            late.append(append(task_id, 'user', 'message', {'text': '追加：夹爪尚未更换。'}))
        return append(task_id, role, kind, body)

    runtime.store.append = append_with_late_message
    asyncio.run(runtime.handoff(task['id'], 'race_test'))
    current = runtime.store.task(task['id'])
    assert current['window_id'] != task['window_id']
    assert current['state']['recorded_through'] == fact['seq']
    assert late[0]['id'] in [event['id'] for event in runtime.store.events(
        task['id'], after=current['state']['recorded_through'])]
    reloaded = Store(tmp_path)
    assert reloaded.resolve(task['id'], fact['ref'])['text'] == 'KX-6186压力更正为31kPa。'
    assert reloaded.task(task['id'])['constraints'] == ['未经复测禁止启动']
    assert any(e['kind'] == 'handoff_completed' for e in reloaded.events(task['id']))


def test_handoff_preserves_verified_short_sources_independently_of_summary(tmp_path):
    runtime = Runtime(Settings(data_dir=tmp_path, memory_backend='llm'), model=SummaryModel())
    task = runtime.store.create_task('继续检验', [], [], [])
    fact = runtime.store.append(task['id'], 'user', 'message', {
        'text': '工艺代码为JN-2706，复测未完成；下一步复核报告。'})
    runtime.store.append(task['id'], 'user', 'message', {'text': '长文材料' * 200})
    asyncio.run(runtime.handoff(task['id'], 'first'))
    anchors = runtime.store.task(task['id'])['state']['source_anchors']
    assert anchors == [{'ref': fact['ref'], 'seq': fact['seq'], 'role': 'user',
                        'text': fact['body']['text']}]
    assert len(anchors) <= 8 and sum(len(a['text']) for a in anchors) <= 800
    # Previously saved text is never trusted over its immutable source.
    forged = {'source_anchors': [{**anchors[0], 'text': '代码未知，等待复测'}]}
    restored = runtime.handoff_anchors(task['id'], forged, [], fact['seq'])
    assert restored[0]['text'] == fact['body']['text']
    asyncio.run(runtime.handoff(task['id'], 'second'))
    assert runtime.store.task(task['id'])['state']['source_anchors'] == anchors
    assert runtime.store.task(task['id'])['state']['latest_user_update'] == anchors[0]
    correction = runtime.store.append(task['id'], 'user', 'message', {'text': '更正：当前压力35kPa。'})
    asyncio.run(runtime.handoff(task['id'], 'third'))
    assert runtime.store.task(task['id'])['state']['latest_user_update']['ref'] == correction['ref']


def test_wire_requests_keep_lora_and_cache_slots_separate(tmp_path):
    async def run():
        settings = Settings(data_dir=tmp_path, primary_adapter_id=0,
                            primary_adapter_version='concise-v2',
                            primary_cache_slot=0, auxiliary_cache_slot=1)
        store = Store(tmp_path)
        task = store.create_task('验证缓存', [], [], [])
        client = ModelClient(settings, store)
        await client.http.aclose()
        requests = []

        def handler(request):
            if request.url.path == '/apply-template':
                return httpx.Response(200, json={'prompt': '测试'})
            if request.url.path == '/tokenize':
                return httpx.Response(200, json={'tokens': [1, 2]})
            if request.url.path == '/health':
                return httpx.Response(200, json={'status': 'ok'})
            if request.url.path == '/props':
                # A one-slot server must fail readiness with a two-slot client.
                return httpx.Response(200, json={'total_slots': 1,
                    'default_generation_settings': {'n_ctx': 8192}})
            if request.url.path == '/lora-adapters':
                return httpx.Response(200, json=[{'id': 0, 'scale': 0}])
            body = json.loads(request.content)
            requests.append(body)
            result = book() if body['response_format']['json_schema']['name'] == 'Handoff' else {
                'action': 'respond', 'response': '已收到', 'tool': '', 'arguments': {}}
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                'message': {'content': json.dumps(result, ensure_ascii=False)}}],
                'usage': {}, 'timings': {}})

        client.http = httpx.AsyncClient(base_url='http://test', transport=httpx.MockTransport(handler))
        prompt = messages('规则', {'current_request': '继续', 'memory_queries_remaining': 0,
                                  'memory_reads_remaining': 0})
        await client.call(task['id'], 'primary', prompt, Decision)
        await client.call(task['id'], 'handoff', prompt, Handoff)
        await client.call(task['id'], 'primary', prompt, Decision)
        assert [r['id_slot'] for r in requests] == [0, 1, 0]
        assert [r['lora'][0]['scale'] for r in requests] == [1, 0, 1]
        assert all(r['cache_prompt'] is True for r in requests)
        empty_schema = requests[1]['response_format']['json_schema']['schema']
        assert empty_schema['properties']['history_entries']['maxItems'] == 0
        source = store.append(task['id'], 'user', 'message', {'text': '压力更正为31kPa'})
        anchor = store.append(task['id'], 'user', 'message', {'text': '下一步复核报告'})
        handoff_prompt = messages('规则', {'records': [{'ref': 'R1', 'text': '更正'}],
            'previous_state': {'history_entries': [{'statement': '先前的更正',
                                                   'evidence_refs': [source['ref']]}],
                               'source_anchors': [{'ref': anchor['ref'], 'text': anchor['body']['text']}]}})
        await client.call(task['id'], 'handoff', handoff_prompt, Handoff)
        wire_schema = requests[-1]['response_format']['json_schema']['schema']
        references = wire_schema['$defs']['HandoffFinding']['properties']['evidence_refs']
        assert references['items']['enum'] == ['R1', source['ref'], anchor['ref']]
        assert references['minItems'] == 1
        # A fabricated source inherited from a handbook is rejected before inference.
        invalid = messages('规则', {'records': [], 'previous_state': {
            'history_entries': [{'statement': '伪造', 'evidence_refs': ['w_other:e_other']}]}})
        before = len(requests)
        with pytest.raises(ValueError):
            await client.call(task['id'], 'handoff', invalid, Handoff)
        assert len(requests) == before
        tampered = messages('规则', {'previous_state': {'source_anchors': [
            {'ref': anchor['ref'], 'text': '伪造为等待复测'}]}})
        with pytest.raises(ValueError, match='differs from its archive'):
            await client.call(task['id'], 'handoff', tampered, Handoff)
        assert len(requests) == before
        assert not (await client.health())['ready']
        await client.close()

    asyncio.run(run())
