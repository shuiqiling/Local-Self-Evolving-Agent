import asyncio
from dataclasses import replace
from lea.config import Settings
from lea.evidence import EvidenceMemory
from lea.archive import source_units
from lea.store import Store
from test_hybrid import FakeEncoder
from test_runtime import FakeModel


def setup(tmp_path):
    store=Store(tmp_path);task=store.create_task('原文独立检索',[],[],[])
    memory=EvidenceMemory(store,FakeModel(),Settings(data_dir=tmp_path),FakeEncoder(),policy={'semantic_min':.5,'keyword_min':.65,'calibrated':False})
    return store,task,memory


def warm(memory,task_id):
    memory.index.sync(task_id)
    async def run():
        while rows:=memory.index.pending():await memory.index.encode_pending(rows)
    asyncio.run(run())


def test_original_evidence_survives_absent_summary_and_carries_conditions(tmp_path):
    store,task,memory=setup(tmp_path)
    event=store.append(task['id'],'user','message',{'text':'设备 AB-731 用于吊装。必须先锁紧两个固定销。未锁紧时禁止升起。'})
    warm(memory,task['id'])
    with store.db() as db:assert db.execute('SELECT COUNT(*) FROM memory_catalog').fetchone()[0]==0
    result=asyncio.run(memory.query(task['id'],'AB-731 可以直接升起吗？'))
    assert 'error' not in result and result['evidence'] and not result['archive_read']
    assert '必须先锁紧两个固定销' in result['evidence'][0]['quote'] and '未锁紧时禁止升起' in result['evidence'][0]['quote']
    assert result['evidence'][0]['event_ref']==event['ref'] and not result['summaries_are_only_entry']
    assert not any(e['kind']=='archive_read_scope' for e in store.events(task['id']))


def test_previous_user_fact_is_not_permanently_excluded_by_auto_retrieval(tmp_path):
    store,task,memory=setup(tmp_path)
    fact=store.append(task['id'],'user','message',{'text':'项目代号是测试鸟。'})
    asyncio.run(memory.query(task['id'],'请记住项目代号',exclude_event_id=fact['id']))
    warm(memory,task['id'])
    result=asyncio.run(memory.query(task['id'],'项目代号是什么？'))
    assert fact['ref'] in result['evidence_refs']


def test_explicit_more_context_read_stays_query_scoped(tmp_path):
    store,task,memory=setup(tmp_path)
    fact=store.append(task['id'],'user','message',{'text':'工单 CC-917 的附件是 offsets.csv。仅允许在离线模式修改。'})
    warm(memory,task['id']);result=asyncio.run(memory.query(task['id'],'CC-917 的附件是什么？'))
    read=memory.read(task['id'],result['request_id'],[result['memory_candidates'][0]['card_id']],'修改条件')
    assert read['archive_read'] and any('离线' in e['quote'] for e in read['evidence'])
    assert all(e['event_ref']==fact['ref'] for e in read['evidence'])


def test_windows_are_complete_sentence_ranges():
    text='对象是样本甲。仅在温度低于20摄氏度时有效。未检查时禁止操作。'
    units=list(source_units(text,'window'))
    assert text[units[0][0]:units[0][1]]=='对象是样本甲。仅在温度低于20摄氏度时有效。'
    assert text[units[1][0]:units[1][1]]=='仅在温度低于20摄氏度时有效。未检查时禁止操作。'
