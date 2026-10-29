import asyncio
import json

import pytest

from lea.config import Settings
from lea.native_kv import reconcile_windows
from lea.runtime import Runtime
from lea.schemas import Decision
from lea.store import Conflict,Store


def metadata(cutoff,total=1):
    return dict(cache_epoch='epoch-test',total_cuts=total,through_seq=cutoff,
                physical_tokens=614,next_absolute_position=3072,
                cuts=[dict(cut=i,seconds=.28,before=3072,after=614,
                    old_tokens_recomputed=0) for i in range(1,total+1)])


def test_native_cut_preserves_late_user_message_and_archive(tmp_path):
    store=Store(tmp_path);task=store.create_task('检验',[],[],[])
    first=store.append(task['id'],'user','message',{'text':'工艺代码JN-2706。'})
    late=store.append(task['id'],'user','message',{'text':'更正：当前压力35kPa。'})
    new=reconcile_windows(store,task['id'],task['window_id'],1,metadata(first['seq']))
    assert new['window_id']!=task['window_id']
    assert new['state']['recorded_through']==first['seq']
    assert new['state']['native_kv']['summary_generated'] is False
    assert store.resolve(task['id'],first['ref'])['body']['text']=='工艺代码JN-2706。'
    assert late['id'] in [e['id'] for e in store.events(task['id'],after=first['seq'])]
    archived=json.loads((tmp_path/'tasks'/task['id']/'windows'/task['window_id']/'handoff.json').read_text())
    assert archived['mode']=='native_kvzip' and archived['history_index']['entries']==[]


def test_native_cut_rejects_stale_constraints_without_window_change(tmp_path):
    store=Store(tmp_path);task=store.create_task('检验',[],[],[])
    first=store.append(task['id'],'user','message',{'text':'压力31kPa。'})
    store.update_constraints(task['id'],['未经复测禁止启动'],1)
    with pytest.raises(Conflict):
        reconcile_windows(store,task['id'],task['window_id'],1,metadata(first['seq']))
    assert len(store.windows(task['id']))==1
    assert store.task(task['id'])['constraints']==['未经复测禁止启动']


def test_lost_ack_reconciliation_rotates_each_native_cut_only_once(tmp_path):
    store=Store(tmp_path);task=store.create_task('检验',[],[],[])
    event=store.append(task['id'],'user','message',{'text':'任务记录'})
    meta=metadata(event['seq'],2)
    current=reconcile_windows(store,task['id'],task['window_id'],1,meta)
    repeated=reconcile_windows(store,task['id'],current['window_id'],1,meta)
    assert repeated['window_id']==current['window_id']
    assert len(store.windows(task['id']))==3
    assert len([e for e in store.events(task['id']) if e['kind']=='handoff_completed'])==2


def test_native_rollback_preserves_new_messages_and_newer_constraints(tmp_path):
    from deployment.kv_service.rollback_state import restore_legacy_cursors
    store=Store(tmp_path);task=store.create_task('检验',[],[],[])
    first=store.append(task['id'],'user','message',{'text':'旧代码JN-2706。'})
    reconcile_windows(store,task['id'],task['window_id'],1,metadata(first['seq']))
    newer=store.append(task['id'],'user','message',{'text':'新压力47kPa。'})
    store.update_constraints(task['id'],['禁止启动'],1)
    assert restore_legacy_cursors(store)==[task['id']]
    current=store.task(task['id'])
    assert current['constraints']==['禁止启动'] and current['constraint_version']==2
    assert current['state']['recorded_through']==0
    assert store.resolve(task['id'],newer['ref'])['body']['text']=='新压力47kPa。'
    assert first['id'] in [e['id'] for e in store.events(task['id'],after=current['state']['recorded_through'])]


def test_runtime_accepts_own_native_cut_and_does_not_generate_a_summary(tmp_path):
    class Native:
        result=None
        def take_result(self,task_id):return self.result
    class Model:
        native_kv=Native()
        async def tokens(self,*args):raise AssertionError('Legacy full-prompt budget path used')
        async def call(self,task_id,role,prompt,schema,**kwargs):
            assert role=='primary'
            payload=json.loads(prompt[-1]['content'])
            self.native_kv.result=metadata(payload['_native_through_seq'])
            return Decision(action='respond',response='已记录。',tool='',arguments={})
    runtime=Runtime(Settings(data_dir=tmp_path,memory_backend='llm'),model=Model())
    task=runtime.store.create_task('检验',[],[],[])
    job=runtime.store.add_job(task['id'],'message',{'text':'记录工艺代码JN-2706。'},request_key='native-test')
    result=asyncio.run(runtime.primary(job))
    assert result['response']=='已记录。'
    assert len(runtime.store.windows(task['id']))==2
    assert runtime.store.task(task['id'])['state']['native_kv']['mode']=='fastkvzip'
    assert not any(e['kind']=='handoff_requested' for e in runtime.store.events(task['id']))
