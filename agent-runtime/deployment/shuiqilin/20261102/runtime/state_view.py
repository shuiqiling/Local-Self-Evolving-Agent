"""Compact views of current observations; never summaries of shared history."""
import copy
import json

VERSION = 'scoped-host-state-v2'
INSTRUCTION = ('仅评估snapshot中的Ubuntu宿主状态；goal、records、handoff_state是共享背景，不是本轮待办。'
               '不要继续历史工具开发。区分瞬时CPU占用与平均负载、宿主磁盘与记忆卷、CPU与非CPU传感器。'
               '历史模型判断不是测量；没有验证依据不能诊断传感器故障或硬件损坏。')


def state_prompt(prompt):
    """Keep history intact in a background message; latest sample is last."""
    out=copy.deepcopy(prompt)
    payload=json.loads(out[-1]['content'])
    current={key:payload[key] for key in ('channel','context_scope','task_id','window_id') if key in payload}
    current.update(snapshot=payload.pop('snapshot'),instruction=payload.pop('instruction'))
    current['request']='只评估本消息snapshot。CPU温度只读snapshot.cpu.temperature_c，非CPU温度不得混用；旧记录不能当当前测量。'
    background={'record_type':'shared_chat_history_not_current_request','shared_context':payload}
    out[-1]['content']=json.dumps(background,ensure_ascii=False,separators=(',',':'))
    out.append({'role':'user','content':json.dumps(current,ensure_ascii=False,separators=(',',':'))})
    # Keep one current instruction, not duplicate request/instruction copies.
    current.pop('request',None)
    out[-1]['content']=json.dumps(current,ensure_ascii=False,separators=(',',':'))
    return out


from active_context import snapshot_view


def select(source, names):
    return {key:copy.deepcopy(source[key]) for key in names if key in source}


def legacy_snapshot_view(snapshot):
    out = select(snapshot, ('collected_at_utc','local_time','timezone','sample_seconds','uptime_seconds'))
    cpu = snapshot.get('cpu') or {}
    out['cpu'] = select(cpu, ('usage_percent','logical_cpus','load_average','temperature_c'))
    out['cpu']['temperature_scope'] = 'maximum_of_cpu_labelled_sensors_only'
    groups = {}
    for sensor in cpu.get('temperature_sensors', []):
        if sensor.get('cpu_sensor') is True:
            continue
        # Preserve unknown identity as unknown; never reinterpret old hwmonN
        # records using today's device numbering.
        device = sensor.get('device') or sensor.get('label') or 'unknown'
        category = 'non_cpu_sensor' if sensor.get('cpu_sensor') is False else 'unclassified_sensor'
        key = (device, category)
        group = groups.setdefault(key, {'device':device,'category':category,'values_c':[]})
        if sensor.get('celsius') is not None:
            group['values_c'].append(sensor['celsius'])
    out['other_temperature_sensors'] = []
    for group in groups.values():
        values = group.pop('values_c')
        if values:
            out['other_temperature_sensors'].append({**group,'minimum_c':min(values),'maximum_c':max(values),'sensor_count':len(values)})
    out['memory'] = select(snapshot.get('memory') or {}, ('total_bytes','available_bytes','used_percent','swap_total_bytes','swap_used_bytes'))
    out['network'] = select(snapshot.get('network') or {}, ('measurement','aggregate_interfaces','rx_mib_per_second','tx_mib_per_second'))
    out['disk'] = {'scope':'host_filesystem', **select(snapshot.get('disk') or {}, ('path','total_bytes','free_bytes','used_percent'))}
    out['memory_storage'] = {'scope':'agent_memory_volume', **select(snapshot.get('memory_storage') or {}, ('limit_gib','used_percent','available_bytes','status','write_allowed','optional_write_allowed'))}
    out['agent'] = select(snapshot.get('agent') or {}, ('automatic_lora_phase','new_windows','windows_per_update','foreground_jobs','idle_seconds'))
    out['view'] = {'version':VERSION,'raw_snapshot_archived':True,'omitted_details':'per-core CPU and per-interface counters; no shared chat records removed'}
    return out
