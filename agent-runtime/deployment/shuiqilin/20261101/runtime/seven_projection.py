"""Deterministic source projection; sampled telemetry never inherits a chat goal."""
from collections import OrderedDict
import math
import json
from auto_memory_data import canonical, sha

VERSION = 'source-projection-v2'
STATE_GOAL = {'目标': '归档这批历史机器采样的观测结果。',
 '必要条件': '这是程序提供的整理任务，不是历史用户原话。只记录采样范围和已测值；缺测保持未知。流量不代表可用带宽，历史数值不代表当前状态。'}
METRICS = {
 'cpu': ('usage_percent', 'temperature_c'),
 'memory': ('used_percent', 'available_bytes', 'swap_used_bytes'),
 'network': ('rx_mib_per_second', 'tx_mib_per_second'),
 'disk': ('used_percent', 'free_bytes'),
}

def decoded(value):
    try: return json.loads(value) if isinstance(value, str) else value
    except ValueError: return value

def state_snapshots(source):
    """Prefer the actual persisted observation over its duplicate chat envelope."""
    by_id = OrderedDict()
    for event in source['events']:
        if event['kind'] not in ('internal_state_input', 'internal_state_observation'): continue
        body = decoded(event['body'])
        if not isinstance(body, dict): continue
        snapshot = decoded(body.get('snapshot'))
        if not isinstance(snapshot, dict): continue
        identifier = body.get('observation_id') or body.get('id') or event['id']
        item = (event, snapshot)
        if identifier not in by_id or event['kind'] == 'internal_state_observation': by_id[identifier] = item
    return sorted(by_id.values(), key=lambda x: (x[1].get('collected_at_utc') or x[0]['created'], x[0]['id']))

def aggregate(source):
    items = state_snapshots(source)
    if not items: return None, []
    value = {'sample_count': len(items), 'first_at': items[0][1].get('collected_at_utc') or items[0][0]['created'],
       'last_at': items[-1][1].get('collected_at_utc') or items[-1][0]['created'],
       'measurement': 'observed_interface_traffic_not_internet_capacity', 'metrics': {}}
    for group, names in METRICS.items():
        value['metrics'][group] = {}
        for name in names:
            values = [s.get(group, {}).get(name) for _, s in items]
            numeric = [x for x in values if type(x) in (int, float) and math.isfinite(x)]
            value['metrics'][group][name] = {'measured': len(numeric), 'missing': len(values)-len(numeric),
              'min': min(numeric) if numeric else None, 'max': max(numeric) if numeric else None,
              'last': values[-1] if type(values[-1]) in (int, float) and math.isfinite(values[-1]) else None}
    anchors = [{'event_id': e['id'], 'body_sha256': sha(e['body'])} for e, _ in items]
    return value, anchors

def project(source, ordinary_project):
    assert sha(canonical(source['events'])) == source['events_sha256']
    # State-model commentary is retained in the archive, never mistaken for measurements.
    ignored = {'internal_state_observation', 'internal_state_input', 'internal_state_result'}
    subset = {**source, 'events': [e for e in source['events'] if e['kind'] not in ignored]}
    subset['events_sha256'] = sha(canonical(subset['events']))
    episodes, coverage = ordinary_project(subset)
    for e in source['events']:
        if e['kind'] in ignored: coverage[e['id']] = {'status': 'state_archive_preserved'}
    value, anchors = aggregate(source)
    if value is not None:
        payload = {'原始记录': [
          {'编号': '用户要求', '类型': '程序提供的归档任务（非历史用户原话）', '内容': STATE_GOAL},
          {'编号': '记录1', '类型': '观测（程序从原始采样计算的批次统计）', '内容': value}]}
        episodes.append({'key': ['internal-state-batch', source['start_exclusive_seq'], source['end_inclusive_seq']],
          'input': payload, 'source_anchors': {'用户要求': {'runtime_policy': VERSION, 'sha256': sha(canonical(STATE_GOAL))},
           '记录1': {'transformation': 'min_max_last_measured_missing', 'sources': anchors}},
          'event_ids': [a['event_id'] for a in anchors], 'state_aggregate': True})
    assert set(coverage) == {e['id'] for e in source['events']}
    return episodes, coverage
