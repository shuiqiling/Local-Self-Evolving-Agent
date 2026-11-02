"""Keep one explicit action's tool prefix stable before and after its receipt."""
import json


def report_view(store, task_id, payload, cutoff):
    from unified_schema import development_request, single_requested_tool
    request = payload.get('current_request', '')
    if development_request(request) or payload.get('pending_operations'):
        return payload
    single = single_requested_tool(request, payload.get('tools', {}))
    receipt = payload.get('last_result') or {}
    if not single:
        return payload
    with store.db() as db:
        job = db.execute("SELECT body FROM jobs WHERE task_id=? AND kind='message' AND status='running' "
                         "AND json_extract(body,'$.text')=? ORDER BY created LIMIT 1", (task_id, request)).fetchone()
    if not job:
        return payload
    user_event = json.loads(job[0]).get('user_event')
    # Apply the same tool view on both sides of execution. Changing the leading
    # tool definitions only after execution would invalidate a reusable KV prefix.
    out = dict(payload)
    out['tools'] = {single: payload['tools'][single]}
    out['requested_action_scope'] = '用户本轮明确限定一次此工具调用；保留其参数说明，不扩展到其他执行工具。'
    if receipt.get('tool') != single or not receipt.get('evidence_ref'):
        return out
    op = next((o for o in store.operations(task_id)
               if o['id'] == receipt.get('operation_id') and o['user_event'] == user_event
               and o['tool'] == single and o['status'] in ('completed', 'failed')), None)
    if not op or (op.get('result') or {}).get('evidence_ref') != receipt['evidence_ref']:
        return out
    try:
        event = store.resolve(task_id, receipt['evidence_ref'])
    except (ValueError, KeyError):
        return out
    if event['seq'] > cutoff or event['role'] != 'tool' or event['kind'] != 'tool_result':
        return out
    # Revalidation adds metadata; every archived receipt field must still match.
    if any(receipt.get(k) != v for k, v in event['body'].items()):
        return out
    # Keep the authorized name for the existing workflow guard. It independently
    # checks the real current-user receipt and permits finish/respond only.
    out['reporting_scope'] = '单次调用已有真实回执；本轮仅报告。失败不能说成功，历史执行回执不能证明当前文件状态。'
    return out
