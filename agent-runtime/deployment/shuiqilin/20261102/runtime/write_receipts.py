"""Observed file contents and narrow, program-generated write acknowledgements."""
import hashlib
import json
import re
from datetime import datetime, timezone

VERSION = 'observed-write-receipts-v1'
MAX_BYTES = 48000
PREVIEW_CHARS = 800


def operative_text(request):
    return re.sub(r'(?:把|内容(?:为|是|：|:))\s*“[^”]*”', '写入指定文本', request)


def literal_contract(request, allowed):
    """Bind only explicit one-write filenames and explicitly verbatim text."""
    from unified_schema import development_request, single_requested_tool
    intent = operative_text(request)
    if development_request(intent) or single_requested_tool(intent, allowed) != 'workspace.write':
        return {}
    matches = list(re.finditer(r'(?:写入|保存到)\s*[`“\"]?([A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,12})[`”\"]?(?=[，,。；;\s]|$)', intent))
    if len(matches) != 1 or intent[matches[0].end():].lstrip().startswith(('或','、')):
        return {}
    path = matches[0][1]
    contract = {'path': path}
    exact = re.search(r'把\s*“([^”]*)”\s*原样(?:写入|保存到)\s*[`“\"]?'+re.escape(path)+r'(?![A-Za-z0-9_./-])', request)
    if exact:
        contract['text'] = exact[1]
    return contract


def validate_literal_write(store, task_id, user_event, allowed, arguments):
    with store.db() as db:
        row = db.execute("SELECT body FROM events WHERE task_id=? AND id=? AND role='user' AND kind='message'", (task_id,user_event)).fetchone()
    if not row:
        return
    contract = literal_contract(json.loads(row[0]).get('text',''), allowed)
    wrong = [field for field, value in contract.items() if arguments.get(field) != value]
    if wrong:
        raise ValueError('写入参数不符合用户本轮明确指定的值：'+','.join(wrong)+'。尚未写入；按原始请求修正。')


def observe_text(path, expected):
    """One bounded binary read. A successful write is not content acceptance."""
    result = {'observed_at': datetime.now(timezone.utc).isoformat(),
              'matches_written_text': None, 'content_acceptance': 'not_evaluated'}
    try:
        with path.open('rb') as source:
            raw = source.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            return {**result, 'status': 'changed', 'matches_written_text': False,
                    'read_limit_reached': True}
        text = raw.decode('utf-8')
        return {**result, 'status': 'matches' if text == expected else 'changed',
                'matches_written_text': text == expected, 'bytes': len(raw),
                'chars': len(text), 'sha256': hashlib.sha256(raw).hexdigest(),
                'text': text[:PREVIEW_CHARS], 'text_complete': len(text) <= PREVIEW_CHARS,
                'scope': '实际文件在观测时的文本；节选不代表全文，写入一致不代表内容满足用户要求。'}
    except FileNotFoundError:
        return {**result, 'status': 'missing', 'matches_written_text': False}
    except (OSError, UnicodeError, ValueError):
        return {**result, 'status': 'unavailable'}


def receipt_only_request(request, allowed):
    """Conservative opt-in from an explicit one-write-and-receipt instruction."""
    from unified_schema import development_request, single_requested_tool
    # In explicit "put this literal text in a file" syntax, quoted words are
    # contents, not an additional instruction to analyze or test the result.
    # Unbalanced or unsupported quotation forms conservatively keep the model.
    request = operative_text(request)
    if development_request(request) or single_requested_tool(request, allowed) != 'workspace.write':
        return False
    if not re.search(r'报告(?:真实)?回执|写入后报告|保存后报告|仅报告写入结果', request):
        return False
    # Content authoring still belongs to the model. Any requested evaluation or
    # substantive final answer keeps the regular model path.
    if re.search(r'[？?]|解释|分析|比较|对比|评价|评估|总结|回答|为什么|怎么办|如何|是否|确认.*(?:满足|符合)|验收|测试|核对|检查', request):
        # "不要重新开发或测试" must not turn an explicit receipt into analysis.
        stripped = re.sub(r'(?:不要|不用|无需|不必|禁止|不需要|别)[^。！？\n]{0,30}(?:测试|检查|核对)', '', request)
        if re.search(r'[？?]|解释|分析|比较|对比|评价|评估|总结|回答|为什么|怎么办|如何|是否|确认.*(?:满足|符合)|验收|测试|核对|检查', stripped):
            return False
    return True


def verified_reply(runtime, job, receipt):
    """No inferred semantics or completion verdict. Bind to the actual operation."""
    store, tid = runtime.store, job['task_id']
    if not isinstance(receipt, dict) or receipt.get('tool') != 'workspace.write' or not receipt.get('ok'):
        return None
    task = store.task(tid)
    if not receipt_only_request(job['body']['text'], task['allowed_tools']):
        return None
    current_job = store.job(job['id'])
    if current_job['status'] != 'running' or current_job['body'] != job['body']:
        return None
    operations = store.operations(tid)
    own = [op for op in operations if op['user_event'] == job['body'].get('user_event')]
    if len(own) != 1 or any(op['status'] not in ('completed', 'failed') for op in operations):
        return None
    op = own[0]
    archived = op.get('result') or {}
    if (op['id'] != receipt.get('operation_id') or op['tool'] != 'workspace.write'
            or op['status'] != 'completed' or not archived.get('ok')
            or archived.get('evidence_ref') != receipt.get('evidence_ref')
            or op['window_id'] != task['window_id']):
        return None
    try:
        event = store.resolve(tid, receipt['evidence_ref'])
    except (KeyError, ValueError):
        return None
    if event['role'] != 'tool' or event['kind'] != 'tool_result':
        return None
    if any(receipt.get(k) != v for k, v in event['body'].items()):
        return None
    user = store.resolve(tid, job['body']['event_ref'])
    with store.db() as db:
        changed = db.execute("SELECT 1 FROM events WHERE task_id=? AND seq>? AND "
                             "(role='user' OR kind='constraint_update') LIMIT 1", (tid, user['seq'])).fetchone()
    if changed:
        return None
    # Resolve a fresh, authorized workspace path. Never re-execute the write.
    try:
        observation = observe_text(runtime.tools.path(tid, op['arguments']['path']), op['arguments']['text'])
    except (KeyError, TypeError, ValueError, PermissionError):
        return None
    proof = store.append(tid, 'tool', 'write_report_observation',
                         {'operation_id': op['id'], 'tool': 'workspace.write', 'observation': observation,
                          'execution_ref': receipt['evidence_ref']})
    # Quote and bound filenames; do not copy untrusted file text into an acknowledgement.
    path = json.dumps(op['arguments']['path'], ensure_ascii=False)
    if len(path) > 70:
        path = path[:65] + '…'
    if observation['status'] == 'matches':
        text = f'程序回执：已写入 {path}（{observation["chars"]} 字符），读回与本次写入一致。内容是否满足全部要求尚未验收。'
    else:
        state = {'changed': '内容已变化', 'missing': '文件已不存在', 'unavailable': '无法读回核对'}[observation['status']]
        text = f'程序回执：历史写入成功，但当前{state}，不能确认文件仍与本次写入一致。内容未验收。'
    if len(text) > runtime.settings.answer_max_chars:
        return None
    return {'response': text, 'answer_truncated': False, 'response_origin': 'runtime_verified_receipt',
            'model_generated': False, 'content_acceptance': 'not_evaluated',
            'sources_provided': [{'source_id': 'W1', 'event_ref': receipt['evidence_ref']},
                                 {'source_id': 'W2', 'event_ref': proof['ref']}],
            'receipt_observation_ref': proof['ref'], 'operation_id': op['id'], 'receipt_version': VERSION}
