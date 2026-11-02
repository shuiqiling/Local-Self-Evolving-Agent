"""Internal observations use the active chat's real archive and window budget."""
import asyncio
import json
import time
from state_view import snapshot_view, state_prompt, INSTRUCTION


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def context_snapshot(snapshot):
    """Keep recurring measurements readable; the complete sample stays archived."""
    cpu = snapshot.get('cpu', {})
    memory = snapshot.get('memory', {})
    network = snapshot.get('network', {})
    return {
        'local_time': snapshot.get('local_time'),
        'cpu': {k: cpu.get(k) for k in ('usage_percent', 'temperature_c', 'load_average')},
        'memory': {k: memory.get(k) for k in ('available_bytes', 'used_percent', 'swap_used_bytes')},
        'network': {k: network.get(k) for k in ('measurement', 'rx_mib_per_second', 'tx_mib_per_second')},
        'disk': {'used_percent': snapshot.get('disk', {}).get('used_percent')},
        'memory_storage': {k: snapshot.get('memory_storage', {}).get(k) for k in ('used_percent', 'status')},
    }


def latest_machine_state(store, task, records):
    """Resolve the newest actual sample; summaries and assessments cannot win."""
    inputs = [r for r in records if r['kind'] == 'internal_state_input']
    if not inputs:
        return task['state'].get('latest_internal_state')
    event = store.resolve(task['id'], inputs[-1]['ref'])
    return {'record_type': 'timestamped_machine_observation_not_user_instruction',
            'event_ref': event['ref'], 'observation_id': event['body']['observation_id'],
            'snapshot': context_snapshot(event['body']['snapshot'])}


class SharedChatContext:
    def __init__(self, runtime, http, autonomy=None):
        self.runtime, self.http = runtime, http
        self.autonomy = autonomy
        self.budget_block = None

    def target(self, db=None):
        if db is None:
            with self.runtime.store.db() as connection:
                return self.target(connection)
        row = db.execute("""SELECT e.task_id FROM events e JOIN tasks t ON t.id=e.task_id
            WHERE e.role='user' AND e.kind IN ('message','constraint_update','task_created')
            ORDER BY e.seq DESC LIMIT 1""").fetchone()
        return row[0] if row else None

    def status(self):
        task_id = self.target()
        task = self.runtime.store.task(task_id) if task_id else None
        return {'mode': 'shared_chat', 'binding': 'latest_external_user_activity',
                'task_id': task_id, 'window_id': task['window_id'] if task else None,
                'working_tokens': self.runtime.settings.working_tokens,
                'counts_real_handoffs': True}

    async def deliver(self, snapshot, identifier, system, schema, limit, on_request,
                      parse_output, expected_external_seq):
        from idle_state_input import external_marker
        from lea.model import BudgetExceeded, messages
        store = self.runtime.store
        task_id = self.target()
        if not task_id:
            return {'status': 'waiting_for_chat_context'}
        started = time.monotonic()
        async with self.runtime.locks[task_id]:
            def valid(db, window=None, constraint=None):
                if external_marker(db)[0] != expected_external_seq or self.target(db) != task_id:
                    return False
                if db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]:
                    return False
                current = store._task(db, task_id)
                return ((window is None or current['window_id'] == window) and
                        (constraint is None or current['constraint_version'] == constraint))

            for rollovers in range(2):
                with store.db() as db:
                    if not valid(db):
                        return {'status': 'preempted_by_foreground_activity'}
                task, records, cutoff = self.runtime.context(task_id, with_cutoff=True)
                budget_key=(task_id,task['window_id'],task['constraint_version'],cutoff,
                            self.runtime.settings.working_tokens,self.runtime.settings.backend_context,limit)
                if self.budget_block and self.budget_block['key']==budget_key:
                    return {k:v for k,v in self.budget_block.items() if k!='key'}
                payload = {'channel': 'internal_state_input', 'context_scope': 'shared_current_chat_window',
                           'task_id': task_id, 'window_id': task['window_id'],
                           'goal': task['goal'], 'constraints': task['constraints'],
                           'handoff_state': task['state'], 'records': records,
                           'snapshot': snapshot_view(snapshot),
                           'instruction': INSTRUCTION}
                if hasattr(self.runtime,'prepare_internal_payload'):
                    payload=self.runtime.prepare_internal_payload(task,payload,cutoff)
                if self.autonomy:
                    from idle_autonomy import SYSTEM, INSTRUCTION as AUTONOMY_INSTRUCTION
                    system=SYSTEM; schema=self.autonomy.schema(task)
                    if hasattr(self.autonomy,'output_limit'):limit=self.autonomy.output_limit(task,limit)
                    parse_output=lambda content:self.autonomy.parse(content,task)
                    payload['initiative_context']=self.autonomy.context(task)
                    payload['instruction']=AUTONOMY_INSTRUCTION
                prompt = messages(system, payload, layout=self.runtime.settings.window_prompt_layout)
                from state_budget import fit, useful_records
                def render(raw):
                    shaped=state_prompt(raw)
                    return self.autonomy.prompt(shaped) if self.autonomy else shaped
                prompt=await fit(self.runtime,task,prompt,cutoff,render)
                count = await self.runtime.model.tokens(prompt)
                if count <= self.runtime.settings.working_tokens and count + limit <= self.runtime.settings.backend_context:
                    self.budget_block=None
                    break
                if rollovers == 1 or not useful_records(records):
                    self.budget_block={'key':budget_key,'status':'context_budget_blocked',
                        'error':'Final state prompt exceeds budget; no empty-window rollover or unchanged-context retry',
                        'input_tokens':count,'working_tokens':self.runtime.settings.working_tokens,
                        'context_task_id':task_id,'context_window_id':task['window_id'],'context_handoffs':rollovers}
                    return {k:v for k,v in self.budget_block.items() if k!='key'}
                await self.runtime.handoff(task_id, 'internal_state_context_limit')
            window, constraint = task['window_id'], task['constraint_version']
            with store.db(True) as db:
                if not valid(db, window, constraint):
                    return {'status': 'preempted_by_foreground_activity'}
                event = store._event(db, store._task(db, task_id), 'internal_state', 'internal_state_input',
                    {'text': canonical({'record_type': 'timestamped_machine_observation_not_user_instruction',
                                        'observation_id': identifier, 'snapshot': context_snapshot(snapshot)}),
                     'observation_id': identifier, 'snapshot': snapshot,
                     'context_refs': [r['ref'] for r in records], 'source_cutoff_seq': cutoff})
                db.execute('''UPDATE internal_state_observations SET context_task_id=?,context_window_id=?,
                    input_event_ref=?,source_cutoff_seq=? WHERE id=?''',
                    (task_id, window, event['ref'], event['seq'], identifier))
            body = {'model': self.runtime.settings.model_name, 'messages': prompt,
                    'temperature': 0, 'cache_prompt': False, 'max_tokens': limit,
                    'response_format': {'type': 'json_schema', 'json_schema':
                        {'name': 'InternalState', 'schema': schema, 'strict': True}}}
            adapter = getattr(self.runtime.model, 'adapter', None)
            if adapter:
                body['lora'] = [{'id': adapter.get('id', 0), 'scale': 1}]
            if hasattr(self.http,'prepare'):body=self.http.prepare(body)
            on_request(body)
            tick = time.monotonic()
            response = await self.http.post('/v1/chat/completions', json=body, timeout=240 if limit>512 else 180)
            response.raise_for_status()
            result = response.json()
            choice = result['choices'][0]
            if choice.get('finish_reason') == 'length':
                raise ValueError('State response truncated')
            output = parse_output(choice['message']['content'])
            elapsed = round(time.monotonic() - tick, 3)
            # The result and its observation status commit together. A queued
            # user message wins even if it arrived just as inference finished.
            with store.db(True) as db:
                if not valid(db, window, constraint):
                    return {'status': 'preempted_by_foreground_activity'}
                reply = store._event(db, store._task(db, task_id), 'internal_state', 'internal_state_result',
                    {'text': canonical({'record_type': 'model_assessment_not_execution_or_verified_fact',
                                        'observation_id': identifier, 'assessment': output}),
                     'observation_id': identifier, 'assessment': output,
                     'input_event_ref': event['ref']})
                db.execute('''UPDATE internal_state_observations SET model_output=?,status='completed',
                    model_seconds=?,result_event_ref=? WHERE id=?''',
                    (canonical(output), elapsed, reply['ref'], identifier))
            store.export(task_id)
            return {'status': 'completed', 'model_output': output, 'model_seconds': elapsed,
                    'delivery_seconds': round(time.monotonic() - started, 3),
                    'context_task_id': task_id, 'context_window_id': window,
                    'input_event_ref': event['ref'], 'result_event_ref': reply['ref'],
                    'input_tokens': count, 'context_handoffs': rollovers}
