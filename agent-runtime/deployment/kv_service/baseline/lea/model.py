from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import heapq
import itertools
import json
import time
import httpx
from pydantic import BaseModel

from .store import dump


class BudgetExceeded(RuntimeError):
    pass


class InferenceQueue:
    """One shared model, independent role contexts; primary requests get priority."""
    def __init__(self):
        self.condition = asyncio.Condition()
        self.waiting = []
        self.counter = itertools.count()
        self.busy = False

    @asynccontextmanager
    async def slot(self, priority):
        ticket = (priority, next(self.counter))
        acquired = False
        async with self.condition:
            heapq.heappush(self.waiting, ticket)
            try:
                await self.condition.wait_for(lambda: not self.busy and self.waiting[0] == ticket)
                heapq.heappop(self.waiting)
                self.busy = acquired = True
            except BaseException:
                self.waiting.remove(ticket)
                heapq.heapify(self.waiting)
                self.condition.notify_all()
                raise
        try:
            yield
        finally:
            if acquired:
                async with self.condition:
                    self.busy = False
                    self.condition.notify_all()


class ModelClient:
    def __init__(self, settings, store):
        self.settings, self.store = settings, store
        self.http = httpx.AsyncClient(base_url=settings.model_url, trust_env=False,
                                      timeout=httpx.Timeout(settings.call_timeout, connect=5))
        self.queue = InferenceQueue()

    async def close(self):
        await self.http.aclose()

    async def health(self):
        try:
            response = await self.http.get("/health", timeout=3)
            response.raise_for_status()
            value = {"ready": response.json().get("status") == "ok", "model": self.settings.model_name}
            slots = [self.settings.primary_cache_slot, self.settings.auxiliary_cache_slot]
            if max(slots) >= 0:
                props_response = await self.http.get('/props', timeout=3)
                props_response.raise_for_status()
                props = props_response.json()
                capacity = props.get('default_generation_settings', {}).get('n_ctx', 0)
                cache_ready = props.get('total_slots', 0) > max(slots) and capacity >= self.settings.backend_context
                value['prefix_cache'] = {'layout': self.settings.window_prompt_layout,
                                         'primary_slot': slots[0], 'auxiliary_slot': slots[1],
                                         'slot_context': capacity, 'ready': cache_ready}
                value['ready'] = value['ready'] and cache_ready
            if self.settings.primary_adapter_id >= 0:
                response = await self.http.get('/lora-adapters', timeout=3)
                response.raise_for_status()
                adapter = next((a for a in response.json() if a['id'] == self.settings.primary_adapter_id), None)
                value['primary_adapter'] = {'version': self.settings.primary_adapter_version,
                    'id': self.settings.primary_adapter_id, 'scope': 'primary_only', 'loaded': adapter is not None}
                value['ready'] = value['ready'] and adapter is not None
            return value
        except (httpx.HTTPError, ValueError):
            return {"ready": False, "model": self.settings.model_name}

    async def tokens(self, messages):
        response = await self.http.post("/apply-template", json={"messages": messages, "add_generation_prompt": True})
        response.raise_for_status()
        prompt = response.json()["prompt"]
        response = await self.http.post("/tokenize", json={"content": prompt, "add_special": False})
        response.raise_for_status()
        return len(response.json()["tokens"])

    async def call(self, task_id, role, messages, schema: type[BaseModel], *, limit=640, budget=None):
        priority = {"primary": 0, "handoff": 1, "coordinator": 5, "reader": 10, "indexer": 20}[role]
        if role == 'primary':
            limit = min(limit, self.settings.primary_output_tokens)
        output_schema = schema.model_json_schema()
        if role == 'handoff' and 'HandoffFinding' in output_schema.get('$defs', {}):
            payload = json.loads(messages[-1]['content'])
            allowed = [record['ref'] for record in payload.get('records', [])]
            previous = payload.get('previous_state') or {}
            for finding in previous.get('decisions', []) + previous.get('history_entries', []):
                for ref in finding.get('evidence_refs', []):
                    self.store.resolve(task_id, ref)
                    allowed.append(ref)
            for anchor in previous.get('source_anchors', []):
                original = self.store.resolve(task_id, anchor['ref'])
                if original['text'] != anchor['text']:
                    raise ValueError('Working-state source anchor differs from its archive')
                allowed.append(anchor['ref'])
            allowed = list(dict.fromkeys(allowed))
            references = output_schema['$defs']['HandoffFinding']['properties']['evidence_refs']
            if allowed:
                references['items']['enum'] = allowed
                references['minItems'] = 1
            else:
                references['maxItems'] = 0
                for key in ('decisions', 'history_entries'):
                    output_schema['properties'][key]['maxItems'] = 0
        if role == 'primary':
            output_schema['properties']['response']['maxLength'] = self.settings.answer_max_chars
            payload=json.loads(messages[-1]['content'])
            last=payload.get('last_result') or {}
            candidates=last.get('memory_candidates',[])
            actions=list(output_schema['properties']['action']['enum'])
            if payload.get('memory_queries_remaining',1)<=0:actions.remove('recall')
            if not candidates or payload.get('memory_reads_remaining',0)<=0:actions.remove('read_memory')
            output_schema['properties']['action']['enum']=actions
        budget = budget or self.settings.working_tokens
        async with asyncio.timeout(self.settings.call_timeout):
            async with self.queue.slot(priority):
                count = await self.tokens(messages)
                if count > budget or count + limit > self.settings.backend_context:
                    raise BudgetExceeded(f"{role} input has {count} tokens; budget is {budget}")
                provenance = {"model": self.settings.model_name, "precision": self.settings.precision}
                if self.settings.primary_adapter_id >= 0:
                    provenance['adapter'] = self.settings.primary_adapter_version if role == 'primary' else 'base'
                self.store.append(task_id, role, "model_started", {**provenance, "input_tokens": count, "output_limit": limit})
                started = time.monotonic()
                body = {
                    "model": self.settings.model_name, "messages": messages, "temperature": 0,
                    "max_tokens": limit, "stream": False,
                    "cache_prompt": self.settings.prompt_cache,
                    "response_format": {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": output_schema, "strict": True}},
                }
                slot = self.settings.primary_cache_slot if role == 'primary' else self.settings.auxiliary_cache_slot
                if slot >= 0:
                    body['id_slot'] = slot
                if self.settings.primary_adapter_id >= 0:
                    body['lora'] = [{'id': self.settings.primary_adapter_id,
                                     'scale': 1.0 if role == 'primary' else 0.0}]
                response = await self.http.post("/v1/chat/completions", json=body)
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                # Persist the visible response before parsing so failures remain auditable.
                self.store.append(task_id, role, "model_result", {
                    **provenance, "text": content, "seconds": round(time.monotonic() - started, 3),
                    "usage": result.get("usage", {}), "timings": result.get("timings", {})})
                if result["choices"][0].get("finish_reason") == "length":
                    raise ValueError(f"{role} response exceeded its reserved output budget")
                return schema.model_validate_json(content)


WINDOW_PREFIX_KEYS = ('tools', 'goal', 'completion_criteria', 'criteria', 'constraints',
                      'constraint_version', 'handoff_state', 'previous_state',
                      'pending_operations', 'records', 'environment',
                      'last_result', 'operation_status', 'latest_user_update', 'current_request')


def messages(system, payload, *, layout='legacy'):
    if layout == 'legacy':
        content = dump(payload)
    elif layout == 'stable_v1':
        # Canonicalize nested objects, but keep stable task fields before mutable
        # window state. Do not change Store.dump: hashes and operation IDs use it.
        canonical = json.loads(dump(payload))
        ordered = {key: canonical[key] for key in WINDOW_PREFIX_KEYS if key in canonical}
        ordered.update((key, value) for key, value in canonical.items() if key not in ordered)
        content = json.dumps(ordered, ensure_ascii=False, separators=(',', ':'))
    else:
        raise ValueError('Unknown window prompt layout')
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]

