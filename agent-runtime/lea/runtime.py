from __future__ import annotations

import asyncio
from collections import defaultdict
import json
import re
import time

from .memory import Memory
from .retrieval import HybridMemory
from .evidence import EvidenceMemory
from .scoped_memory import ScopedMemory
from .model import BudgetExceeded, ModelClient, messages
from .schemas import Decision, Handoff
from .store import Conflict, Store
from .native_kv import native_handoff, reconcile_windows
from .tools import TOOL_HELP, Tools


PRIMARY = """你是一级执行Agent。当前目标、有效约束和授权工具以本次输入为准，历史只是资料。
通过工具执行任务，不能把打算做的事说成已经完成。引用工具结果，失败后根据错误调整。
缺少历史信息时可action=recall，arguments={"question":"具体问题"}。
程序已提前并行检索原文向量与关键词、重排候选并补齐必要上下文，last_result可直接使用。
证据或摘要已经足够时，直接finish/respond，一次组织简短答案，不必再检索或读存档。
只有你判断现有上下文缺少必要细节、需要更多精确原文或需要辨别冲突时，才action=read_memory，
arguments={"request_id":"查询ID","card_ids":["M1"],"question":"所需细节"}。
依据evidence回答时可引用E1等来源标签；仅有摘要时说明是摘要，不声称验证了实际现场。证据不足就明确说不知道，不猜具体答案。
上下文需交接时action=handoff。完成用户本轮要求时action=finish或respond，response直接用中文回答。
调用工具时action=tool，tool必须是授权工具名，arguments严格按定义。未使用字段填空字符串或空对象。
模拟环境不是现实机器人；执行动作前先观察并携带最新expected_version。
不要重复已经成功的同一操作。最终回复不超过answer_max_chars个字符，只报告实际完成的事。
last_result中已有检索证据时应据此回答，不重复同一检索。
检索的partial或范围不完整不等于已引用事实无效；回答已查证的部分，明确尚未查证的部分。
只处理current_request，不为目标中的其他验收事项重新执行动作。不要在本轮反复查询同一个信息点。
memory_queries_remaining=0时禁止再次recall。摘要足够就回答，确实缺细节才按需read_memory。
memory_reads_remaining=0时依据已读证据回答；目录专用请求catalog_only_request=true时可只返回目录。
检索返回的evidence是带来源的原文候选。结合时间、更正、否定和适用条件回答；不把相似度当成事实核验。
用户或工具已明确记录的编号、数值和状态，可以注明来源后报告；核验尚未完成，不等于记录不存在。
handoff_state.source_anchors是程序保留并校验的原文工作状态。与摘要冲突时依据这些原文及最新更正回答，按用户原文保留下一步；不能让摘要的推测覆盖原文。
latest_user_update单列的是交接时最新的用户原文，更正值优先于旧摘要、旧回答和旧检索；不把已作废数值当当前值。
operation_status是持久化执行账本，completed连同path表示该文件已成功处理，不能因为摘要缺项说文件没有记录或重复写入。
回答下一步时保留用户记录的计划与顺序，不新增等待条件或擅自改变流程。
查历史细节使用read_memory。workspace.read只用于用户明确给出或工具确认存在的文件，不能猜文件名来查历史记忆。
只输出符合JSON结构的对象，不输出思考过程。"""
PRIMARY_SCOPED = PRIMARY.replace(
    '程序已提前并行检索原文向量与关键词、重排候选并补齐必要上下文，last_result可直接使用。',
    '程序已用原文向量定位记忆，但last_result默认只有所属片段的摘要索引，没有自动补入原文。'
).replace(
    'arguments={"request_id":"查询ID","card_ids":["M1"],"question":"所需细节"}。',
    'arguments={"request_id":"查询ID","card_ids":["M1"],"question":"当前问题中缺失的具体细节"}。工具只在选中索引对应的原文范围内做向量搜索，返回少量原文和必要条件。'
).replace(
    'last_result中已有检索证据时应据此回答，不重复同一检索。',
    '索引title保留原文主题，先核对对象。summary_pending或summary_rejected=true时只是定位预览，未提及的细节不等于不存在。已有原文证据时据此回答，不重复读取。'
)
HANDOFF = """你是当前一级Agent，正在为下一轮一级生成交接手册和历史索引。
原始记录不会丢失。摘要用于恢复当前任务，不能覆盖或放宽用户约束。
必须准确保留用户已明确给出的编号、当前数值、最新更正和未完成事项；核验未完成不等于已知事实不存在。
next_steps保留记录中明确的后续计划及顺序，不新增等待条件；已经给出的计划不能被你改写成另一条流程。
不得把用户未要求的等待、禁止或推测写成decisions；不能用一次确认收到来证明其他事实。
completed只写有记录支持的已完成事项，next_steps保留未完成目标，重要决定和索引条目必须附真实证据ref。
当前还在准备交接，不能声称换窗已经完成。summary最多50个汉字，completed最多2条，每条最多20字。
decisions只保留真正影响后续动作的决定，没有则[]。history_entries最多2条，避免把每个工具步骤重复列出。
引用只放在evidence_refs中，每条一个记录短标签R1等，不在普通文字里重复引用。空项用[]。
不要把模拟世界当成现实世界，不虚构执行结果。只输出紧凑JSON，不要重复说明。"""


class Runtime:
    def __init__(self, settings, model=None, embedder=None):
        self.settings = settings
        self.store = Store(settings.data_dir)
        self.model = model or ModelClient(settings, self.store)
        self.tools = Tools(self.store)
        self.memory = (ScopedMemory(self.store,self.model,settings,embedder) if settings.memory_backend=='scoped' else
                       EvidenceMemory(self.store,self.model,settings,embedder) if settings.memory_backend=='hybrid' else
                       HybridMemory(self.store,self.model,settings,embedder) if settings.memory_backend=='summary' else
                       Memory(self.store,self.model,settings))
        self.locks = defaultdict(asyncio.Lock)
        self.active = {}
        self.workers = []
        self.closing = False

    async def start(self):
        self.store.recover()
        self.workers = [asyncio.create_task(self.worker('execution')) for _ in range(2)]+[asyncio.create_task(self.worker(kind)) for kind in ('query','verification')]
        if hasattr(self.memory,'start'):await self.memory.start()

    async def close(self):
        self.closing = True
        for child in list(self.active.values()):
            child.cancel()
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        if hasattr(self.memory,'close'):await self.memory.close()
        await self.model.close()

    async def worker(self,kind='execution'):
        while True:
            job = self.store.next_job(kind)
            if job is None:
                await asyncio.sleep(0.3)
                continue
            child = asyncio.create_task(self.dispatch(job))
            self.active[job["id"]] = child
            try:
                result = await child
                self.store.set_job(job["id"], "completed", result)
            except asyncio.CancelledError:
                if self.store.job(job['id'])['status']!='canceled':
                    self.store.set_job(job["id"], "queued" if self.closing else "canceled")
                if self.closing:
                    raise
            except Exception as exc:
                self.store.set_job(job["id"], "error", error=f"{type(exc).__name__}: {exc}")
                self.store.append(job["task_id"], "runtime", "job_error", {"job_id": job["id"], "error": str(exc)})
            finally:
                self.active.pop(job["id"], None)
                self.store.export(job["task_id"])

    def cancel(self, job_id):
        job = self.store.job(job_id)
        if job["status"] not in ("queued", "running"):
            return job
        self.store.set_job(job_id, "canceled")
        if job_id in self.active:
            self.active[job_id].cancel()
        self.store.append(job["task_id"], "user", "job_canceled", {"job_id": job_id})
        return self.store.job(job_id)

    async def dispatch(self, job):
        task_id, body = job["task_id"], job["body"]
        if job['kind']=='query' and self.settings.memory_backend in ('hybrid','scoped'):
            if self.store.job(job['id'])['status']=='canceled':raise asyncio.CancelledError()
            return await self.memory.query(task_id,body['question'],body.get('exhaustive',False),verify=body.get('verify',False),
                read_archive=body.get('read_archive',False),min_score=body.get('min_score'),search_archive=body.get('search_archive',False))
        while not (await self.model.health())["ready"]:
            if self.store.job(job["id"])["status"] == "canceled":
                raise asyncio.CancelledError()
            await asyncio.sleep(2)
        async with self.locks[task_id]:
            if self.store.job(job["id"])["status"] == "canceled":
                raise asyncio.CancelledError()
            if job["kind"] == "query":
                return await self.memory.query(task_id, body["question"], body.get("exhaustive", False))
            if job["kind"] == "handoff":
                return await self.handoff(task_id, "user_requested")
            return await self.primary(job)

    def context(self, task_id, after=None, *, with_cutoff=False):
        task = self.store.task(task_id)
        events = self.store.events(task_id, after=task["state"].get("recorded_through", 0) if after is None else after)
        usable = [e for e in events if e["kind"] in ("message", "assistant_reply", "tool_result", "constraint_update", "retrieval_result", "primary_decision")]
        records = []
        for e in usable:
            text = e['text']
            if e['kind'] == 'retrieval_result':
                text = json.dumps({key:e['body'].get(key) for key in ('request_id','answer_status','answer')}, ensure_ascii=False)
            record = {"ref": e["ref"], "role": e["role"], "kind": e["kind"], "text": text[:1600],
                      "excerpt_only": len(text) > 1600}
            if e['kind'] == 'retrieval_result':
                record['retrieval_id'] = e['body'].get('request_id')
            records.append(record)
        if with_cutoff:
            cutoff = events[-1]['seq'] if events else task['state'].get('recorded_through', 0)
            return task, records, cutoff
        return task, records

    def handoff_anchors(self, task_id, previous_state, records, through_seq):
        # Carry a bounded set of complete source records, independent of the
        # generated handbook. Always resolve again from the immutable archive.
        refs = [item['ref'] for item in previous_state.get('source_anchors', [])]
        refs += [item['ref'] for item in records if item['role'] in ('user', 'tool')]
        sources = {}
        for ref in dict.fromkeys(refs):
            event = self.store.resolve(task_id, ref)
            if event['seq'] > through_seq or event['role'] not in ('user', 'tool'):
                continue
            if event['kind'] not in ('message', 'constraint_update', 'tool_result'):
                continue
            text = event['text']
            if len(text) > 320:
                continue  # Longer material remains in the archive for tool reads.
            sources[ref] = {'ref': ref, 'seq': event['seq'], 'role': event['role'], 'text': text}
        selected, remaining = [], 800
        for item in sorted(sources.values(), key=lambda item: item['seq'], reverse=True):
            if len(item['text']) > remaining:
                continue
            selected.append(item)
            remaining -= len(item['text'])
            if len(selected) == 8:
                break
        return list(reversed(selected))

    async def handoff(self, task_id, reason):
        if getattr(self.model,'native_kv',None):
            return await native_handoff(self,task_id,reason)
        started = time.monotonic()
        task, records, through_seq = self.context(task_id, with_cutoff=True)
        # The cutoff must be the last event actually read, not a later marker:
        # user messages arriving between the read and marker belong to the next window.
        self.store.append(task_id, "primary", "handoff_requested", {"reason": reason})
        aliases = {f"R{index+1}": record["ref"] for index, record in enumerate(records[-10:])}
        compact_records = [{**record, "ref": alias} for alias, record in zip(aliases, records[-10:])]
        payload = {"goal": task["goal"], "criteria": task["criteria"], "constraints": task["constraints"],
                   "previous_state": task["state"], "records": compact_records,
                   "pending_operations": [op for op in self.store.operations(task_id) if op["status"] not in ("completed", "failed")],
                   "omitted_older_record_count": max(0, len(records) - 10)}
        while await self.model.tokens(messages(HANDOFF, payload, layout=self.settings.window_prompt_layout)) > self.settings.working_tokens + self.settings.handoff_reserve:
            if not payload["records"]:
                raise BudgetExceeded("Mandatory handoff state exceeds the reserved context")
            payload["records"].pop(0)
            payload["omitted_older_record_count"] += 1
        self.store.append(task_id, "runtime", "handoff_snapshot", {
            "through_seq": through_seq, "ref_aliases": {record["ref"]: aliases[record["ref"]] for record in payload["records"]}})
        prepared = time.monotonic()
        handbook = await self.model.call(task_id, "handoff", messages(HANDOFF, payload, layout=self.settings.window_prompt_layout), Handoff,
                                         limit=640, budget=self.settings.working_tokens + self.settings.handoff_reserve)
        generated = time.monotonic()
        data = handbook.model_dump()
        visible_aliases = {record["ref"] for record in payload["records"]}
        for finding in data["decisions"] + data["history_entries"]:
            finding["evidence_refs"] = [aliases[ref] if ref in visible_aliases else ref for ref in finding["evidence_refs"]]
        data['source_anchors'] = self.handoff_anchors(task_id, task['state'], records, through_seq)
        user_updates = [item for item in data['source_anchors'] if item['role'] == 'user']
        data['latest_user_update'] = user_updates[-1] if user_updates else None
        result = self.store.commit_handoff(task_id, task["window_id"], through_seq, data, task["constraint_version"])
        committed = time.monotonic()
        self.store.append(task_id, 'runtime', 'handoff_completed', {
            'from_window': task['window_id'], 'to_window': result['window_id'], 'through_seq': through_seq,
            'prompt_layout': self.settings.window_prompt_layout,
            'seconds': round(committed-started, 6), 'prepare_seconds': round(prepared-started, 6),
            'generation_seconds': round(generated-prepared, 6), 'commit_seconds': round(committed-generated, 6)})
        return {"window_id": result["window_id"], "version": result["version"], "summary": handbook.summary}

    async def primary(self, job):
        task_id, body = job["task_id"], job["body"]
        rollovers, last_result = 0, None
        retrieval_ids, latest_retrieval = [], None
        archive_reads=0
        scoped_after = None
        if self.settings.memory_backend == 'scoped':
            with self.store.db() as db:
                row = db.execute('SELECT seq FROM events WHERE task_id=? AND id=?', (task_id,body.get('user_event'))).fetchone()
            # Old original messages must not bypass the explicit memory tool via records.
            scoped_after = row[0]-1 if row else self.store.events(task_id, limit=1, tail=True)[-1]['seq']
        if self.settings.memory_backend in ('hybrid','scoped'):
            question=body['text']
            if len(question)>1500:question=question[:750]+'\n'+question[-750:]
            latest_retrieval=await self.memory.query(task_id,question,exclude_event_id=body.get('user_event'))
            retrieval_ids.append(latest_retrieval['request_id'])
            last_result=self.memory.for_primary(latest_retrieval)
        for step in range(self.settings.max_steps):
            task = self.store.task(task_id)
            after = max(scoped_after, task['state'].get('recorded_through', 0)) if scoped_after is not None else None
            task, records, native_cutoff = self.context(task_id, after=after, with_cutoff=True)
            if last_result and last_result.get('request_id'):
                records = [record for record in records if record.get('retrieval_id') != last_result['request_id']]
            payload = {"goal": task["goal"], "completion_criteria": task["criteria"], "constraints": task["constraints"],
                       "constraint_version": task["constraint_version"], "task_version": task["version"],
                       "handoff_state": task["state"], "current_request": body["text"],
                       "environment": self.tools.observe(task_id),
                       "tools": {name: TOOL_HELP[name] for name in task["allowed_tools"]},
                       "records": records, "last_result": last_result, "step": step + 1,
                       "memory_queries_remaining": max(0, self.settings.max_memory_queries - len(retrieval_ids)),
                       "memory_reads_remaining":max(0,1-archive_reads),"answer_max_chars":self.settings.answer_max_chars,
                       "catalog_only_request":bool(re.search(r'只.{0,8}(?:目录|摘要)|不要.{0,6}(?:原文|存档)',body['text'])),
                       "pending_operations": [op for op in self.store.operations(task_id) if op["status"] not in ("completed", "failed")],
                       "latest_user_update": task["state"].get("latest_user_update"),
                       "operation_status": [{"id": op["id"], "tool": op["tool"], "status": op["status"],
                                             "path": op["arguments"].get("path"),
                                             "evidence_ref": (op.get("result") or {}).get("evidence_ref")}
                                            for op in self.store.operations(task_id)[-10:]]}
            if getattr(self.model,'native_kv',None):
                payload['_native_through_seq']=native_cutoff
                payload['current_request_operations']=[
                    {'id':op['id'],'tool':op['tool'],'status':op['status'],'path':op['arguments'].get('path'),
                     'evidence_ref':(op.get('result') or {}).get('evidence_ref')}
                    for op in self.store.operations(task_id) if op['user_event']==body.get('user_event')]
            prompt = messages(PRIMARY_SCOPED if self.settings.memory_backend=='scoped' else PRIMARY, payload, layout=self.settings.window_prompt_layout)
            if not getattr(self.model,'native_kv',None) and await self.model.tokens(prompt) > self.settings.working_tokens:
                if rollovers >= 2:
                    raise BudgetExceeded("Mandatory request/state is too large; shorten it or raise the working budget")
                await self.handoff(task_id, "working_context_limit")
                rollovers += 1
                continue
            decision = await self.model.call(task_id, "primary", prompt, Decision, limit=self.settings.primary_output_tokens)
            if getattr(self.model,'native_kv',None):
                metadata=self.model.native_kv.take_result(task_id)
                try:
                    task=reconcile_windows(self.store,task_id,task['window_id'],task['constraint_version'],metadata)
                except Conflict:
                    last_result={'warning':'Effective constraints or task state changed; replan with the latest state'}
                    continue
            self.store.append(task_id, "primary", "primary_decision", decision.model_dump())
            # User corrections arriving during generation invalidate this decision.
            current = self.store.task(task_id)
            if current["constraint_version"] != task["constraint_version"] or current["window_id"] != task["window_id"]:
                last_result = {"warning": "Effective constraints or task state changed; replan with the latest state"}
                continue
            if decision.action in ("respond", "finish"):
                if not decision.response.strip():
                    raise ValueError("Primary returned an empty final answer")
                response,clipped=short_answer(decision.response,self.settings.answer_max_chars)
                sources=[{'source_id':f'E{i+1}','event_ref':e['event_ref']} for i,e in enumerate((latest_retrieval or {}).get('evidence',[]))]
                event=self.store.append(task_id,'primary','assistant_reply',{'text':response,'answer_truncated':clipped,'sources_provided':sources})
                return {'response':response,'answer_truncated':clipped,'event_ref':event['ref'],'steps':step+1,'retrieval_ids':retrieval_ids,
                    'archive_reads':archive_reads,'memory_basis':'archive_excerpt' if archive_reads else ('retrieved_context' if (latest_retrieval or {}).get('evidence') else ('summary_index' if (latest_retrieval or {}).get('memory_candidates') else 'none')),
                    'sources_provided':sources}
            if decision.action == "tool":
                try:
                    last_result = self.tools.execute(task_id, task["window_id"], task["constraint_version"],
                                                     body["user_event"], decision.tool, decision.arguments)
                except (PermissionError, ValueError, Conflict) as exc:
                    last_result = {"ok": False, "error": str(exc)}
                    self.store.append(task_id, "runtime", "execution_rejected", last_result)
            elif decision.action == "recall":
                if len(retrieval_ids) >= self.settings.max_memory_queries:
                    # A small model may keep asking for an already retrieved fact.
                    # End visibly with literal evidence, never fake a model answer
                    # or silently run an unbounded number of expensive queries.
                    result = latest_retrieval or {}
                    evidence = sorted(result.get('evidence', []), key=lambda item: item.get('source_type') != 'user_statement')
                    lines=['检索预算已用完。以下为原文节选：' if evidence else '本轮目录检索已完成，以下为记忆摘要：']
                    for item in evidence[:4]:
                        quote = item['quote']
                        lines.append('「' + quote[:300] + ('…（节选）' if len(quote)>300 else '') + '」 [' + item['event_ref'] + ']')
                    if not evidence:
                        lines.extend(card['summary'] for card in result.get('memory_candidates',[])[:2])
                    if not evidence and not result.get('memory_candidates'):lines=['没有达到匹配门槛的候选，当前无法查证。']
                    response,_=short_answer('\n'.join(lines),self.settings.answer_max_chars)
                    event = self.store.append(task_id, 'runtime', 'assistant_reply', {
                        'text':response, 'response_mode':'verified_evidence_fallback', 'query_id':result.get('request_id')})
                    return {'response':response, 'event_ref':event['ref'], 'steps':step+1, 'retrieval_ids':retrieval_ids,
                            'response_mode':'verified_evidence_fallback', 'semantic_review_required':True}
                question = decision.arguments.get("question", "")
                if not isinstance(question, str) or not question.strip():
                    last_result = {"error": "Recall requires a specific question"}
                else:
                    kwargs={'exclude_event_id':body.get('user_event')} if self.settings.memory_backend in ('hybrid','scoped') else {}
                    retrieval = await self.memory.query(task_id, question[:1500], decision.arguments.get("exhaustive", False) is True,**kwargs)
                    latest_retrieval = retrieval
                    retrieval_ids.append(retrieval['request_id'])
                    last_result = self.memory.for_primary(retrieval)
            elif decision.action=='read_memory':
                if self.settings.memory_backend not in ('hybrid','scoped'):
                    last_result={'error':'read_memory requires summary catalog backend'};continue
                if archive_reads>=1:
                    last_result={'error':'本轮原文读取预算已用完，请依据已读证据简短回答。'};continue
                request_id=decision.arguments.get('request_id');card_ids=decision.arguments.get('card_ids')
                if request_id not in retrieval_ids or not isinstance(card_ids,list) or not all(isinstance(x,str) for x in card_ids):
                    last_result={'error':'只能读取本轮recall返回的request_id和card_ids。'};continue
                try:
                    aliases={f'M{i+1}':card['card_id'] for i,card in enumerate(self.memory.selected_cards(latest_retrieval or {}))}
                    card_ids=[aliases.get(identifier,identifier) for identifier in card_ids]
                    self.store.append(task_id,'runtime','memory_read_selection',{'request_id':request_id,'card_aliases':aliases,'selected_card_ids':card_ids})
                    if self.settings.memory_backend=='scoped':
                        latest_retrieval=await self.memory.read(task_id,request_id,card_ids,decision.arguments.get('question',body['text']))
                    else:
                        latest_retrieval=self.memory.read(task_id,request_id,card_ids,decision.arguments.get('question',body['text']))
                    archive_reads+=1;last_result=self.memory.for_primary(latest_retrieval)
                except ValueError as exc:last_result={'error':str(exc)}
            elif decision.action == "handoff":
                if rollovers >= 2:
                    raise BudgetExceeded("Too many handoffs in one request")
                last_result = await self.handoff(task_id, "primary_requested")
                rollovers += 1
        raise RuntimeError("Step limit reached; progress is saved. Inspect results and continue with a new message.")


def short_answer(text,limit):
    if len(text)<=limit:return text,False
    marker='…（已截短）';room=max(0,limit-len(marker));prefix=text[:room]
    boundary=max(prefix.rfind(char) for char in '。！？\n')
    if boundary>=room//2:prefix=prefix[:boundary+1]
    return (prefix+marker)[:limit],True
