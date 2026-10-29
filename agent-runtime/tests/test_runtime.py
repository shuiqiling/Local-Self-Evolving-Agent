import asyncio
from dataclasses import replace
import json
import sqlite3
import threading
import time

from fastapi.testclient import TestClient
import pytest

from lea.app import create_app
from lea.config import Settings
from lea.memory import Memory
from lea.model import BudgetExceeded, InferenceQueue
from lea.runtime import Runtime
from lea.schemas import Decision, Handoff, Reading, SearchPlan, Synthesis
from lea.store import Conflict, Store
from lea.tools import TOOL_HELP, Tools, calculate


def handbook(refs=None):
    return {"summary": "保留进度", "completed": [], "decisions": [], "next_steps": ["继续"],
            "open_questions": [], "topics": ["测试"], "entities": [],
            "history_entries": [{"statement": "历史事实", "evidence_refs": refs}] if refs else []}


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path)


@pytest.fixture
def task(store):
    return store.create_task("验证连续性", ["仅使用任务工作目录"], ["保存证据"], list(TOOL_HELP))


def test_immutable_events_and_cross_task_refs(store, task):
    event = store.append(task["id"], "user", "message", {"text": "原始事实"})
    for sql in ("UPDATE events SET body='{}'", "DELETE FROM events"):
        with pytest.raises(sqlite3.IntegrityError), store.db(True) as db:
            db.execute(sql)
    other = store.create_task("另一任务", [], [], [])
    with pytest.raises(ValueError):
        store.resolve(other["id"], event["ref"])
    assert store.resolve(task["id"], event["ref"])["text"] == "原始事实"


def test_long_archive_export_is_not_cut_off_by_search_limits(store, task):
    with store.db(True) as db:
        for index in range(10005):
            store._event(db, task, "user", "message", {"text": str(index)})
    store.export(task["id"])
    path = store.root / 'tasks' / task['id'] / 'windows' / task['window_id'] / 'conversation.jsonl'
    lines = path.read_text(encoding='utf-8').splitlines()
    assert len(lines) == 10006
    assert json.loads(lines[-1])['body']['text'] == '10004'
    assert store.events(task['id'], limit=2, tail=True)[-1]['body']['text'] == '10004'


def test_handoff_reload_late_messages_and_constraints(store, task):
    fact = store.append(task["id"], "user", "message", {"text": "交接前的事实"})
    late = store.append(task["id"], "user", "message", {"text": "交接期间到达的修正"})
    new = store.commit_handoff(task["id"], task["window_id"], fact["seq"], handbook([fact["ref"]]))
    assert new["version"] == 2 and new["window_id"] != task["window_id"]
    assert new["constraints"] == task["constraints"]
    restored = Store(store.root)
    restored.recover()
    tail = restored.events(task["id"], after=new["state"]["recorded_through"])
    assert late["id"] in [e["id"] for e in tail]
    with pytest.raises(Conflict):
        restored.begin_operation(task["id"], task["window_id"], 1, "request", "calculator", {"expression": "1+1"})


def test_handoff_failure_does_not_switch(store, task, monkeypatch):
    event = store.append(task["id"], "user", "message", {"text": "事实"})
    def fail(*args):
        raise OSError("disk full")
    monkeypatch.setattr("lea.store.atomic_json", fail)
    with pytest.raises(OSError):
        store.commit_handoff(task["id"], task["window_id"], event["seq"], handbook())
    assert store.task(task["id"])["window_id"] == task["window_id"]
    assert store.task(task["id"])["version"] == 1


def test_bad_handoff_references_and_future_cutoff(store, task):
    event = store.events(task["id"])[0]
    with pytest.raises(ValueError):
        store.commit_handoff(task["id"], task["window_id"], event["seq"], handbook(["w_missing:e_missing"]))
    with pytest.raises(ValueError):
        store.commit_handoff(task["id"], task["window_id"], event["seq"] + 100, handbook())
    assert store.task(task["id"])["version"] == 1


def test_operation_idempotency_survives_handoff(store, task):
    tools = Tools(store)
    args = {"path": "result.txt", "text": "513"}
    first = tools.execute(task["id"], task["window_id"], 1, "request", "workspace.write", args)
    path = tools.path(task["id"], "result.txt")
    path.write_text("manual correction")
    through = store.events(task["id"])[-1]["seq"]
    new = store.commit_handoff(task["id"], task["window_id"], through, handbook())
    replay = tools.execute(task["id"], new["window_id"], 1, "request", "workspace.write", dict(reversed(list(args.items()))))
    assert replay["deduplicated"] and replay["operation_id"] == first["operation_id"]
    assert path.read_text() == "manual correction"
    assert len(store.operations(task["id"])) == 1


def test_interrupted_write_is_verified_not_reexecuted(store, task):
    tools = Tools(store)
    args = {"path": "result.txt", "text": "saved before crash"}
    operation_id, _ = store.begin_operation(task["id"], task["window_id"], 1, "request", "workspace.write", args)
    tools.path(task["id"], "result.txt").write_text(args["text"])
    store.recover()
    with pytest.raises(Conflict):
        tools.execute(task["id"], task["window_id"], 1, "request", "workspace.write", args)
    assert tools.reconcile(task["id"], operation_id)["verified_after_restart"]
    assert tools.execute(task["id"], task["window_id"], 1, "request", "workspace.write", args)["deduplicated"]


def test_constraint_version_guard_and_role_permissions(store, task):
    store.update_constraints(task["id"], ["修正后的规则"], 1)
    with pytest.raises(Conflict):
        store.update_constraints(task["id"], ["过时规则"], 1)
    tools = Tools(store)
    with pytest.raises(Conflict):
        tools.execute(task["id"], task["window_id"], 1, "request", "calculator", {"expression": "1+1"})
    with pytest.raises(PermissionError):
        tools.execute(task["id"], task["window_id"], 2, "request", "calculator", {"expression": "1+1"}, role="reader")
    with pytest.raises(PermissionError):
        tools.execute(task["id"], task["window_id"], 2, "request", "shell", {"command": "id"})


def test_workspace_and_calculator_guards(store, task):
    tools = Tools(store)
    with pytest.raises(PermissionError):
        tools.path(task["id"], "../../outside.txt")
    assert calculate("19*27") == 513
    for expression in ("__import__('os').system('id')", "2**999999", "'a'*100", "[x for x in range(3)]"):
        with pytest.raises(ValueError):
            calculate(expression)


def test_world_stale_action_and_recovery(store, task):
    tools = Tools(store)
    args = (task["id"], task["window_id"], 1)
    assert tools.execute(*args, "one", "world.open_door", {"expected_version": 1})["ok"]
    assert not tools.execute(*args, "two", "world.move", {"room": "B", "expected_version": 1})["ok"]
    assert tools.observe(task["id"])["version"] == 2
    assert tools.execute(*args, "three", "world.move", {"room": "B", "expected_version": 2})["ok"]
    assert tools.execute(*args, "four", "world.pick", {"expected_version": 3})["ok"]
    assert Tools(Store(store.root)).observe(task["id"])["holding_cup"]
    with store.db() as db:
        historical = json.loads(db.execute("SELECT body FROM world_versions WHERE task_id=? AND version=1", (task["id"],)).fetchone()[0])
        assert not historical["door_open"] and historical["room"] == "A"


class FakeModel:
    def __init__(self, bad_quote=False, delay=0):
        self.roles, self.bad_quote, self.delay = [], bad_quote, delay
        self.started = threading.Event()

    async def health(self):
        return {"ready": True}

    async def close(self):
        pass

    async def tokens(self, prompt):
        return 100

    async def call(self, task_id, role, prompt, schema, **kwargs):
        self.roles.append(role)
        self.started.set()
        if self.delay:
            await asyncio.sleep(self.delay)
        payload = json.loads(prompt[1]["content"])
        if schema is Decision:
            return Decision(action="finish", response="已回复", tool="", arguments={})
        if schema is Handoff:
            return Handoff(**handbook())
        if schema is SearchPlan:
            return SearchPlan(terms=["通行"], questions=[payload["question"]])
        if schema is Reading:
            record = next((r for r in payload["records"] if "通行" in r["text"]), payload["records"][0])
            quote = "不存在的原文" if self.bad_quote else record["text"]
            return Reading(status="relevant", evidence=[{"event_ref": record["event_ref"], "quote": quote, "relevance": "条件"}], suggested_followups=[])
        if schema is Synthesis:
            return Synthesis(status="supported", findings=[{"statement": "保留原始条件", "evidence_refs": [payload["evidence"][0]["event_ref"]]}], inferences=[], conflicts=[], unresolved_questions=[])
        raise AssertionError(schema)


@pytest.mark.parametrize("bad_quote", [False, True])
def test_three_roles_raw_search_and_quote_validation(store, task, bad_quote):
    fact = store.append(task["id"], "user", "message", {"text": "通行条件：门在上午打开，但未经现场确认不能通行。"})
    store.commit_handoff(task["id"], task["window_id"], fact["seq"], handbook())  # Deliberately omit this fact from the index.
    model = FakeModel(bad_quote=bad_quote)
    result = asyncio.run(Memory(store, model, Settings(data_dir=store.root)).query(task["id"], "查找通行条件"))
    assert model.roles[:2] == ["coordinator", "reader"]
    if bad_quote:
        assert not result["evidence"] and result["warnings"]
    else:
        assert model.roles[-1] == "coordinator"
        assert result["evidence"][0]["event_ref"] == fact["ref"]
        assert result["evidence"][0]["source_type"] == "user_statement"
        assert "未经现场确认" in result["evidence"][0]["quote"]


def test_priority_queue_and_waiter_cancellation():
    async def run():
        queue, order = InferenceQueue(), []
        async def wait(priority):
            async with queue.slot(priority):
                order.append(priority)
        async with queue.slot(0):
            canceled = asyncio.create_task(wait(9))
            tasks = [asyncio.create_task(wait(p)) for p in (10, 5, 0)]
            await asyncio.sleep(0)
            canceled.cancel()
            await asyncio.gather(canceled, return_exceptions=True)
        await asyncio.gather(*tasks)
        assert order == [0, 5, 10]
    asyncio.run(run())


def test_reader_success_is_preserved_if_synthesis_fails(store, task):
    store.append(task["id"], "user", "message", {"text": "通行需要先确认门已打开。"})
    class FailingSynthesis(FakeModel):
        async def call(self, task_id, role, prompt, schema, **kwargs):
            if schema is Synthesis:
                raise TimeoutError("model deadline")
            return await super().call(task_id, role, prompt, schema, **kwargs)
    result = asyncio.run(Memory(store, FailingSynthesis(), Settings(data_dir=store.root)).query(task["id"], "通行条件"))
    assert result["answer_status"] == "partial"
    assert result["evidence"] and result["inspected_scope"]
    assert result["error"] == "model deadline"


def test_model_conclusions_are_searchable_but_never_relabelled_as_observations(store, task):
    event = store.append(task["id"], "primary", "assistant_reply", {"text": "通行建议：我推测可以先观察门。"})
    result = asyncio.run(Memory(store, FakeModel(), Settings(data_dir=store.root)).query(task["id"], "通行建议"))
    assert result["evidence"][0]["event_ref"] == event["ref"]
    assert result["evidence"][0]["source_type"] == "model_conclusion"


def test_retrieval_prefers_original_sources_and_excludes_its_own_plan(store, task):
    original = store.append(task['id'],'user','message',{'text':'通行条件是先确认门已打开。'})
    for _ in range(8):
        store.append(task['id'],'primary','assistant_reply',{'text':'通行条件、通行建议、历史记录：我猜可以直接通行。'})
    plan = store.append(task['id'],'primary','primary_decision',{'action':'recall','response':'通行条件、通行建议、历史记录：待查证的猜测','arguments':{'question':'通行条件'}})
    class CheckingSources(FakeModel):
        assigned = []
        async def call(self, task_id, role, prompt, schema, **kwargs):
            if schema is Reading:
                self.assigned.extend(json.loads(prompt[1]['content'])['records'])
            return await super().call(task_id,role,prompt,schema,**kwargs)
    model = CheckingSources()
    result = asyncio.run(Memory(store,model,Settings(data_dir=store.root)).query(task['id'],'通行条件、通行建议、历史记录'))
    assert result['evidence'][0]['event_ref']==original['ref']
    assert not any('待查证的猜测' in record['text'] for record in model.assigned)


def test_coordinator_dispatches_each_part_of_a_compound_question(store, task):
    store.append(task['id'], 'user', 'message', {'text':'项目代号是测试鸟；适用温度低于30摄氏度。'})
    class AtomicQuestions(FakeModel):
        async def call(self, task_id, role, prompt, schema, **kwargs):
            payload = json.loads(prompt[1]['content'])
            if schema is SearchPlan:
                return SearchPlan(terms=['代号','温度'], questions=['项目代号是什么？','适用温度条件是什么？'])
            if schema is Reading:
                record = next(r for r in payload['records'] if '测试鸟' in r['text'])
                quote = '项目代号是测试鸟' if '代号' in payload['question'] else '适用温度低于30摄氏度'
                return Reading(status='relevant', evidence=[{'event_ref':record['event_ref'],'quote':quote,'relevance':payload['question']}], suggested_followups=[])
            return await super().call(task_id, role, prompt, schema, **kwargs)
    result = asyncio.run(Memory(store,AtomicQuestions(),Settings(data_dir=store.root)).query(task['id'],'项目代号和适用温度是什么？'))
    quotes = '\n'.join(e['quote'] for e in result['evidence'])
    assert '测试鸟' in quotes and '低于30' in quotes
    assert len({scope['question'] for scope in result['inspected_scope']}) == 2


def test_primary_recall_dispatches_all_three_levels_and_keeps_the_original_condition(tmp_path):
    class RecallingModel(FakeModel):
        final_payload = None
        async def call(self, task_id, role, prompt, schema, **kwargs):
            if schema is Decision:
                self.roles.append(role)
                payload = json.loads(prompt[1]['content'])
                if payload['last_result'] is None:
                    return Decision(action='recall', response='', tool='', arguments={'question':'通行条件'})
                self.final_payload = payload
                return Decision(action='finish', response='已找到带来源的通行条件。', tool='', arguments={})
            return await super().call(task_id, role, prompt, schema, **kwargs)
    model = RecallingModel()
    runtime = Runtime(Settings(data_dir=tmp_path,memory_backend="llm"), model=model)
    task = runtime.store.create_task('恢复任务', [], [], [])
    runtime.store.append(task['id'],'user','message',{'text':'通行条件：未经现场确认不能通行。'})
    job = runtime.store.add_job(task['id'],'message',{'text':'查询通行条件'})
    result = asyncio.run(runtime.primary(job))
    assert result['retrieval_ids']
    assert model.roles[:2] == ['primary','coordinator']
    assert model.roles[-2:] == ['coordinator','primary']
    assert model.roles[2:-2] and set(model.roles[2:-2]) == {'reader'}
    assert '未经现场确认不能通行' in model.final_payload['last_result']['evidence'][0]['quote']
    assert not any(record['kind']=='retrieval_result' for record in model.final_payload['records'])


def test_large_retrieval_preview_preserves_findings_and_the_full_archive():
    quote = '原文前缀。' * 100 + '仅在温度低于12摄氏度时适用，不可省略条件。'
    result = {'request_id':'q1','answer_status':'partial','scope_complete':False,
              'supported_facts':[{'statement':'仅在温度低于12摄氏度时适用','evidence_refs':['r1']}],
              'evidence':[{'event_ref':'r1','source_type':'user_statement','quote':quote,'relevance':'原始条件','start':0}]*20,
              'inspected_scope':[{'event_ref':'r1','start':0,'end':len(quote)}]*100,
              'uninspected_reading_tasks':5}
    brief = Memory.for_primary(result)
    assert brief['supported_facts']==result['supported_facts']
    assert brief['evidence'][0]['excerpt_only']
    assert brief['evidence'][0]['quote'] in quote
    assert not brief['inspected_scope']['complete']
    assert brief['omitted_evidence_count']==12
    assert len(json.dumps(brief,ensure_ascii=False)) < len(json.dumps(result,ensure_ascii=False))/3
    assert result['evidence'][0]['quote']==quote


def test_repeated_recall_stops_with_visible_literal_evidence(tmp_path):
    class RepeatingModel(FakeModel):
        async def call(self, task_id, role, prompt, schema, **kwargs):
            if schema is Decision:
                self.roles.append(role)
                return Decision(action='recall',response='',tool='',arguments={'question':'通行条件'})
            return await super().call(task_id,role,prompt,schema,**kwargs)
    model=RepeatingModel()
    runtime=Runtime(Settings(data_dir=tmp_path,memory_backend="llm"),model=model)
    task=runtime.store.create_task('查证通行',[],[],[])
    event=runtime.store.append(task['id'],'user','message',{'text':'通行条件：未确认门已打开之前不可通行。'})
    job=runtime.store.add_job(task['id'],'message',{'text':'查证通行条件'})
    result=asyncio.run(runtime.primary(job))
    assert len(result['retrieval_ids'])==1
    assert result['response_mode']=='verified_evidence_fallback'
    assert result['semantic_review_required']
    assert runtime.store.resolve(task['id'],event['ref'])['text'] in result['response'] and event['ref'] in result['response']
    assert not runtime.store.operations(task['id'])
    assert runtime.store.events(task['id'])[-1]['role']=='runtime'


def test_duplicate_finding_is_not_accepted_as_a_conflict(store,task):
    store.append(task['id'],'user','message',{'text':'通行前需要先观察门。'})
    class RepeatingConflict(FakeModel):
        async def call(self,task_id,role,prompt,schema,**kwargs):
            if schema is Synthesis:
                payload=json.loads(prompt[1]['content'])
                item={'statement':'通行前需要先观察门。','evidence_refs':[payload['evidence'][0]['event_ref']]}
                return Synthesis(status='conflicting',findings=[item],conflicts=[item],inferences=[],unresolved_questions=[])
            return await super().call(task_id,role,prompt,schema,**kwargs)
    result=asyncio.run(Memory(store,RepeatingConflict(),Settings(data_dir=store.root)).query(task['id'],'通行条件'))
    assert result['supported_facts'] and not result['conflicts']
    assert result['answer_status']=='partial' and result['warnings']


def test_pending_operations_and_queries_survive_handoff(store, task):
    operation, _ = store.begin_operation(task["id"], task["window_id"], 1, "request", "workspace.write", {"path": "x.txt", "text": "x"})
    query = store.add_job(task["id"], "query", {"question": "查找先前的限制", "exhaustive": False})
    through = store.events(task["id"])[-1]["seq"]
    store.commit_handoff(task["id"], task["window_id"], through, handbook())
    book = json.loads(store.windows(task["id"])[0]["handoff"])["handoff"]
    assert book["pending_operations"][0]["id"] == operation
    assert book["pending_queries"][0]["job_id"] == query["id"]
    assert book["environment_ref"].endswith(":v1")


def test_automatic_handoff_keeps_the_current_request_and_rules(tmp_path):
    class RotatingModel(FakeModel):
        rotated = False
        payloads = []
        async def tokens(self, prompt):
            if prompt[0]["content"].startswith("你是一级执行Agent") and not self.rotated:
                return 9999
            return 100
        async def call(self, task_id, role, prompt, schema, **kwargs):
            self.payloads.append(json.loads(prompt[1]["content"]))
            if schema is Handoff:
                self.rotated = True
            return await super().call(task_id, role, prompt, schema, **kwargs)
    model = RotatingModel()
    runtime = Runtime(Settings(data_dir=tmp_path), model=model)
    task = runtime.store.create_task("保存进度", ["不能越过任务目录"], [], ["calculator"])
    job = runtime.store.add_job(task["id"], "message", {"text": "本轮继续处理这个请求"})
    result = asyncio.run(runtime.primary(job))
    assert result["response"] == "已回复"
    assert model.roles == ["handoff", "primary"]
    assert model.payloads[-1]["current_request"] == job["body"]["text"]
    assert model.payloads[-1]["constraints"] == ["不能越过任务目录"]
    assert runtime.store.task(task["id"])["version"] == 2


def test_compact_handoff_labels_resolve_to_stable_original_references(tmp_path):
    class CitingModel(FakeModel):
        async def call(self, task_id, role, prompt, schema, **kwargs):
            if schema is Handoff:
                payload = json.loads(prompt[1]['content'])
                return Handoff(**handbook([payload['records'][0]['ref']]))
            return await super().call(task_id, role, prompt, schema, **kwargs)
    runtime = Runtime(Settings(data_dir=tmp_path), model=CitingModel())
    task = runtime.store.create_task('保存来源', [], [], [])
    original = runtime.store.append(task['id'], 'user', 'message', {'text':'可追溯的事实'})
    asyncio.run(runtime.handoff(task['id'], 'test_labels'))
    book = json.loads(runtime.store.windows(task['id'])[0]['handoff'])
    ref = book['history_index']['entries'][0]['evidence_refs'][0]
    assert ref == original['ref']
    assert runtime.store.resolve(task['id'], ref)['text'] == '可追溯的事实'


def test_user_correction_during_handoff_invalidates_the_snapshot(tmp_path):
    model = FakeModel()
    runtime = Runtime(Settings(data_dir=tmp_path), model=model)
    task = runtime.store.create_task("保存进度", ["旧规则"], [], [])
    original_call = model.call
    async def correcting_call(*args, **kwargs):
        runtime.store.update_constraints(task["id"], ["新规则立即生效"], 1)
        return await original_call(*args, **kwargs)
    model.call = correcting_call
    with pytest.raises(Conflict):
        asyncio.run(runtime.handoff(task["id"], "test_race"))
    assert runtime.store.task(task["id"])["window_id"] == task["window_id"]
    assert runtime.store.task(task["id"])["constraints"] == ["新规则立即生效"]


def test_api_idempotence_cancel_and_origin(tmp_path):
    model = FakeModel(delay=20)
    with TestClient(create_app(Settings(data_dir=tmp_path,memory_backend="llm"), model=model)) as client:
        assert client.post("/api/tasks", headers={"Origin": "https://outside.example"}, json={"goal": "denied"}).status_code == 403
        task = client.post("/api/tasks", json={"goal": "测试", "allowed_tools": ["calculator"]}).json()
        path = "/api/tasks/" + task["id"] + "/messages"
        job = client.post(path, json={"text": "你好", "request_id": "same-request"}).json()
        assert client.post(path, json={"text": "你好", "request_id": "same-request"}).json()["id"] == job["id"]
        assert model.started.wait(3)
        assert client.post("/api/jobs/" + job["id"] + "/cancel", json={}).status_code == 200
        time.sleep(0.05)
        assert client.get("/api/jobs/" + job["id"]).json()["status"] == "canceled"
        events = client.get("/api/tasks/" + task["id"] + "/events").json()
        assert sum(e["kind"] == "message" for e in events) == 1
        assert not any(e["kind"] == "assistant_reply" for e in events)
