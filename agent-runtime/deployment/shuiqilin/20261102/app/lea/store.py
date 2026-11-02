from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


def uid(prefix):
    return prefix + "_" + uuid.uuid4().hex[:16]


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as target:
            target.write(dump(value))
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            descriptor = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


class Conflict(RuntimeError):
    pass


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "memory.sqlite3"
        with self.db() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS tasks(
              id TEXT PRIMARY KEY, goal TEXT NOT NULL, status TEXT NOT NULL,
              window_id TEXT NOT NULL, version INTEGER NOT NULL, constraint_version INTEGER NOT NULL,
              constraints TEXT NOT NULL, criteria TEXT NOT NULL, allowed_tools TEXT NOT NULL,
              state TEXT NOT NULL, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS windows(
              id TEXT PRIMARY KEY, task_id TEXT NOT NULL, status TEXT NOT NULL,
              created TEXT NOT NULL, closed TEXT, through_seq INTEGER DEFAULT 0, handoff TEXT);
            CREATE TABLE IF NOT EXISTS events(
              seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL, task_id TEXT NOT NULL,
              window_id TEXT NOT NULL, role TEXT NOT NULL, kind TEXT NOT NULL,
              body TEXT NOT NULL, created TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS events_task ON events(task_id,seq);
            CREATE INDEX IF NOT EXISTS events_window ON events(window_id,seq);
            CREATE TRIGGER IF NOT EXISTS immutable_event_update BEFORE UPDATE ON events
              BEGIN SELECT RAISE(ABORT,'Raw events are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_event_delete BEFORE DELETE ON events
              BEGIN SELECT RAISE(ABORT,'Raw events are immutable'); END;
            CREATE TABLE IF NOT EXISTS state_versions(
              task_id TEXT NOT NULL, version INTEGER NOT NULL, body TEXT NOT NULL,
              PRIMARY KEY(task_id,version));
            CREATE TABLE IF NOT EXISTS constraints(
              task_id TEXT NOT NULL, version INTEGER NOT NULL, body TEXT NOT NULL,
              PRIMARY KEY(task_id,version));
            CREATE TABLE IF NOT EXISTS operations(
              id TEXT PRIMARY KEY, task_id TEXT NOT NULL, window_id TEXT NOT NULL,
              user_event TEXT NOT NULL, tool TEXT NOT NULL, arguments TEXT NOT NULL,
              status TEXT NOT NULL, result TEXT, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs(
              id TEXT PRIMARY KEY, task_id TEXT NOT NULL, kind TEXT NOT NULL, request_key TEXT UNIQUE,
              body TEXT NOT NULL, status TEXT NOT NULL, result TEXT, error TEXT, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS queries(
              id TEXT PRIMARY KEY, task_id TEXT NOT NULL, task_version INTEGER NOT NULL,
              status TEXT NOT NULL, body TEXT NOT NULL, result TEXT, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS world(
              task_id TEXT PRIMARY KEY, version INTEGER NOT NULL, body TEXT NOT NULL, observed TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS world_versions(
              task_id TEXT NOT NULL, version INTEGER NOT NULL, body TEXT NOT NULL, observed TEXT NOT NULL,
              PRIMARY KEY(task_id,version));
            INSERT OR IGNORE INTO world_versions SELECT * FROM world;
            """)

    @contextmanager
    def db(self, write=False):
        db = sqlite3.connect(self.path, timeout=20)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _task(db, task_id):
        row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        result = dict(row)
        for key in ("constraints", "criteria", "allowed_tools", "state"):
            result[key] = json.loads(result[key])
        return result

    def task(self, task_id):
        with self.db() as db:
            return self._task(db, task_id)

    def tasks(self):
        with self.db() as db:
            ids = [r[0] for r in db.execute("SELECT id FROM tasks ORDER BY created DESC")]
            return [self._task(db, task_id) for task_id in ids]

    @staticmethod
    def _event(db, task, role, kind, body):
        event_id = uid("e")
        timestamp = now()
        cursor = db.execute("INSERT INTO events(id,task_id,window_id,role,kind,body,created) VALUES(?,?,?,?,?,?,?)",
                            (event_id, task["id"], task["window_id"], role, kind, dump(body), timestamp))
        return {"id": event_id, "seq": cursor.lastrowid, "task_id": task["id"], "window_id": task["window_id"],
                "role": role, "kind": kind, "body": body, "created": timestamp,
                "ref": task["window_id"] + ":" + event_id}

    def append(self, task_id, role, kind, body):
        with self.db(True) as db:
            return self._event(db, self._task(db, task_id), role, kind, body)

    def create_task(self, goal, constraints, criteria, allowed_tools):
        task_id, window_id = uid("task"), uid("w")
        state = {"summary": "", "completed": [], "next_steps": [], "recorded_through": 0}
        with self.db(True) as db:
            db.execute("INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                       (task_id, goal, "active", window_id, 1, 1, dump(constraints), dump(criteria), dump(allowed_tools), dump(state), now()))
            db.execute("INSERT INTO windows(id,task_id,status,created) VALUES(?,?,?,?)", (window_id, task_id, "active", now()))
            task = self._task(db, task_id)
            self._event(db, task, "user", "task_created", {"goal": goal, "constraints": constraints, "criteria": criteria})
            db.execute("INSERT INTO constraints VALUES(?,?,?)", (task_id, 1, dump({"rules": constraints, "source": "task_created"})))
            db.execute("INSERT INTO state_versions VALUES(?,?,?)", (task_id, 1, dump(task)))
            db.execute("INSERT INTO world VALUES(?,?,?,?)", (task_id, 1, dump({"mode": "simulation", "room": "A", "door_open": False, "cup_room": "B", "holding_cup": False}), now()))
            db.execute("INSERT INTO world_versions SELECT * FROM world WHERE task_id=?", (task_id,))
        self.export(task_id)
        return self.task(task_id)

    @staticmethod
    def event_text(event):
        body = event["body"]
        return body["text"] if isinstance(body, dict) and isinstance(body.get("text"), str) else dump(body)

    @staticmethod
    def _decode_event(row):
        event = dict(row)
        event["body"] = json.loads(event["body"])
        event["ref"] = event["window_id"] + ":" + event["id"]
        event["text"] = Store.event_text(event)
        return event

    def events(self, task_id, after=0, limit=10000, tail=False):
        with self.db() as db:
            self._task(db, task_id)
            order = "DESC" if tail else "ASC"
            events = [self._decode_event(r) for r in db.execute(f"SELECT * FROM events WHERE task_id=? AND seq>? ORDER BY seq {order} LIMIT ?", (task_id, after, limit))]
            return list(reversed(events)) if tail else events

    def resolve(self, task_id, ref):
        window_id, event_id = ref.split(":", 1)
        with self.db() as db:
            row = db.execute("SELECT * FROM events WHERE task_id=? AND window_id=? AND id=?", (task_id, window_id, event_id)).fetchone()
            if row is None:
                raise ValueError("Evidence reference is outside this task or does not exist")
            return self._decode_event(row)

    def windows(self, task_id):
        with self.db() as db:
            return [dict(row) for row in db.execute("SELECT * FROM windows WHERE task_id=? ORDER BY created", (task_id,))]

    def update_constraints(self, task_id, rules, expected_version):
        with self.db(True) as db:
            task = self._task(db, task_id)
            if task["constraint_version"] != expected_version:
                raise Conflict("Constraints changed; reload the current version")
            version = expected_version + 1
            event = self._event(db, task, "user", "constraint_update", {"rules": rules, "version": version})
            db.execute("INSERT INTO constraints VALUES(?,?,?)", (task_id, version, dump({"rules": rules, "source_ref": event["ref"]})))
            db.execute("UPDATE tasks SET constraints=?,constraint_version=? WHERE id=?", (dump(rules), version, task_id))
        self.export(task_id)
        return self.task(task_id)

    def commit_handoff(self, task_id, window_id, through_seq, handoff, expected_constraint_version=None):
        task = self.task(task_id)
        if expected_constraint_version is not None and task["constraint_version"] != expected_constraint_version:
            raise Conflict("Effective constraints changed while the handbook was generated")
        if task["window_id"] != window_id:
            raise Conflict("An obsolete primary window cannot commit a handoff")
        with self.db() as db:
            if not db.execute("SELECT 1 FROM events WHERE task_id=? AND seq=?", (task_id, through_seq)).fetchone():
                raise ValueError("Handoff snapshot does not point to an existing task event")
        refs = [ref for finding in handoff["decisions"] + handoff["history_entries"] for ref in finding["evidence_refs"]]
        for ref in refs:
            if self.resolve(task_id, ref)["seq"] > through_seq:
                raise ValueError("Handoff cites an event beyond its snapshot")
        operations = self.operations(task_id)
        with self.db() as db:
            world = dict(db.execute("SELECT * FROM world WHERE task_id=?", (task_id,)).fetchone())
        handbook = {"schema_version": 1, "task_id": task_id, "window_id": window_id, "recorded_through_event": through_seq,
                    "constraint_version": task["constraint_version"], "created_at": now(),
                    "handoff": {**handoff, "goal": task["goal"], "completion_criteria": task["criteria"],
                                "environment_ref": f"world:{task_id}:v{world['version']}", "environment_observed_at": world["observed"],
                                "pending_operations": [op for op in operations if op["status"] not in ("completed", "failed")],
                                "pending_queries": [{"job_id": job["id"], "status": job["status"], "question": job["body"]["question"]}
                                                    for job in self.jobs(task_id) if job["kind"] == "query" and job["status"] in ("queued", "running")],
                                "completed_operations": [op["id"] for op in operations if op["status"] == "completed"]},
                    "history_index": {"topics": handoff["topics"], "entities": handoff["entities"], "entries": handoff["history_entries"]}}
        # The handbook is durable before the transaction switches window ownership.
        atomic_json(self.root / "tasks" / task_id / "windows" / window_id / "handoff.json", handbook)
        new_window = uid("w")
        state = {**handoff, "recorded_through": through_seq, "previous_window": window_id}
        with self.db(True) as db:
            current = self._task(db, task_id)
            if current["window_id"] != window_id or current["constraint_version"] != task["constraint_version"]:
                raise Conflict("Task or constraints changed during handoff; regenerate it")
            self._event(db, current, "runtime", "handoff_committed", {"next_window": new_window, "through_seq": through_seq})
            db.execute("UPDATE windows SET status='closed',closed=?,through_seq=?,handoff=? WHERE id=?", (now(), through_seq, dump(handbook), window_id))
            db.execute("INSERT INTO windows(id,task_id,status,created) VALUES(?,?,?,?)", (new_window, task_id, "active", now()))
            db.execute("UPDATE tasks SET window_id=?,version=version+1,state=? WHERE id=?", (new_window, dump(state), task_id))
            db.execute("INSERT INTO state_versions VALUES(?,?,?)", (task_id, current["version"] + 1, dump(self._task(db, task_id))))
        self.export(task_id)
        return self.task(task_id)

    def operations(self, task_id):
        with self.db() as db:
            return [{**dict(row), "arguments": json.loads(row["arguments"]), "result": json.loads(row["result"]) if row["result"] else None}
                    for row in db.execute("SELECT * FROM operations WHERE task_id=? ORDER BY created", (task_id,))]

    def begin_operation(self, task_id, window_id, constraint_version, user_event, tool, arguments, *, observation_fingerprint=None):
        key=[task_id,user_event,tool,arguments]
        if observation_fingerprint is not None:
            if tool not in ('workspace.read','workspace.list','world.observe') or not re.fullmatch(r'[0-9a-f]{64}',observation_fingerprint):
                raise ValueError('Observation scope is reserved for bounded read-only tools')
            key.append({'observation_v1':observation_fingerprint})
        operation_id = "op_" + hashlib.sha256(dump(key).encode()).hexdigest()[:24]
        with self.db(True) as db:
            task = self._task(db, task_id)
            if task["window_id"] != window_id or task["constraint_version"] != constraint_version:
                raise Conflict("Stale execution context: reload current task and constraints")
            existing = db.execute("SELECT * FROM operations WHERE id=?", (operation_id,)).fetchone()
            if existing:
                if existing["status"] in ("completed", "failed"):
                    return operation_id, json.loads(existing["result"])
                if observation_fingerprint is not None:
                    # A newly captured read has no external side effect. The same
                    # snapshot can safely finish an interrupted read operation.
                    db.execute("UPDATE operations SET status='running' WHERE id=?",(operation_id,))
                    self._event(db,task,'runtime','observation_restarted',{'operation_id':operation_id,'observation_fingerprint':observation_fingerprint})
                    return operation_id,None
                raise Conflict("Operation has an uncertain result; verify it before retrying")
            if observation_fingerprint is not None:
                old=db.execute("SELECT id FROM operations WHERE task_id=? AND user_event=? AND tool=? AND arguments=? AND status IN ('running','needs_verification')",
                               (task_id,user_event,tool,dump(arguments))).fetchall()
                for row in old:
                    result={'ok':False,'tool':tool,'operation_id':row['id'],'error':'Interrupted read superseded by a fresh observation','superseded_by':operation_id}
                    event=self._event(db,task,'tool','tool_result',result)
                    db.execute("UPDATE operations SET status='failed',result=? WHERE id=?",(dump({**result,'evidence_ref':event['ref']}),row['id']))
            db.execute("INSERT INTO operations VALUES(?,?,?,?,?,?,?,?,?)", (operation_id, task_id, window_id, user_event, tool, dump(arguments), "running", None, now()))
            self._event(db, task, "primary", "tool_request", {"operation_id": operation_id, "tool": tool, "arguments": arguments, **({"observation_fingerprint":observation_fingerprint} if observation_fingerprint else {})})
        return operation_id, None

    def finish_operation(self, task_id, operation_id, result, failed=False):
        with self.db(True) as db:
            event = self._event(db, self._task(db, task_id), "tool", "tool_result", {"operation_id": operation_id, **result})
            db.execute("UPDATE operations SET status=?,result=? WHERE id=? AND task_id=?", ("failed" if failed else "completed", dump({**result, "evidence_ref": event["ref"]}), operation_id, task_id))
        return event

    def add_job(self, task_id, kind, body, request_key=None):
        job_id = uid("job")
        with self.db(True) as db:
            task = self._task(db, task_id)
            key = f"{task_id}:{request_key}" if request_key else None
            if key:
                row = db.execute("SELECT id FROM jobs WHERE request_key=?", (key,)).fetchone()
                if row:
                    return self.job(row[0])
            if kind == "message":
                event = self._event(db, task, "user", "message", {"text": body["text"]})
                body = {**body, "user_event": event["id"], "event_ref": event["ref"]}
                db.execute("UPDATE tasks SET status='active' WHERE id=?", (task_id,))
            db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?)", (job_id, task_id, kind, key, dump(body), "queued", None, None, now()))
        return self.job(job_id)

    def job(self, job_id):
        with self.db() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            result = dict(row)
            result["body"] = json.loads(result["body"])
            result["result"] = json.loads(result["result"]) if result["result"] else None
            return result

    def jobs(self, task_id):
        with self.db() as db:
            ids = [r[0] for r in db.execute("SELECT id FROM jobs WHERE task_id=? ORDER BY created DESC LIMIT 30", (task_id,))]
        return [self.job(job_id) for job_id in ids]

    def next_job(self,kind=None):
        with self.db(True) as db:
            where=(" AND kind='query' AND COALESCE(json_extract(body,'$.verify'),0)!=1" if kind=='query'
                   else (" AND kind='query' AND json_extract(body,'$.verify')=1" if kind=='verification'
                         else (" AND kind!='query'" if kind=='execution' else '')))
            row = db.execute("SELECT id FROM jobs WHERE status='queued'"+where+" ORDER BY CASE kind WHEN 'query' THEN 1 ELSE 0 END,created LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='running' WHERE id=?", (row[0],))
        return self.job(row[0])

    def set_job(self, job_id, status, result=None, error=None):
        with self.db(True) as db:
            db.execute("UPDATE jobs SET status=?,result=?,error=? WHERE id=?", (status, dump(result) if result is not None else None, error, job_id))

    def recover(self):
        with self.db(True) as db:
            db.execute("UPDATE jobs SET status='queued',error='Resuming after process restart' WHERE status='running'")
            db.execute("UPDATE operations SET status='needs_verification' WHERE status='running'")
            db.execute("UPDATE queries SET status='interrupted' WHERE status='running'")
        for task in self.tasks():
            self.export(task["id"])

    def export(self, task_id):
        task = self.task(task_id)
        directory = self.root / "tasks" / task_id
        atomic_json(directory / "current_state.json", task)
        for window in self.windows(task_id):
            path = directory / "windows" / window["id"] / "conversation.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            marker = path.with_suffix(".projection.json")
            with self.db() as db:
                latest = db.execute("SELECT COALESCE(MAX(seq),0) FROM events WHERE window_id=?", (window["id"],)).fetchone()[0]
                try:
                    if path.exists() and json.loads(marker.read_text())["through_seq"] == latest:
                        continue
                except (OSError, ValueError, KeyError):
                    pass
                temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
                try:
                    # Stream complete windows, independently of the retrieval scan cap.
                    with temporary.open("w", encoding="utf-8") as target:
                        for row in db.execute("SELECT * FROM events WHERE window_id=? AND seq<=? ORDER BY seq", (window["id"], latest)):
                            target.write(dump(self._decode_event(row)) + "\n")
                        target.flush()
                        os.fsync(target.fileno())
                    os.replace(temporary, path)
                    atomic_json(marker, {"through_seq": latest})
                finally:
                    temporary.unlink(missing_ok=True)
        atomic_json(directory / "operation_ledger.json", self.operations(task_id))
