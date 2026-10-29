from __future__ import annotations

import ast
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import uuid

from .store import Conflict, dump, now


TOOL_HELP = {
    "workspace.list": '{"path":"."} 列出任务工作目录',
    "workspace.read": '{"path":"notes.txt"} 读取UTF-8文件',
    "workspace.write": '{"path":"notes.txt","text":"内容"} 原子写入文本',
    "calculator": '{"expression":"19*27"} 精确计算',
    "world.observe": '{} 观察模拟环境，返回版本号',
    "world.open_door": '{"expected_version":1} 打开模拟环境A/B之间的门',
    "world.move": '{"room":"B","expected_version":2} 移动到模拟房间A或B',
    "world.pick": '{"expected_version":3} 拿起所在房间的模拟杯子',
}


def calculate(expression):
    if not isinstance(expression, str) or len(expression) > 200:
        raise ValueError("Expression is too long")
    tree = ast.parse(expression, mode="eval")
    if len(list(ast.walk(tree))) > 60:
        raise ValueError("Expression is too complex")
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                  ast.Div: operator.truediv, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in operations:
            value = operations[type(node.op)](evaluate(node.left), evaluate(node.right))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            base, exponent = evaluate(node.left), evaluate(node.right)
            if abs(exponent) > 8 or abs(base) > 10**6:
                raise ValueError("Exponent exceeds the calculator limit")
            value = base**exponent
        else:
            raise ValueError("Only numeric arithmetic is allowed")
        if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 10**15:
            raise ValueError("Result exceeds the calculator limit")
        return value

    return evaluate(tree.body)


class Tools:
    def __init__(self, store):
        self.store = store

    def workspace(self, task_id):
        # IDs originate in the store, never from model-produced filesystem paths.
        self.store.task(task_id)
        root = self.store.root / "workspaces" / task_id
        root.mkdir(parents=True, exist_ok=True)
        return root.resolve()

    def path(self, task_id, relative):
        if not isinstance(relative, str) or len(relative) > 300 or Path(relative).is_absolute():
            raise ValueError("Expected a relative workspace path")
        root = self.workspace(task_id)
        result = (root / relative).resolve()
        if not result.is_relative_to(root):
            raise PermissionError("Path escapes the task workspace")
        return result

    def observe(self, task_id):
        with self.store.db() as db:
            row = db.execute("SELECT * FROM world WHERE task_id=?", (task_id,)).fetchone()
            return {"version": row["version"], "observed_at": row["observed"], **json.loads(row["body"])}

    def execute(self, task_id, window_id, constraint_version, user_event, tool, arguments, role="primary"):
        if role != "primary":
            raise PermissionError("Memory roles cannot use execution tools")
        task = self.store.task(task_id)
        if tool not in TOOL_HELP or tool not in task["allowed_tools"]:
            raise PermissionError("Tool is not authorized for this task")
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object")
        operation_id, previous = self.store.begin_operation(task_id, window_id, constraint_version, user_event, tool, arguments)
        if previous is not None:
            return {**previous, "deduplicated": True, "operation_id": operation_id}
        try:
            result = self._execute(task_id, operation_id, tool, arguments)
            result = {"ok": True, "tool": tool, "value": result, "operation_id": operation_id}
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, PermissionError, Conflict) as exc:
            result = {"ok": False, "tool": tool, "error": str(exc), "operation_id": operation_id}
        event = self.store.finish_operation(task_id, operation_id, result, failed=not result["ok"])
        return {**result, "evidence_ref": event["ref"]}

    def _execute(self, task_id, operation_id, tool, arguments):
        if tool == "calculator":
            return calculate(arguments["expression"])
        if tool.startswith("workspace."):
            path = self.path(task_id, arguments.get("path", "."))
            if tool == "workspace.list":
                names = sorted(p.name for p in path.iterdir())
                return {"entries": names[:100], "truncated": len(names) > 100}
            if tool == "workspace.read":
                if path.stat().st_size > 32000:
                    raise ValueError("File exceeds the text reading limit")
                return {"text": path.read_text(encoding="utf-8")}
            text = arguments["text"]
            if not isinstance(text, str) or len(text) > 12000 or path == self.workspace(task_id):
                raise ValueError("Expected a text file of at most 12000 characters")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name("." + path.name + "." + operation_id + ".tmp")
            try:
                with temporary.open("w", encoding="utf-8") as target:
                    target.write(text)
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            return {"path": str(path.relative_to(self.workspace(task_id))), "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(text.encode()).hexdigest()}
        if tool == "world.observe":
            return self.observe(task_id)
        if type(arguments.get("expected_version")) is not int:
            raise ValueError("expected_version must be an integer world version")
        with self.store.db(True) as db:
            row = db.execute("SELECT * FROM world WHERE task_id=?", (task_id,)).fetchone()
            if arguments.get("expected_version") != row["version"]:
                raise Conflict("World changed: observe again before acting")
            state = json.loads(row["body"])
            if tool == "world.open_door":
                state["door_open"] = True
            elif tool == "world.move":
                room = arguments["room"]
                if room not in ("A", "B") or not state["door_open"]:
                    raise ValueError("Move requires an open door and destination A or B")
                state["room"] = room
            elif tool == "world.pick":
                if state["holding_cup"] or state["cup_room"] != state["room"]:
                    raise ValueError("The cup must be present in the current room")
                state["holding_cup"], state["cup_room"] = True, None
            else:
                raise PermissionError("Unknown world operation")
            state["last_operation_id"] = operation_id
            db.execute("UPDATE world SET version=version+1,body=?,observed=? WHERE task_id=?", (dump(state), now(), task_id))
            db.execute("INSERT INTO world_versions SELECT * FROM world WHERE task_id=?", (task_id,))
        return self.observe(task_id)

    def reconcile(self, task_id, operation_id):
        operation = next((op for op in self.store.operations(task_id) if op["id"] == operation_id), None)
        if not operation or operation["status"] != "needs_verification":
            raise ValueError("Operation does not require verification")
        arguments, tool = operation["arguments"], operation["tool"]
        if tool == "workspace.write":
            path = self.path(task_id, arguments["path"])
            if path.is_file() and path.read_text(encoding="utf-8") == arguments["text"]:
                value = {"path": arguments["path"], "verified_after_restart": True}
            else:
                raise Conflict("Cannot prove this write completed; no automatic retry was made")
        elif tool.startswith("world.") and self.observe(task_id).get("last_operation_id") == operation_id:
            value = self.observe(task_id)
        else:
            raise Conflict("Outcome is uncertain; inspect the external state before retrying")
        result = {"ok": True, "tool": tool, "value": value, "verified_after_restart": True, "operation_id": operation_id}
        self.store.finish_operation(task_id, operation_id, result)
        return result
