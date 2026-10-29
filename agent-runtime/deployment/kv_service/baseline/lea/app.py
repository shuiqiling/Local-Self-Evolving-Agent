from contextlib import asynccontextmanager
import json
import hashlib
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from .config import Settings
from .runtime import Runtime
from .schemas import ConstraintUpdate, Message, Query, TaskCreate, MemoryRead
from .store import Conflict
from .tools import TOOL_HELP


def create_app(settings=None, model=None, embedder=None):
    runtime = Runtime(settings or Settings(), model=model,embedder=embedder)
    source_root = Path(__file__).parent
    build_hash = hashlib.sha256(b"".join(path.name.encode() + path.read_bytes() for path in
        sorted([*source_root.glob("*.py"), *source_root.glob("static/*.html")]))).hexdigest()[:16]

    @asynccontextmanager
    async def lifespan(app):
        await runtime.start()
        yield
        await runtime.close()

    app = FastAPI(title="Local Evolving Agent", version="0.1.0", lifespan=lifespan)
    app.state.runtime = runtime

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin:
            if urlsplit(origin).netloc != request.headers.get("host"):
                return JSONResponse({"error": "Cross-origin mutations are not allowed"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(KeyError)
    async def not_found(request, exc):
        return JSONResponse({"error": "Record not found"}, status_code=404)

    @app.exception_handler(Conflict)
    async def conflict(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=409)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=400)

    @app.exception_handler(PermissionError)
    async def denied(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=403)

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).parent / "static/index.html")

    @app.get("/api/health")
    async def health():
        preparation = {}
        status_file = runtime.settings.model_status_file
        if status_file and Path(status_file).is_file():
            try:
                preparation = json.loads(Path(status_file).read_text())
            except (OSError, ValueError):
                preparation = {"state": "unknown"}
        return {"status": "ok", "build": build_hash, "model": await runtime.model.health(), "precision": runtime.settings.precision, "preparation": preparation,
                "working_tokens": runtime.settings.working_tokens, "backend_context": runtime.settings.backend_context,
                "memory_queries_per_message": runtime.settings.max_memory_queries,
                "answer_max_chars":runtime.settings.answer_max_chars,"primary_output_tokens":runtime.settings.primary_output_tokens,
                "memory_backend":runtime.settings.memory_backend,"embedding":await runtime.memory.health() if hasattr(runtime.memory,'health') else None,
                "reader_concurrency": 1, "execution_environment": "task_workspace_and_simulation"}

    @app.get("/api/tools")
    def tools():
        return TOOL_HELP

    @app.get("/api/tasks")
    def tasks():
        return runtime.store.tasks()

    @app.post("/api/tasks")
    def create_task(body: TaskCreate):
        allowed = body.allowed_tools if body.allowed_tools is not None else list(TOOL_HELP)
        if set(allowed) - set(TOOL_HELP):
            raise ValueError("Unknown execution tools")
        if any(len(rule) > 400 for rule in body.constraints + body.completion_criteria):
            raise ValueError("Each constraint/criterion must fit within 400 characters")
        return runtime.store.create_task(body.goal, body.constraints, body.completion_criteria, allowed)

    @app.get("/api/tasks/{task_id}")
    def task(task_id: str):
        return {"task": runtime.store.task(task_id), "jobs": runtime.store.jobs(task_id),
                "operations": runtime.store.operations(task_id), "windows": runtime.store.windows(task_id),
                "world": runtime.tools.observe(task_id)}

    @app.get("/api/tasks/{task_id}/events")
    def events(task_id: str, after: int = 0, tail: bool = False):
        return runtime.store.events(task_id, after=after, limit=2000, tail=tail)

    @app.get("/api/tasks/{task_id}/evidence")
    def evidence(task_id: str, ref: str):
        return runtime.store.resolve(task_id, ref)

    @app.post("/api/tasks/{task_id}/messages")
    def message(task_id: str, body: Message):
        return runtime.store.add_job(task_id, "message", {"text": body.text}, request_key=body.request_id)

    @app.post("/api/tasks/{task_id}/handoff")
    def handoff(task_id: str):
        return runtime.store.add_job(task_id, "handoff", {})

    @app.post("/api/tasks/{task_id}/queries")
    def query(task_id: str, body: Query):
        return runtime.store.add_job(task_id, "query", body.model_dump())

    @app.post('/api/tasks/{task_id}/memory/read')
    def read_memory(task_id: str, body: MemoryRead):
        if runtime.settings.memory_backend!='hybrid':raise ValueError('Summary catalog is not enabled')
        return runtime.memory.read(task_id,body.request_id,body.card_ids,body.question)

    @app.put("/api/tasks/{task_id}/constraints")
    def constraints(task_id: str, body: ConstraintUpdate):
        if any(len(rule) > 400 for rule in body.constraints):
            raise ValueError("Each constraint must fit within 400 characters")
        return runtime.store.update_constraints(task_id, body.constraints, body.expected_version)

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        return runtime.store.job(job_id)

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel(job_id: str):
        return runtime.cancel(job_id)

    @app.post("/api/tasks/{task_id}/operations/{operation_id}/verify")
    def verify(task_id: str, operation_id: str):
        return runtime.tools.reconcile(task_id, operation_id)

    @app.get("/api/tasks/{task_id}/queries")
    def query_records(task_id: str):
        runtime.store.task(task_id)
        with runtime.store.db() as db:
            return [{**dict(row), "body": json.loads(row["body"]), "result": json.loads(row["result"]) if row["result"] else None}
                    for row in db.execute("SELECT * FROM queries WHERE task_id=? ORDER BY created DESC", (task_id,))]

    @app.get("/api/tasks/{task_id}/world/versions/{version}")
    def world_version(task_id: str, version: int):
        runtime.store.task(task_id)
        with runtime.store.db() as db:
            row = db.execute("SELECT * FROM world_versions WHERE task_id=? AND version=?", (task_id, version)).fetchone()
            if row is None:
                raise KeyError(version)
            return {"version": version, "observed_at": row["observed"], **json.loads(row["body"])}

    @app.get("/api/tasks/{task_id}/files")
    def files(task_id: str, path: str = "."):
        target = runtime.tools.path(task_id, path)
        if target.is_dir():
            return {"entries": [{"name": item.name, "directory": item.is_dir()} for item in sorted(target.iterdir())][:100]}
        if target.stat().st_size > 32000:
            raise ValueError("File exceeds the reading limit")
        return {"text": target.read_text(encoding="utf-8")}

    return app


