from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Decision(StrictModel):
    action: Literal["respond", "tool", "recall", "read_memory", "handoff", "finish"]
    response: str = Field(max_length=2400)
    tool: str = Field(max_length=80)
    arguments: dict


class Finding(StrictModel):
    statement: str = Field(max_length=500)
    evidence_refs: list[str] = Field(max_length=12)


class HandoffFinding(StrictModel):
    statement: str = Field(max_length=100)
    evidence_refs: list[str] = Field(max_length=2)


class Handoff(StrictModel):
    summary: str = Field(max_length=200)
    completed: list[Annotated[str, Field(max_length=100)]] = Field(max_length=3)
    decisions: list[HandoffFinding] = Field(max_length=2)
    next_steps: list[Annotated[str, Field(max_length=100)]] = Field(max_length=2)
    open_questions: list[Annotated[str, Field(max_length=100)]] = Field(max_length=2)
    topics: list[Annotated[str, Field(max_length=24)]] = Field(max_length=4)
    entities: list[Annotated[str, Field(max_length=24)]] = Field(max_length=4)
    history_entries: list[HandoffFinding] = Field(max_length=2)


class SearchPlan(StrictModel):
    terms: list[str] = Field(max_length=6)
    questions: list[Annotated[str, Field(max_length=160)]] = Field(min_length=1, max_length=3)


class Quote(StrictModel):
    event_ref: str
    quote: str = Field(min_length=1, max_length=1000)
    relevance: str = Field(max_length=240)


class Reading(StrictModel):
    evidence: list[Quote] = Field(max_length=8)
    status: Literal["relevant", "partial", "not_found", "unable_to_judge"]
    suggested_followups: list[str] = Field(max_length=4)


class Synthesis(StrictModel):
    status: Literal["supported", "partial", "conflicting", "not_found", "unable_to_judge"]
    findings: list[Finding] = Field(max_length=8)
    inferences: list[str] = Field(max_length=4)
    conflicts: list[Finding] = Field(max_length=6)
    unresolved_questions: list[str] = Field(max_length=6)


class TaskCreate(StrictModel):
    goal: str = Field(min_length=1, max_length=1200)
    constraints: list[str] = Field(default_factory=list, max_length=12)
    completion_criteria: list[str] = Field(default_factory=list, max_length=8)
    allowed_tools: list[str] | None = None


class Message(StrictModel):
    text: str = Field(min_length=1, max_length=6000)
    request_id: str = Field(min_length=1, max_length=100)


class ConstraintUpdate(StrictModel):
    constraints: list[str] = Field(max_length=12)
    expected_version: int


class Query(StrictModel):
    question: str = Field(min_length=1, max_length=1500)
    exhaustive: bool = False
    verify: bool = False
    read_archive: bool = False
    min_score: float | None = Field(default=None, ge=0, le=1)
    search_archive: bool = False


class MemoryRead(StrictModel):
    request_id: str
    card_ids: list[str] = Field(min_length=1, max_length=3)
    question: str = Field(default='', max_length=1500)


class CatalogSummary(StrictModel):
    summary: str = Field(min_length=1, max_length=100)

