"""Public result types shared by the workflow and its callers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class DBAgentError(Exception):
    """Agent-level operational failure."""


@dataclass(frozen=True)
class QueryTrace:
    """One SQL execution attempt and its result."""

    sql: str
    purpose: str | None
    row_count: int
    truncated: bool
    columns: tuple[str, ...]
    rows: list[dict[str, Any]]
    error: str | None = None


@dataclass(frozen=True)
class DBAgentResult:
    """Final outcome from a database question."""

    answer: str
    model: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    traces: list[QueryTrace] | None = None
