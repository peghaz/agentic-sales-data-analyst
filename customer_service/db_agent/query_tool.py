"""Execute one model-requested batch through the existing read-only boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from customer_service.llm.client import LLMResponseError

from .config import DBAgentConfig
from .database import (
    DatabaseAdapter,
    DatabaseExecutionError,
    DatabaseSchema,
    QueryResult,
)
from .prompts import TOOL_NAME
from .types import DBAgentError, QueryTrace
from .validator import SQLValidationError, validate_readonly_sql


@dataclass(frozen=True)
class ToolBatch:
    """State changes from processing the model's tool calls."""

    messages: list[dict[str, Any]]
    traces: list[QueryTrace]
    calls_used: int
    attempts: int


def _parse_arguments(call: dict[str, Any]) -> dict[str, Any]:
    arguments = call.get("arguments", "{}")
    if isinstance(arguments, str):
        parsed = json.loads(arguments)
    elif isinstance(arguments, dict):
        parsed = arguments
    else:
        raise LLMResponseError("Tool arguments must be a JSON object.")
    if not isinstance(parsed, dict):
        raise LLMResponseError("Tool arguments must be a JSON object.")
    return parsed


def _trace_error(sql: str, purpose: str | None, error: str) -> QueryTrace:
    return QueryTrace(
        sql=sql,
        purpose=purpose,
        row_count=0,
        truncated=False,
        columns=(),
        rows=[],
        error=error,
    )


def _tool_reply(call_id: str, content: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(content, ensure_ascii=False, default=str),
    }


def _result_reply(result: QueryResult, max_chars: int) -> dict[str, Any]:
    """Keep the model payload valid JSON even when result rows need trimming."""

    payload: dict[str, Any] = {
        "row_count": result.row_count,
        "truncated": result.truncated,
        "columns": result.columns,
        "rows": result.rows,
    }
    limit = max(200, max_chars)
    while payload["rows"] and len(json.dumps(payload, default=str)) > limit:
        payload["rows"] = payload["rows"][:-1]
    if len(payload["rows"]) < len(result.rows):
        payload["shortened_for_model"] = True
    if len(json.dumps(payload, default=str)) > limit:
        payload["rows"] = []
        payload["shortened_for_model"] = True
    return payload


def execute_tool_calls(
    calls: list[dict[str, Any]],
    *,
    schema: DatabaseSchema,
    adapter: DatabaseAdapter,
    config: DBAgentConfig,
    calls_used: int,
    attempts: int,
) -> ToolBatch:
    """Parse, validate, and run requested SQL without granting write access."""

    max_calls = max(1, config.max_tool_calls)
    messages: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    traces: list[QueryTrace] = []

    for call in calls:
        call_id = str(call.get("id", ""))
        if calls_used >= max_calls:
            error = f"SQL tool-call budget of {max_calls} reached; this query was not executed."
            try:
                skipped = _parse_arguments(call)
            except (json.JSONDecodeError, LLMResponseError):
                skipped = {}
            traces.append(
                _trace_error(
                    str(skipped.get("sql", "")).strip(), skipped.get("purpose"), error
                )
            )
            messages.append(_tool_reply(call_id, {"error": error}))
            continue

        calls_used += 1
        if (call.get("name") or "").lower() != TOOL_NAME:
            error = f"Unsupported tool: {call.get('name')!r}"
            traces.append(_trace_error("", call.get("name"), error))
            messages.append(_tool_reply(call_id, {"error": error}))
            continue

        try:
            payload = _parse_arguments(call)
        except (json.JSONDecodeError, LLMResponseError) as exc:
            attempts += 1
            error = f"Invalid tool arguments: {exc}"
            traces.append(_trace_error("", None, error))
            messages.append(_tool_reply(call_id, {"error": error}))
            if attempts > config.query_retries:
                raise DBAgentError(
                    f"Tool argument parsing failed after {attempts} attempt(s): {error}"
                ) from exc
            corrections.append(
                {
                    "role": "user",
                    "content": f"Tool payload is invalid ({error}). Correct it.",
                }
            )
            continue

        sql = str(payload.get("sql", "")).strip()
        purpose = payload.get("purpose")
        try:
            validated_sql = validate_readonly_sql(sql, catalog=schema)
        except SQLValidationError as exc:
            attempts += 1
            error = str(exc)
            traces.append(_trace_error(sql, purpose, error))
            messages.append(_tool_reply(call_id, {"error": error}))
            if attempts > config.query_retries:
                raise DBAgentError(
                    f"Query validation failed after {attempts} attempt(s): {error}"
                ) from exc
            corrections.append(
                {
                    "role": "user",
                    "content": f"The previous SQL is invalid ({error}). Correct it.",
                }
            )
            continue

        try:
            result = adapter.execute_readonly_query(
                validated_sql, max_rows=config.max_rows
            )
        except DatabaseExecutionError as exc:
            attempts += 1
            error = str(exc)
            traces.append(_trace_error(validated_sql, purpose, error))
            messages.append(_tool_reply(call_id, {"error": error}))
            if attempts > config.query_retries:
                raise DBAgentError(
                    f"Query execution failed after {attempts} attempt(s): {error}"
                ) from exc
            corrections.append(
                {
                    "role": "user",
                    "content": f"Query execution failed ({error}). Correct it.",
                }
            )
            continue

        traces.append(
            QueryTrace(
                sql=result.sql,
                purpose=purpose,
                row_count=result.row_count,
                truncated=result.truncated,
                columns=result.columns,
                rows=result.rows,
            )
        )
        messages.append(
            _tool_reply(call_id, _result_reply(result, config.max_result_chars))
        )

    return ToolBatch(messages + corrections, traces, calls_used, attempts)
