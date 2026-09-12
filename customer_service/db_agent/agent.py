"""Tool-driven orchestration loop for natural-language SQL tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from customer_service.llm.client import LLMResponseError, OpenAILLMClient

from .config import DBAgentConfig
from .database import (
    DatabaseExecutionError,
    DatabaseSchema,
    PostgresDatabaseAdapter,
    QueryResult,
)
from .validator import SQLValidationError, validate_readonly_sql


class DBAgentError(Exception):
    """Agent-level operational failure."""


@dataclass(frozen=True)
class QueryTrace:
    """One SQL execution trace item."""

    sql: str
    purpose: str | None
    row_count: int
    truncated: bool
    columns: tuple[str, ...]
    rows: list[dict[str, Any]]
    error: str | None = None


@dataclass(frozen=True)
class DBAgentResult:
    """Final outcome from a DB ask invocation."""

    answer: str
    model: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    traces: list[QueryTrace] | None = None


class DBAgent:
    """Run a chat loop with a read-only SQL tool."""

    TOOL_NAME = "execute_readonly_sql"

    def __init__(
        self,
        client: OpenAILLMClient,
        adapter: PostgresDatabaseAdapter,
        config: DBAgentConfig,
    ):
        self._client = client
        self._adapter = adapter
        self._config = config

    def ask(self, question: str) -> DBAgentResult:
        if not question or not question.strip():
            raise DBAgentError("Question cannot be empty.")

        schema = self._adapter.inspect_schema(
            allowed_schemas=self._config.allowed_schemas,
            allowed_tables=self._config.allowed_tables,
        )
        system_prompt = self._build_system_prompt(schema)
        tools = [self._tool_schema()]

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        traces: list[QueryTrace] = []
        attempts = 0
        total_latency_ms = 0.0
        max_tool_calls = max(1, self._config.max_tool_calls)
        tool_calls_used = 0

        while tool_calls_used < max_tool_calls:
            response = self._client.ask(
                prompt="",
                messages=messages,
                tools=tools,
                tool_choice="required" if tool_calls_used == 0 else "auto",
                parallel_tool_calls=False,
                temperature=0.0,
                max_tokens=1536,
            )
            total_latency_ms += response.latency_ms
            assistant_message = response.message or {
                "role": "assistant",
                "content": response.content,
            }
            messages.append(assistant_message)

            if not response.tool_calls:
                if response.content:
                    if tool_calls_used == 0:
                        raise DBAgentError(
                            "Model did not emit a tool call. "
                            "Enable tool/function calling for this model."
                        )
                    return DBAgentResult(
                        answer=response.content,
                        model=response.model,
                        latency_ms=total_latency_ms,
                        prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                        total_tokens=response.total_tokens,
                        traces=traces,
                    )

                raise DBAgentError(
                    "Model returned no tool call and no final text. "
                    "Enable tool calling on the model server."
                )

            for call in response.tool_calls:
                if tool_calls_used >= max_tool_calls:
                    error = (
                        f"SQL tool-call budget of {max_tool_calls} reached; "
                        "this query was not executed."
                    )
                    try:
                        skipped_payload = self._parse_tool_arguments(call)
                    except (json.JSONDecodeError, LLMResponseError):
                        skipped_payload = {}
                    traces.append(
                        QueryTrace(
                            sql=str(skipped_payload.get("sql", "")).strip(),
                            purpose=skipped_payload.get("purpose"),
                            row_count=0,
                            truncated=False,
                            columns=(),
                            rows=[],
                            error=error,
                        )
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id", ""),
                            "content": json.dumps({"error": error}),
                        }
                    )
                    continue

                tool_calls_used += 1
                if (call.get("name") or "").lower() != self.TOOL_NAME:
                    traces.append(
                        QueryTrace(
                            sql="",
                            purpose=call.get("name"),
                            row_count=0,
                            truncated=False,
                            columns=(),
                            rows=[],
                            error=f"Unsupported tool: {call.get('name')!r}",
                        )
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id", ""),
                            "content": "Unsupported tool call.",
                        }
                    )
                    continue

                try:
                    payload = self._parse_tool_arguments(call)
                except Exception as exc:
                    attempts += 1
                    err = f"Invalid tool arguments: {exc}"
                    traces.append(
                        QueryTrace(
                            sql="",
                            purpose=None,
                            row_count=0,
                            truncated=False,
                            columns=(),
                            rows=[],
                            error=err,
                        )
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id", ""),
                            "content": json.dumps({"error": err}),
                        }
                    )
                    if attempts > self._config.query_retries:
                        raise DBAgentError(
                            f"Tool argument parsing failed after {attempts} attempt(s): {err}"
                        ) from exc
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Tool payload is invalid ({err}). "
                                "Please provide corrected SQL in the `sql` field."
                            ),
                        }
                    )
                    continue

                sql = str(payload.get("sql", "")).strip()
                purpose = payload.get("purpose")
                call_id = call.get("id", "")
                try:
                    validated_sql = validate_readonly_sql(sql, catalog=schema)
                except SQLValidationError as exc:
                    attempts += 1
                    trace = QueryTrace(
                        sql=sql,
                        purpose=purpose,
                        row_count=0,
                        truncated=False,
                        columns=(),
                        rows=[],
                        error=str(exc),
                    )
                    traces.append(trace)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": json.dumps({"error": str(exc)}),
                        }
                    )
                    if attempts > self._config.query_retries:
                        raise DBAgentError(
                            f"Query validation failed after {attempts} attempt(s): {exc}"
                        ) from exc
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"The previous SQL is invalid ({exc}). "
                                "Please provide corrected SQL and retry."
                            ),
                        }
                    )
                    continue

                try:
                    result = self._adapter.execute_readonly_query(
                        validated_sql,
                        max_rows=self._config.max_rows,
                    )
                    traces.append(self._result_trace(result, purpose=purpose))
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": self._result_to_tool_message(result),
                        }
                    )
                except DatabaseExecutionError as exc:
                    attempts += 1
                    traces.append(
                        QueryTrace(
                            sql=validated_sql,
                            purpose=purpose,
                            row_count=0,
                            truncated=False,
                            columns=(),
                            rows=[],
                            error=str(exc),
                        )
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": json.dumps({"error": str(exc)}),
                        }
                    )
                    if attempts > self._config.query_retries:
                        raise DBAgentError(
                            f"Query execution failed after {attempts} attempt(s): {exc}"
                        ) from exc
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Query execution failed ({exc}). "
                                "Please provide corrected SQL and retry."
                            ),
                        }
                    )

        messages.append(
            {
                "role": "user",
                "content": (
                    f"The SQL tool-call budget of {max_tool_calls} has been used. "
                    "Do not request another tool call. Using only the tool results "
                    "already available, provide the best possible final answer now. "
                    "Clearly identify any part of the question that could not be answered."
                ),
            }
        )
        try:
            final_response = self._client.ask(
                prompt="",
                messages=messages,
                temperature=0.0,
                max_tokens=1536,
            )
        except LLMResponseError as exc:
            raise DBAgentError(
                "SQL tool-call budget was reached, but the model failed to generate "
                f"a final answer: {exc}"
            ) from exc

        total_latency_ms += final_response.latency_ms
        if not final_response.content:
            raise DBAgentError(
                "SQL tool-call budget was reached, but the model did not produce "
                "a final assistant answer."
            )

        return DBAgentResult(
            answer=final_response.content,
            model=final_response.model,
            latency_ms=total_latency_ms,
            prompt_tokens=final_response.prompt_tokens,
            completion_tokens=final_response.completion_tokens,
            total_tokens=final_response.total_tokens,
            traces=traces,
        )

    @staticmethod
    def _parse_tool_arguments(call: dict[str, Any]) -> dict[str, Any]:
        arguments = call.get("arguments", "{}")
        if isinstance(arguments, str):
            parsed = json.loads(arguments)
            if not isinstance(parsed, dict):
                raise LLMResponseError("Tool arguments must be a JSON object.")
            return parsed
        if isinstance(arguments, dict):
            return arguments
        raise LLMResponseError("Tool arguments must be a JSON object.")

    def _result_trace(self, result: QueryResult, purpose: str | None) -> QueryTrace:
        return QueryTrace(
            sql=result.sql,
            purpose=purpose,
            row_count=result.row_count,
            truncated=result.truncated,
            columns=result.columns,
            rows=result.rows,
        )

    def _result_to_tool_message(self, result: QueryResult) -> str:
        payload = {
            "sql": result.sql,
            "row_count": result.row_count,
            "truncated": result.truncated,
            "columns": result.columns,
            "rows": result.rows,
        }
        serialized = json.dumps(payload, ensure_ascii=False)
        if len(serialized) > self._config.max_result_chars:
            return serialized[: self._config.max_result_chars] + "\n...truncated..."
        return serialized

    def _build_system_prompt(self, schema: DatabaseSchema) -> str:
        lines = [
            "You are a PostgreSQL data assistant.",
            "You must answer user questions by first calling the tool `execute_readonly_sql`.",
            "Do not use assumptions, fabricated rows, or SQL that mutates data.",
            "You are only allowed to read from tables listed below.",
            (
                "Call the tool one time at a time. For compound data questions, you may "
                "make multiple sequential tool calls."
            ),
            (
                f"You have at most {max(1, self._config.max_tool_calls)} SQL tool calls. "
                "Plan every requested metric before the first call and stay within that "
                "budget. Do not repeat a metric already returned."
            ),
            (
                "Do not spend a separate exploratory call on date bounds, distributions, "
                "or data shape when the same fact can be derived with a CTE or subquery "
                "inside the analysis query."
            ),
            (
                "Combine related metrics when they fit one flat relational result set. "
                "For differently shaped results, use separate sequential calls instead "
                "of packing rows into json_agg or row_to_json."
            ),
            (
                "Use descriptive, non-reserved names for CTEs and aliases. Once you have "
                "enough data, stop calling tools and answer."
            ),
            (
                "Never add, compare, or rank monetary values across currencies unless "
                "conversion data is available; group monetary results by currency instead."
            ),
            "",
            "Database schema (refreshes on every question):",
        ]
        for table_key in schema.sorted_table_names:
            table = schema.table(*table_key.split(".", 1))
            if table is None:
                continue
            lines.append(f"\n- {table.fqn}")
            for column in table.columns:
                lines.append(f"  - {column.display()}")
            if table.primary_key:
                lines.append(f"  - pk: {', '.join(table.primary_key)}")
            for foreign_key in table.foreign_keys:
                lines.append(
                    "  - fk: "
                    f"{', '.join(foreign_key.columns)} -> "
                    f"{foreign_key.ref_schema}.{foreign_key.ref_table}("
                    f"{', '.join(foreign_key.ref_columns)})"
                )
        lines.append("")
        lines.append(
            "Return a concise Markdown answer after tool output. Use a Markdown table "
            "for multi-row comparisons. If no rows are returned, say so clearly."
        )
        return "\n".join(lines)

    @staticmethod
    def _tool_schema() -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": DBAgent.TOOL_NAME,
                "description": (
                    "Execute a read-only SQL query against the configured PostgreSQL schema."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "The SQL SELECT query."},
                        "purpose": {
                            "type": "string",
                            "description": "Short reason for this query.",
                        },
                    },
                    "required": ["sql"],
                },
            },
        }
