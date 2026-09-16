"""Execute model-requested reads through validated single or federated boundaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from customer_service.llm.client import LLMResponseError

from .config import DBAgentConfig
from .database import DatabaseError, DatabaseSchema, QueryResult, ReadOnlyRoleError
from .federation import FederatedQueryError, FederatedQueryService, SourceQuery
from .gateway import DatabaseGateway, FederatedCatalog
from .prompts import FEDERATED_TOOL_NAME, SCHEMA_TOOL_NAME, TOOL_NAME
from .types import DBAgentError, QueryTrace
from .validator import SQLValidationError, validate_readonly_sql


@dataclass(frozen=True)
class ToolBatch:
    """State changes from processing the model's tool calls."""

    messages: list[dict[str, Any]]
    traces: list[QueryTrace]
    calls_used: int
    metadata_calls: int
    attempts: int
    used_databases: set[str]
    source_failures: dict[str, str]


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


def _trace_error(
    sql: str,
    purpose: str | None,
    error: str,
    *,
    database: str | None = None,
    result_name: str | None = None,
    stage: str = "result",
) -> QueryTrace:
    return QueryTrace(
        sql=sql,
        purpose=purpose,
        row_count=0,
        truncated=False,
        columns=(),
        rows=[],
        error=error,
        database=database,
        result_name=result_name,
        stage=stage,
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


def _schema_payload(catalog: FederatedCatalog, names: list[str]) -> dict[str, Any]:
    databases: dict[str, Any] = {}
    for name in names:
        schema = catalog.schemas.get(name)
        if schema is None:
            raise LLMResponseError(f"Database {name!r} is not available.")
        tables: list[dict[str, Any]] = []
        for table_name in schema.sorted_table_names:
            table = schema.tables[table_name]
            tables.append(
                {
                    "table": table.fqn,
                    "columns": [column.display() for column in table.columns],
                    "primary_key": table.primary_key,
                    "foreign_keys": [
                        {
                            "columns": foreign_key.columns,
                            "references": (
                                f"{foreign_key.ref_schema}.{foreign_key.ref_table}"
                            ),
                            "reference_columns": foreign_key.ref_columns,
                        }
                        for foreign_key in table.foreign_keys
                    ],
                }
            )
        databases[name] = tables
    return {"databases": databases}


def _correction(error: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": f"The previous read-only analysis plan is invalid ({error}). Correct it.",
    }


def execute_tool_calls(
    calls: list[dict[str, Any]],
    *,
    catalog: DatabaseSchema | FederatedCatalog,
    gateway: DatabaseGateway,
    federation: FederatedQueryService,
    config: DBAgentConfig,
    calls_used: int,
    metadata_calls: int,
    attempts: int,
    used_databases: set[str],
    source_failures: dict[str, str] | None = None,
) -> ToolBatch:
    """Parse, validate, and run requested reads without granting write access."""

    max_calls = max(1, config.max_tool_calls)
    messages: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    traces: list[QueryTrace] = []
    used = set(used_databases)
    failures = dict(source_failures or {})

    for call in calls:
        call_id = str(call.get("id", ""))
        name = (call.get("name") or "").lower()
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
            corrections.append(_correction(error))
            continue

        if name == SCHEMA_TOOL_NAME and isinstance(catalog, FederatedCatalog):
            metadata_calls += 1
            if metadata_calls > 3:
                raise DBAgentError(
                    "Detailed schema inspection limit reached without an analytical query."
                )
            raw_names = payload.get("databases")
            if (
                not isinstance(raw_names, list)
                or not raw_names
                or not all(isinstance(item, str) for item in raw_names)
            ):
                error = "databases must be a non-empty list of configured names"
                messages.append(_tool_reply(call_id, {"error": error}))
                corrections.append(_correction(error))
                continue
            try:
                details = _schema_payload(catalog, list(dict.fromkeys(raw_names)))
            except LLMResponseError as exc:
                messages.append(_tool_reply(call_id, {"error": str(exc)}))
                corrections.append(_correction(str(exc)))
                continue
            messages.append(_tool_reply(call_id, details))
            continue

        if calls_used >= max_calls:
            error = f"SQL tool-call budget of {max_calls} reached; this query was not executed."
            traces.append(_trace_error("", payload.get("purpose"), error))
            messages.append(_tool_reply(call_id, {"error": error}))
            continue

        calls_used += 1
        if name == TOOL_NAME and isinstance(catalog, DatabaseSchema):
            sql = str(payload.get("sql", "")).strip()
            purpose = payload.get("purpose")
            database = gateway.default_database
            try:
                validated_sql = validate_readonly_sql(sql, catalog=catalog)
            except SQLValidationError as exc:
                attempts += 1
                error = str(exc)
                traces.append(
                    _trace_error(sql, purpose, error, database=database, stage="source")
                )
                messages.append(_tool_reply(call_id, {"error": error}))
                if attempts > config.query_retries:
                    raise DBAgentError(
                        f"Query validation failed after {attempts} attempt(s): {error}"
                    ) from exc
                corrections.append(_correction(error))
                continue
            try:
                result = gateway.execute_readonly_query(
                    database, validated_sql, max_rows=config.max_rows
                )
            except ReadOnlyRoleError:
                raise
            except DatabaseError as exc:
                attempts += 1
                error = str(exc)
                failures[database] = error
                traces.append(
                    _trace_error(sql, purpose, error, database=database, stage="source")
                )
                messages.append(_tool_reply(call_id, {"error": error}))
                if attempts > config.query_retries:
                    raise DBAgentError(
                        f"Query execution failed after {attempts} attempt(s): {error}"
                    ) from exc
                corrections.append(_correction(error))
                continue

            traces.append(
                QueryTrace(
                    sql=result.sql,
                    purpose=purpose,
                    row_count=result.row_count,
                    truncated=result.truncated,
                    columns=result.columns,
                    rows=result.rows,
                    database=database,
                    stage="result",
                )
            )
            used.add(database)
            failures.pop(database, None)
            messages.append(
                _tool_reply(call_id, _result_reply(result, config.max_result_chars))
            )
            continue

        if name == FEDERATED_TOOL_NAME and isinstance(catalog, FederatedCatalog):
            purpose = payload.get("purpose")
            raw_queries = payload.get("source_queries")
            combine_sql = str(payload.get("combine_sql", "")).strip()
            if not isinstance(raw_queries, list):
                error = "source_queries must be a list"
                messages.append(_tool_reply(call_id, {"error": error}))
                corrections.append(_correction(error))
                continue
            try:
                source_queries = [
                    SourceQuery(
                        result_name=str(item["result_name"]),
                        database=str(item["database"]),
                        sql=str(item["sql"]),
                    )
                    for item in raw_queries
                    if isinstance(item, dict)
                ]
                if len(source_queries) != len(raw_queries):
                    raise KeyError("Each source query must be an object.")
                execution = federation.execute(
                    source_queries,
                    combine_sql,
                    catalog=catalog,
                    purpose=purpose,
                    max_rows=config.max_rows,
                )
            except (KeyError, SQLValidationError, FederatedQueryError) as exc:
                attempts += 1
                error = str(exc)
                if isinstance(exc, FederatedQueryError):
                    traces.extend(exc.traces)
                    failures.update(
                        {
                            trace.database: trace.error
                            for trace in exc.traces
                            if trace.database and trace.error
                        }
                    )
                traces.append(_trace_error(combine_sql, purpose, error))
                messages.append(_tool_reply(call_id, {"error": error}))
                if attempts > config.query_retries:
                    raise DBAgentError(
                        f"Federated query failed after {attempts} attempt(s): {error}"
                    ) from exc
                corrections.append(_correction(error))
                continue

            traces.extend(execution.traces)
            used.update(execution.used_databases)
            for database in execution.used_databases:
                failures.pop(database, None)
            reply = _result_reply(execution.result, config.max_result_chars)
            reply["databases_used"] = sorted(execution.used_databases)
            reply["unavailable_databases"] = sorted(catalog.failures)
            messages.append(_tool_reply(call_id, reply))
            continue

        error = f"Unsupported tool: {call.get('name')!r}"
        traces.append(_trace_error("", call.get("name"), error))
        messages.append(_tool_reply(call_id, {"error": error}))

    return ToolBatch(
        messages=messages + corrections,
        traces=traces,
        calls_used=calls_used,
        metadata_calls=metadata_calls,
        attempts=attempts,
        used_databases=used,
        source_failures=failures,
    )
