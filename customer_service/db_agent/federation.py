"""Exact, ephemeral aggregation over read-only PostgreSQL result sets."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import duckdb
import pandas as pd

from .database import DatabaseError, QueryResult, ReadOnlyRoleError
from .gateway import DatabaseGateway, FederatedCatalog
from .types import QueryTrace
from .validator import validate_federated_sql, validate_readonly_sql

_RESULT_NAME = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


@dataclass(frozen=True)
class SourceQuery:
    """One named query executed against one allowlisted database."""

    result_name: str
    database: str
    sql: str


@dataclass(frozen=True)
class FederatedExecution:
    """Final result and complete source provenance for a federated plan."""

    result: QueryResult
    traces: list[QueryTrace]
    used_databases: set[str]


class FederatedQueryError(DatabaseError):
    """Raised when a federated plan cannot produce an exact result."""

    def __init__(self, message: str, traces: list[QueryTrace] | None = None) -> None:
        super().__init__(message)
        self.traces = traces or []


class DuckDBResultCombiner:
    """Combine source rows in a locked-down, in-memory DuckDB connection."""

    def combine(
        self,
        results: dict[str, QueryResult],
        sql: str,
        *,
        max_rows: int,
    ) -> QueryResult:
        validated = validate_federated_sql(sql, set(results))
        connection = duckdb.connect(
            ":memory:",
            config={
                "enable_external_access": "false",
                "allow_unsigned_extensions": "false",
                "autoinstall_known_extensions": "false",
                "autoload_known_extensions": "false",
            },
        )
        try:
            for name, result in results.items():
                frame = pd.DataFrame(result.rows, columns=result.columns)
                connection.register(name, frame)
            cursor = connection.execute(validated)
            columns = tuple(item[0] for item in cursor.description)
            raw_rows = cursor.fetchmany(max_rows + 1)
        except duckdb.Error as exc:
            raise FederatedQueryError(f"Federated result query failed: {exc}") from exc
        finally:
            connection.close()

        truncated = len(raw_rows) > max_rows
        rows = [
            {
                name: _normalize_value(value)
                for name, value in zip(columns, row, strict=False)
            }
            for row in raw_rows[:max_rows]
        ]
        return QueryResult(
            sql=validated,
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
        )


def _normalize_value(value: Any) -> Any:
    from .database import PostgresDatabaseAdapter

    return PostgresDatabaseAdapter._normalize_value(value)


class FederatedQueryService:
    """Validate, execute, bound, and combine a complete analytical plan."""

    def __init__(
        self,
        gateway: DatabaseGateway,
        *,
        max_intermediate_rows: int,
        max_federated_bytes: int,
        max_concurrency: int,
        combiner: DuckDBResultCombiner | None = None,
    ) -> None:
        self._gateway = gateway
        self._max_intermediate_rows = max_intermediate_rows
        self._max_federated_bytes = max_federated_bytes
        self._max_concurrency = max(1, max_concurrency)
        self._combiner = combiner or DuckDBResultCombiner()

    def execute(
        self,
        source_queries: list[SourceQuery],
        combine_sql: str,
        *,
        catalog: FederatedCatalog,
        purpose: str | None,
        max_rows: int,
    ) -> FederatedExecution:
        if not source_queries:
            raise FederatedQueryError("At least one source query is required.")
        if len(source_queries) > 50:
            raise FederatedQueryError(
                "A federated plan supports at most 50 source queries."
            )

        names: set[str] = set()
        validated: list[SourceQuery] = []
        for query in source_queries:
            if not _RESULT_NAME.fullmatch(query.result_name):
                raise FederatedQueryError(
                    f"Invalid result name {query.result_name!r}; use lowercase letters, "
                    "numbers, and underscores."
                )
            if query.result_name in names:
                raise FederatedQueryError(
                    f"Duplicate result name {query.result_name!r}."
                )
            names.add(query.result_name)
            if query.database in catalog.failures:
                raise FederatedQueryError(
                    f"Database {query.database!r} is unavailable: "
                    f"{catalog.failures[query.database]}"
                )
            schema = catalog.schemas.get(query.database)
            if schema is None:
                raise FederatedQueryError(
                    f"Database {query.database!r} is not available in this catalog."
                )
            validated.append(
                SourceQuery(
                    query.result_name,
                    query.database,
                    validate_readonly_sql(query.sql, schema),
                )
            )

        results: dict[str, QueryResult] = {}
        traces: list[QueryTrace] = []
        failures: list[str] = []
        with ThreadPoolExecutor(
            max_workers=min(self._max_concurrency, len(validated))
        ) as executor:
            future_queries = {
                executor.submit(
                    self._gateway.execute_readonly_query,
                    query.database,
                    query.sql,
                    self._max_intermediate_rows,
                ): query
                for query in validated
            }
            for future in as_completed(future_queries):
                query = future_queries[future]
                try:
                    result = future.result()
                    results[query.result_name] = result
                    trace = QueryTrace(
                        sql=result.sql,
                        purpose=purpose,
                        row_count=result.row_count,
                        truncated=result.truncated,
                        columns=result.columns,
                        rows=result.rows,
                        database=query.database,
                        result_name=query.result_name,
                        stage="source",
                    )
                    traces.append(trace)
                    if result.truncated:
                        failures.append(
                            f"{query.result_name} from {query.database} reached the "
                            "intermediate row limit"
                        )
                except ReadOnlyRoleError:
                    raise
                except DatabaseError as exc:
                    failures.append(f"{query.database}: {exc}")
                    traces.append(
                        QueryTrace(
                            sql=query.sql,
                            purpose=purpose,
                            row_count=0,
                            truncated=False,
                            columns=(),
                            rows=[],
                            error=str(exc),
                            database=query.database,
                            result_name=query.result_name,
                            stage="source",
                        )
                    )

        encoded_size = len(
            json.dumps({k: v.rows for k, v in results.items()}, default=str).encode(
                "utf-8"
            )
        )
        if encoded_size > self._max_federated_bytes:
            failures.append(
                f"Intermediate data exceeded {self._max_federated_bytes} bytes"
            )
        if failures:
            raise FederatedQueryError(
                "Exact federation was not run: "
                + "; ".join(failures)
                + ". Aggregate more data inside each source query or omit unavailable sources.",
                traces,
            )

        try:
            result = self._combiner.combine(results, combine_sql, max_rows=max_rows)
        except FederatedQueryError as exc:
            raise FederatedQueryError(str(exc), traces) from exc
        traces.append(
            QueryTrace(
                sql=result.sql,
                purpose=purpose,
                row_count=result.row_count,
                truncated=result.truncated,
                columns=result.columns,
                rows=result.rows,
                result_name="final_result",
                stage="result",
            )
        )
        return FederatedExecution(
            result=result,
            traces=traces,
            used_databases={query.database for query in validated},
        )
