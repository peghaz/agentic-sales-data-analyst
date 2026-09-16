"""Database abstraction for raw SQL querying and schema introspection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from threading import Lock
from typing import Any

import psycopg2


class DatabaseError(Exception):
    """Base exception for database adapter failures."""


class DatabaseConnectionError(DatabaseError):
    """Raised when the database cannot be reached."""


class DatabaseExecutionError(DatabaseError):
    """Raised when SQL execution fails."""


class DatabaseIntrospectionError(DatabaseError):
    """Raised when schema discovery fails."""


class ReadOnlyRoleError(DatabaseError):
    """Raised when database credentials can mutate the configured source."""


@dataclass(frozen=True)
class ColumnMeta:
    """Metadata for one database column."""

    name: str
    data_type: str
    is_nullable: bool
    character_maximum_length: int | None = None
    numeric_precision: int | None = None
    numeric_scale: int | None = None
    column_default: str | None = None

    def display(self) -> str:
        dtype = self.data_type
        if self.character_maximum_length:
            dtype = f"{dtype}({self.character_maximum_length})"
        elif self.numeric_precision:
            if self.numeric_scale is not None:
                dtype = f"{dtype}({self.numeric_precision},{self.numeric_scale})"
            else:
                dtype = f"{dtype}({self.numeric_precision})"
        details = "NOT NULL" if not self.is_nullable else "NULL"
        return f"{self.name}: {dtype} {details}"


@dataclass(frozen=True)
class ForeignKeyMeta:
    """Foreign-key relation metadata."""

    name: str
    columns: tuple[str, ...]
    ref_schema: str
    ref_table: str
    ref_columns: tuple[str, ...]


@dataclass(frozen=True)
class TableMeta:
    """Schema metadata for a single table."""

    schema: str
    name: str
    columns: tuple[ColumnMeta, ...]
    primary_key: tuple[str, ...] = ()
    unique_constraints: tuple[tuple[str, ...], ...] = ()
    foreign_keys: tuple[ForeignKeyMeta, ...] = ()

    @property
    def fqn(self) -> str:
        return f"{self.schema}.{self.name}"


@dataclass(frozen=True)
class DatabaseSchema:
    """Container for all discoverable tables for one run."""

    tables: dict[str, TableMeta]

    def table(self, schema: str, table: str) -> TableMeta | None:
        return self.tables.get(f"{schema}.{table}")

    def find_by_name(self, table: str) -> list[TableMeta]:
        table = table.lower()
        return [
            t for key, t in self.tables.items() if key.split(".", 1)[1].lower() == table
        ]

    @property
    def sorted_table_names(self) -> list[str]:
        return sorted(self.tables.keys())


@dataclass(frozen=True)
class QueryResult:
    """Normalized SELECT result used by tool output."""

    sql: str
    columns: tuple[str, ...]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool = False


class DatabaseAdapter(ABC):
    """Generic DB adapter contract for the SQL agent."""

    @abstractmethod
    def inspect_schema(
        self,
        allowed_schemas: Iterable[str],
        allowed_tables: Iterable[str] | None = None,
    ) -> DatabaseSchema:
        raise NotImplementedError

    @abstractmethod
    def execute_readonly_query(
        self,
        sql: str,
        max_rows: int,
    ) -> QueryResult:
        raise NotImplementedError


class PostgresDatabaseAdapter(DatabaseAdapter):
    """Raw SQL query adapter for PostgreSQL."""

    def __init__(
        self,
        dsn: str,
        statement_timeout_ms: int = 10000,
        *,
        enforce_readonly_role: bool = True,
    ) -> None:
        self._dsn = dsn
        self._statement_timeout_ms = statement_timeout_ms
        self._enforce_readonly_role = enforce_readonly_role
        self._role_audited = False
        self._audit_lock = Lock()

    @contextmanager
    def _readonly_connection(self) -> Iterator[Any]:
        connection = None
        try:
            connection = psycopg2.connect(self._dsn)
            connection.set_session(readonly=True, autocommit=False)
            with connection.cursor() as cursor:
                cursor.execute("SHOW transaction_read_only")
                row = cursor.fetchone()
                if not row or str(row[0]).lower() not in {"on", "true", "1"}:
                    raise ReadOnlyRoleError(
                        "PostgreSQL did not confirm a read-only transaction."
                    )
                cursor.execute(
                    "SET LOCAL statement_timeout = %s",
                    (self._statement_timeout_ms,),
                )
                if self._enforce_readonly_role and not self._role_audited:
                    with self._audit_lock:
                        if not self._role_audited:
                            self._audit_role(cursor)
                            self._role_audited = True
            yield connection
        finally:
            if connection is not None:
                try:
                    connection.rollback()
                finally:
                    connection.close()

    @staticmethod
    def _audit_role(cursor: Any) -> None:
        cursor.execute(
            """
            SELECT rolsuper, rolcreaterole, rolcreatedb, rolreplication, rolbypassrls
            FROM pg_roles
            WHERE rolname = current_user
            """
        )
        role_flags = cursor.fetchone()
        if not role_flags or any(role_flags):
            raise ReadOnlyRoleError(
                "DB agent credentials must use a non-privileged, SELECT-only role."
            )

        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_roles
                WHERE rolname = ANY(%s)
                  AND pg_has_role(current_user, oid, 'MEMBER')
            )
            """,
            (
                [
                    "pg_execute_server_program",
                    "pg_read_server_files",
                    "pg_signal_backend",
                    "pg_write_all_data",
                    "pg_write_server_files",
                ],
            ),
        )
        if cursor.fetchone()[0]:
            raise ReadOnlyRoleError(
                "DB agent credentials must not inherit server-file, program-execution, "
                "or backend-signaling roles."
            )

        cursor.execute(
            """
            SELECT
                has_database_privilege(current_user, current_database(), 'CREATE')
                OR has_database_privilege(current_user, current_database(), 'TEMP')
            """
        )
        if cursor.fetchone()[0]:
            raise ReadOnlyRoleError(
                "DB agent credentials must not have CREATE or TEMP database privileges."
            )

        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_namespace
                WHERE nspname <> 'information_schema'
                  AND left(nspname, 3) <> 'pg_'
                  AND has_schema_privilege(current_user, oid, 'CREATE')
            )
            """
        )
        if cursor.fetchone()[0]:
            raise ReadOnlyRoleError(
                "DB agent credentials must not have CREATE privileges on user schemas."
            )

        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_class AS relation
                JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
                WHERE namespace.nspname <> 'information_schema'
                  AND left(namespace.nspname, 3) <> 'pg_'
                  AND relation.relkind IN ('r', 'p', 'v', 'm', 'f')
                  AND (
                      has_table_privilege(
                          current_user,
                          relation.oid,
                          'INSERT'
                      )
                      OR has_table_privilege(
                          current_user,
                          relation.oid,
                          'UPDATE'
                      )
                      OR has_table_privilege(
                          current_user,
                          relation.oid,
                          'DELETE'
                      )
                      OR has_table_privilege(
                          current_user,
                          relation.oid,
                          'TRUNCATE'
                      )
                      OR has_table_privilege(
                          current_user,
                          relation.oid,
                          'TRIGGER'
                      )
                      OR has_table_privilege(current_user, relation.oid, 'REFERENCES')
                  )
            )
            """
        )
        if cursor.fetchone()[0]:
            raise ReadOnlyRoleError(
                "DB agent credentials have mutation privileges on a user table or view."
            )

        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_class AS sequence
                JOIN pg_namespace AS namespace ON namespace.oid = sequence.relnamespace
                WHERE namespace.nspname <> 'information_schema'
                  AND left(namespace.nspname, 3) <> 'pg_'
                  AND sequence.relkind = 'S'
                  AND (
                      has_sequence_privilege(
                          current_user,
                          sequence.oid,
                          'USAGE'
                      )
                      OR has_sequence_privilege(
                          current_user,
                          sequence.oid,
                          'UPDATE'
                      )
                  )
            )
            """
        )
        if cursor.fetchone()[0]:
            raise ReadOnlyRoleError(
                "DB agent credentials must not have USAGE or UPDATE on sequences."
            )

    def inspect_schema(
        self,
        allowed_schemas: Iterable[str],
        allowed_tables: Iterable[str] | None = None,
    ) -> DatabaseSchema:
        schemas = [schema for schema in allowed_schemas if schema]
        if not schemas:
            schemas = ["public"]

        allowed_table_set = {table.lower() for table in (allowed_tables or ()) if table}

        try:
            with self._readonly_connection() as connection:  # noqa: SIM117
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT table_schema, table_name
                        FROM information_schema.tables
                        WHERE table_schema = ANY(%s)
                          AND table_type = 'BASE TABLE'
                        ORDER BY table_schema, table_name
                        """,
                        (schemas,),
                    )
                    table_rows = cursor.fetchall()

                    cursor.execute(
                        """
                        SELECT
                            table_schema,
                            table_name,
                            column_name,
                            data_type,
                            is_nullable,
                            character_maximum_length,
                            numeric_precision,
                            numeric_scale,
                            column_default
                        FROM information_schema.columns
                        WHERE table_schema = ANY(%s)
                        ORDER BY table_schema, table_name, ordinal_position
                        """,
                        (schemas,),
                    )
                    column_rows = cursor.fetchall()

                    cursor.execute(
                        """
                        SELECT
                            tc.table_schema,
                            tc.table_name,
                            kcu.column_name,
                            kcu.ordinal_position
                        FROM information_schema.table_constraints tc
                        JOIN information_schema.key_column_usage kcu
                            ON tc.table_schema = kcu.table_schema
                           AND tc.table_name = kcu.table_name
                           AND tc.constraint_name = kcu.constraint_name
                        WHERE tc.table_schema = ANY(%s)
                          AND tc.constraint_type = 'PRIMARY KEY'
                        ORDER BY tc.table_schema, tc.table_name, kcu.ordinal_position
                        """,
                        (schemas,),
                    )
                    pk_rows = cursor.fetchall()

                    cursor.execute(
                        """
                        SELECT
                            tc.table_schema,
                            tc.table_name,
                            tc.constraint_name,
                            kcu.column_name,
                            kcu.ordinal_position
                        FROM information_schema.table_constraints tc
                        JOIN information_schema.key_column_usage kcu
                            ON tc.table_schema = kcu.table_schema
                           AND tc.table_name = kcu.table_name
                           AND tc.constraint_name = kcu.constraint_name
                        WHERE tc.table_schema = ANY(%s)
                          AND tc.constraint_type = 'UNIQUE'
                        ORDER BY tc.table_schema, tc.table_name, tc.constraint_name, kcu.ordinal_position
                        """,
                        (schemas,),
                    )
                    unique_rows = cursor.fetchall()

                    cursor.execute(
                        """
                        SELECT
                            tc.table_schema,
                            tc.table_name,
                            tc.constraint_name,
                            kcu.column_name,
                            ccu.table_schema AS foreign_table_schema,
                            ccu.table_name AS foreign_table_name,
                            ccu.column_name AS foreign_column_name,
                            kcu.ordinal_position
                        FROM information_schema.table_constraints tc
                        JOIN information_schema.key_column_usage kcu
                            ON tc.table_schema = kcu.table_schema
                           AND tc.table_name = kcu.table_name
                           AND tc.constraint_name = kcu.constraint_name
                        JOIN information_schema.constraint_column_usage ccu
                            ON tc.constraint_schema = ccu.constraint_schema
                           AND tc.constraint_name = ccu.constraint_name
                        WHERE tc.table_schema = ANY(%s)
                          AND tc.constraint_type = 'FOREIGN KEY'
                        ORDER BY tc.table_schema, tc.table_name, tc.constraint_name, kcu.ordinal_position
                        """,
                        (schemas,),
                    )
                    fk_rows = cursor.fetchall()

        except psycopg2.Error as exc:
            raise DatabaseIntrospectionError(
                "Could not inspect PostgreSQL schema. Check DB_AGENT credentials and access."
            ) from exc

        return self._build_schema_catalog(
            schemas=schemas,
            allowed_tables=allowed_table_set,
            table_rows=table_rows,
            column_rows=column_rows,
            pk_rows=pk_rows,
            unique_rows=unique_rows,
            fk_rows=fk_rows,
        )

    def execute_readonly_query(self, sql: str, max_rows: int) -> QueryResult:
        bounded_sql = self._bounded_sql(sql, max_rows)
        try:
            with self._readonly_connection() as connection:  # noqa: SIM117
                with connection.cursor() as cursor:
                    cursor.execute(bounded_sql)
                    if cursor.description is None:
                        return QueryResult(
                            sql=sql,
                            columns=(),
                            rows=[],
                            row_count=0,
                            truncated=False,
                        )
                    columns = tuple(desc[0] for desc in cursor.description)
                    rows = cursor.fetchall()
        except psycopg2.Error as exc:
            raise DatabaseExecutionError(str(exc)) from exc

        normalized_rows = [self._normalize_row(columns, row) for row in rows]
        truncated = max_rows > 0 and len(normalized_rows) > max_rows
        if truncated:
            normalized_rows = normalized_rows[:max_rows]
        return QueryResult(
            sql=sql,
            columns=columns,
            rows=normalized_rows,
            row_count=len(normalized_rows),
            truncated=truncated,
        )

    @staticmethod
    def _bounded_sql(sql: str, max_rows: int) -> str:
        if not max_rows or max_rows <= 0:
            return sql
        return f"SELECT * FROM ({sql}) AS _agent_result LIMIT {int(max_rows) + 1}"

    def _build_schema_catalog(
        self,
        schemas: list[str],
        allowed_tables: set[str],
        table_rows: list[tuple[str, str]],
        column_rows: list[tuple[str, ...]],
        pk_rows: list[tuple[str, ...]],
        unique_rows: list[tuple[str, ...]],
        fk_rows: list[tuple[str, ...]],
    ) -> DatabaseSchema:
        tables: dict[str, TableMeta] = {}
        for table_schema, table_name in table_rows:
            if allowed_tables and table_name.lower() not in allowed_tables:
                continue
            key = f"{table_schema}.{table_name}"
            tables[key] = TableMeta(
                schema=table_schema,
                name=table_name,
                columns=(),
            )

        # Fill columns
        for row in column_rows:
            (
                table_schema,
                table_name,
                column_name,
                data_type,
                is_nullable,
                char_max_len,
                numeric_precision,
                numeric_scale,
                column_default,
            ) = row
            key = f"{table_schema}.{table_name}"
            if key not in tables:
                continue
            table = tables[key]
            tables[key] = TableMeta(
                schema=table.schema,
                name=table.name,
                columns=table.columns
                + (
                    ColumnMeta(
                        name=column_name,
                        data_type=data_type,
                        is_nullable=(is_nullable.lower() == "yes"),
                        character_maximum_length=char_max_len,
                        numeric_precision=numeric_precision,
                        numeric_scale=numeric_scale,
                        column_default=column_default,
                    ),
                ),
                primary_key=table.primary_key,
                unique_constraints=table.unique_constraints,
                foreign_keys=table.foreign_keys,
            )

        # Fill PKs
        for table_schema, table_name, column_name, _ordinal in pk_rows:
            key = f"{table_schema}.{table_name}"
            table = tables.get(key)
            if not table:
                continue
            pk = table.primary_key + (column_name,)
            tables[key] = TableMeta(
                schema=table.schema,
                name=table.name,
                columns=table.columns,
                primary_key=pk,
                unique_constraints=table.unique_constraints,
                foreign_keys=table.foreign_keys,
            )

        # Fill unique constraints
        unique_map: dict[str, list[tuple[str, ...]]] = {}
        for (
            table_schema,
            table_name,
            constraint_name,
            column_name,
            _ordinal,
        ) in unique_rows:
            key = f"{table_schema}.{table_name}"
            if key not in tables:
                continue
            ck = f"{key}:{constraint_name}"
            unique_map.setdefault(ck, [])
            unique_map[ck].append((column_name,))
        for row_key, raw_cols in unique_map.items():
            table_key = row_key.split(":", 1)[0]
            table = tables.get(table_key)
            if not table:
                continue
            constraints = set(table.unique_constraints)
            constraints.add(tuple(item[0] for item in raw_cols))
            tables[table_key] = TableMeta(
                schema=table.schema,
                name=table.name,
                columns=table.columns,
                primary_key=table.primary_key,
                unique_constraints=tuple(constraints),
                foreign_keys=table.foreign_keys,
            )

        # Fill FKs
        fk_map: dict[str, dict[str, dict[str, Any]]] = {}
        for (
            table_schema,
            table_name,
            constraint_name,
            column_name,
            foreign_table_schema,
            foreign_table_name,
            foreign_column_name,
            _ordinal,
        ) in fk_rows:
            table_key = f"{table_schema}.{table_name}"
            if table_key not in tables:
                continue
            fk_key = f"{table_key}:{constraint_name}"
            fk_map.setdefault(fk_key, {})
            fk_payload = fk_map[fk_key]
            fk_payload.setdefault("columns", []).append(column_name)
            fk_payload.setdefault("foreign_table_schema", foreign_table_schema)
            fk_payload.setdefault("foreign_table_name", foreign_table_name)
            fk_payload.setdefault("foreign_columns", []).append(foreign_column_name)

        for fk_key, payload in fk_map.items():
            table_key = fk_key.split(":", 1)[0]
            table = tables.get(table_key)
            if not table:
                continue
            foreign = ForeignKeyMeta(
                name=fk_key.split(":", 1)[1],
                columns=tuple(payload["columns"]),
                ref_schema=payload["foreign_table_schema"],
                ref_table=payload["foreign_table_name"],
                ref_columns=tuple(payload["foreign_columns"]),
            )
            tables[table_key] = TableMeta(
                schema=table.schema,
                name=table.name,
                columns=table.columns,
                primary_key=table.primary_key,
                unique_constraints=table.unique_constraints,
                foreign_keys=table.foreign_keys + (foreign,),
            )

        tables = {
            key: value
            for key, value in tables.items()
            if value.columns or any(schema in key for schema in schemas)
        }
        return DatabaseSchema(tables=tables)

    @staticmethod
    def _normalize_row(
        columns: tuple[str, ...], row: tuple[Any, ...]
    ) -> dict[str, Any]:
        normalized = {}
        for key, value in zip(columns, row, strict=False):
            normalized[key] = PostgresDatabaseAdapter._normalize_value(value)
        return normalized

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            return {
                str(key): PostgresDatabaseAdapter._normalize_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [PostgresDatabaseAdapter._normalize_value(item) for item in value]
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return value.hex()
        return value
