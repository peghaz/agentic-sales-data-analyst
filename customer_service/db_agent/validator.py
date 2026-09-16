"""SQL guardrails for the read-only DB agent."""

from __future__ import annotations

from sqlglot import exp, parse
from sqlglot.errors import ParseError

from .database import DatabaseSchema


class SQLValidationError(ValueError):
    """Raised when query is unsafe or schema-incompatible."""


_FORBIDDEN_NODES = tuple(
    node
    for name in (
        "Alter",
        "Analyze",
        "Call",
        "Command",
        "Commit",
        "Copy",
        "Create",
        "Delete",
        "Drop",
        "Execute",
        "Grant",
        "Insert",
        "Into",
        "LoadData",
        "Lock",
        "Merge",
        "Pragma",
        "Rollback",
        "Set",
        "Transaction",
        "TruncateTable",
        "Update",
        "Use",
    )
    if (node := getattr(exp, name, None)) is not None
)
_DANGEROUS_FUNCTIONS = {
    "dblink",
    "dblink_exec",
    "lo_export",
    "lo_import",
    "nextval",
    "pg_advisory_lock",
    "pg_advisory_lock_shared",
    "pg_cancel_backend",
    "pg_create_restore_point",
    "pg_file_rename",
    "pg_file_unlink",
    "pg_file_write",
    "pg_ls_archive_statusdir",
    "pg_ls_dir",
    "pg_ls_logdir",
    "pg_ls_waldir",
    "pg_log_backend_memory_contexts",
    "pg_promote",
    "pg_read_binary_file",
    "pg_read_file",
    "pg_reload_conf",
    "pg_rotate_logfile",
    "pg_stat_file",
    "pg_switch_wal",
    "pg_terminate_backend",
    "pg_wal_replay_pause",
    "pg_wal_replay_resume",
    "set_config",
    "setval",
    "database_to_xml",
    "database_to_xml_and_xmlschema",
    "query_to_xml",
    "query_to_xml_and_xmlschema",
    "query_to_xmlschema",
    "schema_to_xml",
    "schema_to_xml_and_xmlschema",
    "table_to_xml",
    "table_to_xml_and_xmlschema",
}
_EXTERNAL_FUNCTION_PREFIXES = (
    "delta_scan",
    "glob",
    "httpfs",
    "iceberg_scan",
    "parquet_scan",
    "postgres_scan",
    "read_",
    "sqlite_scan",
)


def _parse_single_query(statement: str, dialect: str) -> exp.Query:
    try:
        expressions = [item for item in parse(statement, read=dialect) if item]
    except ParseError as exc:
        raise SQLValidationError(f"SQL could not be parsed: {exc}") from exc
    if len(expressions) != 1:
        raise SQLValidationError("Only one SQL statement is allowed.")
    expression = expressions[0]
    if not isinstance(expression, exp.Query):
        raise SQLValidationError("Only SELECT and WITH queries are allowed.")
    if any(expression.find(node_type) is not None for node_type in _FORBIDDEN_NODES):
        raise SQLValidationError("Potentially mutating SQL detected.")
    _validate_functions(expression, allow_external=False)
    return expression


def _function_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return node.name.lower()
    return node.sql_name().lower()


def _validate_functions(expression: exp.Query, *, allow_external: bool) -> None:
    for function in expression.find_all(exp.Func):
        name = _function_name(function)
        if name in _DANGEROUS_FUNCTIONS or name.startswith("dblink_"):
            raise SQLValidationError(f"Function {name!r} is not allowed.")
        if not allow_external and name.startswith(_EXTERNAL_FUNCTION_PREFIXES):
            raise SQLValidationError(
                f"External-access function {name!r} is not allowed."
            )


def validate_readonly_sql(sql: str, catalog: DatabaseSchema) -> str:
    statement = (sql or "").strip()
    if not statement:
        raise SQLValidationError("SQL cannot be empty.")

    if statement.endswith(";"):
        statement = statement[:-1].strip()
    expression = _parse_single_query(statement, "postgres")
    _validate_referenced_tables(expression, catalog)
    return statement


def validate_federated_sql(sql: str, allowed_relations: set[str]) -> str:
    """Validate an in-memory DuckDB query over named source result relations."""

    statement = (sql or "").strip()
    if not statement:
        raise SQLValidationError("SQL cannot be empty.")
    if statement.endswith(";"):
        statement = statement[:-1].strip()
    expression = _parse_single_query(statement, "duckdb")
    cte_names = {cte.alias_or_name.lower() for cte in expression.find_all(exp.CTE)}
    for table in expression.find_all(exp.Table):
        name = table.name.lower()
        if not name or name in cte_names:
            continue
        if table.db or table.catalog or name not in allowed_relations:
            raise SQLValidationError(
                f"Federated result {table.sql()!r} is not available in this plan."
            )
    return statement


def _validate_referenced_tables(expression: exp.Query, catalog: DatabaseSchema) -> None:
    cte_names = {cte.alias_or_name.lower() for cte in expression.find_all(exp.CTE)}
    for table in expression.find_all(exp.Table):
        table_name = table.name
        schema_name = table.db or None
        if not table_name or (schema_name is None and table_name.lower() in cte_names):
            continue
        if table.catalog:
            raise SQLValidationError(
                "Cross-database relation names are not allowed inside source SQL."
            )
        if schema_name is None:
            matches = catalog.find_by_name(table_name)
            if len(matches) == 1:
                continue
            if len(matches) > 1:
                raise SQLValidationError(
                    f"Ambiguous unqualified table '{table_name}'. Please schema-qualify it."
                )
            raise SQLValidationError(
                f"Table '{table_name}' is not available in this session."
            )
        table = catalog.table(schema_name, table_name)
        if table is None:
            raise SQLValidationError(
                f"Table '{schema_name}.{table_name}' is not in the allowed schema list."
            )
