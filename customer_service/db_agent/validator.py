"""SQL guardrails for the read-only DB agent."""

from __future__ import annotations

import re

from .database import DatabaseSchema


class SQLValidationError(ValueError):
    """Raised when query is unsafe or schema-incompatible."""


_FORBIDDEN_KEYWORDS = (
    r"\binsert\b",
    r"\bupdate\b",
    r"\bdelete\b",
    r"\bdrop\b",
    r"\bcreate\b",
    r"\balter\b",
    r"\btruncate\b",
    r"\bmerge\b",
    r"\bcopy\b",
    r"\bgrant\b",
    r"\brevoke\b",
    r"\bcall\b",
    r"\bdo\b",
    r"\bexecute\b",
    r"\breplace\b",
    r"\bset\b",
    r"\bbegin\b",
    r"\bcommit\b",
    r"\brollback\b",
    r"\bvaccum\b",
    r"\bvaccuum\b",
    r"\banalyze\b",
)


def validate_readonly_sql(sql: str, catalog: DatabaseSchema) -> str:
    statement = (sql or "").strip()
    if not statement:
        raise SQLValidationError("SQL cannot be empty.")

    # remove trailing semi-colon and disallow statement chaining
    if statement.endswith(";"):
        statement = statement[:-1].strip()
    if ";" in statement:
        raise SQLValidationError("Only one SQL statement is allowed.")

    if not re.match(r"(?is)^\s*(with|select)\b", statement):
        raise SQLValidationError("Only SELECT and WITH queries are allowed.")

    for pattern in _FORBIDDEN_KEYWORDS:
        if re.search(pattern, statement, flags=re.IGNORECASE):
            raise SQLValidationError("Potentially mutating SQL detected.")

    if re.search(r"\bselect\b[^;]*\binto\b", statement, flags=re.IGNORECASE):
        raise SQLValidationError("SELECT INTO is not allowed.")

    if re.search(r"\bfor\s+(update|share|key\s+share|no\s+key\s+update)\b", statement, re.IGNORECASE):
        raise SQLValidationError("Row-locking clauses are not allowed.")

    _validate_referenced_tables(statement, catalog)
    return statement


def _validate_referenced_tables(statement: str, catalog: DatabaseSchema) -> None:
    references = set(_extract_referenced_relations(statement))
    cte_names = _extract_cte_names(statement)
    for reference in references:
        schema_name, table_name = _split_relation(reference)
        if schema_name is None and table_name.lower() in cte_names:
            continue
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


def _extract_cte_names(statement: str) -> set[str]:
    """Return query-local relation names declared by a WITH clause."""

    pattern = re.compile(
        r"(?:\bwith\b|,)\s*"
        r"(?:recursive\s+)?"
        r"(?P<name>\"(?:[^\"]|\"\")*\"|[a-z_][a-z0-9_$]*)\s*"
        r"(?:\([^)]*\)\s*)?"
        r"as\s+(?:(?:not\s+)?materialized\s+)?\(",
        flags=re.IGNORECASE,
    )
    names: set[str] = set()
    for match in pattern.finditer(statement):
        name = match.group("name")
        if name.startswith('"') and name.endswith('"'):
            name = name[1:-1].replace('""', '"')
        names.add(name.lower())
    return names


def _extract_referenced_relations(statement: str) -> list[str]:
    pattern = re.compile(r"\b(?:from|join)\s+([^\s,()]+)", flags=re.IGNORECASE)
    found = [match.group(1).strip() for match in pattern.finditer(statement)]
    sanitized: list[str] = []
    for raw in found:
        if raw.startswith("("):
            continue
        if "." in raw and raw.count('"') % 2 == 1:
            continue
        cleaned = raw.rstrip(",")
        if cleaned:
            sanitized.append(cleaned)
    return sanitized


def _split_relation(reference: str) -> tuple[str | None, str]:
    raw_parts = [part.strip() for part in reference.split(".", 1)]
    if not raw_parts:
        raise SQLValidationError(f"Malformed relation reference: {reference}")

    cleaned_parts: list[str] = []
    for part in raw_parts:
        part = part.strip()
        if len(part) >= 2 and (
            (part[0] == part[-1] == '"') or (part[0] == part[-1] == "`")
        ):
            part = part[1:-1]
        cleaned_parts.append(part)

    if len(cleaned_parts) == 2:
        return cleaned_parts[0], cleaned_parts[1]
    return None, cleaned_parts[0]
