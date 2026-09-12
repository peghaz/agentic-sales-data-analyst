"""Configuration helpers for the database agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import quote_plus


def _parse_csv_names(value: str) -> tuple[str, ...]:
    return tuple(
        name.strip() for name in value.split(",") if name and name.strip()
    )


def _parse_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _build_database_url() -> str:
    host = os.getenv("DB_AGENT_DATABASE_HOST", "").strip()
    if host:
        return (
            "postgresql://"
            f"{quote_plus(os.getenv('DB_AGENT_DATABASE_USER', '').strip())}"
            f":{quote_plus(os.getenv('DB_AGENT_DATABASE_PASSWORD', '').strip())}"
            f"@{host}:"
            f"{os.getenv('DB_AGENT_DATABASE_PORT', os.getenv('POSTGRES_PORT', '5432')).strip()}/"
            f"{os.getenv('DB_AGENT_DATABASE_NAME', os.getenv('POSTGRES_DB', 'dbagent')).strip()}"
        )
    return os.getenv(
        "DB_AGENT_DATABASE_URL",
        "".join(
            [
                "postgresql://",
                f"{quote_plus(os.getenv('POSTGRES_USER', 'postgres').strip())}",
                f":{quote_plus(os.getenv('POSTGRES_PASSWORD', '').strip())}",
                f"@{os.getenv('POSTGRES_HOST', 'localhost').strip()}:",
                f"{os.getenv('POSTGRES_PORT', '5432').strip()}/",
                f"{os.getenv('POSTGRES_DB', 'dbagent').strip()}",
            ]
        ),
    )


@dataclass(frozen=True)
class DBAgentConfig:
    """Runtime configuration for DB query agent execution."""

    database_url: str
    allowed_schemas: tuple[str, ...] = ("public",)
    allowed_tables: tuple[str, ...] = ()
    max_rows: int = 200
    max_result_chars: int = 50000
    statement_timeout_ms: int = 10000
    max_tool_calls: int = 5
    query_retries: int = 2

    @classmethod
    def from_env(cls) -> "DBAgentConfig":
        database_url = _build_database_url().strip()
        if not database_url:
            raise ValueError(
                "Could not determine a database URL. Set DB_AGENT_DATABASE_URL or POSTGRES_*."
            )

        allowed_schemas = _parse_csv_names(
            os.getenv("DB_AGENT_ALLOWED_SCHEMAS", "public")
        )
        if not allowed_schemas:
            allowed_schemas = ("public",)

        allowed_tables = _parse_csv_names(
            os.getenv("DB_AGENT_ALLOWED_TABLES", "")
        )

        return cls(
            database_url=database_url,
            allowed_schemas=tuple(s.strip() for s in allowed_schemas),
            allowed_tables=tuple(t.strip() for t in allowed_tables),
            max_rows=_parse_int("DB_AGENT_MAX_ROWS", 200),
            max_result_chars=_parse_int("DB_AGENT_MAX_RESULT_CHARS", 50000),
            statement_timeout_ms=_parse_int("DB_AGENT_STATEMENT_TIMEOUT_MS", 10000),
            max_tool_calls=_parse_int("DB_AGENT_MAX_TOOL_CALLS", 5),
            query_retries=_parse_int("DB_AGENT_QUERY_RETRIES", 2),
        )

