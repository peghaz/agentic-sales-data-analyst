"""Configuration helpers for the database agent."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from urllib.parse import quote, quote_plus, urlsplit, urlunsplit

_DATABASE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")


def _parse_csv_names(value: str) -> tuple[str, ...]:
    return tuple(name.strip() for name in value.split(",") if name and name.strip())


def _parse_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _parse_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _parse_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of true, false, 1, 0, yes, no, on, or off.")


def _parse_database_names() -> tuple[str, ...]:
    raw = os.getenv("DATABASES_AVAILABLE", "").strip()
    if not raw:
        return ()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "DATABASES_AVAILABLE must be a JSON list of database names."
        ) from exc
    if not isinstance(value, list) or not value:
        raise ValueError("DATABASES_AVAILABLE must be a non-empty JSON list.")
    names: list[str] = []
    for item in value:
        if not isinstance(item, str) or not _DATABASE_NAME.fullmatch(item):
            raise ValueError(
                "DATABASES_AVAILABLE entries must use lowercase letters, numbers, "
                "hyphens, or underscores."
            )
        if item in names:
            raise ValueError(f"DATABASES_AVAILABLE contains duplicate name {item!r}.")
        names.append(item)
    if len(names) > 25:
        raise ValueError("DATABASES_AVAILABLE supports at most 25 databases.")
    return tuple(names)


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
    model_max_tokens: int = 4096
    enable_thinking: bool = True
    profile_name: str = "sales"
    available_databases: tuple[str, ...] = ()
    enforce_readonly_role: bool = True
    schema_cache_ttl_seconds: int = 300
    max_database_concurrency: int = 4
    max_intermediate_rows: int = 10000
    max_federated_bytes: int = 50000000

    @property
    def is_federated(self) -> bool:
        return len(self.available_databases) > 1

    def database_url_for(self, database: str) -> str:
        """Return the configured PostgreSQL URL with an allowlisted database path."""

        if self.available_databases and database not in self.available_databases:
            raise ValueError(f"Database {database!r} is not in DATABASES_AVAILABLE.")
        parsed = urlsplit(self.database_url)
        if parsed.scheme not in {"postgres", "postgresql"}:
            raise ValueError("DB_AGENT_DATABASE_URL must be a PostgreSQL URL.")
        return urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                f"/{quote(database, safe='')}",
                parsed.query,
                "",
            )
        )

    @classmethod
    def from_env(cls) -> DBAgentConfig:
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

        allowed_tables = _parse_csv_names(os.getenv("DB_AGENT_ALLOWED_TABLES", ""))

        return cls(
            database_url=database_url,
            allowed_schemas=tuple(s.strip() for s in allowed_schemas),
            allowed_tables=tuple(t.strip() for t in allowed_tables),
            max_rows=_parse_int("DB_AGENT_MAX_ROWS", 200),
            max_result_chars=_parse_int("DB_AGENT_MAX_RESULT_CHARS", 50000),
            statement_timeout_ms=_parse_int("DB_AGENT_STATEMENT_TIMEOUT_MS", 10000),
            max_tool_calls=_parse_int("DB_AGENT_MAX_TOOL_CALLS", 5),
            query_retries=_parse_int("DB_AGENT_QUERY_RETRIES", 2),
            model_max_tokens=_parse_positive_int("DB_AGENT_MODEL_MAX_TOKENS", 4096),
            enable_thinking=_parse_bool("DB_AGENT_ENABLE_THINKING", True),
            profile_name=os.getenv("DB_AGENT_PROFILE", "sales").strip() or "sales",
            available_databases=_parse_database_names(),
            enforce_readonly_role=_parse_bool("DB_AGENT_ENFORCE_READONLY_ROLE", True),
            schema_cache_ttl_seconds=_parse_positive_int(
                "DB_AGENT_SCHEMA_CACHE_TTL_SECONDS", 300
            ),
            max_database_concurrency=_parse_positive_int(
                "DB_AGENT_MAX_DATABASE_CONCURRENCY", 4
            ),
            max_intermediate_rows=_parse_positive_int(
                "DB_AGENT_MAX_INTERMEDIATE_ROWS", 10000
            ),
            max_federated_bytes=_parse_positive_int(
                "DB_AGENT_MAX_FEDERATED_BYTES", 50000000
            ),
        )
