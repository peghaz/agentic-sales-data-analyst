"""Database routing and catalog caching for one or many read-only sources."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock

from customer_service.llm.profile import (
    DatabaseRelationship,
    DatabaseSourceProfile,
    DomainProfile,
)

from .config import DBAgentConfig
from .database import (
    DatabaseAdapter,
    DatabaseError,
    DatabaseIntrospectionError,
    DatabaseSchema,
    PostgresDatabaseAdapter,
    QueryResult,
    ReadOnlyRoleError,
)
from .types import DatabaseCoverage


@dataclass(frozen=True)
class DatabaseTarget:
    """Resolved runtime configuration for one physical database."""

    name: str
    description: str
    allowed_schemas: tuple[str, ...]
    allowed_tables: tuple[str, ...]


@dataclass(frozen=True)
class FederatedCatalog:
    """Available schemas plus failures for the current catalog snapshot."""

    schemas: dict[str, DatabaseSchema]
    targets: dict[str, DatabaseTarget]
    failures: dict[str, str]
    relationships: tuple[DatabaseRelationship, ...] = ()

    @property
    def database_names(self) -> tuple[str, ...]:
        return tuple(self.targets)

    def coverage(
        self,
        used: set[str] | None = None,
        runtime_failures: dict[str, str] | None = None,
    ) -> list[DatabaseCoverage]:
        used = used or set()
        failures = {**self.failures, **(runtime_failures or {})}
        return [
            DatabaseCoverage(
                database=name,
                status=(
                    "unavailable"
                    if name in failures
                    else "used"
                    if name in used
                    else "available"
                ),
                detail=failures.get(name),
            )
            for name in self.targets
        ]


class DatabaseGateway:
    """SOLID boundary used by the workflow instead of concrete connections."""

    def __init__(
        self,
        adapters: dict[str, DatabaseAdapter],
        targets: dict[str, DatabaseTarget],
        *,
        relationships: tuple[DatabaseRelationship, ...] = (),
        cache_ttl_seconds: int = 300,
        max_concurrency: int = 4,
    ) -> None:
        if not adapters or adapters.keys() != targets.keys():
            raise ValueError(
                "Database adapters and targets must define the same sources."
            )
        self._adapters = adapters
        self._targets = targets
        self._relationships = relationships
        self._cache_ttl_seconds = cache_ttl_seconds
        self._max_concurrency = max(1, min(max_concurrency, len(adapters)))
        self._catalog: FederatedCatalog | None = None
        self._catalog_loaded_at = 0.0
        self._catalog_lock = Lock()

    @property
    def is_federated(self) -> bool:
        return len(self._adapters) > 1

    @property
    def default_database(self) -> str:
        return next(iter(self._adapters))

    def inspect_catalog(self, *, refresh: bool = False) -> FederatedCatalog:
        now = time.monotonic()
        with self._catalog_lock:
            if (
                not refresh
                and self._catalog is not None
                and now - self._catalog_loaded_at < self._cache_ttl_seconds
            ):
                return self._catalog

            schemas: dict[str, DatabaseSchema] = {}
            failures: dict[str, str] = {}
            with ThreadPoolExecutor(max_workers=self._max_concurrency) as executor:
                future_names = {
                    executor.submit(
                        self._adapters[name].inspect_schema,
                        allowed_schemas=target.allowed_schemas,
                        allowed_tables=target.allowed_tables,
                    ): name
                    for name, target in self._targets.items()
                }
                for future in as_completed(future_names):
                    name = future_names[future]
                    try:
                        schemas[name] = future.result()
                    except ReadOnlyRoleError:
                        raise
                    except DatabaseError as exc:
                        failures[name] = str(exc)

            if not schemas:
                detail = "; ".join(
                    f"{name}: {error}" for name, error in failures.items()
                )
                raise DatabaseIntrospectionError(
                    f"Could not inspect any configured PostgreSQL database. {detail}"
                )
            self._validate_relationships(schemas)
            self._catalog = FederatedCatalog(
                schemas=schemas,
                targets=self._targets,
                failures=failures,
                relationships=self._relationships,
            )
            self._catalog_loaded_at = now
            return self._catalog

    def _validate_relationships(self, schemas: dict[str, DatabaseSchema]) -> None:
        for relationship in self._relationships:
            for endpoint in (relationship.left, relationship.right):
                database, schema_name, table_name, column_name = endpoint.split(".")
                schema = schemas.get(database)
                if schema is None:
                    continue
                table = schema.table(schema_name, table_name)
                if table is None or column_name not in {
                    column.name for column in table.columns
                }:
                    raise DatabaseIntrospectionError(
                        f"Relationship {relationship.name!r} references unavailable "
                        f"column {endpoint!r}."
                    )

    def execute_readonly_query(
        self, database: str, sql: str, max_rows: int
    ) -> QueryResult:
        adapter = self._adapters.get(database)
        if adapter is None:
            raise DatabaseError(f"Database {database!r} is not configured.")
        return adapter.execute_readonly_query(sql, max_rows)


def build_database_gateway(
    config: DBAgentConfig, profile: DomainProfile
) -> DatabaseGateway:
    """Compose physical adapters from environment and profile configuration."""

    if config.available_databases:
        requested = set(config.available_databases)
        declared = {source.name for source in profile.sources.databases}
        if requested != declared:
            missing = sorted(requested - declared)
            extra = sorted(declared - requested)
            details = []
            if missing:
                details.append(f"missing from sources.toml: {', '.join(missing)}")
            if extra:
                details.append(f"not listed in DATABASES_AVAILABLE: {', '.join(extra)}")
            raise ValueError(
                "Database source configuration mismatch (" + "; ".join(details) + ")."
            )
        sources = profile.sources.databases
    else:
        if len(profile.sources.databases) != 1:
            raise ValueError(
                "DATABASES_AVAILABLE is required when sources.toml declares multiple "
                "databases."
            )
        source = profile.sources.databases[0]
        sources = (
            DatabaseSourceProfile(
                name=source.name,
                description=source.description,
                allowed_schemas=config.allowed_schemas,
                allowed_tables=config.allowed_tables,
            ),
        )

    targets = {
        source.name: DatabaseTarget(
            name=source.name,
            description=source.description,
            allowed_schemas=source.allowed_schemas,
            allowed_tables=source.allowed_tables,
        )
        for source in sources
    }
    adapters: dict[str, DatabaseAdapter] = {}
    for source in sources:
        dsn = (
            config.database_url_for(source.name)
            if config.available_databases
            else config.database_url
        )
        adapters[source.name] = PostgresDatabaseAdapter(
            dsn=dsn,
            statement_timeout_ms=config.statement_timeout_ms,
            enforce_readonly_role=config.enforce_readonly_role,
        )
    return DatabaseGateway(
        adapters,
        targets,
        relationships=profile.sources.relationships,
        cache_ttl_seconds=config.schema_cache_ttl_seconds,
        max_concurrency=config.max_database_concurrency,
    )
