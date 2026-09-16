"""Federated analytics and immutable PostgreSQL boundary tests."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from customer_service.db_agent.agent import DBAgent
from customer_service.db_agent.config import DBAgentConfig
from customer_service.db_agent.database import (
    ColumnMeta,
    DatabaseExecutionError,
    DatabaseSchema,
    PostgresDatabaseAdapter,
    QueryResult,
    ReadOnlyRoleError,
    TableMeta,
)
from customer_service.db_agent.federation import (
    FederatedQueryError,
    FederatedQueryService,
    SourceQuery,
)
from customer_service.db_agent.gateway import (
    DatabaseGateway,
    DatabaseTarget,
    build_database_gateway,
)
from customer_service.db_agent.prompts import (
    FEDERATED_TOOL_NAME,
    SCHEMA_TOOL_NAME,
)
from customer_service.db_agent.validator import (
    SQLValidationError,
    validate_federated_sql,
    validate_readonly_sql,
)
from customer_service.llm.client import LLMResponse
from customer_service.llm.profile import (
    DatabaseRelationship,
    DatabaseSourceProfile,
    SourceProfile,
    load_domain_profile,
)


def _schema(table: str, *columns: str) -> DatabaseSchema:
    return DatabaseSchema(
        tables={
            f"public.{table}": TableMeta(
                schema="public",
                name=table,
                columns=tuple(
                    ColumnMeta(name=name, data_type="text", is_nullable=False)
                    for name in columns
                ),
            )
        }
    )


class _Adapter:
    def __init__(
        self,
        schema: DatabaseSchema | None,
        results: dict[str, QueryResult] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.schema = schema
        self.results = results or {}
        self.error = error
        self.executed: list[str] = []

    def inspect_schema(self, **_: Any) -> DatabaseSchema:
        if self.error:
            raise self.error
        assert self.schema is not None
        return self.schema

    def execute_readonly_query(self, sql: str, max_rows: int) -> QueryResult:
        if self.error:
            raise self.error
        self.executed.append(sql)
        return self.results[sql]


def _gateway(*, second_error: Exception | None = None) -> DatabaseGateway:
    sales_schema = _schema("orders", "customer_id", "revenue")
    support_schema = _schema("calls", "customer_id", "tickets")
    sales_sql = "SELECT customer_id, revenue FROM public.orders"
    support_sql = "SELECT customer_id, tickets FROM public.calls"
    adapters = {
        "sales": _Adapter(
            sales_schema,
            {
                sales_sql: QueryResult(
                    sql=sales_sql,
                    columns=("customer_id", "revenue"),
                    rows=[
                        {"customer_id": 1, "revenue": "10.50"},
                        {"customer_id": 2, "revenue": "20.25"},
                    ],
                    row_count=2,
                )
            },
        ),
        "support": _Adapter(
            support_schema,
            {
                support_sql: QueryResult(
                    sql=support_sql,
                    columns=("customer_id", "tickets"),
                    rows=[
                        {"customer_id": 1, "tickets": 3},
                        {"customer_id": 2, "tickets": 1},
                    ],
                    row_count=2,
                )
            },
            second_error,
        ),
    }
    targets = {
        "sales": DatabaseTarget("sales", "Sales records", ("public",), ()),
        "support": DatabaseTarget("support", "Support records", ("public",), ()),
    }
    relationship = DatabaseRelationship(
        name="customer_identity",
        left="sales.public.orders.customer_id",
        right="support.public.calls.customer_id",
        cardinality="one-to-one",
        description="Shared customer identifier",
    )
    return DatabaseGateway(
        adapters,
        targets,
        relationships=(relationship,),
        max_concurrency=2,
    )


def _profile():
    source_profile = SourceProfile(
        databases=(
            DatabaseSourceProfile("sales", "Sales", ("public",), ()),
            DatabaseSourceProfile("support", "Support", ("public",), ()),
        ),
        relationships=(
            DatabaseRelationship(
                "customer_identity",
                "sales.public.orders.customer_id",
                "support.public.calls.customer_id",
                "one-to-one",
                "Shared customer identifier",
            ),
        ),
    )
    return replace(load_domain_profile("sales"), sources=source_profile)


class MultiDatabaseConfigTests(SimpleTestCase):
    @patch.dict(
        os.environ,
        {
            "DB_AGENT_DATABASE_URL": (
                "postgresql://reader:secret@db.internal/original?sslmode=require"
            ),
            "DATABASES_AVAILABLE": '["sales", "support"]',
        },
        clear=True,
    )
    def test_database_list_builds_allowlisted_urls_and_secure_defaults(self):
        config = DBAgentConfig.from_env()

        self.assertEqual(config.available_databases, ("sales", "support"))
        self.assertTrue(config.enforce_readonly_role)
        self.assertEqual(
            config.database_url_for("support"),
            "postgresql://reader:secret@db.internal/support?sslmode=require",
        )
        with self.assertRaisesRegex(ValueError, "not in DATABASES_AVAILABLE"):
            config.database_url_for("secret")

    def test_database_list_rejects_malformed_duplicate_and_oversized_values(self):
        values = (
            "sales,support",
            '["sales", "sales"]',
            json.dumps([f"db{index}" for index in range(26)]),
            '["../secret"]',
        )
        for value in values:
            with (
                self.subTest(value=value),
                patch.dict(
                    os.environ,
                    {
                        "DB_AGENT_DATABASE_URL": "postgresql://reader@host/base",
                        "DATABASES_AVAILABLE": value,
                    },
                    clear=True,
                ),
                self.assertRaises(ValueError),
            ):
                DBAgentConfig.from_env()

    @patch.dict(
        os.environ,
        {
            "DB_AGENT_DATABASE_URL": "postgresql://reader@host/base",
            "DB_AGENT_ENFORCE_READONLY_ROLE": "false",
        },
        clear=True,
    )
    def test_privilege_audit_has_explicit_development_escape_hatch(self):
        config = DBAgentConfig.from_env()

        self.assertFalse(config.enforce_readonly_role)

    def test_profile_database_names_must_match_environment_exactly(self):
        profile = load_domain_profile("sales")
        config = DBAgentConfig(
            database_url="postgresql://reader@host/base",
            available_databases=("sales", "support"),
        )

        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            build_database_gateway(config, profile)

    def test_multi_source_profile_requires_explicit_database_list(self):
        profile = load_domain_profile("sales")
        profile = replace(
            profile,
            sources=SourceProfile(
                databases=(
                    profile.sources.databases[0],
                    DatabaseSourceProfile("support", "Support", ("public",), ()),
                ),
                relationships=(),
            ),
        )
        config = DBAgentConfig(database_url="postgresql://reader@host/base")

        with self.assertRaisesRegex(ValueError, "DATABASES_AVAILABLE is required"):
            build_database_gateway(config, profile)


class ReadOnlyBoundaryTests(SimpleTestCase):
    def test_connection_is_read_only_and_always_rolled_back(self):
        connection = MagicMock()
        cursor = MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor
        cursor.fetchone.return_value = ("on",)
        cursor.description = (("value",),)
        cursor.fetchall.return_value = [(1,)]
        adapter = PostgresDatabaseAdapter(
            "postgresql://unused", enforce_readonly_role=False
        )

        with patch(
            "customer_service.db_agent.database.psycopg2.connect",
            return_value=connection,
        ):
            result = adapter.execute_readonly_query("SELECT 1 AS value", 10)

        connection.set_session.assert_called_once_with(readonly=True, autocommit=False)
        connection.rollback.assert_called_once()
        connection.close.assert_called_once()
        connection.commit.assert_not_called()
        self.assertEqual(result.rows, [{"value": 1}])
        executed = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertEqual(executed[0], "SHOW transaction_read_only")
        self.assertTrue(executed[-1].startswith("SELECT * FROM (SELECT 1"))

    def test_select_only_role_passes_all_privilege_audits(self):
        connection = MagicMock()
        cursor = MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor
        cursor.fetchone.side_effect = [
            ("on",),
            (False, False, False, False, False),
            (False,),
            (False,),
            (False,),
            (False,),
            (False,),
        ]
        cursor.description = (("value",),)
        cursor.fetchall.return_value = [(1,)]
        adapter = PostgresDatabaseAdapter("postgresql://unused")

        with patch(
            "customer_service.db_agent.database.psycopg2.connect",
            return_value=connection,
        ):
            result = adapter.execute_readonly_query("SELECT 1 AS value", 10)

        self.assertEqual(result.rows, [{"value": 1}])
        connection.commit.assert_not_called()
        connection.rollback.assert_called_once()
        executed = [
            call.args[0].strip().upper() for call in cursor.execute.call_args_list
        ]
        self.assertTrue(
            all(sql.startswith(("SELECT", "SHOW", "SET LOCAL")) for sql in executed)
        )

    def test_overprivileged_role_is_rejected_before_user_sql(self):
        connection = MagicMock()
        cursor = MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor
        cursor.fetchone.side_effect = [("on",), (True, False, False, False, False)]
        adapter = PostgresDatabaseAdapter("postgresql://unused")

        with (
            patch(
                "customer_service.db_agent.database.psycopg2.connect",
                return_value=connection,
            ),
            self.assertRaises(ReadOnlyRoleError),
        ):
            adapter.execute_readonly_query("SELECT 1", 10)

        executed = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertFalse(any("_agent_result" in sql for sql in executed))
        connection.rollback.assert_called_once()
        connection.commit.assert_not_called()

    def test_parser_rejects_mutation_cross_database_and_dangerous_functions(self):
        catalog = _schema("orders", "id")
        unsafe = (
            "WITH changed AS (DELETE FROM public.orders RETURNING *) SELECT * FROM changed",
            "SELECT nextval('orders_id_seq')",
            "SELECT pg_read_file('/etc/passwd')",
            "SELECT query_to_xml('SELECT * FROM private.secrets', true, false, '')",
            "SELECT * FROM other.public.orders",
            "SELECT * FROM public.orders FOR UPDATE",
            "SELECT 1; DELETE FROM public.orders",
        )
        for sql in unsafe:
            with self.subTest(sql=sql), self.assertRaises(SQLValidationError):
                validate_readonly_sql(sql, catalog)

    def test_local_combiner_rejects_external_access_and_unknown_relations(self):
        for sql in (
            "SELECT * FROM read_csv('/etc/passwd')",
            "SELECT * FROM missing_result",
            "ATTACH 'file.db' AS secrets",
        ):
            with self.subTest(sql=sql), self.assertRaises(SQLValidationError):
                validate_federated_sql(sql, {"sales_result"})


class FederationTests(SimpleTestCase):
    def test_overprivileged_source_fails_closed_instead_of_becoming_partial(self):
        gateway = _gateway(second_error=ReadOnlyRoleError("unsafe role"))

        with self.assertRaisesRegex(ReadOnlyRoleError, "unsafe role"):
            gateway.inspect_catalog()

    def test_readonly_transaction_failure_aborts_federated_execution(self):
        gateway = _gateway()
        catalog = gateway.inspect_catalog()
        gateway._adapters["support"].error = ReadOnlyRoleError("not read only")
        service = FederatedQueryService(
            gateway,
            max_intermediate_rows=100,
            max_federated_bytes=100000,
            max_concurrency=2,
        )

        with self.assertRaisesRegex(ReadOnlyRoleError, "not read only"):
            service.execute(
                [
                    SourceQuery(
                        "support_rows",
                        "support",
                        "SELECT customer_id, tickets FROM public.calls",
                    )
                ],
                "SELECT * FROM support_rows",
                catalog=catalog,
                purpose="Support",
                max_rows=20,
            )

    def test_declared_relationship_must_reference_real_allowed_columns(self):
        gateway = _gateway()
        gateway._relationships = (
            DatabaseRelationship(
                "broken",
                "sales.public.orders.missing_customer_id",
                "support.public.calls.customer_id",
                "one-to-one",
                "Invalid mapping",
            ),
        )

        with self.assertRaisesRegex(Exception, "missing_customer_id"):
            gateway.inspect_catalog(refresh=True)

    def test_different_schemas_are_joined_and_aggregated_exactly(self):
        gateway = _gateway()
        catalog = gateway.inspect_catalog()
        service = FederatedQueryService(
            gateway,
            max_intermediate_rows=100,
            max_federated_bytes=100000,
            max_concurrency=2,
        )

        execution = service.execute(
            [
                SourceQuery(
                    "sales_rows",
                    "sales",
                    "SELECT customer_id, revenue FROM public.orders",
                ),
                SourceQuery(
                    "support_rows",
                    "support",
                    "SELECT customer_id, tickets FROM public.calls",
                ),
            ],
            """
            SELECT
                sum(cast(s.revenue AS decimal(18,2))) AS total_revenue,
                sum(c.tickets) AS total_tickets
            FROM sales_rows s
            JOIN support_rows c USING (customer_id)
            """,
            catalog=catalog,
            purpose="Combine sales and support",
            max_rows=20,
        )

        self.assertEqual(
            execution.result.rows,
            [{"total_revenue": "30.75", "total_tickets": 4}],
        )
        self.assertEqual(execution.used_databases, {"sales", "support"})
        self.assertEqual(
            sorted(trace.stage for trace in execution.traces),
            ["result", "source", "source"],
        )

    def test_truncated_intermediate_data_refuses_exact_result(self):
        gateway = _gateway()
        adapter = gateway._adapters["sales"]  # Test-only inspection of fake source.
        result = next(iter(adapter.results.values()))
        adapter.results[result.sql] = replace(result, truncated=True)
        service = FederatedQueryService(
            gateway,
            max_intermediate_rows=1,
            max_federated_bytes=100000,
            max_concurrency=2,
        )

        with self.assertRaisesRegex(FederatedQueryError, "row limit"):
            service.execute(
                [
                    SourceQuery(
                        "sales_rows",
                        "sales",
                        "SELECT customer_id, revenue FROM public.orders",
                    )
                ],
                "SELECT * FROM sales_rows",
                catalog=gateway.inspect_catalog(),
                purpose="Sales",
                max_rows=20,
            )

    def test_unavailable_database_is_retained_in_coverage(self):
        gateway = _gateway(second_error=DatabaseExecutionError("offline"))
        catalog = gateway.inspect_catalog()

        coverage = catalog.coverage({"sales"})

        self.assertEqual(
            [(item.database, item.status) for item in coverage],
            [("sales", "used"), ("support", "unavailable")],
        )


def _tool_response(name: str, arguments: dict[str, Any], call_id: str) -> LLMResponse:
    encoded = json.dumps(arguments)
    return LLMResponse(
        content="",
        model="test-model",
        latency_ms=1.0,
        message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": encoded},
                }
            ],
        },
        tool_calls=[{"id": call_id, "name": name, "arguments": encoded}],
    )


class _Client:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def ask(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.calls.append({"prompt": prompt, **kwargs})
        return self.responses.pop(0)


class FederatedWorkflowTests(SimpleTestCase):
    def test_workflow_inspects_selected_schemas_then_runs_federated_plan(self):
        source_queries = [
            {
                "result_name": "sales_rows",
                "database": "sales",
                "sql": "SELECT customer_id, revenue FROM public.orders",
            },
            {
                "result_name": "support_rows",
                "database": "support",
                "sql": "SELECT customer_id, tickets FROM public.calls",
            },
        ]
        client = _Client(
            [
                _tool_response(
                    SCHEMA_TOOL_NAME,
                    {"databases": ["sales", "support"]},
                    "schema-call",
                ),
                _tool_response(
                    FEDERATED_TOOL_NAME,
                    {
                        "purpose": "Executive summary",
                        "source_queries": source_queries,
                        "combine_sql": (
                            "SELECT sum(cast(revenue AS decimal(18,2))) AS revenue "
                            "FROM sales_rows"
                        ),
                    },
                    "query-call",
                ),
                LLMResponse(
                    content="Revenue was 30.75.",
                    model="test-model",
                    latency_ms=1.0,
                    message={"role": "assistant", "content": "Revenue was 30.75."},
                ),
            ]
        )
        config = DBAgentConfig(
            database_url="postgresql://unused/base",
            available_databases=("sales", "support"),
            max_tool_calls=2,
        )
        agent = DBAgent(
            client=client,
            adapter=None,
            config=config,
            profile=_profile(),
            gateway=_gateway(),
        )

        result = agent.ask("Give me a combined executive report")

        self.assertEqual(result.answer, "Revenue was 30.75.")
        self.assertEqual(
            [(item.database, item.status) for item in result.coverage or []],
            [("sales", "used"), ("support", "used")],
        )
        self.assertEqual(len(result.traces or []), 3)
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(
            [tool["function"]["name"] for tool in client.calls[0]["tools"]],
            [SCHEMA_TOOL_NAME, FEDERATED_TOOL_NAME],
        )

    def test_runtime_source_failure_remains_visible_after_partial_retry(self):
        gateway = _gateway()
        gateway.inspect_catalog()
        gateway._adapters["support"].error = DatabaseExecutionError("offline")
        first_plan = {
            "purpose": "Combined summary",
            "source_queries": [
                {
                    "result_name": "sales_rows",
                    "database": "sales",
                    "sql": "SELECT customer_id, revenue FROM public.orders",
                },
                {
                    "result_name": "support_rows",
                    "database": "support",
                    "sql": "SELECT customer_id, tickets FROM public.calls",
                },
            ],
            "combine_sql": "SELECT * FROM sales_rows",
        }
        partial_plan = {
            "purpose": "Partial sales summary",
            "source_queries": [
                {
                    "result_name": "sales_rows",
                    "database": "sales",
                    "sql": "SELECT customer_id, revenue FROM public.orders",
                }
            ],
            "combine_sql": "SELECT * FROM sales_rows",
        }
        client = _Client(
            [
                _tool_response(
                    SCHEMA_TOOL_NAME,
                    {"databases": ["sales", "support"]},
                    "schema-call",
                ),
                _tool_response(FEDERATED_TOOL_NAME, first_plan, "failed-plan"),
                _tool_response(FEDERATED_TOOL_NAME, partial_plan, "partial-plan"),
                LLMResponse(
                    content="Sales results are partial because support is offline.",
                    model="test-model",
                    latency_ms=1.0,
                    message={
                        "role": "assistant",
                        "content": "Sales results are partial because support is offline.",
                    },
                ),
            ]
        )
        config = DBAgentConfig(
            database_url="postgresql://unused/base",
            available_databases=("sales", "support"),
            max_tool_calls=2,
        )
        agent = DBAgent(
            client=client,
            adapter=None,
            config=config,
            profile=_profile(),
            gateway=gateway,
        )

        result = agent.ask("Give me a combined report")

        self.assertEqual(
            [(item.database, item.status) for item in result.coverage or []],
            [("sales", "used"), ("support", "unavailable")],
        )
