import json
from datetime import date
from decimal import Decimal
from typing import Any

from django.test import SimpleTestCase

from customer_service.db_agent.agent import DBAgent
from customer_service.db_agent.config import DBAgentConfig
from customer_service.db_agent.database import (
    ColumnMeta,
    DatabaseSchema,
    PostgresDatabaseAdapter,
    QueryResult,
    TableMeta,
)
from customer_service.db_agent.validator import (
    SQLValidationError,
    validate_readonly_sql,
)
from customer_service.llm.client import LLMResponse


def _catalog() -> DatabaseSchema:
    return DatabaseSchema(
        tables={
            "public.orders": TableMeta(
                schema="public",
                name="orders",
                columns=(
                    ColumnMeta(
                        name="id",
                        data_type="integer",
                        is_nullable=False,
                    ),
                ),
                primary_key=("id",),
            )
        }
    )


def _tool_response(*calls: tuple[str, str]) -> LLMResponse:
    tool_calls = [
        {
            "id": call_id,
            "name": DBAgent.TOOL_NAME,
            "arguments": json.dumps({"sql": sql, "purpose": f"Run {call_id}"}),
        }
        for call_id, sql in calls
    ]
    message_calls = [
        {
            "id": call["id"],
            "type": "function",
            "function": {
                "name": call["name"],
                "arguments": call["arguments"],
            },
        }
        for call in tool_calls
    ]
    return LLMResponse(
        content="",
        model="test-model",
        latency_ms=1.0,
        message={"role": "assistant", "content": "", "tool_calls": message_calls},
        tool_calls=tool_calls,
    )


def _text_response(content: str = "Final answer") -> LLMResponse:
    return LLMResponse(
        content=content,
        model="test-model",
        latency_ms=1.0,
        message={"role": "assistant", "content": content},
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )


class _FakeClient:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def ask(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.calls.append({"prompt": prompt, **kwargs})
        return self.responses.pop(0)


class _FakeAdapter:
    def __init__(self) -> None:
        self.executed_sql: list[str] = []

    def inspect_schema(self, **kwargs: Any) -> DatabaseSchema:
        return _catalog()

    def execute_readonly_query(self, sql: str, max_rows: int) -> QueryResult:
        self.executed_sql.append(sql)
        return QueryResult(
            sql=sql,
            columns=("value",),
            rows=[{"value": len(self.executed_sql)}],
            row_count=1,
        )


def _config(max_tool_calls: int) -> DBAgentConfig:
    return DBAgentConfig(
        database_url="postgresql://unused",
        max_tool_calls=max_tool_calls,
    )


class DBAgentLoopTests(SimpleTestCase):
    def test_tool_budget_allows_separate_final_answer_turn(self):
        client = _FakeClient(
            [
                _tool_response(("call-1", "SELECT 1 FROM public.orders")),
                _tool_response(("call-2", "SELECT 2 FROM public.orders")),
                _text_response(),
            ]
        )
        adapter = _FakeAdapter()
        agent = DBAgent(client=client, adapter=adapter, config=_config(2))

        result = agent.ask("Summarize orders")

        self.assertEqual(result.answer, "Final answer")
        self.assertEqual(len(result.traces or []), 2)
        self.assertEqual(len(client.calls), 3)
        self.assertIn("tools", client.calls[0])
        self.assertIn("tools", client.calls[1])
        self.assertNotIn("tools", client.calls[2])
        self.assertIn(
            "tool-call budget of 2 has been used",
            client.calls[2]["messages"][-1]["content"],
        )

    def test_answer_before_budget_does_not_add_finalization_turn(self):
        client = _FakeClient(
            [
                _tool_response(("call-1", "SELECT 1 FROM public.orders")),
                _text_response("One order."),
            ]
        )
        agent = DBAgent(
            client=client,
            adapter=_FakeAdapter(),
            config=_config(5),
        )

        result = agent.ask("How many orders?")

        self.assertEqual(result.answer, "One order.")
        self.assertEqual(len(client.calls), 2)
        self.assertIn("tools", client.calls[1])

    def test_excess_parallel_call_is_answered_but_not_executed(self):
        client = _FakeClient(
            [
                _tool_response(
                    ("call-1", "SELECT 1 FROM public.orders"),
                    ("call-2", "SELECT 2 FROM public.orders"),
                ),
                _text_response(),
            ]
        )
        adapter = _FakeAdapter()
        agent = DBAgent(client=client, adapter=adapter, config=_config(1))

        result = agent.ask("Summarize orders")

        self.assertEqual(adapter.executed_sql, ["SELECT 1 FROM public.orders"])
        self.assertEqual(len(result.traces or []), 2)
        self.assertIn("was not executed", result.traces[-1].error or "")
        final_messages = client.calls[-1]["messages"]
        tool_message_ids = [
            message["tool_call_id"]
            for message in final_messages
            if message["role"] == "tool"
        ]
        self.assertEqual(tool_message_ids, ["call-1", "call-2"])

    def test_system_prompt_prefers_flat_results_and_safe_aliases(self):
        agent = DBAgent(
            client=_FakeClient([]),
            adapter=_FakeAdapter(),
            config=_config(5),
        )

        prompt = agent._build_system_prompt(_catalog())

        self.assertIn("flat relational result set", prompt)
        self.assertIn("json_agg", prompt)
        self.assertIn("non-reserved names", prompt)
        self.assertIn("at most 5 SQL tool calls", prompt)
        self.assertIn("Do not spend a separate exploratory call", prompt)


class SQLValidatorCTETests(SimpleTestCase):
    def test_allows_cte_relation_names(self):
        sql = (
            "WITH bounds AS (SELECT max(id) AS max_id FROM public.orders) "
            "SELECT orders.id FROM public.orders "
            "CROSS JOIN bounds WHERE orders.id <= bounds.max_id"
        )

        self.assertEqual(validate_readonly_sql(sql, _catalog()), sql)

    def test_allows_multiple_and_recursive_ctes(self):
        sql = (
            "WITH RECURSIVE order_ids(id) AS ("
            "SELECT id FROM public.orders UNION ALL "
            "SELECT orders.id FROM public.orders JOIN order_ids ON false"
            "), totals AS (SELECT count(*) AS total FROM order_ids) "
            "SELECT total FROM totals"
        )

        self.assertEqual(validate_readonly_sql(sql, _catalog()), sql)

    def test_still_rejects_unavailable_table_inside_cte(self):
        sql = (
            "WITH leaked AS (SELECT id FROM public.secret_orders) "
            "SELECT id FROM leaked"
        )

        with self.assertRaisesRegex(SQLValidationError, "secret_orders"):
            validate_readonly_sql(sql, _catalog())


class DatabaseNormalizationTests(SimpleTestCase):
    def test_recursively_normalizes_nested_postgres_values(self):
        value = {
            "groups": [
                {
                    "amount": Decimal("12.34"),
                    "on": date(2026, 9, 12),
                    "payload": b"sales",
                }
            ],
            "totals": (Decimal("56.78"),),
        }

        normalized = PostgresDatabaseAdapter._normalize_value(value)

        self.assertEqual(
            normalized,
            {
                "groups": [
                    {
                        "amount": "12.34",
                        "on": "2026-09-12",
                        "payload": "sales",
                    }
                ],
                "totals": ["56.78"],
            },
        )
