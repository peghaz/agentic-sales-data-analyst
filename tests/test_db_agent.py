"""Read-only agent workflow and validation tests."""

import json
import os
from datetime import date
from decimal import Decimal
from typing import Any
from unittest.mock import patch

from django.test import SimpleTestCase

from customer_service.db_agent.agent import DBAgent, QueryTrace
from customer_service.db_agent.config import DBAgentConfig
from customer_service.db_agent.database import (
    ColumnMeta,
    DatabaseSchema,
    PostgresDatabaseAdapter,
    QueryResult,
    TableMeta,
)
from customer_service.db_agent.memory import remember_turn
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


class DBAgentConfigTests(SimpleTestCase):
    @patch.dict(
        os.environ,
        {"DB_AGENT_DATABASE_URL": "postgresql://unused"},
        clear=True,
    )
    def test_model_defaults_keep_thinking_on_with_4096_token_budget(self):
        config = DBAgentConfig.from_env()

        self.assertEqual(config.model_max_tokens, 4096)
        self.assertTrue(config.enable_thinking)

    @patch.dict(
        os.environ,
        {
            "DB_AGENT_DATABASE_URL": "postgresql://unused",
            "DB_AGENT_MODEL_MAX_TOKENS": "8192",
            "DB_AGENT_ENABLE_THINKING": "off",
        },
        clear=True,
    )
    def test_model_settings_can_be_overridden(self):
        config = DBAgentConfig.from_env()

        self.assertEqual(config.model_max_tokens, 8192)
        self.assertFalse(config.enable_thinking)

    def test_invalid_model_token_budget_is_rejected(self):
        for value in ("0", "-1", "many"):
            with (
                self.subTest(value=value),
                patch.dict(
                    os.environ,
                    {
                        "DB_AGENT_DATABASE_URL": "postgresql://unused",
                        "DB_AGENT_MODEL_MAX_TOKENS": value,
                    },
                    clear=True,
                ),
                self.assertRaisesRegex(
                    ValueError, "DB_AGENT_MODEL_MAX_TOKENS must be a positive integer"
                ),
            ):
                DBAgentConfig.from_env()

    @patch.dict(
        os.environ,
        {
            "DB_AGENT_DATABASE_URL": "postgresql://unused",
            "DB_AGENT_ENABLE_THINKING": "sometimes",
        },
        clear=True,
    )
    def test_invalid_thinking_value_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "DB_AGENT_ENABLE_THINKING must be one of"
        ):
            DBAgentConfig.from_env()


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
        for call in client.calls:
            self.assertEqual(call["max_tokens"], 4096)
            self.assertEqual(
                call["extra_body"],
                {"chat_template_kwargs": {"enable_thinking": True}},
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
        self.assertIn("non-technical business user", prompt)
        self.assertIn("If a follow-up reference is unclear", prompt)

    def test_follow_up_receives_prior_answer_and_query_evidence(self):
        client = _FakeClient(
            [
                _tool_response(("call-1", "SELECT 1 FROM public.orders")),
                _text_response("There is one order."),
                _text_response("That means one purchase was recorded."),
            ]
        )
        agent = DBAgent(client=client, adapter=_FakeAdapter(), config=_config(2))

        agent.ask("How many orders?", thread_id="chat-a")
        follow_up = agent.ask("What does that mean?", thread_id="chat-a")

        self.assertEqual(follow_up.traces, [])
        second_messages = client.calls[2]["messages"]
        self.assertIn({"role": "user", "content": "How many orders?"}, second_messages)
        self.assertIn(
            {"role": "assistant", "content": "There is one order."}, second_messages
        )
        self.assertTrue(
            any(
                "SELECT 1 FROM public.orders" in message["content"]
                for message in second_messages
            )
        )
        self.assertEqual(client.calls[2]["tool_choice"], "auto")

    def test_threads_do_not_share_conversation_context(self):
        client = _FakeClient(
            [
                _tool_response(("call-1", "SELECT 1 FROM public.orders")),
                _text_response("First answer"),
                _tool_response(("call-2", "SELECT 2 FROM public.orders")),
                _text_response("Second answer"),
            ]
        )
        agent = DBAgent(client=client, adapter=_FakeAdapter(), config=_config(1))

        agent.ask("First question", thread_id="chat-a")
        agent.ask("Second question", thread_id="chat-b")

        second_messages = client.calls[2]["messages"]
        self.assertFalse(
            any("First question" in message["content"] for message in second_messages)
        )
        self.assertEqual(client.calls[2]["tool_choice"], "required")

    def test_history_is_limited_to_three_completed_turns(self):
        client = _FakeClient(
            [
                _tool_response(("call-1", "SELECT 1 FROM public.orders")),
                _text_response("Answer 0"),
                _text_response("Answer 1"),
                _text_response("Answer 2"),
                _text_response("Answer 3"),
                _text_response("Answer 4"),
            ]
        )
        agent = DBAgent(client=client, adapter=_FakeAdapter(), config=_config(1))

        for index in range(5):
            agent.ask(f"Question {index}", thread_id="chat-a")

        final_messages = client.calls[-1]["messages"]
        self.assertFalse(
            any(msg.get("content") == "Question 0" for msg in final_messages)
        )
        for index in (1, 2, 3, 4):
            self.assertTrue(
                any(msg.get("content") == f"Question {index}" for msg in final_messages)
            )

    def test_failed_turn_does_not_become_follow_up_history(self):
        client = _FakeClient(
            [
                _tool_response(("bad", "DELETE FROM public.orders")),
                _tool_response(("good", "SELECT 1 FROM public.orders")),
                _text_response("Recovered answer"),
            ]
        )
        config = DBAgentConfig(
            database_url="postgresql://unused", max_tool_calls=1, query_retries=0
        )
        agent = DBAgent(client=client, adapter=_FakeAdapter(), config=config)

        with self.assertRaisesRegex(Exception, "validation failed"):
            agent.ask("Failed question", thread_id="chat-a")
        result = agent.ask("New question", thread_id="chat-a")

        self.assertEqual(result.answer, "Recovered answer")
        self.assertFalse(
            any(
                message.get("content") == "Failed question"
                for message in client.calls[1]["messages"]
            )
        )

    def test_data_follow_up_runs_new_validated_query(self):
        client = _FakeClient(
            [
                _tool_response(("first", "SELECT 1 FROM public.orders")),
                _text_response("All orders were counted."),
                _tool_response(("second", "SELECT 2 FROM public.orders")),
                _text_response("The filtered count is two."),
            ]
        )
        adapter = _FakeAdapter()
        agent = DBAgent(client=client, adapter=adapter, config=_config(1))

        agent.ask("Count all orders", thread_id="chat-a")
        follow_up = agent.ask("What about the filtered set?", thread_id="chat-a")

        self.assertEqual(len(adapter.executed_sql), 2)
        self.assertEqual(follow_up.traces[0].rows[0]["value"], 2)
        self.assertIn(
            {"role": "user", "content": "Count all orders"},
            client.calls[2]["messages"],
        )

    def test_forced_retry_requires_a_fresh_query(self):
        client = _FakeClient(
            [
                _tool_response(("first", "SELECT 1 FROM public.orders")),
                _text_response("One order."),
                _tool_response(("retry", "SELECT 2 FROM public.orders")),
                _text_response("Two orders."),
            ]
        )
        agent = DBAgent(client=client, adapter=_FakeAdapter(), config=_config(1))

        agent.ask("Count orders", thread_id="chat-a")
        agent.ask("Count orders", thread_id="chat-a", force_query=True)

        self.assertEqual(client.calls[2]["tool_choice"], "required")

    def test_invalid_sql_can_be_corrected_inside_one_turn(self):
        client = _FakeClient(
            [
                _tool_response(("bad", "DELETE FROM public.orders")),
                _tool_response(("good", "SELECT 1 FROM public.orders")),
                _text_response("One order."),
            ]
        )
        agent = DBAgent(client=client, adapter=_FakeAdapter(), config=_config(2))

        result = agent.ask("How many orders?")

        self.assertEqual(result.answer, "One order.")
        self.assertEqual(len(result.traces), 2)
        self.assertIsNotNone(result.traces[0].error)
        self.assertIsNone(result.traces[1].error)


class MemoryTests(SimpleTestCase):
    def test_explanation_keeps_latest_query_evidence(self):
        trace = QueryTrace(
            sql="SELECT 1 FROM public.orders",
            purpose="Count orders",
            row_count=1,
            truncated=False,
            columns=("orders",),
            rows=[{"orders": 1}],
        )

        first = remember_turn([], "How many orders?", "One order.", [trace])
        second = remember_turn(first, "What does that mean?", "One purchase.", [])

        self.assertEqual(second[-1]["evidence"], first[-1]["evidence"])


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
            "WITH leaked AS (SELECT id FROM public.secret_orders) SELECT id FROM leaked"
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
