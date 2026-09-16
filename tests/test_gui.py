"""Streamlit rendering and chat interaction tests."""

import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from django.test import SimpleTestCase
from streamlit.testing.v1 import AppTest

from customer_service.db_agent.agent import DBAgentResult, QueryTrace
from customer_service.db_agent.config import DBAgentConfig
from customer_service.db_agent.database import DatabaseSchema
from customer_service.llm.client import LLMConfig, LLMResponseError
from customer_service.llm.profile import (
    DomainProfileError,
    PromptCategory,
    load_domain_profile,
)

GUI_PATH = Path(__file__).resolve().parents[1] / "gui.py"


def _assistant_turn(traces: list[QueryTrace]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "## Sales summary\n\nThe analysis completed.",
        "traces": traces,
        "model": "test-model",
        "endpoint": "http://test/v1",
        "latency_ms": 1.0,
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
        "error": None,
    }


def _render(traces: list[QueryTrace]) -> AppTest:
    app = AppTest.from_file(str(GUI_PATH)).run(timeout=10)
    app.session_state["conversation"] = [_assistant_turn(traces)]
    return app.run(timeout=10)


class GUIDomainProfileTests(SimpleTestCase):
    def test_alternate_profile_drives_branding_examples_and_input_copy(self):
        animal_profile = replace(
            load_domain_profile("sales"),
            name="animals",
            app_name="Animal Population Analyst",
            page_icon="🐾",
            page_caption="Population findings across species and habitats.",
            welcome_message="Ask about animal populations and habitats.",
            chat_placeholder="Ask a question about wildlife data",
            analysis_status="Analyzing wildlife data...",
            data_source_label="wildlife data",
            agent_instructions="You are an animal population analyst.",
            example_categories=(
                PromptCategory(
                    label="Population",
                    prompts=("Compare annual population by species.",),
                ),
            ),
        )
        fake_result = DBAgentResult(
            answer="Wolf populations increased.",
            model="test-model",
            latency_ms=1.0,
            traces=[],
        )
        with (
            patch.dict(os.environ, {"DB_AGENT_PROFILE": "animals"}),
            patch(
                "customer_service.llm.profile.load_domain_profile",
                return_value=animal_profile,
            ) as load_profile,
            patch(
                "customer_service.db_agent.agent.DBAgent.ask",
                return_value=fake_result,
            ),
        ):
            app = AppTest.from_file(str(GUI_PATH)).run(timeout=10)
            app.chat_input[0].set_value("Count wolves").run(timeout=10)

        self.assertEqual(list(app.exception), [])
        self.assertEqual(app.title[0].value, "Animal Population Analyst")
        self.assertIn(
            "Population findings across species and habitats.",
            [item.value for item in app.caption],
        )
        self.assertIn(
            "Ask about animal populations and habitats.",
            [item.value for item in app.markdown],
        )
        self.assertIn(
            "Compare annual population by species.",
            [button.label for button in app.button],
        )
        self.assertEqual(
            app.chat_input[0].placeholder, "Ask a question about wildlife data"
        )
        self.assertEqual(app.status[0].label, "Analyzing wildlife data...")
        load_profile.assert_any_call("animals")
        self.assertTrue(
            all(call.args == ("animals",) for call in load_profile.call_args_list)
        )

    def test_invalid_profile_shows_actionable_configuration_state(self):
        with (
            patch(
                "customer_service.llm.profile.load_domain_profile",
                side_effect=DomainProfileError(
                    "Missing required section: Page caption"
                ),
            ),
            patch("customer_service.gui_logging.log_analysis_failure") as log_failure,
        ):
            app = AppTest.from_file(str(GUI_PATH)).run(timeout=10)

        self.assertEqual(list(app.exception), [])
        self.assertEqual(
            [item.value for item in app.error],
            ["The selected domain profile could not be loaded."],
        )
        self.assertIn(
            "Check DB_AGENT_PROFILE",
            " ".join(item.value for item in app.caption),
        )
        self.assertIn("Missing required section", app.code[0].value)
        log_failure.assert_called_once()


class GUITraceRenderingTests(SimpleTestCase):
    def test_monthly_results_show_chart_and_readable_table_headers(self):
        trace = QueryTrace(
            sql="SELECT month, revenue FROM sales",
            purpose="Monthly revenue",
            row_count=2,
            truncated=False,
            columns=("month", "revenue"),
            rows=[
                {"month": "2026-01", "revenue": 10},
                {"month": "2026-02", "revenue": 20},
            ],
        )

        app = _render([trace])

        self.assertEqual(list(app.exception), [])
        self.assertEqual(len(app.get("vega_lite_chart")), 1)
        self.assertEqual(list(app.dataframe[0].value.columns), ["Month", "Revenue"])

    def test_limited_result_shows_caveat_and_no_chart(self):
        trace = QueryTrace(
            sql="SELECT month, revenue FROM sales",
            purpose="Monthly revenue",
            row_count=2,
            truncated=True,
            columns=("month", "revenue"),
            rows=[
                {"month": "2026-01", "revenue": 10},
                {"month": "2026-02", "revenue": 20},
            ],
        )

        app = _render([trace])

        self.assertEqual(list(app.exception), [])
        self.assertEqual(len(app.get("vega_lite_chart")), 0)
        self.assertIn(
            "This result was limited to the rows shown; conclusions may be partial.",
            [item.value for item in app.warning],
        )

    def test_empty_result_is_explained(self):
        trace = QueryTrace(
            sql="SELECT month FROM sales WHERE false",
            purpose="Monthly sales",
            row_count=0,
            truncated=False,
            columns=("month",),
            rows=[],
        )

        app = _render([trace])

        self.assertEqual(list(app.exception), [])
        self.assertIn(
            "No matching data was found.",
            [item.value for item in app.caption],
        )

    def test_clear_chat_starts_a_new_thread(self):
        app = AppTest.from_file(str(GUI_PATH)).run(timeout=10)
        old_thread_id = app.session_state["thread_id"]

        app.button[0].click().run(timeout=10)

        self.assertEqual(list(app.exception), [])
        self.assertNotEqual(app.session_state["thread_id"], old_thread_id)
        self.assertEqual(len(app.session_state["conversation"]), 1)

    def test_chat_questions_reuse_one_thread_id(self):
        fake_result = DBAgentResult(
            answer="A short finding.", model="test-model", latency_ms=1.0, traces=[]
        )
        with patch(
            "customer_service.db_agent.agent.DBAgent.ask", return_value=fake_result
        ) as ask:
            app = AppTest.from_file(str(GUI_PATH)).run(timeout=10)
            app.chat_input[0].set_value("First question").run(timeout=10)
            app.chat_input[0].set_value("And for GBP?").run(timeout=10)

        self.assertEqual(list(app.exception), [])
        self.assertEqual(ask.call_count, 2)
        self.assertEqual(
            ask.call_args_list[0].kwargs["thread_id"],
            ask.call_args_list[1].kwargs["thread_id"],
        )
        self.assertEqual(len(app.session_state["conversation"]), 5)

    def test_retry_requests_fresh_data(self):
        fake_result = DBAgentResult(
            answer="A short finding.", model="test-model", latency_ms=1.0, traces=[]
        )
        with patch(
            "customer_service.db_agent.agent.DBAgent.ask", return_value=fake_result
        ) as ask:
            app = AppTest.from_file(str(GUI_PATH)).run(timeout=10)
            app.chat_input[0].set_value("Count orders").run(timeout=10)
            retry = next(button for button in app.button if button.label == "Retry")
            retry.click().run(timeout=10)

        self.assertEqual(list(app.exception), [])
        self.assertEqual(ask.call_count, 2)
        self.assertTrue(ask.call_args_list[1].kwargs["force_query"])

    def test_nested_result_fields_render_as_separate_tables(self):
        trace = QueryTrace(
            sql="SELECT summary",
            purpose="Sales summary",
            row_count=1,
            truncated=False,
            columns=("revenue_by_currency", "top_shop", "active_customers"),
            rows=[
                {
                    "revenue_by_currency": [
                        {"currency": "EUR", "revenue": 790522.56},
                        {"currency": "USD", "revenue": 538851.08},
                    ],
                    "top_shop": {"name": "Shop 3", "currency": "EUR"},
                    "active_customers": 145,
                }
            ],
        )

        app = _render([trace])

        self.assertEqual(list(app.exception), [])
        self.assertEqual(len(app.dataframe), 3)
        self.assertEqual(app.dataframe[0].value.iloc[0]["Active customers"], 145)
        self.assertEqual(app.dataframe[1].value.iloc[0]["Currency"], "EUR")
        self.assertEqual(app.metric[0].label, "Active customers")
        self.assertEqual(app.metric[0].value, "145")
        markdown = [element.value for element in app.markdown]
        self.assertIn("**Summary**", markdown)
        self.assertIn("**Revenue by currency**", markdown)
        self.assertIn("**Top shop**", markdown)
        self.assertIn("##### Data behind this answer", markdown)
        self.assertIn("**Step 1** · 1 row", markdown)

    def test_recovered_failure_is_collapsed_and_not_shown_as_error(self):
        failed = QueryTrace(
            sql="WITH window AS (SELECT 1) SELECT * FROM window",
            purpose="Sales summary",
            row_count=0,
            truncated=False,
            columns=(),
            rows=[],
            error='syntax error at or near "window"',
        )
        corrected = QueryTrace(
            sql="WITH sales_window AS (SELECT 1) SELECT * FROM sales_window",
            purpose="Sales summary",
            row_count=1,
            truncated=False,
            columns=("value",),
            rows=[{"value": 1}],
        )

        app = _render([failed, corrected])

        self.assertEqual(list(app.exception), [])
        self.assertEqual(len(app.error), 0)
        self.assertIn(
            "How this was calculated",
            [expander.label for expander in app.expander],
        )
        self.assertIn(
            "**Step 1 · corrected after retry** · 0 rows",
            [item.value for item in app.markdown],
        )
        self.assertEqual(list(app.warning), [])

    def test_unrecovered_failure_remains_visible(self):
        failed = QueryTrace(
            sql="SELECT * FROM unavailable",
            purpose="Sales summary",
            row_count=0,
            truncated=False,
            columns=(),
            rows=[],
            error="Table is unavailable",
        )

        app = _render([failed])

        self.assertEqual(list(app.exception), [])
        self.assertEqual(
            [warning.value for warning in app.warning],
            [
                (
                    "Part of this analysis could not be completed. "
                    "See how this was calculated for details."
                )
            ],
        )
        self.assertIn("How this was calculated", [item.label for item in app.expander])


class GUIAgentIntegrationTests(SimpleTestCase):
    def test_empty_model_response_survives_real_graph_and_logs_error_id(self):
        """Exercise Streamlit through the real LangGraph, without external services."""

        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="",
                        tool_calls=None,
                        reasoning="private model reasoning",
                    ),
                    finish_reason="length",
                )
            ],
            model="test-model",
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=4096,
                total_tokens=4196,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=4096),
            ),
        )
        with (
            patch(
                "customer_service.db_agent.config.DBAgentConfig.from_env",
                return_value=DBAgentConfig(database_url="postgresql://unused"),
            ),
            patch(
                "customer_service.llm.client.LLMConfig.from_env",
                return_value=LLMConfig(
                    model="test-model",
                    base_url="http://model.invalid/v1",
                    api_key="EMPTY",
                ),
            ),
            patch("customer_service.llm.client.OpenAI") as openai_class,
            patch(
                "customer_service.db_agent.database.PostgresDatabaseAdapter.inspect_schema",
                return_value=DatabaseSchema(tables={}),
            ) as inspect_schema,
            patch(
                "customer_service.db_agent.database.PostgresDatabaseAdapter.execute_readonly_query"
            ) as execute_query,
            patch("customer_service.gui_logging.log_analysis_failure") as log_failure,
        ):
            create = openai_class.return_value.chat.completions.create
            create.return_value = completion
            app = AppTest.from_file(str(GUI_PATH)).run(timeout=10)
            app.chat_input[0].set_value("Summarize sales").run(timeout=10)

        self.assertEqual(list(app.exception), [])
        self.assertEqual(len(app.error), 1)
        self.assertIn(
            "The analysis service did not respond as expected.", app.error[0].value
        )
        failure = app.session_state["conversation"][-1]
        self.assertIsInstance(failure["error_id"], str)
        self.assertEqual(len(failure["error_id"]), 8)
        self.assertIn("finish_reason='length'", failure["error"])
        self.assertIn("completion_tokens=4096", failure["error"])
        self.assertIn("reasoning_tokens=4096", failure["error"])
        self.assertNotIn("private model reasoning", failure["error"])
        self.assertIn(
            f"Error ID: {failure['error_id']}", [item.value for item in app.caption]
        )
        self.assertEqual(log_failure.call_count, 1)
        logged_exception = log_failure.call_args.args[0]
        self.assertIsInstance(logged_exception, LLMResponseError)
        self.assertEqual(log_failure.call_args.kwargs["error_id"], failure["error_id"])
        inspect_schema.assert_called_once()
        execute_query.assert_not_called()
        create.assert_called_once()
        request = create.call_args.kwargs
        self.assertEqual(request["max_tokens"], 4096)
        self.assertEqual(
            request["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": True}},
        )
