from pathlib import Path
from typing import Any

from django.test import SimpleTestCase
from streamlit.testing.v1 import AppTest

from customer_service.db_agent.agent import QueryTrace

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


class GUITraceRenderingTests(SimpleTestCase):
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
        self.assertEqual(app.dataframe[0].value.iloc[0]["active_customers"], 145)
        self.assertEqual(app.dataframe[1].value.iloc[0]["currency"], "EUR")
        markdown = [element.value for element in app.markdown]
        self.assertIn("**Summary**", markdown)
        self.assertIn("**Revenue by currency**", markdown)
        self.assertIn("**Top shop**", markdown)
        self.assertIn("**Query 1** · **1 row**", markdown)

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
            "Query 1 · corrected after retry",
            [expander.label for expander in app.expander],
        )
        self.assertIn(
            "This generated query failed and was corrected later.",
            [warning.value for warning in app.warning],
        )

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
            [error.value for error in app.error],
            ["Table is unavailable"],
        )
