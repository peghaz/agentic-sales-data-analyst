"""Deterministic analysis-card and chart-selection tests."""

from django.test import SimpleTestCase

from customer_service.db_agent.presentation import charts_for_trace, metrics_for_traces
from customer_service.db_agent.types import QueryTrace


def _trace(rows, columns, *, truncated=False):
    return QueryTrace(
        sql="SELECT ...",
        purpose="Sales trend",
        row_count=len(rows),
        truncated=truncated,
        columns=columns,
        rows=rows,
    )


class PresentationTests(SimpleTestCase):
    def test_metric_cards_are_scalar_and_currency_scoped(self):
        trace = _trace(
            [{"currency": "GBP", "revenue": "12345.67", "orders": 34, "shop_id": 2}],
            ("currency", "revenue", "orders", "shop_id"),
        )

        metrics = metrics_for_traces([trace])

        self.assertEqual(
            [(metric.label, metric.value) for metric in metrics],
            [("Revenue · GBP", "12,345.67"), ("Orders · GBP", "34")],
        )

    def test_small_rates_are_not_rounded_to_zero_or_nonfinite(self):
        trace = _trace(
            [{"failure_rate": "0.0043", "other_rate": "NaN"}],
            ("failure_rate", "other_rate"),
        )

        self.assertEqual(
            [(metric.label, metric.value) for metric in metrics_for_traces([trace])],
            [("Failure rate", "0.0043")],
        )

    def test_monthly_currency_series_create_separate_line_charts(self):
        trace = _trace(
            [
                {"month": "2026-01", "currency": "EUR", "revenue": "10"},
                {"month": "2026-02", "currency": "EUR", "revenue": "20"},
                {"month": "2026-01", "currency": "USD", "revenue": "30"},
                {"month": "2026-02", "currency": "USD", "revenue": "40"},
            ],
            ("month", "currency", "revenue"),
        )

        charts = charts_for_trace(trace)

        self.assertEqual([chart.kind for chart in charts], ["line", "line"])
        self.assertEqual(
            [chart.title for chart in charts],
            ["Revenue over time · EUR", "Revenue over time · USD"],
        )
        self.assertEqual([chart.rows[0]["revenue"] for chart in charts], [10.0, 30.0])

    def test_ranked_categories_get_bar_chart_but_ambiguous_rows_do_not(self):
        ranked = _trace(
            [
                {"shop_name": "North", "orders": 5},
                {"shop_name": "South", "orders": 3},
            ],
            ("shop_name", "orders"),
        )
        ambiguous = _trace(
            [
                {"month": "2026-01", "shop_name": "North", "revenue": 5},
                {"month": "2026-02", "shop_name": "South", "revenue": 6},
            ],
            ("month", "shop_name", "revenue"),
        )

        self.assertEqual(charts_for_trace(ranked)[0].kind, "bar")
        self.assertEqual(charts_for_trace(ambiguous), [])
        self.assertEqual(
            charts_for_trace(_trace(ranked.rows, ranked.columns, truncated=True)), []
        )
