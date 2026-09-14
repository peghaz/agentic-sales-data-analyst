"""Conservative, deterministic view hints derived from verified query results."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from .types import QueryTrace

_METRIC_WORDS = (
    "count",
    "orders",
    "customers",
    "revenue",
    "sales",
    "amount",
    "total",
    "average",
    "avg",
    "aov",
    "profit",
    "margin",
    "rate",
    "cost",
)
_TIME_NAMES = ("month", "date", "day", "week", "year")
_CATEGORY_NAMES = (
    "shop",
    "product",
    "category",
    "customer",
    "method",
    "carrier",
    "country",
    "name",
)


@dataclass(frozen=True)
class Metric:
    label: str
    value: str


@dataclass(frozen=True)
class ChartSpec:
    title: str
    kind: Literal["line", "bar"]
    x: str
    y: str
    rows: list[dict[str, Any]]


def display_label(name: str) -> str:
    return name.replace("_", " ").strip().capitalize()


def _number(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return number if number.is_finite() else None


def _is_metric_name(name: str) -> bool:
    lower = name.lower()
    return (
        lower != "id"
        and not lower.endswith("_id")
        and any(word in lower for word in _METRIC_WORDS)
    )


def _format_number(value: Decimal) -> str:
    if value == value.to_integral_value():
        return f"{value:,.0f}"
    if abs(value) < 1:
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return f"{value:,.2f}"


def metrics_for_traces(traces: list[QueryTrace], limit: int = 6) -> list[Metric]:
    """Highlight unambiguous scalar metrics, never totals across currencies."""

    metrics: list[Metric] = []
    seen: set[str] = set()
    for trace in traces:
        if trace.error or trace.truncated or trace.row_count != 1 or not trace.rows:
            continue
        row = trace.rows[0]
        currency = row.get("currency")
        for name, value in row.items():
            if not _is_metric_name(name):
                continue
            number = _number(value)
            if number is None:
                continue
            label = display_label(name)
            if currency:
                label = f"{label} · {currency}"
            if label in seen:
                continue
            seen.add(label)
            metrics.append(Metric(label, _format_number(number)))
            if len(metrics) >= limit:
                return metrics
    return metrics


def _axis(columns: tuple[str, ...], words: tuple[str, ...]) -> str | None:
    return next(
        (name for name in columns if any(word in name.lower() for word in words)),
        None,
    )


def _chart_for_rows(
    rows: list[dict[str, Any]], columns: tuple[str, ...], currency: str | None
) -> ChartSpec | None:
    if not 2 <= len(rows) <= 100:
        return None
    time_axis = _axis(columns, _TIME_NAMES)
    category_axis = _axis(columns, _CATEGORY_NAMES)
    x = time_axis or category_axis
    if not x:
        return None
    if time_axis is None and len(rows) > 15:
        return None
    if len({str(row.get(x)) for row in rows}) != len(rows):
        return None

    other_dimensions = [
        name
        for name in columns
        if name not in {x, "currency"}
        and not _is_metric_name(name)
        and any(row.get(name) is not None for row in rows)
    ]
    if other_dimensions:
        return None
    y = next(
        (
            name
            for name in columns
            if name != x
            and _is_metric_name(name)
            and all(_number(row.get(name)) is not None for row in rows)
        ),
        None,
    )
    if not y:
        return None
    suffix = "over time" if time_axis else f"by {display_label(x).lower()}"
    title = f"{display_label(y)} {suffix}"
    if currency:
        title += f" · {currency}"
    chart_rows = [
        {x: row[x], y: float(_number(row[y]))}  # type: ignore[arg-type]
        for row in rows
    ]
    return ChartSpec(title, "line" if time_axis else "bar", x, y, chart_rows)


def charts_for_trace(trace: QueryTrace) -> list[ChartSpec]:
    """Only chart flat, complete data with a clear axis and monetary partition."""

    if trace.error or trace.truncated or not trace.rows:
        return []
    if any(
        isinstance(value, (dict, list, tuple))
        for row in trace.rows
        for value in row.values()
    ):
        return []
    if "currency" not in trace.columns:
        chart = _chart_for_rows(trace.rows, trace.columns, None)
        return [chart] if chart else []

    currencies = sorted({str(row.get("currency")) for row in trace.rows})
    charts: list[ChartSpec] = []
    for currency in currencies:
        rows = [row for row in trace.rows if str(row.get("currency")) == currency]
        chart = _chart_for_rows(rows, trace.columns, currency)
        if chart:
            charts.append(chart)
    return charts
