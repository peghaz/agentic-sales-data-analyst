from __future__ import annotations

import csv
import io
import json
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from customer_service.db_agent import DBAgent, DBAgentError, QueryTrace
from customer_service.db_agent.config import DBAgentConfig
from customer_service.db_agent.database import DatabaseError, PostgresDatabaseAdapter
from customer_service.llm.client import LLMError, OpenAILLMClient

load_dotenv()

WELCOME_MESSAGE = (
    "Ask a sales question to explore customers, orders, products, payments, "
    "and fulfillment."
)

EXAMPLE_PROMPTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Executive",
        (
            (
                "Give me an executive sales summary for the latest 12 months in the data: "
                "revenue by currency, orders, active customers, average order value, top "
                "shop, and top product."
            ),
            (
                "Compare monthly revenue and order volume by shop for the latest 12 months, "
                "including month-over-month change and keeping currencies separate."
            ),
        ),
    ),
    (
        "Customers",
        (
            (
                "Rank the top 10 customers by lifetime spend, showing order count, average "
                "order value, last purchase date, and keeping currencies separate."
            ),
            (
                "Build a customer retention view by signup month: customers acquired and "
                "how many purchased again within 30, 60, and 90 days."
            ),
            (
                "Find high-value customers at risk: at least 5 paid or shipped orders, but "
                "no purchase in the 90 days before the latest order in the dataset."
            ),
        ),
    ),
    (
        "Products",
        (
            (
                "Which product categories deliver the highest estimated gross profit and "
                "margin percentage, using product cost and line-item sales and keeping "
                "currencies separate?"
            ),
            (
                "Find the product pairs most frequently bought together, with pair count "
                "and combined sales by currency."
            ),
        ),
    ),
    (
        "Operations",
        (
            (
                "Compare payment failure rates by payment method and shop, including "
                "attempts, failed payments, and failed amount by currency."
            ),
            (
                "Compare carrier performance by destination country: shipment count, "
                "average and 90th-percentile delivery time, return rate, and shipping cost "
                "by currency."
            ),
            (
                "Show cancellation and refund rates by shop and month, with affected order "
                "value by currency."
            ),
        ),
    ),
)


def _initial_conversation() -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": WELCOME_MESSAGE,
            "traces": [],
            "model": None,
            "endpoint": None,
            "latency_ms": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "error": None,
        }
    ]


def _format_error_for_user(exc: Exception) -> str:
    if isinstance(exc, LLMError):
        return (
            "LLM connection or response validation failed. "
            "Check your model endpoint and make sure tool-calling is enabled."
        )
    if isinstance(exc, DatabaseError):
        return (
            "Could not inspect or query the database. "
            "Check DB credentials, network, and table permissions."
        )
    if isinstance(exc, DBAgentError):
        return f"Agent flow error: {exc}"
    if isinstance(exc, ValueError):
        return str(exc)
    return "Something blocked this turn. Check .env and service availability."


def _build_runtime() -> tuple[OpenAILLMClient, DBAgent, DBAgentConfig]:
    config = DBAgentConfig.from_env()
    client = OpenAILLMClient()
    adapter = PostgresDatabaseAdapter(
        dsn=config.database_url,
        statement_timeout_ms=config.statement_timeout_ms,
    )
    agent = DBAgent(client=client, adapter=adapter, config=config)
    return client, agent, config


@st.cache_resource(show_spinner=False)
def _get_runtime() -> tuple[OpenAILLMClient, DBAgent, DBAgentConfig]:
    return _build_runtime()


def _rows_to_csv(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        return b""

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(
        {
            key: _json_text(value) if _is_nested(value) else value
            for key, value in row.items()
        }
        for row in rows
    )
    return output.getvalue().encode("utf-8")


def _is_nested(value: Any) -> bool:
    return isinstance(value, (dict, list, tuple))


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _table_safe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: _json_text(value) if _is_nested(value) else value
            for key, value in row.items()
        }
        for row in rows
    ]


def _nested_table_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return _table_safe_rows([value])
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        if all(isinstance(item, dict) for item in value):
            return _table_safe_rows(list(value))
        return [
            {"value": _json_text(item) if _is_nested(item) else item}
            for item in value
        ]
    return [{"value": value}]


def _display_label(column: str) -> str:
    return column.replace("_", " ").strip().capitalize()


def _render_trace_rows(rows: list[dict[str, Any]]) -> None:
    nested_cells: list[tuple[int, str, Any]] = []
    scalar_rows: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        scalar_row: dict[str, Any] = {}
        for column, value in row.items():
            if _is_nested(value):
                nested_cells.append((row_index, column, value))
            else:
                scalar_row[column] = value
        if scalar_row:
            scalar_rows.append(scalar_row)

    if not nested_cells:
        st.dataframe(rows, width="stretch", hide_index=True)
        return

    if scalar_rows:
        st.markdown("**Summary**")
        st.dataframe(scalar_rows, width="stretch", hide_index=True)

    for row_index, column, value in nested_cells:
        label = _display_label(column)
        if len(rows) > 1:
            label = f"{label} · Row {row_index + 1}"
        st.markdown(f"**{label}**")
        nested_rows = _nested_table_rows(value)
        if nested_rows:
            st.dataframe(nested_rows, width="stretch", hide_index=True)
        else:
            st.caption("No values returned.")


def _previous_user_query(conversation: list[dict[str, Any]], idx: int) -> str:
    for j in range(idx - 1, -1, -1):
        msg = conversation[j]
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _run_question(question: str) -> None:
    if not question or not question.strip():
        st.warning("Type a question before sending.")
        return

    st.session_state.conversation.append({"role": "user", "content": question})

    with st.status("Analyzing sales data...", expanded=False):
        try:
            client, agent, config = _get_runtime()
            result = agent.ask(question)
            st.session_state.conversation.append(
                {
                    "role": "assistant",
                    "content": result.answer,
                    "traces": result.traces or [],
                    "model": result.model,
                    "endpoint": client.config.base_url,
                    "latency_ms": result.latency_ms,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.total_tokens,
                    "allowed_rows": config.max_rows,
                    "statement_timeout_ms": config.statement_timeout_ms,
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001 - UI boundary maps failures for users.
            st.session_state.conversation.append(
                {
                    "role": "assistant",
                    "content": _format_error_for_user(exc),
                    "traces": [],
                    "model": None,
                    "endpoint": None,
                    "latency_ms": None,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "allowed_rows": None,
                    "statement_timeout_ms": None,
                    "error": str(exc),
                }
            )


def _render_trace(
    trace: QueryTrace,
    turn_index: int,
    trace_index: int,
    *,
    recovered: bool,
) -> None:
    row_label = "row" if trace.row_count == 1 else "rows"
    if recovered:
        with st.expander(
            f"Query {trace_index + 1} · corrected after retry",
            expanded=False,
        ):
            if trace.purpose:
                st.caption(f"Purpose: {trace.purpose}")
            st.warning("This generated query failed and was corrected later.")
            if trace.error:
                st.code(trace.error, language="text")
            if trace.sql:
                st.code(trace.sql, language="sql")
        return

    with st.container(border=True):
        st.markdown(
            f"**Query {trace_index + 1}** · **{trace.row_count} {row_label}**"
            + ("  *(truncated)*" if trace.truncated else ""),
        )
        if trace.purpose:
            st.caption(f"Purpose: {trace.purpose}")

        if trace.error:
            st.error(trace.error)

        if trace.columns:
            if trace.rows:
                _render_trace_rows(trace.rows)

                csv_data = _rows_to_csv(trace.rows)
                st.download_button(
                    "Download CSV",
                    csv_data,
                    file_name=f"trace_{turn_index + 1}_{trace_index + 1}.csv",
                    mime="text/csv",
                    key=f"trace-download-{turn_index}-{trace_index}",
                )
            else:
                st.caption("No rows returned.")

        with st.expander("How this was calculated", expanded=False):
            if trace.sql:
                st.code(trace.sql, language="sql")
            else:
                st.caption("No SQL captured for this trace.")
            st.caption(f"Truncated: {'yes' if trace.truncated else 'no'}")
            st.caption(f"Rows: {trace.row_count}")


def _render_conversation() -> None:
    for turn_index, msg in enumerate(st.session_state.conversation):
        if msg.get("role") == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
            continue

        with st.chat_message("assistant"):
            if msg.get("error"):
                st.error(msg["content"])
                with st.expander("Debug detail", expanded=False):
                    st.caption(msg["error"])
            else:
                st.markdown(msg["content"])

            if msg.get("traces"):
                st.markdown("#### Analysis records")
                traces = msg.get("traces", [])
                for trace_index, trace in enumerate(traces):
                    recovered = bool(
                        trace.error
                        and any(
                            not later_trace.error
                            and (
                                trace.purpose is None
                                or later_trace.purpose == trace.purpose
                            )
                            for later_trace in traces[trace_index + 1 :]
                        )
                    )
                    _render_trace(
                        trace,
                        turn_index,
                        trace_index,
                        recovered=recovered,
                    )
            elif msg.get("model"):
                st.caption("No SQL trace was produced for this response.")

            if msg.get("model"):
                with st.expander("Run metadata", expanded=False):
                    st.caption(f"Model: {msg['model']}")
                    st.caption(f"Endpoint: {msg['endpoint']}")
                    if msg.get("latency_ms") is not None:
                        st.caption(f"Latency: {msg['latency_ms']:.2f} ms")
                    if msg.get("prompt_tokens") is not None:
                        st.caption(
                            f"Tokens: prompt={msg['prompt_tokens']}, "
                            f"completion={msg['completion_tokens']}, total={msg['total_tokens']}"
                        )

            prev_question = _previous_user_query(st.session_state.conversation, turn_index)
            if prev_question:
                cols = st.columns([4, 1])
                with cols[1]:
                    if st.button(
                        "Retry",
                        key=f"retry-{turn_index}",
                        width="stretch",
                    ):
                        st.session_state.queued_prompt = prev_question
                        st.rerun()


def _sidebar() -> None:
    with st.sidebar:
        st.header("Sales Data Analyst")
        if st.button("Clear chat", width="stretch"):
            st.session_state.conversation = _initial_conversation()
            st.rerun()

        st.caption(
            "Read-only mode is enabled. Questions are answered with schema-aware, "
            "SELECT-only SQL."
        )

        with st.expander("Try a showcase question", expanded=True):
            for category_index, (category, prompts) in enumerate(EXAMPLE_PROMPTS):
                st.markdown(f"**{category}**")
                for prompt_index, prompt in enumerate(prompts):
                    if st.button(
                        prompt,
                        key=f"example-{category_index}-{prompt_index}",
                        width="stretch",
                    ):
                        st.session_state.queued_prompt = prompt
                        st.rerun()


st.set_page_config(
    page_title="Sales Data Analyst",
    page_icon="📊",
    layout="wide",
)

if "conversation" not in st.session_state:
    st.session_state.conversation = _initial_conversation()

if "queued_prompt" not in st.session_state:
    st.session_state.queued_prompt = None

st.title("Sales Data Analyst")
st.caption("Ask questions about customers, revenue, products, payments, and fulfillment.")
_sidebar()

queued_prompt = st.session_state.get("queued_prompt")
if queued_prompt:
    st.session_state.queued_prompt = None
    _run_question(queued_prompt)

rendered_trigger = st.chat_input("Ask a question about your sales data")
if rendered_trigger:
    _run_question(rendered_trigger)

_render_conversation()
