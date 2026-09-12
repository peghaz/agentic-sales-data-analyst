from __future__ import annotations

import csv
import io
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from customer_service.db_agent import DBAgent, DBAgentError, QueryTrace
from customer_service.db_agent.config import DBAgentConfig
from customer_service.db_agent.database import DatabaseError, PostgresDatabaseAdapter
from customer_service.llm.client import LLMError, OpenAILLMClient

load_dotenv()


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

        :root {
            --paper: #0c111d;
            --panel: #121b2d;
            --ink: #edf1f8;
            --muted: #8ea0bb;
            --accent: #7bb8ff;
            --danger: #ff9a5f;
            --radius: 14px;
            --focus: 0 0 0 3px rgba(123, 184, 255, 0.32);
        }

        html, body {
            font-family: 'IBM Plex Sans', 'Avenir Next', 'Segoe UI', sans-serif;
            color: var(--ink);
        }

        .block-container {
            max-width: 1080px;
            padding-top: 2rem;
            padding-bottom: 2rem;
            animation: sweep-in 380ms cubic-bezier(0.16, 1, 0.3, 1);
        }

        .stApp {
            background:
                radial-gradient(circle at 20% 8%, rgba(123, 184, 255, 0.13), transparent 34%),
                radial-gradient(circle at 80% 12%, rgba(20, 64, 114, 0.17), transparent 30%),
                linear-gradient(175deg, #090f1a 0%, #0b1321 42%, #0a1320 100%);
        }

        .assistant-msg {
            border: 1px solid rgba(151, 176, 220, 0.2);
            border-radius: var(--radius);
            background: var(--panel);
            padding: 0.8rem 1rem;
            box-shadow: 0 12px 25px rgba(0, 0, 0, 0.22);
        }

        [data-testid='stChatMessage'] {
            gap: 0.5rem;
        }

        .trace-box {
            border: 1px dashed rgba(141, 162, 188, 0.45);
            border-radius: 11px;
            padding: 0.8rem;
            margin: 0.6rem 0 0.2rem;
            background: rgba(9, 16, 28, 0.66);
        }

        .section-kicker {
            letter-spacing: 0.08em;
            font-size: 0.78rem;
            color: var(--muted);
            text-transform: uppercase;
        }

        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 999px;
            background: linear-gradient(180deg, #7edb8d, #2fbc7d);
            display: inline-block;
            margin-right: 0.35rem;
            box-shadow: 0 0 0 3px rgba(126, 219, 141, 0.14);
        }

        .stButton>button {
            border-radius: 10px;
            font-weight: 600;
            border: 1px solid #2c435f;
            background: rgba(19, 43, 73, 0.72);
        }

        .stButton>button:hover {
            border-color: var(--accent);
            background: linear-gradient(180deg, #1f4f84, #173c63);
            transform: translateY(-1px);
        }

        .stButton>button:focus-visible,
        input:focus-visible {
            outline: none;
            box-shadow: var(--focus);
        }

        .table-tools {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        @keyframes sweep-in {
            from {
                opacity: 0;
                transform: translateY(10px);
                filter: blur(2px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
                filter: blur(0);
            }
        }

        @media (max-width: 760px) {
            .table-tools {
                flex-direction: column;
                align-items: flex-start;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


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

    with st.status("Running DB agent...", expanded=False):
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
        except Exception as exc:
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


def _render_trace(trace: QueryTrace, turn_index: int, trace_index: int) -> None:
    with st.container():
        st.markdown("<div class='trace-box'>", unsafe_allow_html=True)
        st.markdown(
            f"**Query {trace_index + 1}** · **{trace.row_count} rows**"
            + ("  *(truncated)*" if trace.truncated else ""),
        )
        if trace.purpose:
            st.caption(f"Purpose: {trace.purpose}")

        if trace.error:
            st.error(trace.error)

        if trace.columns:
            if trace.rows:
                st.dataframe(
                    trace.rows,
                    use_container_width=True,
                    hide_index=True,
                )

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
        st.markdown("</div>", unsafe_allow_html=True)


def _render_conversation() -> None:
    for turn_index, msg in enumerate(st.session_state.conversation):
        if msg.get("role") == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
            continue

        with st.chat_message("assistant"):
            with st.container():
                if msg.get("error"):
                    st.markdown("<div class='assistant-msg'>", unsafe_allow_html=True)
                    st.error(msg["content"])
                    if msg.get("error"):
                        with st.expander("Debug detail", expanded=False):
                            st.caption(msg.get("error"))
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<div class='assistant-msg'>{msg['content']}</div>",
                        unsafe_allow_html=True,
                    )

                if msg.get("traces"):
                    st.markdown("")
                    st.markdown("#### Analysis records")
                    for trace_index, trace in enumerate(msg.get("traces", [])):
                        _render_trace(trace, turn_index, trace_index)
                else:
                    st.info("No SQL trace was produced.")

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

                st.markdown("<div class='section-kicker'>&nbsp;</div>", unsafe_allow_html=True)
                st.markdown("<div class='section-kicker'>&nbsp;</div>", unsafe_allow_html=True)
                prev_question = _previous_user_query(st.session_state.conversation, turn_index)
                if prev_question:
                    cols = st.columns([4, 1])
                    with cols[1]:
                        if st.button(
                            "Retry",
                            key=f"retry-{turn_index}",
                            use_container_width=True,
                        ):
                            st.session_state.queued_prompt = prev_question
                            st.rerun()


def _sidebar() -> None:
    with st.sidebar:
        st.markdown("## DB Analyst Chat")
        if st.button("Clear chat", use_container_width=True):
            st.session_state.conversation = [
                {
                    "role": "assistant",
                    "content": "Ready to run. Ask a question to trigger schema-aware SQL execution.",
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
            st.rerun()

        st.caption(
            "Read-only mode enabled by default. The agent inspects schema each turn "
            "and sends SELECT-only SQL through the configured model tool-call API."
        )

        st.markdown("#### Example prompts")
        examples = [
            "How many active customers are in the system?",
            "Top 5 customers by total purchases this month",
            "List products with low stock",
        ]
        for i, prompt in enumerate(examples):
            if st.button(
                prompt,
                key=f"example-{i}",
                use_container_width=True,
            ):
                st.session_state.queued_prompt = prompt
                st.rerun()


st.set_page_config(
    page_title="DB Agent Analyst Chat",
    page_icon="🧮",
    layout="wide",
)

_inject_styles()

if "conversation" not in st.session_state:
    st.session_state.conversation = [
        {
            "role": "assistant",
            "content": "Ready to run. Ask a question to trigger schema-aware SQL execution.",
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

if "queued_prompt" not in st.session_state:
    st.session_state.queued_prompt = None

st.title("DB Analyst Ledger")
st.caption("Chat with your local PostgreSQL schema through read-only SQL prompts.")
_sidebar()

queued_prompt = st.session_state.get("queued_prompt")
if queued_prompt:
    st.session_state.queued_prompt = None
    _run_question(queued_prompt)

rendered_trigger = st.chat_input("Ask for a count, list, summary, or trend")
if rendered_trigger:
    _run_question(rendered_trigger)

_render_conversation()
