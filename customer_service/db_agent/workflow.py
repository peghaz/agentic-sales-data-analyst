"""LangGraph orchestration for one bounded, read-only analytical turn."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from customer_service.llm.client import LLMResponse, LLMResponseError, OpenAILLMClient

from .config import DBAgentConfig
from .database import DatabaseAdapter, DatabaseSchema
from .memory import evidence_message, history_messages, remember_turn
from .prompts import build_system_prompt, final_answer_instruction, tool_schema
from .query_tool import execute_tool_calls
from .types import DBAgentError, QueryTrace


class AgentState(TypedDict, total=False):
    """Checkpointed state; working fields are reset at the start of each turn."""

    question: str
    force_query: bool
    history: list[dict[str, str]]
    schema: DatabaseSchema
    messages: list[dict[str, Any]]
    traces: list[QueryTrace]
    tool_calls_used: int
    attempts: int
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    pending_tool_calls: list[dict[str, Any]]
    answer: str
    model: str


def _add_usage(current: int | None, incoming: int | None) -> int | None:
    if current is None and incoming is None:
        return None
    return (current or 0) + (incoming or 0)


def _response_updates(state: AgentState, response: LLMResponse) -> dict[str, Any]:
    """Accumulate one model response without checkpointing the raw SDK object."""

    return {
        "messages": [
            *state["messages"],
            response.message or {"role": "assistant", "content": response.content},
        ],
        "pending_tool_calls": response.tool_calls,
        "answer": response.content if not response.tool_calls else "",
        "model": response.model,
        "latency_ms": state["latency_ms"] + response.latency_ms,
        "prompt_tokens": _add_usage(state["prompt_tokens"], response.prompt_tokens),
        "completion_tokens": _add_usage(
            state["completion_tokens"], response.completion_tokens
        ),
        "total_tokens": _add_usage(state["total_tokens"], response.total_tokens),
    }


def build_workflow(
    client: OpenAILLMClient,
    adapter: DatabaseAdapter,
    config: DBAgentConfig,
    checkpointer: InMemorySaver,
) -> Any:
    """Compile an explicit model → validated query → answer workflow."""

    max_calls = max(1, config.max_tool_calls)
    tool_definitions = [tool_schema()]

    def prepare(state: AgentState) -> dict[str, Any]:
        schema = adapter.inspect_schema(
            allowed_schemas=config.allowed_schemas,
            allowed_tables=config.allowed_tables,
        )
        history = state.get("history", [])
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt(schema, config)}
        ]
        messages.extend(history_messages(history))
        evidence = evidence_message(history)
        if evidence:
            messages.append({"role": "user", "content": evidence})
        messages.append({"role": "user", "content": state["question"]})
        return {
            "schema": schema,
            "messages": messages,
            "traces": [],
            "tool_calls_used": 0,
            "attempts": 0,
            "latency_ms": 0.0,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "pending_tool_calls": [],
            "answer": "",
            "model": "",
        }

    def model(state: AgentState) -> dict[str, Any]:
        response = client.ask(
            prompt="",
            messages=state["messages"],
            tools=tool_definitions,
            tool_choice=(
                "required"
                if state["tool_calls_used"] == 0
                and (state.get("force_query", False) or not state.get("history"))
                else "auto"
            ),
            parallel_tool_calls=False,
            temperature=0.0,
            max_tokens=1536,
        )
        if not response.tool_calls and not response.content:
            raise DBAgentError(
                "Model returned no tool call and no answer. Enable tool calling on "
                "the model server."
            )
        if (
            not response.tool_calls
            and state["tool_calls_used"] == 0
            and (state.get("force_query", False) or not state.get("history"))
        ):
            raise DBAgentError(
                "Model did not emit a tool call. Enable tool/function calling for this model."
            )
        return _response_updates(state, response)

    def route_model(state: AgentState) -> Literal["query", "complete"]:
        return "query" if state["pending_tool_calls"] else "complete"

    def query(state: AgentState) -> dict[str, Any]:
        batch = execute_tool_calls(
            state["pending_tool_calls"],
            schema=state["schema"],
            adapter=adapter,
            config=config,
            calls_used=state["tool_calls_used"],
            attempts=state["attempts"],
        )
        return {
            "messages": [*state["messages"], *batch.messages],
            "traces": [*state["traces"], *batch.traces],
            "tool_calls_used": batch.calls_used,
            "attempts": batch.attempts,
            "pending_tool_calls": [],
        }

    def route_query(state: AgentState) -> Literal["model", "finalize"]:
        return "finalize" if state["tool_calls_used"] >= max_calls else "model"

    def finalize(state: AgentState) -> dict[str, Any]:
        messages = [
            *state["messages"],
            {"role": "user", "content": final_answer_instruction(max_calls)},
        ]
        try:
            response = client.ask(
                prompt="",
                messages=messages,
                temperature=0.0,
                max_tokens=1536,
            )
        except LLMResponseError as exc:
            raise DBAgentError(
                "SQL tool-call budget was reached, but the model failed to generate "
                f"a final answer: {exc}"
            ) from exc
        if not response.content:
            raise DBAgentError(
                "SQL tool-call budget was reached, but the model did not produce "
                "a final assistant answer."
            )
        return _response_updates({**state, "messages": messages}, response)

    def complete(state: AgentState) -> dict[str, Any]:
        return {
            "history": remember_turn(
                state.get("history", []),
                state["question"],
                state["answer"],
                state["traces"],
            )
        }

    graph = StateGraph(AgentState)
    graph.add_node("prepare", prepare)
    graph.add_node("model", model)
    graph.add_node("query", query)
    graph.add_node("finalize", finalize)
    graph.add_node("complete", complete)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "model")
    graph.add_conditional_edges("model", route_model)
    graph.add_conditional_edges("query", route_query)
    graph.add_edge("finalize", "complete")
    graph.add_edge("complete", END)
    return graph.compile(checkpointer=checkpointer)
