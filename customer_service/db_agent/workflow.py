"""LangGraph orchestration for one bounded, read-only analytical turn."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from customer_service.llm.client import LLMResponse, LLMResponseError, OpenAILLMClient
from customer_service.llm.profile import DomainProfile

from .config import DBAgentConfig
from .database import DatabaseSchema
from .federation import FederatedQueryService
from .gateway import DatabaseGateway, FederatedCatalog
from .memory import evidence_message, history_messages, remember_turn
from .prompts import build_system_prompt, final_answer_instruction, tool_schemas
from .query_tool import execute_tool_calls
from .types import DatabaseCoverage, DBAgentError, QueryTrace


class AgentState(TypedDict, total=False):
    """Checkpointed state; working fields are reset at the start of each turn."""

    question: str
    force_query: bool
    history: list[dict[str, str]]
    catalog: DatabaseSchema | FederatedCatalog
    source_catalog: FederatedCatalog
    messages: list[dict[str, Any]]
    traces: list[QueryTrace]
    tool_calls_used: int
    metadata_calls: int
    attempts: int
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    pending_tool_calls: list[dict[str, Any]]
    answer: str
    model: str
    used_databases: set[str]
    source_failures: dict[str, str]
    coverage: list[DatabaseCoverage]


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
    gateway: DatabaseGateway,
    config: DBAgentConfig,
    profile: DomainProfile,
    checkpointer: InMemorySaver,
) -> Any:
    """Compile an explicit model → validated query → answer workflow."""

    max_calls = max(1, config.max_tool_calls)
    federation = FederatedQueryService(
        gateway,
        max_intermediate_rows=config.max_intermediate_rows,
        max_federated_bytes=config.max_federated_bytes,
        max_concurrency=config.max_database_concurrency,
    )
    model_options = {
        "temperature": 0.0,
        "max_tokens": config.model_max_tokens,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": config.enable_thinking}
        },
    }

    def prepare(state: AgentState) -> dict[str, Any]:
        source_catalog = gateway.inspect_catalog()
        catalog: DatabaseSchema | FederatedCatalog = (
            source_catalog
            if gateway.is_federated
            else source_catalog.schemas[gateway.default_database]
        )
        history = state.get("history", [])
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": build_system_prompt(catalog, config, profile),
            }
        ]
        messages.extend(history_messages(history))
        evidence = evidence_message(history)
        if evidence:
            messages.append({"role": "user", "content": evidence})
        messages.append({"role": "user", "content": state["question"]})
        return {
            "catalog": catalog,
            "source_catalog": source_catalog,
            "messages": messages,
            "traces": [],
            "tool_calls_used": 0,
            "metadata_calls": 0,
            "attempts": 0,
            "latency_ms": 0.0,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "pending_tool_calls": [],
            "answer": "",
            "model": "",
            "used_databases": set(),
            "source_failures": {},
            "coverage": source_catalog.coverage(),
        }

    def model(state: AgentState) -> dict[str, Any]:
        response = client.ask(
            prompt="",
            messages=state["messages"],
            tools=tool_schemas(state["catalog"]),
            tool_choice=(
                "required"
                if state["tool_calls_used"] == 0
                and (state.get("force_query", False) or not state.get("history"))
                else "auto"
            ),
            parallel_tool_calls=False,
            **model_options,
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
            catalog=state["catalog"],
            gateway=gateway,
            federation=federation,
            config=config,
            calls_used=state["tool_calls_used"],
            metadata_calls=state["metadata_calls"],
            attempts=state["attempts"],
            used_databases=state["used_databases"],
            source_failures=state["source_failures"],
        )
        return {
            "messages": [*state["messages"], *batch.messages],
            "traces": [*state["traces"], *batch.traces],
            "tool_calls_used": batch.calls_used,
            "metadata_calls": batch.metadata_calls,
            "attempts": batch.attempts,
            "pending_tool_calls": [],
            "used_databases": batch.used_databases,
            "source_failures": batch.source_failures,
            "coverage": state["source_catalog"].coverage(
                batch.used_databases, batch.source_failures
            ),
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
                **model_options,
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
