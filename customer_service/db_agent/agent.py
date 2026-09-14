"""Small public facade over the stateful, read-only LangGraph workflow."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from customer_service.llm.client import OpenAILLMClient

from .config import DBAgentConfig
from .database import DatabaseAdapter, DatabaseSchema
from .prompts import TOOL_NAME, build_system_prompt, tool_schema
from .types import DBAgentError, DBAgentResult, QueryTrace
from .workflow import build_workflow


class DBAgent:
    """Answer questions with a validated SQL tool and optional conversation state."""

    TOOL_NAME = TOOL_NAME

    def __init__(
        self,
        client: OpenAILLMClient,
        adapter: DatabaseAdapter,
        config: DBAgentConfig,
    ) -> None:
        self._config = config
        self._checkpointer = InMemorySaver(
            serde=JsonPlusSerializer(
                allowed_msgpack_modules=[
                    ("customer_service.db_agent.database", "ColumnMeta"),
                    ("customer_service.db_agent.database", "ForeignKeyMeta"),
                    ("customer_service.db_agent.database", "TableMeta"),
                    ("customer_service.db_agent.database", "DatabaseSchema"),
                    ("customer_service.db_agent.types", "QueryTrace"),
                ]
            )
        )
        self._graph = build_workflow(client, adapter, config, self._checkpointer)

    def ask(
        self,
        question: str,
        *,
        thread_id: str | None = None,
        force_query: bool = False,
    ) -> DBAgentResult:
        """Run one question; reuse a thread ID and optionally require fresh data."""

        if not question or not question.strip():
            raise DBAgentError("Question cannot be empty.")
        if thread_id is not None and not thread_id.strip():
            raise DBAgentError("Conversation ID cannot be empty.")

        state = self._graph.invoke(
            {"question": question.strip(), "force_query": force_query},
            config={
                "configurable": {"thread_id": thread_id or uuid4().hex},
                "recursion_limit": 2 * max(1, self._config.max_tool_calls) + 8,
            },
        )
        return DBAgentResult(
            answer=state["answer"],
            model=state["model"],
            latency_ms=state["latency_ms"],
            prompt_tokens=state["prompt_tokens"],
            completion_tokens=state["completion_tokens"],
            total_tokens=state["total_tokens"],
            traces=state["traces"],
        )

    def _build_system_prompt(self, schema: DatabaseSchema) -> str:
        """Compatibility helper for existing prompt tests and integrations."""

        return build_system_prompt(schema, self._config)

    @staticmethod
    def _tool_schema() -> dict[str, Any]:
        return tool_schema()


__all__ = ["DBAgent", "DBAgentError", "DBAgentResult", "QueryTrace"]
