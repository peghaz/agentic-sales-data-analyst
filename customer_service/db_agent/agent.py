"""Small public facade over the stateful, read-only LangGraph workflow."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from customer_service.llm.client import OpenAILLMClient
from customer_service.llm.profile import DomainProfile, load_domain_profile

from .config import DBAgentConfig
from .database import DatabaseAdapter, DatabaseSchema
from .gateway import DatabaseGateway, DatabaseTarget
from .prompts import TOOL_NAME, build_system_prompt, tool_schema
from .types import DBAgentError, DBAgentResult, QueryTrace
from .workflow import build_workflow


class DBAgent:
    """Answer questions with a validated SQL tool and optional conversation state."""

    TOOL_NAME = TOOL_NAME

    def __init__(
        self,
        client: OpenAILLMClient,
        adapter: DatabaseAdapter | None,
        config: DBAgentConfig,
        profile: DomainProfile | None = None,
        *,
        gateway: DatabaseGateway | None = None,
    ) -> None:
        self._config = config
        self._profile = profile or load_domain_profile(config.profile_name)
        if gateway is None:
            if adapter is None:
                raise ValueError("Either adapter or gateway must be provided.")
            source = self._profile.sources.databases[0]
            target = DatabaseTarget(
                name=source.name,
                description=source.description,
                allowed_schemas=config.allowed_schemas,
                allowed_tables=config.allowed_tables,
            )
            gateway = DatabaseGateway(
                {source.name: adapter},
                {source.name: target},
                cache_ttl_seconds=config.schema_cache_ttl_seconds,
            )
        self._gateway = gateway
        self._checkpointer = InMemorySaver(
            serde=JsonPlusSerializer(
                allowed_msgpack_modules=[
                    ("customer_service.db_agent.database", "ColumnMeta"),
                    ("customer_service.db_agent.database", "ForeignKeyMeta"),
                    ("customer_service.db_agent.database", "TableMeta"),
                    ("customer_service.db_agent.database", "DatabaseSchema"),
                    ("customer_service.db_agent.types", "QueryTrace"),
                    ("customer_service.db_agent.types", "DatabaseCoverage"),
                    ("customer_service.db_agent.gateway", "DatabaseTarget"),
                    ("customer_service.db_agent.gateway", "FederatedCatalog"),
                    ("customer_service.llm.profile", "DatabaseRelationship"),
                ]
            )
        )
        self._graph = build_workflow(
            client, self._gateway, config, self._profile, self._checkpointer
        )

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
                "recursion_limit": 2 * max(1, self._config.max_tool_calls) + 14,
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
            coverage=state["coverage"],
        )

    def _build_system_prompt(self, schema: DatabaseSchema) -> str:
        """Compatibility helper for existing prompt tests and integrations."""

        return build_system_prompt(schema, self._config, self._profile)

    @staticmethod
    def _tool_schema() -> dict[str, Any]:
        return tool_schema()


__all__ = ["DBAgent", "DBAgentError", "DBAgentResult", "QueryTrace"]
