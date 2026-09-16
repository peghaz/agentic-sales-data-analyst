"""Shared composition root for CLI and Streamlit agent runtimes."""

from __future__ import annotations

from dataclasses import dataclass

from customer_service.llm.client import OpenAILLMClient
from customer_service.llm.profile import DomainProfile, load_domain_profile

from .agent import DBAgent
from .config import DBAgentConfig
from .gateway import DatabaseGateway, build_database_gateway


@dataclass(frozen=True)
class AgentRuntime:
    """Fully composed dependencies for one application process."""

    client: OpenAILLMClient
    agent: DBAgent
    config: DBAgentConfig
    profile: DomainProfile
    gateway: DatabaseGateway


def build_agent_runtime(
    *,
    config: DBAgentConfig | None = None,
    profile: DomainProfile | None = None,
    client: OpenAILLMClient | None = None,
) -> AgentRuntime:
    resolved_config = config or DBAgentConfig.from_env()
    resolved_profile = profile or load_domain_profile(resolved_config.profile_name)
    resolved_client = client or OpenAILLMClient()
    gateway = build_database_gateway(resolved_config, resolved_profile)
    agent = DBAgent(
        client=resolved_client,
        adapter=None,
        config=resolved_config,
        profile=resolved_profile,
        gateway=gateway,
    )
    return AgentRuntime(
        client=resolved_client,
        agent=agent,
        config=resolved_config,
        profile=resolved_profile,
        gateway=gateway,
    )
