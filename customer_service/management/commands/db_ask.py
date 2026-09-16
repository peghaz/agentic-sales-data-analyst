"""Management command for NL -> SQL using function-calling tools."""

from __future__ import annotations

import json
from dataclasses import replace

from django.core.management.base import BaseCommand, CommandError

from customer_service.db_agent.agent import DBAgent, DBAgentError
from customer_service.db_agent.config import DBAgentConfig
from customer_service.db_agent.database import (
    DatabaseError,
    PostgresDatabaseAdapter,
)
from customer_service.llm.client import LLMError, OpenAILLMClient


class Command(BaseCommand):
    help = "Ask questions against the configured PostgreSQL database with SQL tool-calling."

    def add_arguments(self, parser):
        parser.add_argument(
            "question",
            nargs="+",
            help="Question to answer against the database",
        )
        parser.add_argument(
            "--sql-limit",
            type=int,
            help="Override max rows per SQL execution.",
        )
        parser.add_argument(
            "--max-tool-calls",
            type=int,
            help=(
                "Override the maximum number of SQL tool calls. "
                "The final answer turn is not counted."
            ),
        )

    def handle(self, *args, **options):
        question = " ".join(options["question"]).strip()
        if not question:
            raise CommandError("Question cannot be empty.")

        try:
            config = DBAgentConfig.from_env()
            if options.get("sql_limit"):
                config = replace(
                    config,
                    max_rows=options["sql_limit"],
                    max_tool_calls=options.get("max_tool_calls")
                    or config.max_tool_calls,
                )
            elif options.get("max_tool_calls"):
                config = replace(
                    config,
                    max_tool_calls=options["max_tool_calls"],
                )
            client = OpenAILLMClient()
            adapter = PostgresDatabaseAdapter(
                dsn=config.database_url,
                statement_timeout_ms=config.statement_timeout_ms,
            )
            agent = DBAgent(client=client, adapter=adapter, config=config)
            result = agent.ask(question)
        except (LLMError, DatabaseError, DBAgentError, ValueError) as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(f"Model: {result.model}")
        self.stdout.write(f"Endpoint: {client.config.base_url}")
        if result.total_tokens is not None:
            self.stdout.write(
                f"Token usage: prompt={result.prompt_tokens}, "
                f"completion={result.completion_tokens}, total={result.total_tokens}"
            )
        self.stdout.write(f"Latency: {result.latency_ms:.2f} ms")
        self.stdout.write("")

        if result.traces:
            self.stdout.write(self.style.WARNING("SQL trace:"))
            for index, trace in enumerate(result.traces, start=1):
                self.stdout.write("")
                self.stdout.write(f"{index}. purpose: {trace.purpose or 'n/a'}")
                self.stdout.write(
                    f"   rows: {trace.row_count} {'(truncated)' if trace.truncated else ''}"
                )
                if trace.sql:
                    self.stdout.write(f"   sql: {trace.sql}")
                if trace.error:
                    self.stdout.write(self.style.ERROR(f"   error: {trace.error}"))
                if trace.columns:
                    preview = trace.rows[:3]
                    if preview:
                        self.stdout.write(
                            f"   sample: {json.dumps(preview, ensure_ascii=False)}"
                        )

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Answer:"))
        self.stdout.write(result.answer)
