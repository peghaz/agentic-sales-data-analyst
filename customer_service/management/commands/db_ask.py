"""Management command for NL -> SQL using function-calling tools."""

from __future__ import annotations

import json
from dataclasses import replace

from django.core.management.base import BaseCommand, CommandError

from customer_service.db_agent.agent import DBAgentError
from customer_service.db_agent.config import DBAgentConfig
from customer_service.db_agent.database import DatabaseError
from customer_service.db_agent.runtime import build_agent_runtime
from customer_service.llm.client import LLMError


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
            runtime = build_agent_runtime(config=config)
            result = runtime.agent.ask(question)
        except (LLMError, DatabaseError, DBAgentError, ValueError) as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(f"Model: {result.model}")
        self.stdout.write(f"Endpoint: {runtime.client.config.base_url}")
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
                if trace.database:
                    self.stdout.write(f"   database: {trace.database}")
                self.stdout.write(f"   stage: {trace.stage}")
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

        if result.coverage:
            self.stdout.write("")
            self.stdout.write(self.style.WARNING("Data coverage:"))
            for item in result.coverage:
                detail = f" ({item.detail})" if item.detail else ""
                self.stdout.write(f"- {item.database}: {item.status}{detail}")

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Answer:"))
        self.stdout.write(result.answer)
