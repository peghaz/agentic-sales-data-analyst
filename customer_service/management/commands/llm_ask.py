"""Management command for quick manual checks against configured LLM."""

from django.core.management.base import BaseCommand, CommandError

from customer_service.llm.client import (
    LLMError,
    OpenAILLMClient,
)


class Command(BaseCommand):
    help = "Run a prompt against the configured OpenAI-compatible LLM."

    def add_arguments(self, parser):
        parser.add_argument(
            "prompt",
            nargs="+",
            help="Prompt text to send to the model",
        )
        parser.add_argument(
            "--system",
            help="Optional system prompt",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            help="Temperature override (defaults to LLM_CONFIG value)",
        )
        parser.add_argument(
            "--max-tokens",
            type=int,
            help="Maximum output tokens (defaults to LLM_CONFIG value)",
        )
        parser.add_argument(
            "--top-p",
            type=float,
            help="Top-p nucleus sampling value",
        )
        parser.add_argument(
            "--stop",
            action="append",
            help="Stop sequence(s); provide multiple --stop options if needed",
        )

    def handle(self, *args, **options):
        prompt = " ".join(options["prompt"]).strip()
        if not prompt:
            raise CommandError("Prompt cannot be empty.")

        try:
            client = OpenAILLMClient()
            response = client.ask(
                prompt,
                system_prompt=options.get("system"),
                temperature=options.get("temperature"),
                max_tokens=options.get("max_tokens"),
                top_p=options.get("top_p"),
                stop=options.get("stop"),
            )
        except LLMError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(f"Model: {response.model}")
        self.stdout.write(f"Endpoint: {client.config.base_url}")
        if response.total_tokens is not None:
            self.stdout.write(
                f"Token usage: prompt={response.prompt_tokens}, "
                f"completion={response.completion_tokens}, total={response.total_tokens}"
            )
        self.stdout.write(f"Latency: {response.latency_ms:.2f} ms")
        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS(response.content))
