"""LLM client, configuration, and command tests."""

import os
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase

from customer_service.llm.client import (
    LLMConfig,
    LLMConfigurationError,
    LLMResponse,
    LLMResponseError,
    OpenAILLMClient,
)


class LLMConfigTests(SimpleTestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_requires_model_name(self):
        with self.assertRaisesRegex(
            LLMConfigurationError, "LLM_MODEL_NAME must be set"
        ):
            LLMConfig.from_env()

    @patch.dict(
        os.environ,
        {"LLM_MODEL_NAME": "qwen-3.8-27b"},
        clear=True,
    )
    def test_uses_local_defaults(self):
        config = LLMConfig.from_env()

        self.assertEqual(config.base_url, "http://127.0.0.1:8000/v1")

    @patch.dict(
        os.environ,
        {
            "LLM_MODEL_NAME": "qwen-3.8-27b",
            "LLM_HOST": " 192.168.1.50 ",
            "LLM_PORT": " 9000 ",
        },
        clear=True,
    )
    def test_builds_endpoint_from_remote_host_and_port(self):
        config = LLMConfig.from_env()

        self.assertEqual(config.base_url, "http://192.168.1.50:9000/v1")

    @patch.dict(
        os.environ,
        {
            "LLM_MODEL_NAME": "qwen-3.8-27b",
            "LLM_HOST": "192.168.1.50",
            "LLM_PORT": "9000",
            "LLM_API_BASE_URL": " https://llm.example.com/custom/v1/ ",
        },
        clear=True,
    )
    def test_full_base_url_takes_precedence_and_is_normalized(self):
        config = LLMConfig.from_env()

        self.assertEqual(config.base_url, "https://llm.example.com/custom/v1")

    @patch.dict(
        os.environ,
        {"LLM_MODEL_NAME": "qwen-3.8-27b", "LLM_API_MODE": "unsupported"},
        clear=True,
    )
    def test_rejects_unsupported_api_mode(self):
        with self.assertRaisesRegex(
            LLMConfigurationError,
            "LLM_API_MODE must be either 'completion' or 'chat'",
        ):
            LLMConfig.from_env()


class OpenAILLMClientTests(SimpleTestCase):
    @patch("customer_service.llm.client.OpenAI")
    def test_empty_reasoning_only_reply_reports_safe_diagnostics(self, openai_class):
        sdk_client = openai_class.return_value
        sdk_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=None,
                        reasoning_content="private model reasoning",
                    ),
                    finish_reason="length",
                )
            ],
            model="test-model",
            usage=SimpleNamespace(completion_tokens=1536),
        )
        config = LLMConfig(
            model="test-model",
            base_url="http://model.invalid/v1",
            api_key="EMPTY",
        )

        with self.assertRaises(LLMResponseError) as caught:
            OpenAILLMClient(config).ask("Question")

        message = str(caught.exception)
        self.assertIn("LLM returned empty response content", message)
        self.assertIn("finish_reason='length'", message)
        self.assertIn("completion_tokens=1536", message)
        self.assertIn("reasoning_tokens=None", message)
        self.assertIn("reasoning_present=True", message)
        self.assertIn("DB_AGENT_MODEL_MAX_TOKENS", message)
        self.assertNotIn("private model reasoning", message)
        self.assertEqual(sdk_client.chat.completions.create.call_count, 1)

    @patch("customer_service.llm.client.OpenAI")
    def test_empty_reply_detects_vllm_reasoning_and_token_metadata(self, openai_class):
        sdk_client = openai_class.return_value
        sdk_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=None,
                        reasoning="private vLLM reasoning",
                    ),
                    finish_reason="length",
                )
            ],
            model="test-model",
            usage=SimpleNamespace(
                completion_tokens=4096,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=4096),
            ),
        )
        config = LLMConfig(
            model="test-model",
            base_url="http://model.invalid/v1",
            api_key="EMPTY",
        )

        with self.assertRaises(LLMResponseError) as caught:
            OpenAILLMClient(config).ask("Question")

        message = str(caught.exception)
        self.assertIn("after reaching the output-token limit", message)
        self.assertIn("completion_tokens=4096", message)
        self.assertIn("reasoning_tokens=4096", message)
        self.assertIn("reasoning_present=True", message)
        self.assertNotIn("private vLLM reasoning", message)
        self.assertEqual(sdk_client.chat.completions.create.call_count, 1)

    @patch("customer_service.llm.client.OpenAI")
    def test_unparseable_tool_call_is_distinguished_from_empty_reply(
        self, openai_class
    ):
        sdk_client = openai_class.return_value
        sdk_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[SimpleNamespace(id="call-1", function=None)],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            model="test-model",
            usage=None,
        )
        config = LLMConfig(
            model="test-model",
            base_url="http://model.invalid/v1",
            api_key="EMPTY",
        )

        with self.assertRaises(LLMResponseError) as caught:
            OpenAILLMClient(config).ask("Question")

        message = str(caught.exception)
        self.assertIn("tool calls that could not be parsed", message)
        self.assertIn("raw_tool_calls=1", message)
        self.assertIn("finish_reason='tool_calls'", message)

    @patch("customer_service.llm.client.OpenAI")
    def test_completion_mode_sends_prompt_and_extracts_text(self, openai_class):
        sdk_client = openai_class.return_value
        sdk_client.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(text=" Generated text ")],
            model="completion-model",
            usage=SimpleNamespace(
                prompt_tokens=4,
                completion_tokens=2,
                total_tokens=6,
            ),
        )
        config = LLMConfig(
            model="completion-model",
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            api_mode="completion",
        )

        response = OpenAILLMClient(config).ask(
            "Question",
            system_prompt="Instructions",
            top_p=0.8,
            stop=["[SQL]"],
        )

        request = sdk_client.completions.create.call_args.kwargs
        self.assertEqual(request["prompt"], "Instructions\n\nQuestion")
        self.assertEqual(request["top_p"], 0.8)
        self.assertEqual(request["stop"], ["[SQL]"])
        self.assertEqual(response.content, "Generated text")
        self.assertEqual(response.total_tokens, 6)
        sdk_client.chat.completions.create.assert_not_called()

    @patch("customer_service.llm.client.OpenAI")
    def test_chat_mode_sends_messages_and_extracts_content(self, openai_class):
        sdk_client = openai_class.return_value
        sdk_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=" Chat response "))
            ],
            model="chat-model",
            usage=None,
        )
        config = LLMConfig(
            model="chat-model",
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            api_mode="chat",
        )

        response = OpenAILLMClient(config).ask(
            "Hello",
            system_prompt="Be concise.",
        )

        request = sdk_client.chat.completions.create.call_args.kwargs
        self.assertEqual(
            request["messages"],
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello"},
            ],
        )
        self.assertEqual(response.content, "Chat response")
        sdk_client.completions.create.assert_not_called()


class LLMAskCommandTests(SimpleTestCase):
    @patch("customer_service.management.commands.llm_ask.OpenAILLMClient")
    def test_reports_resolved_remote_endpoint(self, client_class):
        client = client_class.return_value
        client.config.base_url = "http://192.168.1.50:8000/v1"
        client.ask.return_value = LLMResponse(
            content="Remote response",
            model="qwen-3.8-27b",
            latency_ms=12.5,
        )
        stdout = StringIO()

        call_command("llm_ask", "Hello", stdout=stdout)

        self.assertIn("Endpoint: http://192.168.1.50:8000/v1", stdout.getvalue())

    @patch.dict(
        os.environ,
        {"LLM_MODEL_NAME": "qwen-3.8-27b", "LLM_API_MODE": "unsupported"},
        clear=True,
    )
    def test_reports_configuration_errors(self):
        with self.assertRaisesRegex(
            CommandError,
            "LLM_API_MODE must be either 'completion' or 'chat'",
        ):
            call_command("llm_ask", "Hello")

    @patch.dict(os.environ, {}, clear=True)
    def test_reports_missing_model_name(self):
        with self.assertRaisesRegex(CommandError, "LLM_MODEL_NAME must be set"):
            call_command("llm_ask", "Hello")
