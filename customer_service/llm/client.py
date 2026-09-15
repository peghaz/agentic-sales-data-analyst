"""OpenAI-compatible LLM client helpers."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from openai import APIConnectionError, APITimeoutError, OpenAI, OpenAIError


class LLMError(Exception):
    """Base error for LLM integration failures."""


class LLMConnectionError(LLMError):
    """Raised when the client cannot reach the LLM endpoint."""


class LLMResponseError(LLMError):
    """Raised when the LLM response is missing or malformed."""


class LLMConfigurationError(LLMError):
    """Raised when the LLM client configuration is invalid."""


LLMApiMode = Literal["completion", "chat"]


@dataclass(frozen=True)
class LLMConfig:
    """Configuration loaded from environment variables."""

    model: str
    base_url: str
    api_key: str
    api_mode: LLMApiMode = "chat"
    timeout_seconds: float = 60.0
    default_temperature: float = 0.0
    default_max_tokens: int = 1024

    @classmethod
    def from_env(cls) -> LLMConfig:
        model = os.getenv("LLM_MODEL_NAME", "").strip()
        if not model:
            raise LLMConfigurationError("LLM_MODEL_NAME must be set.")
        api_mode_value = os.getenv("LLM_API_MODE", "chat").strip().lower() or "chat"
        if api_mode_value not in {"completion", "chat"}:
            raise LLMConfigurationError(
                "LLM_API_MODE must be either 'completion' or 'chat'; "
                f"received {api_mode_value!r}."
            )
        api_mode = cast(LLMApiMode, api_mode_value)

        base_url = os.getenv("LLM_API_BASE_URL")
        if base_url:
            base_url = base_url.strip().rstrip("/")
        else:
            host = os.getenv("LLM_HOST", "127.0.0.1").strip() or "127.0.0.1"
            port = os.getenv("LLM_PORT", "8000").strip() or "8000"
            base_url = f"http://{host}:{port}/v1"

        api_key = (
            os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
        ).strip()

        try:
            timeout_seconds = float(
                os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "60").strip()
            )
        except ValueError:
            timeout_seconds = 60.0

        return cls(
            model=model,
            base_url=base_url,
            api_key=api_key,
            api_mode=api_mode,
            timeout_seconds=timeout_seconds,
        )


@dataclass(frozen=True)
class LLMResponse:
    """Normalized response object returned by the LLM client."""

    content: str
    model: str
    latency_ms: float
    message: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    raw: Any | None = None


class OpenAILLMClient:
    """Thin OpenAI SDK wrapper backed by env-configured settings."""

    def __init__(self, config: LLMConfig | None = None):
        self._config = config or LLMConfig.from_env()
        self._client = OpenAI(
            api_key=self._config.api_key,
            base_url=self._config.base_url,
            timeout=self._config.timeout_seconds,
        )

    @property
    def config(self) -> LLMConfig:
        return self._config

    def ask(
        self,
        prompt: str,
        *,
        messages: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a completion request and return normalized output."""

        if messages is None and not prompt.strip():
            raise LLMResponseError("Prompt cannot be empty.")

        if self._config.api_mode == "completion" and tools:
            raise LLMConfigurationError(
                "Tool calling requires LLM_API_MODE=chat. Set LLM_API_MODE=chat."
            )

        request_payload: dict[str, Any] = {
            "model": self._config.model,
            "temperature": (
                temperature
                if temperature is not None
                else self._config.default_temperature
            ),
            "max_tokens": (
                max_tokens
                if max_tokens is not None
                else self._config.default_max_tokens
            ),
        }
        if self._config.api_mode == "chat":
            if messages is None:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
            elif not messages:
                raise LLMResponseError("At least one message is required.")
            request_payload["messages"] = messages
        else:
            if messages is not None:
                raise LLMResponseError(
                    "Prompt-level messages are only supported in chat mode."
                )
            completion_prompt = prompt
            if system_prompt and system_prompt.strip():
                completion_prompt = f"{system_prompt.strip()}\n\n{prompt}"
            request_payload["prompt"] = completion_prompt

        if tools:
            request_payload["tools"] = tools
            request_payload["tool_choice"] = tool_choice or "auto"
            request_payload["parallel_tool_calls"] = parallel_tool_calls

        if top_p is not None:
            request_payload["top_p"] = top_p
        if stop is not None:
            request_payload["stop"] = stop

        request_payload.update(kwargs)

        start = time.perf_counter()
        try:
            if self._config.api_mode == "chat":
                completion = self._client.chat.completions.create(**request_payload)
            else:
                completion = self._client.completions.create(**request_payload)
        except (APIConnectionError, APITimeoutError) as exc:
            raise LLMConnectionError(
                f"Could not connect to {self._config.base_url}. "
                "Verify the configured endpoint is running and reachable, then retry."
            ) from exc
        except OpenAIError as exc:
            raise LLMResponseError(f"OpenAI request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000

        if not getattr(completion, "choices", None):
            raise LLMResponseError("LLM returned no choices.")

        first_choice = completion.choices[0]
        message = getattr(first_choice, "message", None)
        usage = getattr(completion, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        total_tokens = getattr(usage, "total_tokens", None) if usage else None
        completion_details = (
            getattr(usage, "completion_tokens_details", None) if usage else None
        )
        reasoning_tokens = (
            getattr(completion_details, "reasoning_tokens", None)
            if completion_details
            else None
        )
        tool_calls = self._extract_tool_calls(message)
        if self._config.api_mode == "chat":
            raw_content = self._message_field(message, "content")
            content = raw_content.strip() if isinstance(raw_content, str) else ""
        else:
            content = (getattr(first_choice, "text", None) or "").strip()
        if not content and not tool_calls:
            raw_calls = self._message_field(message, "tool_calls") or []
            raw_call_count = (
                len(raw_calls) if isinstance(raw_calls, (list, tuple)) else 1
            )
            reasoning = self._message_field(
                message, "reasoning_content"
            ) or self._message_field(message, "reasoning")
            finish_reason = getattr(first_choice, "finish_reason", None)
            if raw_call_count:
                issue = "LLM returned tool calls that could not be parsed"
            elif finish_reason == "length":
                issue = (
                    "LLM returned empty response content after reaching the "
                    "output-token limit"
                )
            else:
                issue = "LLM returned empty response content"
            hint = (
                " Increase DB_AGENT_MODEL_MAX_TOKENS or set "
                "DB_AGENT_ENABLE_THINKING=false."
                if finish_reason == "length"
                else ""
            )
            raise LLMResponseError(
                f"{issue} (model={getattr(completion, 'model', None) or self._config.model!r}, "
                f"finish_reason={finish_reason!r}, "
                f"completion_tokens={completion_tokens!r}, "
                f"reasoning_tokens={reasoning_tokens!r}, "
                f"reasoning_present={bool(reasoning) or bool(reasoning_tokens)}, "
                f"raw_tool_calls={raw_call_count}).{hint}"
            )
        response_message = self._build_response_message(message)

        return LLMResponse(
            content=content,
            model=completion.model or self._config.model,
            latency_ms=latency_ms,
            message=response_message,
            tool_calls=tool_calls,
            finish_reason=getattr(first_choice, "finish_reason", None),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw=completion,
        )

    @staticmethod
    def _message_field(message: Any, name: str) -> Any:
        if isinstance(message, dict):
            return message.get(name)
        return getattr(message, name, None)

    @staticmethod
    def _extract_tool_calls(message: Any) -> list[dict[str, Any]]:
        raw_tool_calls = OpenAILLMClient._message_field(message, "tool_calls") or []
        normalized: list[dict[str, Any]] = []
        for raw_call in raw_tool_calls:
            call_id = getattr(raw_call, "id", "")
            call_name: str | None = None
            raw_args: Any = None

            if isinstance(raw_call, dict):
                call_id = raw_call.get("id", call_id)
                function = raw_call.get("function")
                call_name = raw_call.get("name")
                if function is None:
                    call_name = call_name or raw_call.get("name")
            else:
                function = getattr(raw_call, "function", None)
                call_name = getattr(raw_call, "name", None)

            if isinstance(function, dict):
                call_name = call_name or function.get("name")
                raw_args = function.get("arguments")
            elif function is not None:
                call_name = call_name or getattr(function, "name", None)
                raw_args = getattr(function, "arguments", None)

            if isinstance(raw_call, dict):
                raw_args = raw_call.get("arguments", raw_args)

            if not call_name:
                continue

            args = raw_args
            if isinstance(args, str):
                arg_source = args
            else:
                arg_source = json.dumps(args or {})
            normalized.append(
                {
                    "id": call_id,
                    "name": call_name,
                    "arguments": arg_source,
                }
            )
        return normalized

    @staticmethod
    def _to_openai_tool_call(tool_call: dict[str, Any]) -> dict[str, Any] | None:
        call_name = tool_call.get("name")
        if not call_name:
            return None
        arguments = tool_call.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {})
        return {
            "id": tool_call.get("id", ""),
            "type": "function",
            "function": {"name": call_name, "arguments": arguments},
        }

    @staticmethod
    def _build_response_message(message: Any) -> dict[str, Any]:
        if message is None:
            return {"role": "assistant", "content": ""}
        if isinstance(message, dict):
            role = message.get("role", "assistant")
            content = message.get("content", "")
            payload = {"role": role, "content": content}
            tool_calls = message.get("tool_calls")
            if tool_calls:
                normalized_tool_calls: list[dict[str, Any]] = []
                for raw_call in tool_calls:
                    if (
                        isinstance(raw_call, dict)
                        and raw_call.get("type") == "function"
                        and isinstance(raw_call.get("function"), dict)
                    ):
                        normalized_tool_calls.append(raw_call)
                        continue

                    normalized = OpenAILLMClient._to_openai_tool_call(
                        raw_call if isinstance(raw_call, dict) else {}
                    )
                    if normalized is not None:
                        normalized_tool_calls.append(normalized)

                if normalized_tool_calls:
                    payload["tool_calls"] = normalized_tool_calls
            return payload
        role = getattr(message, "role", "assistant")
        content = message.content or ""
        payload = {"role": role, "content": content}
        tool_calls = OpenAILLMClient._extract_tool_calls(message)
        if tool_calls:
            normalized_tool_calls: list[dict[str, Any]] = []
            for raw_call in tool_calls:
                normalized = OpenAILLMClient._to_openai_tool_call(raw_call)
                if normalized is not None:
                    normalized_tool_calls.append(normalized)
            if normalized_tool_calls:
                payload["tool_calls"] = normalized_tool_calls
        return payload
