"""LLM request helpers."""

from typing import Any

from openai import APIError, AsyncOpenAI, OpenAI

from dite.app.config import ChatCompletionProfileConfig


def _translate_reasoning_controls(
    client: OpenAI | AsyncOpenAI,
    profile: ChatCompletionProfileConfig,
) -> dict[str, Any]:
    """Translate generic reasoning controls into provider-specific request fields."""
    if profile.reasoning_mode == "default" and profile.thinking_budget is None:
        return {}

    base_url = str(getattr(client, "base_url", "")).casefold()
    if "siliconflow.cn" in base_url:
        extra_body: dict[str, Any] = {}
        if profile.reasoning_mode != "default":
            extra_body["enable_thinking"] = profile.reasoning_mode == "on"
        if profile.thinking_budget is not None:
            extra_body["thinking_budget"] = profile.thinking_budget
        return extra_body

    return {}


def build_chat_completion_kwargs(
    *,
    client: OpenAI | AsyncOpenAI,
    model: str,
    messages: list[dict[str, Any]],
    profile: ChatCompletionProfileConfig | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Build chat completion kwargs from task intent plus provider adaptation."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        **overrides,
    }

    if profile is None:
        return kwargs

    kwargs.setdefault("max_tokens", profile.max_tokens)

    extra_body = dict(kwargs.get("extra_body") or {})
    extra_body.update(_translate_reasoning_controls(client, profile))
    if extra_body:
        kwargs["extra_body"] = extra_body

    return kwargs


def format_api_error(exc: APIError) -> str:
    """Extract the highest-signal fields from provider API errors."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        message = body.get("message") or getattr(exc, "message", None) or str(exc)
        details = []

        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            details.append(f"status={status_code}")

        code = body.get("code") or getattr(exc, "code", None)
        if code not in (None, ""):
            details.append(f"code={code}")

        error_type = body.get("type") or getattr(exc, "type", None)
        if error_type:
            details.append(f"type={error_type}")

        param = body.get("param") or getattr(exc, "param", None)
        if param:
            details.append(f"param={param}")

        request_id = getattr(exc, "request_id", None)
        if request_id:
            details.append(f"request_id={request_id}")

        if details:
            return f"{message} ({', '.join(details)})"
        return str(message)

    return str(exc)


def should_retry_api_error(exc: APIError) -> bool:
    """Return whether the provider error is worth retrying."""
    status_code = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    error_code = (
        body.get("code") if isinstance(body, dict) else getattr(exc, "code", None)
    )

    if status_code is None:
        return True
    if status_code >= 500:
        return True
    return error_code in {50507}
