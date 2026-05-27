"""Shared low-level helpers."""

from .api import (
    AsyncRequestRuntime,
    ChatCompletionRequest,
    ChatCompletionResult,
    build_httpx_limits,
    build_httpx_timeout,
    build_sync_openai_client,
)
from .hash import compute_file_hash
from .llm import (
    build_chat_completion_kwargs,
    format_api_error,
    should_retry_api_error,
)
from .log import (
    LogLevel,
    get_console,
    get_logger,
    setup_logging,
    silence_docling_logging,
)

__all__ = [
    "AsyncRequestRuntime",
    "ChatCompletionRequest",
    "ChatCompletionResult",
    "build_httpx_limits",
    "build_httpx_timeout",
    "build_sync_openai_client",
    "compute_file_hash",
    "build_chat_completion_kwargs",
    "format_api_error",
    "should_retry_api_error",
    "LogLevel",
    "get_console",
    "get_logger",
    "setup_logging",
    "silence_docling_logging",
]
