import asyncio

from dite.config import Config
from dite.utils.api_runtime import (
    AsyncRequestRuntime,
    ChatCompletionRequest,
    build_httpx_limits,
    build_httpx_timeout,
)


def test_build_httpx_timeout_and_limits_from_config() -> None:
    config = Config()
    config.api.connect_timeout_sec = 1.5
    config.api.read_timeout_sec = 20.0
    config.api.write_timeout_sec = 7.0
    config.api.pool_timeout_sec = 3.5
    config.api.max_connections = 18
    config.api.max_keepalive_connections = 9
    config.api.keepalive_expiry_sec = 11.0

    timeout = build_httpx_timeout(config)
    limits = build_httpx_limits(config)

    assert timeout.connect == 1.5
    assert timeout.read == 20.0
    assert timeout.write == 7.0
    assert timeout.pool == 3.5
    assert limits.max_connections == 18
    assert limits.max_keepalive_connections == 9
    assert limits.keepalive_expiry == 11.0


def test_async_request_runtime_clamps_effective_limits() -> None:
    config = Config()
    config.api.max_connections = 3
    config.processing.cluster_naming_workers = 10
    config.processing.vlm_api_workers = 8
    config.processing.vlm_pages_per_document = 6

    runtime = AsyncRequestRuntime(config)

    assert runtime.effective_cluster_workers == 3
    assert runtime.effective_vlm_workers == 3
    assert runtime.effective_vlm_pages_per_document == 3


def test_async_request_runtime_vlm_batch_respects_global_and_local_limits(
    monkeypatch,
) -> None:
    tracker = {"current": 0, "max": 0, "closed": False, "builds": 0}

    class _Completions:
        async def create(self, **kwargs):
            del kwargs
            tracker["current"] += 1
            tracker["max"] = max(tracker["max"], tracker["current"])
            await asyncio.sleep(0.02)
            tracker["current"] -= 1

            class _Message:
                content = "page text"

            class _Choice:
                message = _Message()

            class _Response:
                choices = [_Choice()]

            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

        async def close(self) -> None:
            tracker["closed"] = True

    def fake_build_async_client(config):
        del config
        tracker["builds"] += 1
        return _Client()

    monkeypatch.setattr(
        "dite.utils.api_runtime._build_async_openai_client",
        fake_build_async_client,
    )

    config = Config()
    config.api.api_key = "dummy"
    config.api.base_url = "https://api.example.com/v1"
    config.api.max_connections = 8
    config.processing.vlm_api_workers = 4
    config.processing.vlm_pages_per_document = 2

    requests = [
        ChatCompletionRequest(
            kwargs={
                "model": "dummy-model",
                "messages": [{"role": "user", "content": "page"}],
            }
        )
        for _ in range(4)
    ]

    with AsyncRequestRuntime(config) as runtime:
        results = runtime.run_vlm_page_batch(requests, per_document_limit=4)

    assert len(results) == 4
    assert all(result.content == "page text" for result in results)
    assert tracker["max"] == 2
    assert tracker["closed"] is True
    assert tracker["builds"] == 1


def test_async_request_runtime_reuses_single_client_across_batches(monkeypatch) -> None:
    tracker = {"builds": 0, "closed": False}

    class _Completions:
        async def create(self, **kwargs):
            del kwargs

            class _Message:
                content = "ok"

            class _Choice:
                message = _Message()

            class _Response:
                choices = [_Choice()]

            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

        async def close(self) -> None:
            tracker["closed"] = True

    def fake_build_async_client(config):
        del config
        tracker["builds"] += 1
        return _Client()

    monkeypatch.setattr(
        "dite.utils.api_runtime._build_async_openai_client",
        fake_build_async_client,
    )

    config = Config()
    config.api.api_key = "dummy"
    config.api.base_url = "https://api.example.com/v1"
    request = ChatCompletionRequest(
        kwargs={
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "page"}],
        }
    )

    with AsyncRequestRuntime(config) as runtime:
        first = runtime.run_vlm_page_batch([request], per_document_limit=1)
        second = runtime.run_cluster_naming_batch([request])

    assert first[0].content == "ok"
    assert second[0].content == "ok"
    assert tracker["builds"] == 1
    assert tracker["closed"] is True


def test_async_request_runtime_start_is_idempotent(monkeypatch) -> None:
    tracker = {"builds": 0, "closed": False}

    class _Client:
        class _Chat:
            class _Completions:
                async def create(self, **kwargs):
                    del kwargs
                    raise AssertionError("create should not run")

            completions = _Completions()

        chat = _Chat()

        async def close(self) -> None:
            tracker["closed"] = True

    def fake_build_async_client(config):
        del config
        tracker["builds"] += 1
        return _Client()

    monkeypatch.setattr(
        "dite.utils.api_runtime._build_async_openai_client",
        fake_build_async_client,
    )

    config = Config()
    config.api.api_key = "dummy"
    config.api.base_url = "https://api.example.com/v1"

    runtime = AsyncRequestRuntime(config)
    runtime.start()
    runtime.start()
    runtime.close()

    assert tracker["builds"] == 1
    assert tracker["closed"] is True


def test_async_request_runtime_close_is_idempotent(monkeypatch) -> None:
    tracker = {"builds": 0, "closed": 0}

    class _Client:
        class _Chat:
            class _Completions:
                async def create(self, **kwargs):
                    del kwargs
                    raise AssertionError("create should not run")

            completions = _Completions()

        chat = _Chat()

        async def close(self) -> None:
            tracker["closed"] += 1

    def fake_build_async_client(config):
        del config
        tracker["builds"] += 1
        return _Client()

    monkeypatch.setattr(
        "dite.utils.api_runtime._build_async_openai_client",
        fake_build_async_client,
    )

    config = Config()
    config.api.api_key = "dummy"
    config.api.base_url = "https://api.example.com/v1"

    runtime = AsyncRequestRuntime(config)
    runtime.start()
    runtime.close()
    runtime.close()

    assert tracker["builds"] == 1
    assert tracker["closed"] == 1
    assert runtime._loop is None
    assert runtime._thread is None
    assert runtime._client is None


def test_async_request_runtime_shutdown_runtime_closes_executor(monkeypatch) -> None:
    runtime = AsyncRequestRuntime(Config())
    calls: list[tuple[str, float | None]] = []

    class _Client:
        async def close(self) -> None:
            calls.append(("client_close", None))

    class _Loop:
        async def shutdown_asyncgens(self) -> None:
            calls.append(("shutdown_asyncgens", None))

        async def shutdown_default_executor(self, timeout=None) -> None:
            calls.append(("shutdown_default_executor", timeout))

    runtime._client = _Client()
    loop = _Loop()

    monkeypatch.setattr("asyncio.get_running_loop", lambda: loop)

    asyncio.run(runtime._shutdown_runtime())

    assert calls == [
        ("client_close", None),
        ("shutdown_asyncgens", None),
        ("shutdown_default_executor", 5.0),
    ]


def test_async_request_runtime_cannot_restart_after_close(monkeypatch) -> None:
    class _Client:
        class _Chat:
            class _Completions:
                async def create(self, **kwargs):
                    del kwargs
                    raise AssertionError("create should not run")

            completions = _Completions()

        chat = _Chat()

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        "dite.utils.api_runtime._build_async_openai_client",
        lambda config: _Client(),
    )

    config = Config()
    config.api.api_key = "dummy"
    config.api.base_url = "https://api.example.com/v1"

    runtime = AsyncRequestRuntime(config)
    runtime.start()
    runtime.close()

    try:
        runtime.start()
    except RuntimeError as exc:
        assert str(exc) == "async request runtime is already closed"
    else:
        raise AssertionError("runtime.start() should fail after close")
