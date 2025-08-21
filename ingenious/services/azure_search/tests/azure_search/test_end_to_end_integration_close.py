"""Provider end-to-end (direct) key-path creation and lifecycle closure.

Why:
- Verify that from provider construction through retrieval and `provider.close()`,
  all closeable clients are properly awaited and no resource leaks occur.
- Assert that client creation is delegated to `client_init` (key-path via config).

Key entry points:
- `AzureSearchProvider(settings).retrieve()` and `.close()`.

Side effects:
- Patches `build_search_pipeline` to a minimal closeable stub to avoid network.
- Patches **the provider import site** `make_search_client` so the provider uses
  our stub rerank client (important because provider imports the symbol).

Usage:
- Run with pytest. Network-free. Focused on integration seams and lifecycle.
"""

from __future__ import annotations

from typing import Any, List
from types import SimpleNamespace

import pytest
from unittest.mock import patch

from ingenious.config.main_settings import IngeniousSettings
from ingenious.config.models import AzureSearchSettings, ModelSettings
from ingenious.services.azure_search.provider import AzureSearchProvider


class _CloseTrackedClient:
    """Rerank client stub that records whether close() was awaited."""

    def __init__(self) -> None:
        """Initialize close flag."""
        self.closed: bool = False

    async def search(self, *args: Any, **kwargs: Any) -> Any:
        """Return an empty async iterator for safety.

        Returns:
            An async iterator that immediately terminates.
        """
        class _Iter:
            def __aiter__(self) -> "_Iter":
                return self

            async def __anext__(self) -> Any:
                raise StopAsyncIteration

        return _Iter()

    async def close(self) -> None:
        """Record closure and provide awaitable close()."""
        self.closed = True


class _StubPipeline:
    """Close-tracked pipeline stub returning a single doc from fuser path."""

    def __init__(self, *_: Any, **__: Any) -> None:
        """Initialize with predictable fuser/retriever behavior."""
        class _Retr:
            async def search_lexical(self, _q: str) -> list[dict[str, Any]]:
                """Return a single lexical result."""
                return [{"id": "L1", "content": "lex", "@search.score": 0.2}]

            async def search_vector(self, _q: str) -> list[dict[str, Any]]:
                """Return a single vector result."""
                return [{"id": "V1", "content": "vec", "@search.score": 0.3}]

            async def close(self) -> None:
                """No-op close for retriever."""
                return None

        class _Fuser:
            async def fuse(
                self, _q: str, _lex: list[dict[str, Any]], _vec: list[dict[str, Any]]
            ) -> list[dict[str, Any]]:
                """Return a minimal fused list."""
                return [{"id": "F", "_fused_score": 0.4, "content": "fused"}]

            async def close(self) -> None:
                """No-op close for fuser."""
                return None

        self.retriever = _Retr()
        self.fuser = _Fuser()
        self.answer_generator = None  # generation disabled for this test
        self._closed: bool = False

    async def close(self) -> None:
        """Mark pipeline closed."""
        self._closed = True


def _settings() -> IngeniousSettings:
    """Construct valid settings for the provider constructor.

    Returns:
        Settings with two AOAI models and one Azure Search service.
    """
    s = IngeniousSettings.model_construct()
    s.models = [
        ModelSettings(
            model="text-embedding-3-small",
            deployment="emb",
            api_key="ok",
            base_url="https://aoai.example.com",
            api_version="2024-02-01",
        ),
        ModelSettings(
            model="gpt-4o",
            deployment="chat",
            api_key="ok",
            base_url="https://aoai.example.com",
            api_version="2024-02-01",
        ),
    ]
    s.azure_search_services = [
        AzureSearchSettings(
            service="svc",
            endpoint="https://s.example.net",
            key="sk",
            index_name="idx",
            semantic_ranking=False,  # keep reranker logically unused in retrieve()
        )
    ]
    return s


@pytest.mark.asyncio
async def test_end_to_end_kb_direct_uses_client_init_key_path_and_closes_clients() -> None:
    """Assert provider delegates to client_init and closes all clients on close()."""
    rerank_stub = _CloseTrackedClient()
    pipeline_stub = _StubPipeline()

    make_calls: list[bool] = []

    def _mk_sc(*_a: Any, **_k: Any) -> _CloseTrackedClient:
        """Spy that returns a close-tracked rerank client stub.

        Returns:
            The rerank client stub that supports `search` and `close`.
        """
        make_calls.append(True)
        return rerank_stub

    # IMPORTANT: patch the **provider import site** for `make_search_client`.
    with patch(
        "ingenious.services.azure_search.provider.make_search_client", new=_mk_sc
    ), patch(
        "ingenious.services.azure_search.provider.build_search_pipeline",
        new=lambda *_a, **_k: pipeline_stub,
    ):
        provider = AzureSearchProvider(_settings(), enable_answer_generation=False)

        # Execute a retrieval (uses the stub pipeline; reranker will be a stub too).
        out: list[dict[str, Any]] = await provider.retrieve("q", top_k=1)
        assert out and out[0]["id"] == "F"

        # Assert delegation to client_init for rerank client construction.
        assert make_calls, "Expected client_init.make_search_client to be called."

        # Ensure close() is awaited on both pipeline and rerank client.
        await provider.close()
        assert pipeline_stub._closed is True
        assert rerank_stub.closed is True
