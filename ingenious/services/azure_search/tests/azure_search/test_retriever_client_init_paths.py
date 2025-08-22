"""Retriever ↔ client_init integration: delegation and parameter mapping.

Why:
- Ensure `AzureSearchRetriever` builds both the SearchClient and the AsyncOpenAI
  embedding client via `client_init` at construction time.
- Verify that the mapping (endpoint/index/version) is forwarded correctly.
- Ensure vector path does not call embeddings on empty query.

Key entry point:
- `AzureSearchRetriever(config)` constructor + `search_vector("")` fast-path.

Usage:
- Run with pytest. No network calls; client_init is patched to stubs/spies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest
from pydantic import SecretStr
from unittest.mock import patch

from ingenious.services.azure_search.components.retrieval import AzureSearchRetriever
from ingenious.services.azure_search.config import SearchConfig


class _SpyOpenAI:
    """Spy AsyncOpenAI with an embeddings.create counter."""

    def __init__(self) -> None:
        """Initialize with zero calls and a dummy embeddings namespace."""
        self._calls: int = 0

        class _Emb:
            def __init__(self, parent: "_SpyOpenAI") -> None:
                self._p = parent

            async def create(self, *args: Any, **kwargs: Any) -> Any:
                self._p._calls += 1
                # Minimal shape; never used on empty query path.
                return {"data": [{"embedding": [0.1, 0.2]}]}

        self.embeddings = _Emb(self)

    @property
    def calls(self) -> int:
        """Return how many times embeddings.create was called."""
        return self._calls

    async def close(self) -> None:
        """Provide an awaitable close method for retriever.close()."""
        return None


class _CloseableSearchClient:
    """Closeable SearchClient stub."""

    async def search(self, *args: Any, **kwargs: Any) -> Any:
        """Return an empty async iterator (unused in this test)."""
        class _Iter:
            def __aiter__(self) -> "_Iter":
                return self

            async def __anext__(self) -> Any:
                raise StopAsyncIteration

        return _Iter()

    async def close(self) -> None:
        """Provide an awaitable close method."""
        return None


@pytest.mark.asyncio
async def test_retriever_vector_and_lexical_clients_both_built_through_client_init() -> None:
    """Assert retriever obtains clients via client_init and maps parameters."""
    # Build a fully valid SearchConfig (no answer generation needed here).
    cfg = SearchConfig(
        search_endpoint="https://s.example.net",
        search_key=SecretStr("sk"),
        search_index_name="idx",
        openai_endpoint="https://aoai.example.com",
        openai_key=SecretStr("ok"),
        openai_version="2025-01-01",
        embedding_deployment_name="emb",
        generation_deployment_name="chat",
        use_semantic_ranking=False,
    )

    captured_sc: List[Tuple[str, str]] = []
    captured_aoai: List[Tuple[str, str]] = []  # endpoint, version

    def _mk_sc(passed_cfg: SearchConfig, **_k: Any) -> _CloseableSearchClient:
        """Spy factory that records mapping of endpoint/index to SearchClient."""
        captured_sc.append((passed_cfg.search_endpoint, passed_cfg.search_index_name))
        return _CloseableSearchClient()

    def _mk_aoai(passed_cfg: SearchConfig, **_k: Any) -> _SpyOpenAI:
        """Spy factory that records mapping of endpoint/version to AsyncOpenAI."""
        captured_aoai.append((passed_cfg.openai_endpoint, passed_cfg.openai_version))
        return _SpyOpenAI()

    with patch(
        "ingenious.services.azure_search.client_init.make_async_search_client", new=_mk_sc
    ), patch(
        "ingenious.services.azure_search.client_init.make_async_openai_client",
        new=_mk_aoai,
    ):
        retriever = AzureSearchRetriever(cfg)

        # Mapping assertions
        assert captured_sc == [("https://s.example.net", "idx")]
        assert captured_aoai == [("https://aoai.example.com", "2025-01-01")]

        # Empty-query fast path: must not call embeddings at all.
        out: List[Dict[str, Any]] = await retriever.search_vector("")
        assert out == []
        # Confirm embeddings.create was never called.
        spy: _SpyOpenAI = retriever._embedding_client  # type: ignore[attr-defined]
        assert isinstance(spy, _SpyOpenAI)
        assert spy.calls == 0

        # Cleanup
        await retriever.close()
