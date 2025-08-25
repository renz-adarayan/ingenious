"""AzureSearchRetriever — vector + lexical retrieval helpers.

This module implements the hybrid (BM25 + vector) retrieval stage used by the
Advanced Azure AI Search pipeline. It is designed to work with the async Azure
Search SDK and an async OpenAI/Azure OpenAI client for embeddings. The code
intentionally avoids heavy imports and keeps dependency touch points localized.

Key entry points:
- `AzureSearchRetriever.search_lexical()`: BM25 (keyword) retrieval.
- `AzureSearchRetriever.search_vector()`: Vector retrieval using embeddings.
- `AzureSearchRetriever.close()`: Graceful client shutdown.

I/O/Deps/Side effects:
- Expects an async Azure `SearchClient` (aio) and a model client exposing either
  `client.embeddings.create(...)` (attribute style) or `client.embeddings().create(...)`
  (method style). Tests may inject AsyncMock-based stubs; this module handles both.
- Returns lists of plain dict rows enriched with `_retrieval_*` diagnostics.

Usage:
    retriever = AzureSearchRetriever(config, search_client, embedding_client)
    lexical = await retriever.search_lexical("query")
    vector  = await retriever.search_vector("query")
    await retriever.close()
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any, cast

from azure.search.documents.models import QueryType, VectorizedQuery

if TYPE_CHECKING:
    from ingenious.services.azure_search.config import SearchConfig

LOG_NAME = "ingenious.services.azure_search.retrieval"
RETRIEVAL_TYPE_LEXICAL = "lexical_bm25"
RETRIEVAL_TYPE_VECTOR = "vector_dense"
STATUS_TOO_MANY_REQUESTS = 429

logger = logging.getLogger(LOG_NAME)


class AzureSearchRetriever:
    """Hybrid BM25 + vector retriever.

    The retriever issues two parallel queries:
    1) A lexical (BM25) search via `SearchClient.search`.
    2) A vector search using an embeddings call followed by a vectorized query.

    Notes on compatibility:
    Some test stubs (and SDK shims) expose the OpenAI client with `embeddings`
    as a *method* returning an object that has `.create(...)`, while others
    expose it as an *attribute* with `.create(...)` directly. The vector path
    supports both shapes to keep tests and integrations compatible.
    """

    def __init__(
        self,
        config: SearchConfig,
        search_client: Any | None = None,
        embedding_client: Any | None = None,
    ) -> None:
        """Initialize the retriever with config and dependency clients.

        Args:
            config: Validated `SearchConfig` object.
            search_client: Async Azure Search client (aio). May be None in tests.
            embedding_client: Async OpenAI/Azure OpenAI client, used for embeddings.
        """
        self._cfg = config
        self._search_client = search_client
        self._embedding_client = embedding_client

    # ------------------------------ Internals --------------------------------

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """Heuristically detect a 429 rate-limit error from various SDKs.

        Why:
            Some tests simulate a 429 on embeddings and expect a `RuntimeError`
            to be raised (distinct from other OpenAI errors that should bubble).

        Args:
            exc: The exception raised by the embeddings call.

        Returns:
            True if the exception appears to be a 429/rate limit, else False.
        """
        code = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        if isinstance(code, int) and code == STATUS_TOO_MANY_REQUESTS:
            return True
        resp = getattr(exc, "response", None)
        resp_code = getattr(resp, "status_code", None)
        if isinstance(resp_code, int) and resp_code == STATUS_TOO_MANY_REQUESTS:
            return True
        text = str(exc).lower()
        return "429" in text or "rate limit" in text or "ratelimit" in text

    async def _resolve_embeddings_client(self) -> Any | None:
        """Resolve an embeddings client that has an async `.create(...)` method.

        Supports both attribute-style (`client.embeddings.create`) and
        method-style (`client.embeddings().create`) access patterns. Avoids
        misclassifying AsyncMock instances as callables that must be invoked.

        Returns:
            An object with a `.create(...)` coroutine method, or None if not found.
        """
        if self._embedding_client is None:
            return None

        embeddings_attr: Any = getattr(self._embedding_client, "embeddings", None)
        if embeddings_attr is None:
            return None

        if hasattr(embeddings_attr, "create"):
            return embeddings_attr

        if callable(embeddings_attr):
            try:
                maybe_client: Any = embeddings_attr()
            except Exception:
                return None
            if inspect.isawaitable(maybe_client):
                try:
                    maybe_client = await maybe_client
                except Exception:
                    return None
            if hasattr(maybe_client, "create"):
                return maybe_client

        return None

    # ------------------------------ Lexical ----------------------------------

    async def search_lexical(self, query: str) -> list[dict[str, Any]]:
        """Run a BM25 (keyword) search.

        Args:
            query: The user query string.

        Returns:
            A list of result dicts. Each row includes `_retrieval_type` and
            `_retrieval_score` derived from `@search.score`.
        """
        logger.debug(
            f"search_lexical({query!r}) with top_k={self._cfg.top_k_retrieval}"
        )
        if not query or not query.strip():
            return []

        if self._search_client is None:
            return []

        # Some stub environments may not define QueryType.SIMPLE; omit the arg
        # in that case.
        params: dict[str, Any] = {
            "search_text": query,
            "top": self._cfg.top_k_retrieval,
        }
        qt = getattr(QueryType, "SIMPLE", None)
        if qt is not None:
            params["query_type"] = qt

        results = await self._search_client.search(**params)

        out: list[dict[str, Any]] = []
        async for row in results:
            d = dict(row)
            d["_retrieval_type"] = RETRIEVAL_TYPE_LEXICAL
            d["_retrieval_score"] = d.get("@search.score", 0.0)
            out.append(d)
        logger.debug(f"search_lexical -> {len(out)} rows")
        return out

    # ------------------------------- Vector ----------------------------------

    async def search_vector(self, query: str) -> list[dict[str, Any]]:
        """Run a vector search (embed → vectorized query).

        Error policy:
            * If the embeddings step raises a *rate-limit* (429), raise
              `RuntimeError` so higher layers can retry.
            * For other embedding errors, re-raise the original exception type.
            * If the embeddings facility is entirely unavailable, return `[]`
              so the pipeline can still proceed with lexical results.

        Args:
            query: The user query string.

        Returns:
            A list of result dicts with `_retrieval_type` and `_retrieval_score`.
            Returns `[]` on blank input or when embeddings are unavailable.
        """
        logger.debug(f"search_vector({query!r}) with top_k={self._cfg.top_k_retrieval}")
        if not query or not query.strip():
            return []

        embeddings_client = await self._resolve_embeddings_client()
        if embeddings_client is None or not hasattr(embeddings_client, "create"):
            return []

        try:
            emb_resp: Any = await embeddings_client.create(
                input=[query],
                model=self._cfg.embedding_deployment_name,
            )
        except Exception as exc:
            if self._is_rate_limit_error(exc):
                raise RuntimeError("Embedding request was rate-limited (429).") from exc
            raise

        data: Any = getattr(emb_resp, "data", None)
        vec: Any = None
        if isinstance(data, list) and data:
            first = data[0]
            vec = getattr(first, "embedding", None)
            if vec is None and isinstance(first, dict):
                vec = first.get("embedding")
        elif isinstance(emb_resp, dict):
            first = (emb_resp.get("data") or [None])[0]
            if isinstance(first, dict):
                vec = first.get("embedding")

        if not isinstance(vec, list) or not vec:
            return []

        # Prefer passing a string 'fields' (ensures tests can assert equality
        # against config.vector_field). If an SDK requires a list[str], retry
        # accordingly. Use a cast to satisfy static typing for the fallback.
        fields_str = self._cfg.vector_field
        try:
            vq = VectorizedQuery(
                vector=vec,
                k_nearest_neighbors=self._cfg.top_k_retrieval,
                fields=fields_str,
                exhaustive=True,
            )
        except (TypeError, ValueError):
            vq = VectorizedQuery(
                vector=vec,
                k_nearest_neighbors=self._cfg.top_k_retrieval,
                fields=cast(Any, [fields_str]),
                exhaustive=True,
            )

        if self._search_client is None:
            return []

        results = await self._search_client.search(
            search_text=None,
            vector_queries=[vq],
            top=self._cfg.top_k_retrieval,
        )

        out: list[dict[str, Any]] = []
        async for row in results:
            d = dict(row)
            d["_retrieval_type"] = RETRIEVAL_TYPE_VECTOR
            d["_retrieval_score"] = d.get("@search.score", 0.0)
            out.append(d)
        logger.debug(f"search_vector -> {len(out)} rows")
        return out

    # -------------------------------- Close ----------------------------------

    async def close(self) -> None:
        """Close underlying clients, tolerating both sync and async close methods.

        This ensures both the search and embedding clients release any resources,
        independent of whether they expose an async or sync `close()` API.
        """

        async def _aclose(x: Any) -> None:
            """Close a client that may expose sync/async `close()`.

            Args:
                x: The client instance (or None).

            Returns:
                None. Errors are intentionally not raised here.
            """
            if not x:
                return
            close_fn = getattr(x, "close", None)
            if close_fn:
                res = close_fn()
                if inspect.isawaitable(res):
                    await res

        await _aclose(self._search_client)
        await _aclose(self._embedding_client)
