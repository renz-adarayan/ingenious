"""
Advanced Azure AI Search pipeline orchestration.

This module implements the multi‑stage search flow used by the Knowledge Base
agent: L1 retrieval (BM25 + vector) → DAT fusion → (optional) Semantic
Ranker (L2) → (optional) RAG answer generation. It centralizes the execution
and client lifecycle, and provides a small factory to compose the pipeline
with clients from `client_init`.

Why/what:
- Keep heavy SDK imports out of import time.
- Provide robust, observable, and testable retrieval/fusion/ranking steps.
- Normalize returned chunks for downstream consumers (stable `content`/`snippet`).

Usage:
    pipeline = build_search_pipeline(cfg)
    top_chunks = await pipeline.retrieve(query="...", top_k=5)
    # or, if enabled:
    answer_dict = await pipeline.get_answer("...")

Key entry points:
- `AdvancedSearchPipeline.retrieve()`
- `AdvancedSearchPipeline.answer()` / `get_answer()`
- `build_search_pipeline(config)`
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any

from azure.search.documents.models import QueryType

from ingenious.services.azure_search.components.fusion import DynamicRankFuser
from ingenious.services.azure_search.components.generation import AnswerGenerator
from ingenious.services.azure_search.components.retrieval import AzureSearchRetriever
from ingenious.services.azure_search.config import SearchConfig
from ingenious.services.retrieval.errors import GenerationDisabledError

if TYPE_CHECKING:
    from azure.search.documents.aio import SearchClient

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

LOGGER_NAME = "ingenious.services.azure_search.pipeline"
SEMANTIC_RERANK_HEAD_MAX = 50

logger = logging.getLogger(LOGGER_NAME)


class _NullAsyncSearchClient:
    """Minimal async client used in tests if a rerank client wasn't injected.

    This stub provides the `search` coroutine and a `close` coroutine so tests
    can rely on the presence of these attributes without pulling real SDKs.
    """

    async def search(self, *args: Any, **kwargs: Any) -> Any:
        """Return an async-iterable that yields no rows."""

        class _Empty:
            def __aiter__(self) -> "_Empty":
                return self

            async def __anext__(self) -> Any:
                raise StopAsyncIteration

        return _Empty()

    async def close(self) -> None:
        """No-op close to satisfy the interface."""
        return None


class AdvancedSearchPipeline:
    """
    Orchestrates: L1 → DAT → (optional) L2 → (optional) RAG.

    The pipeline owns the retriever, fuser, and (optionally) the answer
    generator. The semantic rerank step reuses the search client unless a
    dedicated client is injected.

    Args:
        config: Valid `SearchConfig` describing services and behavior.
        retriever: The BM25 + vector retriever.
        fuser: The DAT fuser (LLM-backed alpha estimator).
        answer_generator: Optional RAG answer generator.
        rerank_client: Optional client used for Azure Semantic Ranker.
    """

    _config: SearchConfig
    retriever: AzureSearchRetriever
    fuser: DynamicRankFuser
    answer_generator: AnswerGenerator | None
    _rerank_client: Any

    def __init__(
        self,
        config: SearchConfig,
        retriever: AzureSearchRetriever,
        fuser: DynamicRankFuser,
        answer_generator: AnswerGenerator | None = None,
        rerank_client: Any | None = None,
    ) -> None:
        """Initialize the pipeline with required components."""
        self._config = config
        self.retriever = retriever
        self.fuser = fuser
        self.answer_generator = answer_generator
        self._rerank_client = rerank_client or _NullAsyncSearchClient()

    # --------------------------- Internal helpers ---------------------------

    async def _apply_semantic_ranking(
        self, query: str, fused_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Apply Azure Semantic Ranker (L2) to the head of the results.

        The method forms an OR filter over ID equality clauses for the top-N
        results (bounded by `SEMANTIC_RERANK_HEAD_MAX`). It preserves unmatched
        docs and falls back to fused scores on any error.

        Args:
            query: The user query string.
            fused_results: The list of fused results from DAT.

        Returns:
            The reranked list with `_final_score` set for all items.
        """
        head = fused_results[:SEMANTIC_RERANK_HEAD_MAX]
        tail = fused_results[SEMANTIC_RERANK_HEAD_MAX:]

        if not head:
            return fused_results

        id_field = self._config.id_field
        ids = [str(r[id_field]) for r in head if id_field in r]
        if not ids:
            for r in fused_results:
                r["_final_score"] = r.get("_fused_score", 0.0)
            return fused_results

        def _quote(v: str) -> str:
            return "'" + v.replace("'", "''") + "'"

        filt = " or ".join(f"{id_field} eq {_quote(i)}" for i in ids)
        try:
            results = await self._rerank_client.search(
                search_text=query,
                filter=filt,
                query_type=QueryType.SEMANTIC,
                semantic_configuration_name=self._config.semantic_configuration_name,
                top=len(ids),
            )

            by_id = {str(r[id_field]): r for r in head if id_field in r}
            matched: set[str] = set()
            reranked: list[dict[str, Any]] = []

            async for row in results:
                rid = str(row.get(id_field))
                base = by_id.get(rid)
                if not base:
                    continue
                merged = base.copy()
                merged.update(row)
                merged["_final_score"] = merged.get("@search.reranker_score")
                reranked.append(merged)
                matched.add(rid)

            # preserve unmatched head items with fused scores
            for r in head:
                rid = str(r.get(id_field))
                if rid not in matched:
                    keep = r.copy()
                    keep["_final_score"] = keep.get("_fused_score", 0.0)
                    reranked.append(keep)

            return reranked + tail
        except Exception as exc:  # pragma: no cover - exercised in tests
            logger.error(
                "Semantic Ranking failed (%s). Falling back to fused scores.", exc
            )
            for r in fused_results:
                r["_final_score"] = r.get("_fused_score", 0.0)
            return fused_results

    # ----------------------------- Public API -------------------------------

    async def retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """
        Run L1 retrieval → DAT fusion → optional L2; then clean & return top_k.

        Args:
            query: The user query string.
            top_k: The number of items to return after ranking/cleanup.

        Returns:
            A list of cleaned rows (at most `top_k`).
        """
        logger.debug("retrieve(query=%r, top_k=%s) start", query, top_k)
        if top_k <= 0:
            return []
        if not query or not query.strip():
            return []

        # Concurrency with cancellation: if one branch fails, cancel the sibling.
        lex_task = asyncio.create_task(self.retriever.search_lexical(query))
        vec_task = asyncio.create_task(self.retriever.search_vector(query))
        try:
            lex, vec = await asyncio.gather(lex_task, vec_task)
            logger.debug("L1 results: lex=%d vec=%d", len(lex), len(vec))
        except Exception:
            for t in (lex_task, vec_task):
                if not t.cancelled():
                    t.cancel()
            with contextlib.suppress(Exception):
                await asyncio.gather(lex_task, vec_task)
            raise

        # DAT fusion
        try:
            fused = await self.fuser.fuse(query, lex, vec)
            logger.debug("DAT fused=%d", len(fused))
        except Exception as exc:
            logger.exception("DAT fusion failed")
            raise RuntimeError("DAT Fusion failed.") from exc

        # Optional L2
        if self._config.use_semantic_ranking:
            ranked = await self._apply_semantic_ranking(query, fused)
            logger.debug("L2 ranked=%d", len(ranked))
        else:
            ranked = fused
            for r in ranked:
                r["_final_score"] = r.get("_fused_score", 0.0)

        head = ranked[: max(0, int(top_k))]
        logger.debug("head=%d", len(head))
        return self._clean_sources(head)

    async def answer(self, query: str) -> dict[str, Any]:
        """
        Full RAG: retrieve/rank then generate an answer.

        Requires `enable_answer_generation=True` in `SearchConfig`.

        Args:
            query: The user's question.

        Returns:
            A dict with "answer" and "source_chunks" keys.

        Raises:
            GenerationDisabledError: If generation is disabled or misconfigured.
        """
        if not self._config.enable_answer_generation:
            raise GenerationDisabledError(
                "get_answer() requires enable_answer_generation=True. "
                "Construct SearchConfig(..., enable_answer_generation=True)."
            )
        if self.answer_generator is None:
            raise GenerationDisabledError("AnswerGenerator is not configured.")

        if not query or not query.strip():
            return {
                "answer": "Please enter a question so I can search the knowledge base.",
                "source_chunks": [],
            }

        top = await self.retrieve(query, self._config.top_n_final)
        if not top:
            return {
                "answer": (
                    "I could not find any relevant information in the knowledge base "
                    "to answer your question."
                ),
                "source_chunks": [],
            }

        ans = await self.answer_generator.generate(query, top)
        return {"answer": ans, "source_chunks": top}

    async def get_answer(self, query: str) -> dict[str, Any]:
        """Back-compat alias for `answer()`."""
        return await self.answer(query)

    # ------------------------------- Lifecycle -------------------------------

    def _extract_snippet(self, row: dict[str, Any]) -> str:
        """
        Extract a short, plain-text snippet from Azure captions if present.

        Azure Search may return `@search.captions` as a list or dict. This
        helper normalizes that to a simple string so downstream consumers can
        rely on a stable `snippet` field.

        Args:
            row: A single result row as returned by the SDK.

        Returns:
            A best-effort plain-text snippet, or an empty string when unavailable.
        """
        cap = row.get("@search.captions")
        if not cap:
            return ""
        try:
            # Common shapes: list[{"text": "...", ...}], {"text": "..."} or list[str]
            if isinstance(cap, list) and cap:
                first = cap[0]
                if isinstance(first, dict):
                    txt = first.get("text") or first.get("caption") or ""
                    return str(txt)
                return str(first)
            if isinstance(cap, dict):
                txt = cap.get("text") or cap.get("caption") or ""
                return str(txt)
            return str(cap)
        except Exception:
            return ""

    def _clean_sources(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Normalize and trim fields on the final chunks.

        Behavior:
        - Preserve the configured content field and also **alias** it to a stable
          `"content"` key if different and not already present.
        - Preserve a stable `"snippet"` by extracting from `@search.captions`
          when a snippet is not already present.
        - Remove verbose/transient fields (scores, captions, vector field).

        Args:
            chunks: Raw rows emitted by retrieval/ranking.

        Returns:
            Cleaned rows suitable for downstream formatting.
        """
        out: list[dict[str, Any]] = []
        cfg_content_field = self._config.content_field
        vec_field = self._config.vector_field

        for c in chunks:
            d = c.copy()

            # 1) Map Azure captions -> stable 'snippet' if the caller didn't supply one
            if "snippet" not in d:
                snippet = self._extract_snippet(c)
                if snippet:
                    d["snippet"] = snippet

            # 2) Alias configured content field to a stable 'content' key for consumers
            if cfg_content_field != "content" and "content" not in d:
                if cfg_content_field in d:
                    try:
                        d["content"] = d[cfg_content_field]
                    except Exception:
                        # Defensive: ignore aliasing failure and continue cleanup.
                        pass

            # 3) Strip transient/verbose fields
            for k in (
                "_retrieval_score",
                "_normalized_score",
                "_fused_score",
                "@search.score",
                "@search.captions",
                "@search.reranker_score",
                vec_field,
            ):
                d.pop(k, None)

            out.append(d)
        return out

    async def close(self) -> None:
        """
        Close underlying clients gracefully (best effort).

        Ensures that retriever, fuser, generator, and rerank client are closed
        if they expose a `close()` method (sync or async).
        """

        async def _aclose(x: Any) -> None:
            if not x:
                return
            closer = getattr(x, "close", None)
            if not closer:
                return
            try:
                res = closer()
                if asyncio.iscoroutine(res):
                    await res
            except Exception:  # pragma: no cover - defensive
                logger.exception("Error closing a pipeline component.")

        await asyncio.gather(
            _aclose(self.retriever),
            _aclose(self.fuser),
            _aclose(self.answer_generator),
            _aclose(self._rerank_client),
            return_exceptions=True,
        )


# ----------------------------- Factory function ------------------------------


def build_search_pipeline(config: SearchConfig) -> AdvancedSearchPipeline:
    """
    Compose the pipeline with clients produced via client_init factories.

    Validates semantic ranking configuration and constructs shared clients
    (Search + AOAI) used by retriever, DAT fuser, and (optionally) generator.

    Args:
        config: Validated `SearchConfig`.

    Returns:
        An initialized `AdvancedSearchPipeline`.

    Raises:
        ValueError: If semantic ranking is enabled without a configuration name.
    """
    if config.use_semantic_ranking and not config.semantic_configuration_name:
        raise ValueError(
            "Configuration Error: 'use_semantic_ranking' is True, but "
            "'semantic_configuration_name' is not provided."
        )

    from ingenious.services.azure_search.client_init import (
        make_async_openai_client,
        make_async_search_client,
    )

    # Create shared clients
    search_client: SearchClient | Any = make_async_search_client(config)
    rerank_client: SearchClient | Any = search_client  # reuse unless dedicated client
    llm_client: Any = make_async_openai_client(config)

    # Compose components
    retriever = AzureSearchRetriever(
        config=config, search_client=search_client, embedding_client=llm_client
    )
    fuser = DynamicRankFuser(config=config, llm_client=llm_client)
    generator = (
        AnswerGenerator(config=config, llm_client=llm_client)
        if config.enable_answer_generation
        else None
    )

    # Assemble pipeline
    return AdvancedSearchPipeline(
        config=config,
        retriever=retriever,
        fuser=fuser,
        answer_generator=generator,
        rerank_client=rerank_client,
    )
