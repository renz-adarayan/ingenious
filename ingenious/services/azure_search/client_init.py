"""Create Azure AI service clients via the central factory (async)."""

from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from azure.search.documents.aio import SearchClient
    from openai import AsyncAzureOpenAI
    from .config import SearchConfig

# Allow tests to monkeypatch either this symbol or the accessor.
AzureClientFactory = None  # type: ignore
__all__ = ["async_", "make_async_openai_client"]

# Internal cache to avoid repeated imports; test can also patch this in place.
_FACTORY_SINGLETON: Any | None = None


def _get_factory() -> Any:
    """Return a factory class/obj, preferring a locally patched symbol if present."""
    global _FACTORY_SINGLETON
    if AzureClientFactory is not None:
        return AzureClientFactory
    if _FACTORY_SINGLETON is None:
        from ingenious.client.azure import AzureClientFactory as _F
        _FACTORY_SINGLETON = _F
    return _FACTORY_SINGLETON


def make_async_search_client(cfg: "SearchConfig", **client_options: Any) -> "SearchClient":
    """Create the async Azure Search client via AzureClientFactory.

    Any keyword args in `client_options` are forwarded to the underlying SDK ctor.
    """
    factory = _get_factory()
    return factory.create_async_search_client(
        index_name=cfg.search_index_name,
        config={
            "endpoint": cfg.search_endpoint,
            "search_key": cfg.search_key.get_secret_value(),  # unwrap SecretStr
        },
        **client_options,
    )

def make_async_openai_client(cfg: "SearchConfig", **client_options: Any) -> "AsyncAzureOpenAI":
    """Create the async Azure OpenAI client via AzureClientFactory.

    Any keyword args in `client_options` are forwarded to the underlying SDK ctor.
    We default max_retries to 3 if not provided (keeps tests and sensible prod default).
    """
    factory = _get_factory()

    return factory.create_async_openai_client(
        config={
            "openai_endpoint": cfg.openai_endpoint,
            "openai_key": cfg.openai_key.get_secret_value(),  # unwrap SecretStr
        },
        api_version=cfg.openai_version,
        **client_options,
    )
