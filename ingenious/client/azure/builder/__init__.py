from __future__ import annotations

from typing import Any

__all__ = (
    # sync builders (preserve API)
    "BlobClientBuilder",
    "BlobServiceClientBuilder",
    "AzureOpenAIClientBuilder",
    "AzureOpenAIChatCompletionClientBuilder",
    "AzureSearchClientBuilder",
    "CosmosClientBuilder",
    "AzureSqlClientBuilder",
    "AzureSqlClientBuilderWithAuth",
    # async builders (new)
    "AzureSearchAsyncClientBuilder",
    "AsyncAzureOpenAIClientBuilder",
)


def __getattr__(name: str) -> Any:
    # Import on first access; keeps import-time side effects minimal.
    if name in ("BlobClientBuilder", "BlobServiceClientBuilder"):
        from .blob_client import BlobClientBuilder, BlobServiceClientBuilder

        return {
            "BlobClientBuilder": BlobClientBuilder,
            "BlobServiceClientBuilder": BlobServiceClientBuilder,
        }[name]

    if name == "AzureOpenAIClientBuilder":
        from .openai_client import AzureOpenAIClientBuilder

        return AzureOpenAIClientBuilder

    if name == "AzureOpenAIChatCompletionClientBuilder":
        from .openai_chat_completions_client import (
            AzureOpenAIChatCompletionClientBuilder,
        )

        return AzureOpenAIChatCompletionClientBuilder

    if name == "AzureSearchClientBuilder":
        from .search_client import AzureSearchClientBuilder

        return AzureSearchClientBuilder

    if name == "CosmosClientBuilder":
        from .cosmos_client import CosmosClientBuilder

        return CosmosClientBuilder

    if name in ("AzureSqlClientBuilder", "AzureSqlClientBuilderWithAuth"):
        from .sql_client import AzureSqlClientBuilder, AzureSqlClientBuilderWithAuth

        return {
            "AzureSqlClientBuilder": AzureSqlClientBuilder,
            "AzureSqlClientBuilderWithAuth": AzureSqlClientBuilderWithAuth,
        }[name]

    if name == "AzureSearchAsyncClientBuilder":
        from .search_client_async import AzureSearchAsyncClientBuilder

        return AzureSearchAsyncClientBuilder

    if name == "AsyncAzureOpenAIClientBuilder":
        from .openai_client_async import AsyncAzureOpenAIClientBuilder

        return AsyncAzureOpenAIClientBuilder

    raise AttributeError(name)
