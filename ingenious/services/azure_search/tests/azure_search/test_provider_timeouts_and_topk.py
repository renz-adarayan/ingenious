"""
Timeout propagation and top_k=0 short-circuit — provider delegates to pipeline.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from ingenious.config.main_settings import IngeniousSettings
from ingenious.config.models import AzureSearchSettings, ModelSettings
from ingenious.services.azure_search.provider import AzureSearchProvider


def _settings(use_semantic: bool = True) -> IngeniousSettings:
    s = IngeniousSettings.model_construct()
    s.models = [
        ModelSettings(
            model="text-embedding-3-small",
            deployment="emb",
            api_key="K",
            base_url="https://oai",
        ),
        ModelSettings(
            model="gpt-4o", deployment="chat", api_key="K", base_url="https://oai"
        ),
    ]
    s.azure_search_services = [
        AzureSearchSettings(
            service="svc",
            endpoint="https://search.example.net",
            key="SK",
            index_name="idx",
            use_semantic_ranking=use_semantic,
        ),
    ]
    return s


@pytest.mark.asyncio
async def test_provider_retrieve_timeout_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    pipeline = MagicMock()
    pipeline.retrieve = AsyncMock(side_effect=asyncio.TimeoutError("vec timeout"))

    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.build_search_pipeline",
        lambda _cfg: pipeline,
        raising=False,
    )

    prov = AzureSearchProvider(settings)
    with pytest.raises(asyncio.TimeoutError):
        await prov.retrieve("q")
    await prov.close()


@pytest.mark.asyncio
async def test_provider_retrieve_topk_zero_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(use_semantic=False)
    pipeline = MagicMock()
    pipeline.retrieve = AsyncMock(return_value=[])

    monkeypatch.setattr(
        "ingenious.services.azure_search.provider.build_search_pipeline",
        lambda _cfg: pipeline,
        raising=False,
    )

    prov = AzureSearchProvider(settings)
    out = await prov.retrieve("q", top_k=0)
    assert out == []
    await prov.close()
