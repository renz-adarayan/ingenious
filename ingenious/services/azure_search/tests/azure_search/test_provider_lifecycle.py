"""Test the AzureSearchProvider's resource lifecycle management.

These tests verify that the AzureSearchProvider correctly manages the
lifecycle of its underlying resources, such as Azure SDK clients.
Properly closing these resources is crucial to prevent connection leaks
and ensure graceful application shutdown. This suite focuses on the
`close()` method's behavior.
"""

from __future__ import annotations

import pytest

from ingenious.services.azure_search.config import SearchConfig
from ingenious.services.azure_search.provider import AzureSearchProvider


class _DummyPipeline:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_provider_close_calls_all_underlying_clients(
    config: SearchConfig,
) -> None:
    p = _DummyPipeline()
    provider = AzureSearchProvider(settings_or_config=config, pipeline=p)
    await provider.close()
    assert p.closed is True
