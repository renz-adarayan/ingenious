# tests/test_async_builders_and_factory.py
import importlib
import sys

import pytest


def _reload(module_name: str):
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    else:
        __import__(module_name)
    return sys.modules[module_name]


def test_async_search_builder_api_key_path():
    builder_mod = _reload("ingenious.client.azure.builder.search_client_async")
    AzureSearchAsyncClientBuilder = builder_mod.AzureSearchAsyncClientBuilder

    cfg = {"search_endpoint": "https://example.search.windows.net", "search_key": "S_KEY"}
    b = AzureSearchAsyncClientBuilder.from_config(cfg, index_name="idx")
    client = b.build()

    from azure.core.credentials import AzureKeyCredential  # stubbed
    assert client.endpoint == "https://example.search.windows.net"
    assert client.index_name == "idx"
    assert isinstance(client.credential, AzureKeyCredential)


def test_async_search_builder_prefers_token_over_key():
    builder_mod = _reload("ingenious.client.azure.builder.search_client_async")
    AzureSearchAsyncClientBuilder = builder_mod.AzureSearchAsyncClientBuilder

    # Provide BOTH SP credentials and a key; must prefer TokenCredential (AAD)
    cfg = {
        "search_endpoint": "https://example.search.windows.net",
        "search_key": "S_KEY",
        "client_id": "cid",
        "client_secret": "csec",
        "tenant_id": "tid",
    }
    b = AzureSearchAsyncClientBuilder.from_config(cfg, index_name="idx", client_options={"user_agent": "ua"})
    client = b.build()

    from azure.identity.aio import ClientSecretCredential  # stubbed
    assert isinstance(client.credential, ClientSecretCredential)
    assert client.kwargs.get("user_agent") == "ua"


def test_async_openai_builder_api_key_path():
    builder_mod = _reload("ingenious.client.azure.builder.openai_client_async")
    AsyncAzureOpenAIClientBuilder = builder_mod.AsyncAzureOpenAIClientBuilder

    cfg = {"openai_endpoint": "https://example.openai.azure.com", "openai_key": "O_KEY", "api_version": "2024-02-01"}
    b = AsyncAzureOpenAIClientBuilder.from_config(cfg)
    client = b.build()

    assert client.azure_endpoint == "https://example.openai.azure.com"
    assert client.api_version == "2024-02-01"
    assert client.api_key == "O_KEY"
    assert client.azure_ad_token_provider is None


def test_async_openai_builder_aad_path():
    builder_mod = _reload("ingenious.client.azure.builder.openai_client_async")
    AsyncAzureOpenAIClientBuilder = builder_mod.AsyncAzureOpenAIClientBuilder

    cfg = {"openai_endpoint": "https://example.openai.azure.com"}  # no key => use AAD default
    b = AsyncAzureOpenAIClientBuilder.from_config(cfg, api_version="2024-02-01")
    client = b.build()

    assert client.azure_endpoint == "https://example.openai.azure.com"
    assert client.api_version == "2024-02-01"
    assert client.azure_ad_token_provider is not None
    assert callable(client.azure_ad_token_provider)
    assert client.azure_ad_token_provider() == "aad-bearer-token"


def test_factory_async_methods_select_and_forward():
    fac_mod = _reload("ingenious.client.azure.azure_client_builder_factory")
    AzureClientFactory = fac_mod.AzureClientFactory

    # Search client (key path)
    cfg_search = {"search_endpoint": "https://ex.search.windows.net", "search_key": "K"}
    s_client = AzureClientFactory.create_async_search_client(index_name="myindex", config=cfg_search)
    assert s_client.index_name == "myindex"

    # OpenAI client (key path)
    cfg_aoai = {"openai_endpoint": "https://ex.openai.azure.com", "openai_key": "K", "api_version": "2024-02-01"}
    o_client = AzureClientFactory.create_async_openai_client(config=cfg_aoai)
    assert o_client.api_key == "K"


@pytest.mark.asyncio
async def test_create_async_search_client_prefers_token():
    # Async test: ensure factory prefers token when both are present
    fac_mod = _reload("ingenious.client.azure.azure_client_builder_factory")
    AzureClientFactory = fac_mod.AzureClientFactory

    cfg = {
        "search_endpoint": "https://example.search.windows.net",
        "search_key": "KEY",
        "client_id": "cid",
        "client_secret": "csec",
        "tenant_id": "tid",
    }
    client = AzureClientFactory.create_async_search_client(index_name="idx", config=cfg)
    from azure.identity.aio import ClientSecretCredential
    assert isinstance(client.credential, ClientSecretCredential)
    # Exercise async method to satisfy pytest-asyncio path
    await client.close()
