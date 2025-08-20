# tests/test_auth_aliasing_and_client_init.py
import importlib
import sys

from pydantic import SecretStr


def _reload(module_name: str):
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    else:
        __import__(module_name)
    return sys.modules[module_name]


def test_auth_aliasing_maps_keys_and_endpoint():
    auth_mod = _reload("ingenious.config.auth_config")
    AzureAuthConfig = auth_mod.AzureAuthConfig

    cfg1 = {"search_key": "S"}
    a1 = AzureAuthConfig.from_config(cfg1)
    assert a1.api_key == "S"

    cfg2 = {"openai_key": "O", "openai_endpoint": "https://aoai"}
    a2 = AzureAuthConfig.from_config(cfg2)
    assert a2.api_key == "O"
    assert a2.endpoint == "https://aoai"


def test_client_init_delegates_to_factory(monkeypatch):
    fac_mod = _reload("ingenious.client.azure.azure_client_builder_factory")
    called = {"search": False, "openai": False}

    def fake_search(index_name, config=None, **opts):
        called["search"] = True
        return object()

    def fake_openai(config=None, api_version=None, **opts):
        called["openai"] = True
        return object()

    monkeypatch.setattr(fac_mod.AzureClientFactory, "create_async_search_client", staticmethod(fake_search))
    monkeypatch.setattr(fac_mod.AzureClientFactory, "create_async_openai_client", staticmethod(fake_openai))

    client_init = _reload("ingenious.services.azure_search.client_init")
    cfg_mod = _reload("ingenious.services.azure_search.config")
    SearchConfig = cfg_mod.SearchConfig

    cfg = SearchConfig(
        search_endpoint="https://s",
        search_key=SecretStr("sk"),
        search_index_name="idx",
        openai_endpoint="https://o",
        openai_key=SecretStr("ok"),
        openai_version="2024-02-01",
        embedding_deployment_name="emb",
        generation_deployment_name="gen",
        use_semantic_ranking=False,
        top_k_retrieval=5,
        top_n_final=3,
    )

    sc = client_init.make_search_client(cfg)
    oc = client_init.make_async_openai_client(cfg)
    assert called["search"] is True
    assert called["openai"] is True
    assert sc is not None and oc is not None
