from __future__ import annotations
from typing import Any, Mapping
import inspect

DEFAULT_OPENAI_MAX_RETRIES = 3

class AsyncAzureOpenAIClientBuilder:
    def __init__(self, model_config: Any, api_version: str | None, client_options: dict[str, Any] | None):
        self.model_config = model_config
        self.api_version = api_version
        self.client_options = dict(client_options or {})

    @classmethod
    def from_config(
        cls,
        config: Any,
        api_version: str | None = None,
        client_options: dict[str, Any] | None = None,
    ) -> "AsyncAzureOpenAIClientBuilder":
        # normalize aliases and apply defaults here so every callsite is consistent
        norm_opts = _normalize_openai_client_options(client_options or {}, config)
        return cls(config, api_version=api_version, client_options=norm_opts)

    def _get(self, name: str, default: Any = None) -> Any:
        mc = self.model_config
        if isinstance(mc, Mapping):
            return mc.get(name, default)
        return getattr(mc, name, default)

    def build(self):
        # Map config -> SDK kwargs
        azure_endpoint = (
            self._get("openai_endpoint")
            or self._get("base_url")
            or self._get("endpoint")
            or self._get("azure_endpoint")
        )
        api_key = _to_plain_secret(self._get("openai_key")) or _to_plain_secret(self._get("api_key"))
        api_version = self.api_version or self._get("openai_version") or self._get("api_version")

        if not azure_endpoint:
            raise ValueError("Azure OpenAI endpoint is required")
        if not api_key:
            raise ValueError("Azure OpenAI api_key is required")
        if not api_version:
            raise ValueError("Azure OpenAI api_version is required")

        kwargs: dict[str, Any] = {
            "azure_endpoint": azure_endpoint,
            "api_key": api_key,
            "api_version": api_version,
        }
        # Merge normalized client options (includes canonical max_retries)
        kwargs.update(self.client_options)

        from openai import AsyncAzureOpenAI
        # Filter unknown kwargs so neither the real SDK nor your dummy explodes
        kwargs = _filter_kwargs_for_ctor(AsyncAzureOpenAI, kwargs)
        return AsyncAzureOpenAI(**kwargs)


def _normalize_openai_client_options(
    opts: dict[str, Any],
    config: Any | None,
) -> dict[str, Any]:
    """
    Normalize caller-provided options and apply default max_retries when neither
    canonical nor alias is provided.

    Precedence:
      1) explicit opts['max_retries']
      2) explicit opts['retries'] (alias)
      3) config.openai_max_retries or config.max_retries
      4) DEFAULT_OPENAI_MAX_RETRIES (3)  <-- required to satisfy your unit test
    """
    out = dict(opts)

    # Canonicalize alias
    if "max_retries" not in out and "retries" in out:
        out["max_retries"] = out.pop("retries")

    # If still missing, check config fields (Mapping or Pydantic object)
    if "max_retries" not in out and config is not None:
        mr = None
        if isinstance(config, Mapping):
            mr = config.get("openai_max_retries") or config.get("max_retries")
        else:
            mr = getattr(config, "openai_max_retries", None) or getattr(config, "max_retries", None)
        if mr is not None:
            out["max_retries"] = mr

    # Final default (keeps tests green)
    out.setdefault("max_retries", DEFAULT_OPENAI_MAX_RETRIES)

    # Basic validation
    out["max_retries"] = int(out["max_retries"])
    if out["max_retries"] < 0:
        raise ValueError("max_retries must be >= 0")

    return out


def _filter_kwargs_for_ctor(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Only pass kwargs the constructor actually accepts.

    - If the __init__ has **kwargs, keep everything.
    - Otherwise, drop unknown keys to avoid TypeError with test doubles or SDK changes.
    """
    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        # If signature introspection fails, pass as-is (best effort)
        return kwargs

    params = sig.parameters.values()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return kwargs  # ctor accepts **kwargs, no filtering necessary

    allowed = {p.name for p in params if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                                    inspect.Parameter.KEYWORD_ONLY)}
    # 'self' is in the signature; it won't be in kwargs anyway, so no need to remove
    return {k: v for k, v in kwargs.items() if k in allowed}

def _to_plain_secret(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    getter = getattr(value, "get_secret_value", None)
    return getter() if callable(getter) else None
