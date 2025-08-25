# ingenious/client/azure/builder/openai_client_async.py
from __future__ import annotations

import inspect
from typing import Any, Mapping, Optional

from ingenious.config.auth_config import AzureAuthConfig

DEFAULT_OPENAI_MAX_RETRIES = 3
_COGS_SCOPE = "https://cognitiveservices.azure.com/.default"


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """Return the first non-empty attr/mapping value among aliases."""
    for n in names:
        if isinstance(obj, Mapping):
            if n in obj and obj[n] not in (None, ""):
                return obj[n]
        else:
            v = getattr(obj, n, None)
            if v not in (None, ""):
                return v
    return default


def _to_plain_secret(value: Any) -> Optional[str]:
    """Unwrap pydantic SecretStr or return the string directly."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    getter = getattr(value, "get_secret_value", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            return None
    return None


def _normalize_openai_client_options(
    opts: Mapping[str, Any] | None, config: Any | None
) -> dict[str, Any]:
    """
    Normalize caller-provided options and apply default max_retries.

    Precedence:
      1) explicit opts['max_retries']
      2) explicit opts['retries'] (alias)
      3) config.openai_max_retries or config.max_retries
      4) DEFAULT_OPENAI_MAX_RETRIES (3)
    """
    out = dict(opts or {})

    # Canonicalize alias
    if "max_retries" not in out and "retries" in out:
        out["max_retries"] = out.pop("retries")

    # Pull from config if still missing
    if "max_retries" not in out and config is not None:
        mr = None
        if isinstance(config, Mapping):
            mr = config.get("openai_max_retries") or config.get("max_retries")
        else:
            mr = getattr(config, "openai_max_retries", None) or getattr(
                config, "max_retries", None
            )
        if mr is not None:
            out["max_retries"] = mr

    out.setdefault("max_retries", DEFAULT_OPENAI_MAX_RETRIES)

    # Basic validation
    out["max_retries"] = int(out["max_retries"])
    if out["max_retries"] < 0:
        raise ValueError("max_retries must be >= 0")

    return out


def _filter_kwargs_for_ctor(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Only pass kwargs the constructor actually accepts.

    - If __init__ has **kwargs, keep everything.
    - Otherwise, drop unknown keys to avoid TypeError with strict dummies / SDK changes.
    """
    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return kwargs

    params = sig.parameters.values()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return kwargs  # ctor accepts **kwargs

    allowed = {
        p.name
        for p in params
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {k: v for k, v in kwargs.items() if k in allowed}


class AsyncAzureOpenAIClientBuilder:
    """
    Builder for `openai.AsyncAzureOpenAI`.

    Auth precedence (per spec): **AAD token provider > API key**.
    """

    def __init__(
        self,
        model_config: Any,
        api_version: str | None,
        client_options: dict[str, Any] | None,
    ):
        self._cfg = model_config
        self._api_version = api_version
        self._client_options = dict(client_options or {})

    @classmethod
    def from_config(
        cls,
        config: Any,
        api_version: str | None = None,
        client_options: Mapping[str, Any] | None = None,
    ) -> "AsyncAzureOpenAIClientBuilder":
        norm_opts = _normalize_openai_client_options(client_options, config)
        return cls(config, api_version=api_version, client_options=norm_opts)

    def build(self):
        # Resolve endpoint & version
        azure_endpoint = _get(
            self._cfg, "openai_endpoint", "base_url", "endpoint", "azure_endpoint"
        )
        if not azure_endpoint:
            raise ValueError("Azure OpenAI endpoint is required")

        # Take explicit api_version override, else config, else auth default
        api_version = (
            self._api_version
            or _get(self._cfg, "openai_version", "api_version")
            or AzureAuthConfig.from_config(self._cfg).api_version
        )
        if not api_version:
            raise ValueError("Azure OpenAI api_version is required")

        # Prefer AAD token provider over a key
        auth = AzureAuthConfig.from_config(self._cfg)
        token_provider = auth.to_openai_async_token_provider_or_none(_COGS_SCOPE)

        # Resolve API key (optional)
        api_key = _to_plain_secret(_get(self._cfg, "openai_key", "api_key"))

        # Common kwargs for both auth paths
        kwargs: dict[str, Any] = {
            "azure_endpoint": azure_endpoint,
            "api_version": api_version,
        }
        kwargs.update(self._client_options)

        if token_provider:
            kwargs["azure_ad_token_provider"] = token_provider
        elif api_key:
            kwargs["api_key"] = api_key
        else:
            # Should not normally happen: no token provider and no key
            raise ValueError(
                "No authentication available for Azure OpenAI (no AAD token provider and no API key)."
            )

        from openai import AsyncAzureOpenAI  # import here to keep builders import-light

        kwargs = _filter_kwargs_for_ctor(AsyncAzureOpenAI, kwargs)
        return AsyncAzureOpenAI(**kwargs)
