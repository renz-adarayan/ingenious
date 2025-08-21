# ingenious/config/auth_config.py
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Mapping, Optional

from ingenious.common.enums import AuthenticationMethod


def _get(obj: Any, *names: str) -> Optional[Any]:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        for n in names:
            if n in obj and obj[n] is not None:
                return obj[n]
        return None
    for n in names:
        val = getattr(obj, n, None)
        if val is not None:
            return val
    return None


class AzureAuthConfig:
    """
    Centralized auth configuration for Azure client builders.

    Fields:
      - authentication_method: AuthenticationMethod
      - api_key: Optional[str] (TOKEN/API key)
      - client_id, client_secret, tenant_id: Optional[str] (AAD/SPN or MSI client_id)
      - endpoint: Optional[str] (used by some callers)
      - openai_key / openai_endpoint aliases are recognized in from_config()
      - search_key alias recognized in from_config()
    """

    def __init__(
        self,
        authentication_method: AuthenticationMethod = AuthenticationMethod.DEFAULT_CREDENTIAL,
        api_key: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        tenant_id: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> None:
        self.authentication_method = authentication_method
        self.api_key = api_key
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.endpoint = endpoint

        # Optional AOAI specifics (may be read by consumers)
        self.openai_key: Optional[str] = api_key if api_key else None
        self.openai_endpoint: Optional[str] = endpoint if endpoint else None

        # Optional API version if present in config
        self.api_version: Optional[str] = None

    @classmethod
    def default_credential(cls) -> "AzureAuthConfig":
        return cls(authentication_method=AuthenticationMethod.DEFAULT_CREDENTIAL)

    @classmethod
    def from_config(cls, config: Any) -> "AzureAuthConfig":
        # Aliases for API key
        api_key = _get(config, "api_key", "key", "search_key", "openai_key", "token")
        if api_key is not None:
            api_key = str(api_key)

        # Endpoint aliases (not required everywhere)
        endpoint = _get(config, "endpoint", "base_url", "url", "openai_endpoint")

        # AAD/SPN/MSI
        client_id = _get(config, "client_id")
        client_secret = _get(config, "client_secret")
        tenant_id = _get(config, "tenant_id")

        # Optional explicit method
        explicit = _get(config, "authentication_method")
        if isinstance(explicit, str):
            try:
                explicit = AuthenticationMethod[explicit]
            except Exception:
                explicit = None

        # Optional API version (useful for AOAI)
        api_version = _get(config, "openai_version", "api_version")

        # Precedence: TokenCredential (AAD/MSI) > API key
        if client_id and client_secret and tenant_id:
            method = AuthenticationMethod.CLIENT_ID_AND_SECRET
        elif client_id and not client_secret and not tenant_id:
            method = AuthenticationMethod.MSI
        elif api_key:
            method = AuthenticationMethod.TOKEN
        else:
            method = AuthenticationMethod.DEFAULT_CREDENTIAL

        # Respect explicit method only if it doesn't demote SPN
        if isinstance(explicit, AuthenticationMethod):
            if method != AuthenticationMethod.CLIENT_ID_AND_SECRET:
                method = explicit

        inst = cls(
            authentication_method=method,
            api_key=api_key,
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
            endpoint=endpoint,
        )
        inst.openai_key = api_key
        inst.openai_endpoint = endpoint
        inst.api_version = str(api_version) if api_version else None
        return inst

    def validate_for_method(self) -> None:
        if self.authentication_method == AuthenticationMethod.TOKEN:
            if not self.api_key:
                raise ValueError("API key is required for TOKEN authentication.")
        elif self.authentication_method == AuthenticationMethod.CLIENT_ID_AND_SECRET:
            if not (self.client_id and self.client_secret and self.tenant_id):
                raise ValueError(
                    "CLIENT_ID_AND_SECRET requires client_id, client_secret, and tenant_id."
                )
        # MSI/DEFAULT have no additional required fields.

    # -------------------------- Async helpers --------------------------

    def to_openai_async_token_provider_or_none(
        self,
        scope: str,
    ) -> Optional[Callable[[], str]]:
        """
        Return a **synchronous** callable that yields a bearer token string
        suitable for passing as `azure_ad_token_provider` to
        `openai.AsyncAzureOpenAI`. Returns None iff key-based auth should be used.

        Strategy:
          1) If explicit TOKEN w/ api_key → return None (we will use key auth).
          2) Try sync `azure.identity` path (preferred).
          3) Else try `azure.identity.aio` and **wrap** the provider so it's sync.
          4) If neither import works, raise helpful ImportError.
        """
        # If explicit key path, don't build a provider.
        if self.authentication_method == AuthenticationMethod.TOKEN and self.api_key:
            return None

        # Try sync azure.identity first
        try:
            from azure.identity import (  # type: ignore
                DefaultAzureCredential as SyncDefaultAzureCredential,
                ClientSecretCredential as SyncClientSecretCredential,
                ManagedIdentityCredential as SyncManagedIdentityCredential,
                get_bearer_token_provider as get_sync_bearer_token_provider,
            )

            if (
                self.authentication_method == AuthenticationMethod.CLIENT_ID_AND_SECRET
                and self.tenant_id
                and self.client_id
                and self.client_secret
            ):
                cred = SyncClientSecretCredential(
                    tenant_id=str(self.tenant_id),
                    client_id=str(self.client_id),
                    client_secret=str(self.client_secret),
                )
            elif self.authentication_method == AuthenticationMethod.MSI and self.client_id:
                cred = SyncManagedIdentityCredential(client_id=str(self.client_id))
            else:
                # DEFAULT or unspecified → DefaultAzureCredential
                cred = SyncDefaultAzureCredential(
                    exclude_interactive_browser_credential=True
                )
            return get_sync_bearer_token_provider(cred, scope)
        except Exception:
            # Fall back to aio
            pass

        # Try aio azure.identity.aio and wrap to sync callable
        try:
            from azure.identity.aio import (  # type: ignore
                DefaultAzureCredential as AioDefaultAzureCredential,
                ClientSecretCredential as AioClientSecretCredential,
                ManagedIdentityCredential as AioManagedIdentityCredential,
                get_bearer_token_provider as get_aio_bearer_token_provider,
            )

            if (
                self.authentication_method == AuthenticationMethod.CLIENT_ID_AND_SECRET
                and self.tenant_id
                and self.client_id
                and self.client_secret
            ):
                aio_cred = AioClientSecretCredential(
                    tenant_id=str(self.tenant_id),
                    client_id=str(self.client_id),
                    client_secret=str(self.client_secret),
                )
            elif self.authentication_method == AuthenticationMethod.MSI and self.client_id:
                aio_cred = AioManagedIdentityCredential(client_id=str(self.client_id))
            else:
                aio_cred = AioDefaultAzureCredential(
                    exclude_interactive_browser_credential=True
                )
            aio_provider = get_aio_bearer_token_provider(aio_cred, scope)
        except Exception as e:  # pragma: no cover
            # Neither sync nor aio identity path is available
            raise ImportError(
                "Async Azure OpenAI with AAD requires 'azure-identity'. "
                "Install with: pip install azure-identity"
            ) from e

        # Wrap the aio provider into a sync callable returning string
        def _sync_provider() -> str:
            token_or_coro = aio_provider()
            if inspect.isawaitable(token_or_coro):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Create a private loop to avoid 'running loop' errors
                        new_loop = asyncio.new_event_loop()
                        try:
                            return new_loop.run_until_complete(token_or_coro)  # type: ignore[return-value]
                        finally:
                            new_loop.close()
                    return loop.run_until_complete(token_or_coro)  # type: ignore[return-value]
                except RuntimeError:
                    # No loop set yet
                    return asyncio.run(token_or_coro)  # type: ignore[return-value]
            # Some stubs may return a plain string (tests)
            return token_or_coro  # type: ignore[return-value]

        return _sync_provider
