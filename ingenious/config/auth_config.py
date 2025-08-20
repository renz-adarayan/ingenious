# ingenious/config/auth_config.py
from __future__ import annotations

from typing import Any, Mapping, Optional

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

        # Precedence: TokenCredential (AAD) > API key
        if client_id and client_secret and tenant_id:
            method = AuthenticationMethod.CLIENT_ID_AND_SECRET
        elif client_id and not client_secret and not tenant_id:
            method = AuthenticationMethod.MSI
        elif api_key:
            method = AuthenticationMethod.TOKEN
        else:
            method = AuthenticationMethod.DEFAULT_CREDENTIAL

        # Respect explicit method only if it doesn't demote AAD in presence of SPN
        if isinstance(explicit, AuthenticationMethod):
            if method != AuthenticationMethod.CLIENT_ID_AND_SECRET:
                method = explicit

        return cls(
            authentication_method=method,
            api_key=api_key,
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
            endpoint=endpoint,
        )

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

    def to_openai_async_token_provider_or_none(self, scope: str):
        """
        Returns a callable suitable for openai.AsyncAzureOpenAI(azure_ad_token_provider=...)
        or None if key-based auth should be used.
        """
        # If explicit key path, don't build a provider.
        if self.authentication_method == AuthenticationMethod.TOKEN and self.api_key:
            return None

        try:
            from azure.identity.aio import (  # type: ignore
                DefaultAzureCredential,
                ManagedIdentityCredential,
                ClientSecretCredential,
                get_bearer_token_provider,
            )
        except Exception:
            # azure-identity not available
            return None

        if self.authentication_method == AuthenticationMethod.CLIENT_ID_AND_SECRET:
            cred = ClientSecretCredential(
                tenant_id=self.tenant_id,
                client_id=self.client_id,
                client_secret=self.client_secret,
            )
        elif self.authentication_method == AuthenticationMethod.MSI:
            cred = (
                ManagedIdentityCredential(client_id=self.client_id)
                if self.client_id
                else ManagedIdentityCredential()
            )
        else:
            cred = DefaultAzureCredential()

        return get_bearer_token_provider(cred, scope)
