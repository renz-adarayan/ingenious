from abc import ABC, abstractmethod
from typing import Any, Optional, Union

from azure.core.credentials import AzureKeyCredential, TokenCredential
from azure.identity import (
    ClientSecretCredential,
    DefaultAzureCredential,
    ManagedIdentityCredential,
)

from ingenious.common.enums import AuthenticationMethod
from ingenious.config.auth_config import AzureAuthConfig


class AzureClientBuilder(ABC):
    """Abstract base class for Azure client builders with authentication support."""

    def __init__(self, auth_config: Optional[AzureAuthConfig] = None):
        self.auth_config = auth_config or AzureAuthConfig.default_credential()
        self._credential = None  # Lazy-loaded credential cache

    @classmethod
    def from_config(cls, config: Any):
        """
        Create builder instance from a configuration object.

        Args:
            config: Configuration object (either legacy or new format)

        Returns:
            Builder instance with authentication configuration extracted
        """
        auth_config = AzureAuthConfig.from_config(config)
        return cls(auth_config=auth_config)

    @property
    def credential(self) -> Union[TokenCredential, AzureKeyCredential]:
        """
        Get the appropriate credential based on authentication method.
        Cached after first access for efficiency.

        Returns:
            TokenCredential: For Azure AD authentication (DEFAULT_CREDENTIAL, MSI, CLIENT_ID_AND_SECRET)
            AzureKeyCredential: For API key authentication (TOKEN)
        """
        if self._credential is None:
            # Validate authentication configuration
            self.auth_config.validate_for_method()

            if (
                self.auth_config.authentication_method
                == AuthenticationMethod.DEFAULT_CREDENTIAL
            ):
                self._credential = DefaultAzureCredential()

            elif self.auth_config.authentication_method == AuthenticationMethod.MSI:
                if not self.auth_config.client_id:
                    # Use system-assigned managed identity
                    self._credential = ManagedIdentityCredential()
                else:
                    # Use user-assigned managed identity
                    self._credential = ManagedIdentityCredential(
                        client_id=self.auth_config.client_id
                    )

            elif (
                self.auth_config.authentication_method
                == AuthenticationMethod.CLIENT_ID_AND_SECRET
            ):
                # Type assertion since validation ensures these are not None
                assert self.auth_config.client_id is not None
                assert self.auth_config.client_secret is not None
                assert self.auth_config.tenant_id is not None
                self._credential = ClientSecretCredential(
                    tenant_id=self.auth_config.tenant_id,
                    client_id=self.auth_config.client_id,
                    client_secret=self.auth_config.client_secret,
                )

            elif self.auth_config.authentication_method == AuthenticationMethod.TOKEN:
                # Use AzureKeyCredential for consistent API key handling
                assert self.auth_config.api_key is not None
                self._credential = AzureKeyCredential(self.auth_config.api_key)

            else:
                raise ValueError(
                    f"Unsupported authentication method: {self.auth_config.authentication_method}"
                )

        return self._credential

    @property
    def api_key(self) -> str:
        """
        Get the raw API key string for special cases (like connection strings).

        Returns:
            str: Raw API key/token value

        Raises:
            ValueError: If authentication method is not TOKEN or api_key is missing
        """
        if self.auth_config.authentication_method != AuthenticationMethod.TOKEN:
            raise ValueError(
                f"API key requires TOKEN authentication method, "
                f"got {self.auth_config.authentication_method}"
            )

        # Use the centralized validation to ensure consistency
        self.auth_config.validate_for_method()

        # Type assertion is safe here because validation ensures api_key is not None
        assert self.auth_config.api_key is not None
        return self.auth_config.api_key

    @property
    def key_credential(self) -> AzureKeyCredential:
        """
        Get AzureKeyCredential specifically for services that only accept AzureKeyCredential.
        """
        if self.auth_config.authentication_method != AuthenticationMethod.TOKEN:
            raise ValueError(
                "key_credential property is only valid for TOKEN authentication method"
            )

        # Use the centralized validation to ensure consistency
        self.auth_config.validate_for_method()

        # Type assertion is safe here because validation ensures api_key is not None
        assert self.auth_config.api_key is not None
        return AzureKeyCredential(self.auth_config.api_key)

    @property
    def token_credential(self) -> TokenCredential:
        """
        Get TokenCredential specifically for services that only accept TokenCredential.

        Returns:
            TokenCredential: Credential object for Azure AD authentication

        Raises:
            ValueError: If authentication method is TOKEN (use api_key property instead)
        """
        if self.auth_config.authentication_method == AuthenticationMethod.TOKEN:
            raise ValueError(
                "TOKEN authentication method should use api_key property "
                "for services that need raw API key strings"
            )

        # For non-TOKEN methods, credential will always be TokenCredential
        cred = self.credential
        if not isinstance(cred, TokenCredential):
            raise ValueError(f"Expected TokenCredential but got {type(cred)}")

        return cred

    @abstractmethod
    def build(self) -> Any:
        """Build and return the Azure client."""
        pass
