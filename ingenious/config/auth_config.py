from typing import Any, Optional

from pydantic import BaseModel, Field

from ingenious.common.enums import AuthenticationMethod


class AzureAuthConfig(BaseModel):
    """Configuration class for Azure authentication parameters."""

    authentication_method: AuthenticationMethod = Field(
        default=AuthenticationMethod.DEFAULT_CREDENTIAL,
        description="Azure authentication method to use",
    )
    client_id: Optional[str] = Field(
        default=None,
        description="Azure client ID for MSI or service principal authentication",
    )
    client_secret: Optional[str] = Field(
        default=None,
        description="Azure client secret for service principal authentication",
    )
    tenant_id: Optional[str] = Field(
        default=None, description="Azure tenant ID for service principal authentication"
    )
    api_key: Optional[str] = Field(
        default=None, description="API key for token-based authentication"
    )

    @classmethod
    def from_config(cls, config: Any) -> "AzureAuthConfig":
        """
        Create AzureAuthConfig from any configuration object.

        Args:
            config: Configuration object (either legacy or new format)

        Returns:
            AzureAuthConfig: Extracted authentication configuration
        """
        default_auth_method = AuthenticationMethod.DEFAULT_CREDENTIAL
        if hasattr(config, "authentication_method"):
            default_auth_method = config.authentication_method

        return cls(
            authentication_method=getattr(
                config, "authentication_method", default_auth_method
            ),
            client_id=getattr(config, "client_id", None),
            client_secret=getattr(config, "client_secret", None),
            tenant_id=getattr(config, "tenant_id", None),
            api_key=getattr(config, "api_key", None)
            or getattr(config, "token", None)
            or getattr(config, "key", None),
        )

    @classmethod
    def default_credential(cls) -> "AzureAuthConfig":
        """Create AzureAuthConfig for DefaultAzureCredential authentication."""
        return cls(authentication_method=AuthenticationMethod.DEFAULT_CREDENTIAL)

    @classmethod
    def managed_identity(cls, client_id: Optional[str] = None) -> "AzureAuthConfig":
        """Create AzureAuthConfig for Managed Identity authentication."""
        return cls(authentication_method=AuthenticationMethod.MSI, client_id=client_id)

    @classmethod
    def service_principal(
        cls, client_id: str, client_secret: str, tenant_id: str
    ) -> "AzureAuthConfig":
        """Create AzureAuthConfig for Service Principal authentication."""
        return cls(
            authentication_method=AuthenticationMethod.CLIENT_ID_AND_SECRET,
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
        )

    @classmethod
    def token_auth(cls, api_key: str) -> "AzureAuthConfig":
        """Create AzureAuthConfig for token/API key authentication."""
        return cls(authentication_method=AuthenticationMethod.TOKEN, api_key=api_key)

    def validate_for_method(self) -> None:
        """Validate that required parameters are present for the authentication method."""
        if self.authentication_method == AuthenticationMethod.CLIENT_ID_AND_SECRET:
            if not all([self.client_id, self.client_secret, self.tenant_id]):
                raise ValueError(
                    "client_id, client_secret, and tenant_id are required "
                    "for CLIENT_ID_AND_SECRET authentication method"
                )
        elif self.authentication_method == AuthenticationMethod.TOKEN:
            if not self.api_key:
                raise ValueError("api_key is required for TOKEN authentication method")

    def get_azure_auth_config(self) -> "AzureAuthConfig":
        """Get AzureAuthConfig for this configuration."""
        return self
