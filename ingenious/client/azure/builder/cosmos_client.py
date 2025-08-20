from typing import Union

from azure.cosmos import CosmosClient

from ingenious.client.azure.builder.base import AzureClientBuilder
from ingenious.common.enums import AuthenticationMethod
from ingenious.config.auth_config import AzureAuthConfig
from ingenious.config.models import CosmosSettings
from ingenious.models.config import CosmosConfig


class CosmosClientBuilder(AzureClientBuilder):
    """Builder for Azure Cosmos DB clients with multiple authentication methods."""

    def __init__(self, cosmos_config: Union[CosmosConfig, CosmosSettings]):
        auth_config = self._create_auth_config_from_chat_history_config(cosmos_config)
        super().__init__(auth_config=auth_config)
        self.uri = cosmos_config.uri

    def _create_auth_config_from_chat_history_config(self, cosmos_config):
        """Create AzureAuthConfig from chat history configuration."""
        return AzureAuthConfig.from_config(cosmos_config)

    def build(self) -> CosmosClient:
        """
        Build Azure Cosmos DB client based on configuration.

        Returns:
            CosmosClient: Configured Azure Cosmos DB client
        """
        # Configure client based on credential type
        if self.auth_config.authentication_method == AuthenticationMethod.TOKEN:
            # Cosmos DB expects raw string for API key, not AzureKeyCredential
            return CosmosClient(
                url=self.uri,
                credential=self.api_key,  # Use raw string property
            )
        else:
            # Use Azure AD authentication - credential will be TokenCredential
            from azure.core.credentials import TokenCredential

            if not isinstance(self.credential, TokenCredential):
                raise ValueError(
                    f"Expected TokenCredential for Azure AD auth, got {type(self.credential)}"
                )

            return CosmosClient(url=self.uri, credential=self.credential)
