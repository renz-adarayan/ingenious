from typing import Optional, Union

from azure.search.documents import SearchClient

from ingenious.client.azure.builder.base import AzureClientBuilder
from ingenious.config.auth_config import AzureAuthConfig
from ingenious.config.models import AzureSearchSettings
from ingenious.models.config import AzureSearchConfig


class AzureSearchClientBuilder(AzureClientBuilder):
    """Builder for Azure Search clients with multiple authentication methods."""

    def __init__(
        self,
        search_config: Union[AzureSearchConfig, AzureSearchSettings],
        index_name: Optional[str] = None,
    ):
        # Extract authentication parameters from config
        auth_config = self._create_auth_config_from_search_config(search_config)
        super().__init__(auth_config=auth_config)
        self.search_config = search_config
        self.index_name = index_name

    def _create_auth_config_from_search_config(self, search_config):
        """Create AzureAuthConfig from search configuration."""
        return AzureAuthConfig.from_config(search_config)

    def build(self) -> SearchClient:
        """
        Build Azure Search client based on search configuration.

        Returns:
            SearchClient: Configured Azure Search client
        """
        if not self.index_name:
            raise ValueError("Index name is required for SearchClient")

        # Get endpoint from search config
        endpoint = getattr(self.search_config, "endpoint", None)

        if not endpoint:
            service = getattr(self.search_config, "service", None)
            endpoint = f"https://{service}.search.windows.net" if service else None

        if not endpoint:
            raise ValueError("Endpoint is required for search client")

        return SearchClient(
            endpoint=endpoint, index_name=self.index_name, credential=self.credential
        )
