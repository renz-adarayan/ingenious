"""
Azure Client Factory for building various Azure service clients.

This module provides a centralized factory for creating Azure service clients
with appropriate authentication methods based on configuration.
"""

from typing import TYPE_CHECKING, Any, Optional, Union

import pyodbc
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.storage.blob import BlobClient, BlobServiceClient
from openai import AzureOpenAI

# Optional imports with fallbacks
try:
    from azure.cosmos import CosmosClient

    HAS_COSMOS = True
except ImportError:
    HAS_COSMOS = False

try:
    from azure.search.documents import SearchClient

    HAS_SEARCH = True
except ImportError:
    HAS_SEARCH = False

# Type imports for type checking
if TYPE_CHECKING:
    if HAS_COSMOS:
        from azure.cosmos import CosmosClient
    else:
        CosmosClient = Any

    if HAS_SEARCH:
        from azure.search.documents import SearchClient
    else:
        SearchClient = Any

# Support both old and new configuration systems
from ingenious.common.enums import AuthenticationMethod
from ingenious.config.models import (
    AzureSearchSettings,
    AzureSqlSettings,
    CosmosSettings,
    FileStorageContainerSettings,
    ModelSettings,
)
from ingenious.models.config import (
    AzureSearchConfig,
    AzureSqlConfig,
    CosmosConfig,
    FileStorageContainer,
    ModelConfig,
)

from .builder.blob_client import BlobClientBuilder, BlobServiceClientBuilder
from .builder.openai_chat_completions_client import (
    AzureOpenAIChatCompletionClientBuilder,
)

# Import all the builders
from .builder.openai_client import AzureOpenAIClientBuilder
from .builder.sql_client import AzureSqlClientBuilder, AzureSqlClientBuilderWithAuth

# Optional builder imports
try:
    from .builder.search_client import AzureSearchClientBuilder
except ImportError:
    AzureSearchClientBuilder = None

try:
    from .builder.cosmos_client import CosmosClientBuilder
except ImportError:
    CosmosClientBuilder = None


class AzureClientFactory:
    """Factory class for creating Azure service clients with proper authentication."""

    @staticmethod
    def create_openai_client(
        model_config: Union[ModelConfig, ModelSettings],
    ) -> AzureOpenAI:
        """
        Create an Azure OpenAI client from model configuration.

        Args:
            model_config: Model configuration containing authentication details

        Returns:
            AzureOpenAI: Configured Azure OpenAI client
        """
        builder = AzureOpenAIClientBuilder(model_config)
        return builder.build()

    @staticmethod
    def create_openai_client_from_params(
        model: str,
        base_url: str,
        api_version: str,
        deployment: Optional[str] = None,
        api_key: Optional[str] = None,
        authentication_method: AuthenticationMethod = AuthenticationMethod.DEFAULT_CREDENTIAL,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> AzureOpenAI:
        """
        Create an Azure OpenAI client with direct parameters.

        This method is useful when you don't have a ModelConfig object but want to
        create a client with specific parameters.

        Args:
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
            base_url: Azure OpenAI endpoint URL
            api_version: Azure OpenAI API version
            deployment: Azure deployment name. If None, uses model name
            api_key: API key for authentication. Required if not using default credential
            authentication_method: Authentication method
            client_id: Client ID for MSI or CLIENT_ID_AND_SECRET authentication
            client_secret: Client secret for CLIENT_ID_AND_SECRET authentication
            tenant_id: Tenant ID for CLIENT_ID_AND_SECRET authentication

        Returns:
            AzureOpenAI: Configured Azure OpenAI client

        Raises:
            ValueError: If required parameters are missing
        """
        model_settings = ModelSettings(
            model=model,
            api_type="rest",
            base_url=base_url,
            api_version=api_version,
            deployment=deployment or model,
            api_key=api_key or "",
            authentication_method=authentication_method,
            client_id=client_id or "",
            client_secret=client_secret or "",
            tenant_id=tenant_id or "",
        )
        builder = AzureOpenAIClientBuilder(model_settings)
        return builder.build()

    @staticmethod
    def create_openai_chat_completion_client(
        model_config: Union[ModelConfig, ModelSettings],
    ) -> AzureOpenAIChatCompletionClient:
        """
        Create an Azure OpenAI Chat Completion client from model configuration.

        Args:
            model_config: Model configuration containing authentication details

        Returns:
            AzureOpenAIChatCompletionClient: Configured Azure OpenAI Chat Completion client
        """
        builder = AzureOpenAIChatCompletionClientBuilder(model_config)
        return builder.build()

    @staticmethod
    def create_openai_chat_completion_client_from_params(
        model: str,
        base_url: str,
        api_version: str,
        deployment: Optional[str] = None,
        api_key: Optional[str] = None,
        authentication_method: AuthenticationMethod = AuthenticationMethod.DEFAULT_CREDENTIAL,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> AzureOpenAIChatCompletionClient:
        """
        Create an Azure OpenAI Chat Completion client with direct parameters.

        This method is useful when you don't have a ModelConfig object but want to
        create a client with specific parameters.

        Args:
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
            base_url: Azure OpenAI endpoint URL
            api_version: Azure OpenAI API version
            deployment: Azure deployment name. If None, uses model name
            api_key: API key for authentication. Required if not using default credential
            authentication_method: Authentication method
            client_id: Client ID for MSI or CLIENT_ID_AND_SECRET authentication
            client_secret: Client secret for CLIENT_ID_AND_SECRET authentication
            tenant_id: Tenant ID for CLIENT_ID_AND_SECRET authentication

        Returns:
            AzureOpenAIChatCompletionClient: Configured Azure OpenAI Chat Completion client

        Raises:
            ValueError: If required parameters are missing
        """
        model_settings = ModelSettings(
            model=model,
            api_type="rest",
            base_url=base_url,
            api_version=api_version,
            deployment=deployment or model,
            api_key=api_key or "",
            authentication_method=authentication_method,
            client_id=client_id or "",
            client_secret=client_secret or "",
            tenant_id=tenant_id or "",
        )
        builder = AzureOpenAIChatCompletionClientBuilder(model_settings)
        return builder.build()

    @staticmethod
    def create_blob_service_client(
        file_storage_config: Union[FileStorageContainer, FileStorageContainerSettings],
    ) -> BlobServiceClient:
        """
        Create an Azure Blob Service client from file storage configuration.

        Args:
            file_storage_config: File storage configuration containing authentication details

        Returns:
            BlobServiceClient: Configured Azure Blob Service client
        """
        builder = BlobServiceClientBuilder(file_storage_config)
        return builder.build()

    @staticmethod
    def create_blob_service_client_from_params(
        account_url: str,
        authentication_method: AuthenticationMethod = AuthenticationMethod.DEFAULT_CREDENTIAL,
        token: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> BlobServiceClient:
        """
        Create an Azure Blob Service client with direct parameters.

        Args:
            account_url: Storage account URL
            authentication_method: Authentication method
            token: API key/SAS token/connection string for TOKEN authentication
            client_id: Client ID for MSI authentication

        Returns:
            BlobServiceClient: Configured Azure Blob Service client
        """
        file_storage_settings = FileStorageContainerSettings(
            enable=True,
            storage_type="azure",
            container_name="",  # Not required for service client
            path="./",
            add_sub_folders=True,
            url=account_url,
            client_id=client_id or "",
            token=token or "",
            authentication_method=authentication_method,
        )
        builder = BlobServiceClientBuilder(file_storage_settings)
        return builder.build()

    @staticmethod
    def create_blob_client(
        file_storage_config: Union[FileStorageContainer, FileStorageContainerSettings],
        blob_name: str,
        container_name: Optional[str] = None,
    ) -> BlobClient:
        """
        Create an Azure Blob client from file storage configuration.

        Args:
            file_storage_config: File storage configuration containing authentication details
            blob_name: Name of the blob
            container_name: Container name (optional, will use config if not provided)

        Returns:
            BlobClient: Configured Azure Blob client
        """
        builder = BlobClientBuilder(file_storage_config, container_name, blob_name)
        return builder.build()

    @staticmethod
    def create_blob_client_from_params(
        account_url: str,
        blob_name: str,
        container_name: str,
        authentication_method: AuthenticationMethod = AuthenticationMethod.DEFAULT_CREDENTIAL,
        token: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> BlobClient:
        """
        Create an Azure Blob client with direct parameters.

        Args:
            account_url: Storage account URL
            blob_name: Name of the blob
            container_name: Name of the container
            authentication_method: Authentication method
            token: API key/SAS token/connection string for TOKEN authentication
            client_id: Client ID for MSI authentication

        Returns:
            BlobClient: Configured Azure Blob client
        """
        file_storage_settings = FileStorageContainerSettings(
            enable=True,
            storage_type="azure",
            container_name=container_name,
            path="./",
            add_sub_folders=True,
            url=account_url,
            client_id=client_id or "",
            token=token or "",
            authentication_method=authentication_method,
        )
        builder = BlobClientBuilder(file_storage_settings, container_name, blob_name)
        return builder.build()

    @staticmethod
    def create_cosmos_client(
        cosmos_config: Union[CosmosConfig, CosmosSettings],
    ) -> Any:
        """
        Create an Azure Cosmos DB client.

        Args:
            endpoint: Cosmos DB endpoint URL
            authentication_method: Authentication method to use
            api_key: API key for TOKEN authentication
            client_id: Client ID for MSI or CLIENT_ID_AND_SECRET authentication
            client_secret: Client secret for CLIENT_ID_AND_SECRET authentication
            tenant_id: Tenant ID for CLIENT_ID_AND_SECRET authentication

        Returns:
            CosmosClient: Configured Azure Cosmos DB client
        """
        if not HAS_COSMOS:
            raise ImportError(
                "azure-cosmos is required for Cosmos DB functionality. "
                "Install with: pip install azure-cosmos"
            )

        if CosmosClientBuilder is None:
            raise ImportError(
                "CosmosClientBuilder is not available. "
                "azure-cosmos package is required."
            )

        builder = CosmosClientBuilder(cosmos_config)
        return builder.build()

    @staticmethod
    def create_search_client(
        search_config: Union[AzureSearchConfig, AzureSearchSettings], index_name: str
    ) -> Any:
        """
        Create an Azure Search client from search configuration.

        Args:
            search_config: Search configuration containing authentication details
            index_name: Name of the search index

        Returns:
            SearchClient: Configured Azure Search client
        """
        if not HAS_SEARCH:
            raise ImportError(
                "azure-search-documents is required for Azure Search functionality. "
                "Install with: pip install azure-search-documents"
            )

        if AzureSearchClientBuilder is None:
            raise ImportError(
                "AzureSearchClientBuilder is not available. "
                "azure-search-documents package is required."
            )

        builder = AzureSearchClientBuilder(search_config, index_name)
        return builder.build()

    @staticmethod
    def create_search_client_from_params(
        endpoint: str,
        index_name: str,
        api_key: str,
        service: Optional[str] = None,
        authentication_method: AuthenticationMethod = AuthenticationMethod.DEFAULT_CREDENTIAL,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Any:
        """
        Create an Azure Search client with direct parameters.

        Args:
            endpoint: Azure Search service endpoint URL
            index_name: Name of the search index
            api_key: Azure Search service API key
            service: Azure Search service name (optional)

        Returns:
            SearchClient: Configured Azure Search client
        """
        if not HAS_SEARCH:
            raise ImportError(
                "azure-search-documents is required for Azure Search functionality. "
                "Install with: pip install azure-search-documents"
            )

        if AzureSearchClientBuilder is None:
            raise ImportError(
                "AzureSearchClientBuilder is not available. "
                "azure-search-documents package is required."
            )

        search_settings = AzureSearchSettings(
            service=service or "",
            endpoint=endpoint,
            key=api_key,
            client_id=client_id or "",
            client_secret=client_secret or "",
            tenant_id=tenant_id or "",
            authentication_method=authentication_method,
        )
        builder = AzureSearchClientBuilder(search_settings, index_name)
        return builder.build()

    @staticmethod
    def create_sql_client(
        sql_config: Union[AzureSqlConfig, AzureSqlSettings],
    ) -> pyodbc.Connection:
        """
        Create an Azure SQL client from SQL configuration.

        Args:
            sql_config: SQL configuration containing connection details

        Returns:
            pyodbc.Connection: Configured Azure SQL connection
        """
        builder = AzureSqlClientBuilder(sql_config)
        return builder.build()

    @staticmethod
    def create_sql_client_from_params(
        database_name: str,
        connection_string: str,
        table_name: Optional[str] = None,
    ) -> pyodbc.Connection:
        """
        Create an Azure SQL client with direct parameters.

        Args:
            database_name: Azure SQL database name
            connection_string: Azure SQL connection string
            table_name: Default table name for operations (optional)

        Returns:
            pyodbc.Connection: Configured Azure SQL connection
        """
        sql_settings = AzureSqlSettings(
            database_name=database_name,
            table_name=table_name or "",
            database_connection_string=connection_string,
        )
        builder = AzureSqlClientBuilder(sql_settings)
        return builder.build()

    @staticmethod
    def create_sql_client_with_auth(
        server: str,
        database: str,
        authentication_method: AuthenticationMethod = AuthenticationMethod.DEFAULT_CREDENTIAL,
        username: Optional[str] = None,
        password: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> pyodbc.Connection:
        """
        Create an Azure SQL client with explicit authentication configuration.

        Args:
            server: SQL Server name
            database: Database name
            authentication_method: Authentication method to use
            username: Username for SQL authentication
            password: Password for SQL authentication
            client_id: Client ID for MSI or CLIENT_ID_AND_SECRET authentication
            client_secret: Client secret for CLIENT_ID_AND_SECRET authentication
            tenant_id: Tenant ID for CLIENT_ID_AND_SECRET authentication

        Returns:
            pyodbc.Connection: Configured Azure SQL connection
        """
        builder = AzureSqlClientBuilderWithAuth(
            server=server,
            database=database,
            authentication_method=authentication_method,
            username=username,
            password=password,
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
        )
        return builder.build()

    @staticmethod
    def create_sql_client_with_auth_from_params(
        server: str,
        database: str,
        authentication_method: AuthenticationMethod = AuthenticationMethod.DEFAULT_CREDENTIAL,
        username: Optional[str] = None,
        password: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> pyodbc.Connection:
        """
        Create an Azure SQL client with explicit authentication configuration from direct parameters.

        This is an alias for create_sql_client_with_auth since it already accepts direct parameters.

        Args:
            server: SQL Server name
            database: Database name
            authentication_method: Authentication method to use
            username: Username for SQL authentication
            password: Password for SQL authentication
            client_id: Client ID for MSI or CLIENT_ID_AND_SECRET authentication
            client_secret: Client secret for CLIENT_ID_AND_SECRET authentication
            tenant_id: Tenant ID for CLIENT_ID_AND_SECRET authentication

        Returns:
            pyodbc.Connection: Configured Azure SQL connection
        """
        return AzureClientFactory.create_sql_client_with_auth(
            server=server,
            database=database,
            authentication_method=authentication_method,
            username=username,
            password=password,
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
        )
