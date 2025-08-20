# ingenious/client/azure/azure_client_builder_factory.py
"""
Azure Client Factory for building various Azure service clients.

This module provides a centralized factory for creating Azure service clients
with appropriate authentication methods based on configuration. All optional
Azure SDK imports are lazy (inside methods) to keep import-time dependencies minimal.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

# Type-only imports (no runtime dependency)
if TYPE_CHECKING:
    # Azure OpenAI (sync/async)
    from openai import AzureOpenAI, AsyncAzureOpenAI  # type: ignore[missing-import]
    from autogen_ext.models.openai import AzureOpenAIChatCompletionClient  # type: ignore[missing-import]
    # Azure Search (sync/async)
    from azure.search.documents import SearchClient as SyncSearchClient  # type: ignore[missing-import]
    from azure.search.documents.aio import SearchClient as AsyncSearchClient  # type: ignore[missing-import]
    # Azure Blob
    from azure.storage.blob import BlobServiceClient, BlobClient  # type: ignore[missing-import]
    # Cosmos
    from azure.cosmos import CosmosClient  # type: ignore[missing-import]
    # SQL
    import pyodbc  # type: ignore[missing-import]

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


class AzureClientFactory:
    """Factory class for creating Azure service clients with proper authentication."""

    # --------------------------- OpenAI (sync) ---------------------------

    @staticmethod
    def create_openai_client(
        model_config: Union[ModelConfig, ModelSettings],
    ) -> "AzureOpenAI":
        """Create an Azure OpenAI client from model configuration."""
        try:
            # Lazy import: builder and SDK only when needed
            from .builder.openai_client import AzureOpenAIClientBuilder  # type: ignore
        except Exception as e:
            raise ImportError(
                "Azure OpenAI (sync) requires the 'openai' package. "
                "Install with: pip install openai"
            ) from e

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
    ) -> "AzureOpenAI":
        """Create an Azure OpenAI client with direct parameters."""
        try:
            from .builder.openai_client import AzureOpenAIClientBuilder  # type: ignore
        except Exception as e:
            raise ImportError(
                "Azure OpenAI (sync) requires the 'openai' package. "
                "Install with: pip install openai"
            ) from e

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

    # ---------------- OpenAI Chat Completions client (sync, optional) ----------------

    @staticmethod
    def create_openai_chat_completion_client(
        model_config: Union[ModelConfig, ModelSettings],
    ) -> "AzureOpenAIChatCompletionClient":
        """Create an Azure OpenAI Chat Completion client from model configuration."""
        try:
            # Builder depends on autogen_ext; import lazily
            from .builder.openai_chat_completions_client import (  # type: ignore
                AzureOpenAIChatCompletionClientBuilder,
            )
        except Exception as e:
            raise ImportError(
                "Azure OpenAI Chat Completions client requires 'autogen-ext'. "
                "Install with: pip install autogen-ext"
            ) from e

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
    ) -> "AzureOpenAIChatCompletionClient":
        """Create an Azure OpenAI Chat Completion client with direct parameters."""
        try:
            from .builder.openai_chat_completions_client import (  # type: ignore
                AzureOpenAIChatCompletionClientBuilder,
            )
        except Exception as e:
            raise ImportError(
                "Azure OpenAI Chat Completions client requires 'autogen-ext'. "
                "Install with: pip install autogen-ext"
            ) from e

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

    # --------------------------- Blob (sync) ---------------------------

    @staticmethod
    def create_blob_service_client(
        file_storage_config: Union[FileStorageContainer, FileStorageContainerSettings],
    ) -> "BlobServiceClient":
        """Create an Azure Blob Service client from file storage configuration."""
        try:
            from .builder.blob_client import BlobServiceClientBuilder  # type: ignore
        except Exception as e:
            raise ImportError(
                "Azure Blob client requires 'azure-storage-blob'. "
                "Install with: pip install azure-storage-blob"
            ) from e

        builder = BlobServiceClientBuilder(file_storage_config)
        return builder.build()

    @staticmethod
    def create_blob_service_client_from_params(
        account_url: str,
        authentication_method: AuthenticationMethod = AuthenticationMethod.DEFAULT_CREDENTIAL,
        token: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> "BlobServiceClient":
        """Create an Azure Blob Service client with direct parameters."""
        try:
            from .builder.blob_client import BlobServiceClientBuilder  # type: ignore
        except Exception as e:
            raise ImportError(
                "Azure Blob client requires 'azure-storage-blob'. "
                "Install with: pip install azure-storage-blob"
            ) from e

        file_storage_settings = FileStorageContainerSettings(
            enable=True,
            storage_type="azure",
            container_name="",
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
    ) -> "BlobClient":
        """Create an Azure Blob client from file storage configuration."""
        try:
            from .builder.blob_client import BlobClientBuilder  # type: ignore
        except Exception as e:
            raise ImportError(
                "Azure Blob client requires 'azure-storage-blob'. "
                "Install with: pip install azure-storage-blob"
            ) from e

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
    ) -> "BlobClient":
        """Create an Azure Blob client with direct parameters."""
        try:
            from .builder.blob_client import BlobClientBuilder  # type: ignore
        except Exception as e:
            raise ImportError(
                "Azure Blob client requires 'azure-storage-blob'. "
                "Install with: pip install azure-storage-blob"
            ) from e

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

    # --------------------------- Cosmos (sync, optional) ---------------------------

    @staticmethod
    def create_cosmos_client(
        cosmos_config: Union[CosmosConfig, CosmosSettings],
    ) -> "CosmosClient":
        """Create an Azure Cosmos DB client."""
        try:
            from .builder.cosmos_client import CosmosClientBuilder  # type: ignore
        except Exception as e:
            raise ImportError(
                "Cosmos DB requires 'azure-cosmos'. Install with: pip install azure-cosmos"
            ) from e

        builder = CosmosClientBuilder(cosmos_config)
        return builder.build()

    # --------------------------- Search (sync) ---------------------------

    @staticmethod
    def create_search_client(
        search_config: Union[AzureSearchConfig, AzureSearchSettings], index_name: str
    ) -> "SyncSearchClient":
        """Create an Azure Search client from search configuration."""
        try:
            from .builder.search_client import AzureSearchClientBuilder  # type: ignore
        except Exception as e:
            raise ImportError(
                "Azure Search (sync) requires 'azure-search-documents'. "
                "Install with: pip install azure-search-documents"
            ) from e

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
    ) -> "SyncSearchClient":
        """Create an Azure Search client with direct parameters."""
        try:
            from .builder.search_client import AzureSearchClientBuilder  # type: ignore
        except Exception as e:
            raise ImportError(
                "Azure Search (sync) requires 'azure-search-documents'. "
                "Install with: pip install azure-search-documents"
            ) from e

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

    # --------------------------- SQL (sync) ---------------------------

    @staticmethod
    def create_sql_client(
        sql_config: Union[AzureSqlConfig, AzureSqlSettings],
    ) -> "pyodbc.Connection":
        """Create an Azure SQL client from SQL configuration."""
        try:
            from .builder.sql_client import AzureSqlClientBuilder  # type: ignore
        except Exception as e:
            raise ImportError(
                "Azure SQL requires 'pyodbc' and appropriate drivers."
            ) from e

        builder = AzureSqlClientBuilder(sql_config)
        return builder.build()

    @staticmethod
    def create_sql_client_from_params(
        database_name: str,
        connection_string: str,
        table_name: Optional[str] = None,
    ) -> "pyodbc.Connection":
        """Create an Azure SQL client with direct parameters."""
        try:
            from .builder.sql_client import AzureSqlClientBuilder  # type: ignore
        except Exception as e:
            raise ImportError(
                "Azure SQL requires 'pyodbc' and appropriate drivers."
            ) from e

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
    ) -> "pyodbc.Connection":
        """Create an Azure SQL client with explicit authentication configuration."""
        try:
            from .builder.sql_client import AzureSqlClientBuilderWithAuth  # type: ignore
        except Exception as e:
            raise ImportError(
                "Azure SQL requires 'pyodbc' and appropriate drivers."
            ) from e

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
    ) -> "pyodbc.Connection":
        """Alias for create_sql_client_with_auth using direct parameters."""
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

    # --------------------------- Async builders ---------------------------

    @staticmethod
    def create_async_search_client(
        index_name: str,
        config: Optional[Mapping[str, Any] | AzureSearchConfig | AzureSearchSettings] = None,
        **client_options: Any,
    ) -> "AsyncSearchClient":
        """Create an async Azure Search client (azure.search.documents.aio.SearchClient)."""
        try:
            from .builder.search_client_async import AzureSearchAsyncClientBuilder  # type: ignore
        except Exception as e:
            raise ImportError(
                "Async Azure Search requires: pip install azure-search-documents azure-identity"
            ) from e

        builder = AzureSearchAsyncClientBuilder.from_config(
            config, index_name=index_name, client_options=client_options
        )
        return builder.build()

    @staticmethod
    def create_async_openai_client(
        config: Optional[Mapping[str, Any] | ModelConfig | ModelSettings] = None,
        api_version: Optional[str] = None,
        **client_options: Any,
    ) -> "AsyncAzureOpenAI":
        """Create an async Azure OpenAI client (openai.AsyncAzureOpenAI)."""
        try:
            from .builder.openai_client_async import AsyncAzureOpenAIClientBuilder  # type: ignore
        except Exception as e:
            raise ImportError(
                "Async Azure OpenAI requires: pip install openai azure-identity"
            ) from e

        builder = AsyncAzureOpenAIClientBuilder.from_config(
            config, api_version=api_version, client_options=client_options
        )
        return builder.build()
