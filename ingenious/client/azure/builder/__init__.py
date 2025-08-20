"""
Azure Client Builders Package

This package provides builders for creating Azure service clients with
different authentication methods based on configuration.

Available Builders:
- AzureOpenAIClientBuilder: For Azure OpenAI clients
- AzureOpenAIChatCompletionClientBuilder: For AutoGen Azure OpenAI chat completion clients
- BlobServiceClientBuilder: For Azure Blob Storage service clients
- BlobContainerClientBuilder: For Azure Blob Storage container clients
- BlobClientBuilder: For Azure Blob Storage blob clients
- CosmosClientBuilder: For Azure Cosmos DB clients
- CosmosDatabaseClientBuilder: For Azure Cosmos DB database clients
- CosmosContainerClientBuilder: For Azure Cosmos DB container clients
- AzureSearchClientBuilder: For Azure Search clients
- AzureSearchIndexClientBuilder: For Azure Search index clients
- KeyVaultSecretClientBuilder: For Azure Key Vault secret clients
- AzureSqlClientBuilder: For Azure SQL clients

Factory Class:
- AzureClientFactory: Centralized factory for creating all Azure clients

Authentication Methods Supported:
- DEFAULT_CREDENTIAL: Uses Azure Default Credential chain
- MSI: Uses Managed Identity (system or user-assigned)
- CLIENT_ID_AND_SECRET: Uses Service Principal authentication
- TOKEN: Uses API key or connection string authentication
"""

from .blob_client import BlobClientBuilder, BlobServiceClientBuilder
from .cosmos_client import CosmosClientBuilder
from .openai_chat_completions_client import AzureOpenAIChatCompletionClientBuilder
from .openai_client import AzureOpenAIClientBuilder
from .search_client import AzureSearchClientBuilder
from .sql_client import AzureSqlClientBuilder, AzureSqlClientBuilderWithAuth

__all__ = [
    # OpenAI builders
    "AzureOpenAIClientBuilder",
    "AzureOpenAIChatCompletionClientBuilder",
    # Blob Storage builders
    "BlobServiceClientBuilder",
    "BlobClientBuilder",
    # Cosmos DB builders
    "CosmosClientBuilder",
    # Search builders
    "AzureSearchClientBuilder",
    # SQL builders
    "AzureSqlClientBuilder",
    "AzureSqlClientBuilderWithAuth",
]
