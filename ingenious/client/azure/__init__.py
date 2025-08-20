"""
Azure Client Module

This module provides a centralized factory for creating Azure service clients
with appropriate authentication methods based on configuration.

The main entry point is AzureClientFactory, which provides convenient methods
for creating common Azure clients. For advanced use cases or custom builders,
import from the builder submodule.

Usage:
    from ingenious.client.azure import AzureClientFactory

    # Create an Azure OpenAI client
    openai_client = AzureClientFactory.create_openai_client(model_config)

    # Create an Azure Blob Storage client
    blob_client = AzureClientFactory.create_blob_service_client(storage_config)

    # For advanced builder usage:
    from ingenious.client.azure.builder import AzureOpenAIClientBuilder
"""

from .azure_client_builder_factory import AzureClientFactory

# Export only the factory for most users
__all__ = [
    "AzureClientFactory",
]
