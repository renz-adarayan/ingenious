"""
Tests for AzureClientFactory.

This module tests the Azure client factory functionality, including:
- OpenAI client creation with various authentication methods
- Blob storage client creation
- Search client creation (when available)
- Cosmos client creation (when available)
- SQL client creation
- Proper error handling for optional dependencies
"""

from unittest.mock import Mock, patch

import pytest

from ingenious.client.azure.azure_client_builder_factory import (
    HAS_COSMOS,
    HAS_SEARCH,
    AzureClientFactory,
)
from ingenious.common.enums import AuthenticationMethod
from ingenious.config.models import (
    AzureSearchSettings,
    AzureSqlSettings,
    FileStorageContainerSettings,
    ModelSettings,
)
from ingenious.models.config import (
    ModelConfig,
)


class TestAzureClientFactory:
    """Test cases for Azure Client Factory."""

    # OpenAI Client Tests
    @patch(
        "ingenious.client.azure.azure_client_builder_factory.AzureOpenAIClientBuilder"
    )
    def test_create_openai_client_with_model_config(self, mock_builder_class):
        """Test creating OpenAI client with ModelConfig."""
        # Arrange
        mock_builder = Mock()
        mock_client = Mock()
        mock_builder.build.return_value = mock_client
        mock_builder_class.return_value = mock_builder

        # Build a proper ModelConfig with mocked config and profile
        mock_config_ns = Mock()
        mock_config_ns.model = "gpt-4"
        mock_config_ns.api_type = "rest"
        mock_config_ns.api_version = "2023-03-15-preview"
        mock_config_ns.deployment = "test-deployment"

        mock_profile = Mock()
        mock_profile.deployment = "test-deployment"
        mock_profile.api_version = "2023-03-15-preview"
        mock_profile.authentication_method = AuthenticationMethod.DEFAULT_CREDENTIAL
        mock_profile.client_id = None
        mock_profile.client_secret = None
        mock_profile.tenant_id = None
        mock_profile.api_key = None
        mock_profile.base_url = "https://test.openai.azure.com/"

        model_config = ModelConfig(mock_config_ns, mock_profile)

        # Act
        result = AzureClientFactory.create_openai_client(model_config)

        # Assert
        mock_builder_class.assert_called_once_with(model_config)
        mock_builder.build.assert_called_once()
        assert result == mock_client

    @patch(
        "ingenious.client.azure.azure_client_builder_factory.AzureOpenAIClientBuilder"
    )
    def test_create_openai_client_with_model_settings(self, mock_builder_class):
        """Test creating OpenAI client with ModelSettings."""
        # Arrange
        mock_builder = Mock()
        mock_client = Mock()
        mock_builder.build.return_value = mock_client
        mock_builder_class.return_value = mock_builder

        model_settings = ModelSettings(
            model="gpt-4",
            base_url="https://test.openai.azure.com/",
            api_version="2023-05-15",
            deployment="gpt-4",
            api_key="test-key",
            authentication_method=AuthenticationMethod.TOKEN,
            client_id="",
            client_secret="",
            tenant_id="",
            api_type="rest",
        )

        # Act
        result = AzureClientFactory.create_openai_client(model_settings)

        # Assert
        mock_builder_class.assert_called_once_with(model_settings)
        mock_builder.build.assert_called_once()
        assert result == mock_client

    @patch(
        "ingenious.client.azure.azure_client_builder_factory.AzureOpenAIClientBuilder"
    )
    def test_create_openai_client_from_params_default_credential(
        self, mock_builder_class
    ):
        """Test creating OpenAI client from parameters with default credential."""
        # Arrange
        mock_builder = Mock()
        mock_client = Mock()
        mock_builder.build.return_value = mock_client
        mock_builder_class.return_value = mock_builder

        # Act
        result = AzureClientFactory.create_openai_client_from_params(
            model="gpt-4",
            base_url="https://test.openai.azure.com/",
            api_version="2023-05-15",
            authentication_method=AuthenticationMethod.DEFAULT_CREDENTIAL,
        )

        # Assert
        mock_builder_class.assert_called_once()
        mock_builder.build.assert_called_once()
        assert result == mock_client

        # Check that ModelSettings was created with correct parameters
        call_args = mock_builder_class.call_args[0][0]
        assert call_args.model == "gpt-4"
        assert call_args.base_url == "https://test.openai.azure.com/"
        assert call_args.api_version == "2023-05-15"
        assert (
            call_args.authentication_method == AuthenticationMethod.DEFAULT_CREDENTIAL
        )

    @patch(
        "ingenious.client.azure.azure_client_builder_factory.AzureOpenAIClientBuilder"
    )
    def test_create_openai_client_from_params_with_api_key(self, mock_builder_class):
        """Test creating OpenAI client from parameters with API key."""
        # Arrange
        mock_builder = Mock()
        mock_client = Mock()
        mock_builder.build.return_value = mock_client
        mock_builder_class.return_value = mock_builder

        # Act
        result = AzureClientFactory.create_openai_client_from_params(
            model="gpt-4",
            base_url="https://test.openai.azure.com/",
            api_version="2023-05-15",
            api_key="test-api-key",
            authentication_method=AuthenticationMethod.TOKEN,
        )

        # Assert
        mock_builder_class.assert_called_once()
        mock_builder.build.assert_called_once()
        assert result == mock_client

        # Check that ModelSettings was created with correct parameters
        call_args = mock_builder_class.call_args[0][0]
        assert call_args.api_key == "test-api-key"
        assert call_args.authentication_method == AuthenticationMethod.TOKEN

    @patch(
        "ingenious.client.azure.azure_client_builder_factory.AzureOpenAIChatCompletionClientBuilder"
    )
    def test_create_openai_chat_completion_client(self, mock_builder_class):
        """Test creating OpenAI chat completion client."""
        # Arrange
        mock_builder = Mock()
        mock_client = Mock()
        mock_builder.build.return_value = mock_client
        mock_builder_class.return_value = mock_builder

        # Build a proper ModelConfig with mocked config and profile
        mock_config_ns = Mock()
        mock_config_ns.model = "gpt-4"
        mock_config_ns.api_type = "rest"
        mock_config_ns.api_version = "2023-05-15"
        mock_config_ns.deployment = "gpt-4"

        mock_profile = Mock()
        mock_profile.deployment = "gpt-4"
        mock_profile.api_version = "2023-05-15"
        mock_profile.authentication_method = AuthenticationMethod.TOKEN
        mock_profile.client_id = "test-client-id"
        mock_profile.client_secret = "test-client-secret"
        mock_profile.tenant_id = "test-tenant-id"
        mock_profile.api_key = "test-key"
        mock_profile.base_url = "https://test.openai.azure.com/"

        model_config = ModelConfig(mock_config_ns, mock_profile)

        # Act
        result = AzureClientFactory.create_openai_chat_completion_client(model_config)

        # Assert
        mock_builder_class.assert_called_once_with(model_config)
        mock_builder.build.assert_called_once()
        assert result == mock_client

    @patch(
        "ingenious.client.azure.azure_client_builder_factory.AzureOpenAIChatCompletionClientBuilder"
    )
    def test_create_openai_chat_completion_client_from_params(self, mock_builder_class):
        """Test creating OpenAI chat completion client from parameters."""
        # Arrange
        mock_builder = Mock()
        mock_client = Mock()
        mock_builder.build.return_value = mock_client
        mock_builder_class.return_value = mock_builder

        # Act
        result = AzureClientFactory.create_openai_chat_completion_client_from_params(
            model="gpt-4",
            base_url="https://test.openai.azure.com/",
            api_version="2023-05-15",
            deployment="gpt-4-deployment",
            api_key="test-key",
            authentication_method=AuthenticationMethod.TOKEN,
        )

        # Assert
        mock_builder_class.assert_called_once()
        mock_builder.build.assert_called_once()
        assert result == mock_client

    # Blob Storage Client Tests
    @patch(
        "ingenious.client.azure.azure_client_builder_factory.BlobServiceClientBuilder"
    )
    def test_create_blob_service_client(self, mock_builder_class):
        """Test creating blob service client."""
        # Arrange
        mock_builder = Mock()
        mock_client = Mock()
        mock_builder.build.return_value = mock_client
        mock_builder_class.return_value = mock_builder

        storage_config = FileStorageContainerSettings(
            enable=True,
            storage_type="azure",
            container_name="test-container",
            path="./",
            add_sub_folders=True,
            url="https://teststorage.blob.core.windows.net/",
            client_id="",
            token="",
            authentication_method=AuthenticationMethod.DEFAULT_CREDENTIAL,
        )

        # Act
        result = AzureClientFactory.create_blob_service_client(storage_config)

        # Assert
        mock_builder_class.assert_called_once_with(storage_config)
        mock_builder.build.assert_called_once()
        assert result == mock_client

    @patch(
        "ingenious.client.azure.azure_client_builder_factory.BlobServiceClientBuilder"
    )
    def test_create_blob_service_client_from_params(self, mock_builder_class):
        """Test creating blob service client from parameters."""
        # Arrange
        mock_builder = Mock()
        mock_client = Mock()
        mock_builder.build.return_value = mock_client
        mock_builder_class.return_value = mock_builder

        # Act
        result = AzureClientFactory.create_blob_service_client_from_params(
            account_url="https://teststorage.blob.core.windows.net/",
            authentication_method=AuthenticationMethod.DEFAULT_CREDENTIAL,
            client_id="test-client-id",
        )

        # Assert
        mock_builder_class.assert_called_once()
        mock_builder.build.assert_called_once()
        assert result == mock_client

    @patch("ingenious.client.azure.azure_client_builder_factory.BlobClientBuilder")
    def test_create_blob_client(self, mock_builder_class):
        """Test creating blob client."""
        # Arrange
        mock_builder = Mock()
        mock_client = Mock()
        mock_builder.build.return_value = mock_client
        mock_builder_class.return_value = mock_builder

        storage_config = FileStorageContainerSettings(
            enable=True,
            storage_type="azure",
            container_name="test-container",
            path="./",
            add_sub_folders=True,
            url="https://teststorage.blob.core.windows.net/",
            client_id="",
            token="",
            authentication_method=AuthenticationMethod.DEFAULT_CREDENTIAL,
        )

        # Act
        result = AzureClientFactory.create_blob_client(
            storage_config,
            blob_name="test-blob.txt",
            container_name="override-container",
        )

        # Assert
        mock_builder_class.assert_called_once_with(
            storage_config, "override-container", "test-blob.txt"
        )
        mock_builder.build.assert_called_once()
        assert result == mock_client

    @patch("ingenious.client.azure.azure_client_builder_factory.BlobClientBuilder")
    def test_create_blob_client_from_params(self, mock_builder_class):
        """Test creating blob client from parameters."""
        # Arrange
        mock_builder = Mock()
        mock_client = Mock()
        mock_builder.build.return_value = mock_client
        mock_builder_class.return_value = mock_builder

        # Act
        result = AzureClientFactory.create_blob_client_from_params(
            account_url="https://teststorage.blob.core.windows.net/",
            blob_name="test-blob.txt",
            container_name="test-container",
            authentication_method=AuthenticationMethod.TOKEN,
            token="test-token",
        )

        # Assert
        mock_builder_class.assert_called_once()
        mock_builder.build.assert_called_once()
        assert result == mock_client

    # Cosmos Client Tests
    def test_create_cosmos_client_without_cosmos_package(self):
        """Test creating Cosmos client when package is not available."""
        # This test will pass if HAS_COSMOS is False
        if not HAS_COSMOS:
            with pytest.raises(ImportError, match="azure-cosmos is required"):
                AzureClientFactory.create_cosmos_client(
                    endpoint="https://test-cosmos.documents.azure.com:443/",
                    authentication_method=AuthenticationMethod.DEFAULT_CREDENTIAL,
                )

    def test_create_cosmos_client_not_implemented(self):
        """Test that Cosmos client creation returns NotImplemented."""
        # This test will check the current NotImplemented behavior
        if HAS_COSMOS:
            result = AzureClientFactory.create_cosmos_client(
                endpoint="https://test-cosmos.documents.azure.com:443/",
                authentication_method=AuthenticationMethod.DEFAULT_CREDENTIAL,
            )
            assert result == NotImplemented

    # Search Client Tests
    @pytest.mark.skipif(not HAS_SEARCH, reason="azure-search-documents not available")
    @patch(
        "ingenious.client.azure.azure_client_builder_factory.AzureSearchClientBuilder"
    )
    def test_create_search_client_with_search_package(self, mock_builder_class):
        """Test creating search client when package is available."""
        # Arrange
        mock_builder = Mock()
        mock_client = Mock()
        mock_builder.build.return_value = mock_client
        mock_builder_class.return_value = mock_builder
        search_config = AzureSearchSettings(
            service="test-search",
            endpoint="https://test-search.search.windows.net",
            key="test-key",
            client_id="",
            client_secret="",
            tenant_id="",
            authentication_method=AuthenticationMethod.DEFAULT_CREDENTIAL,
        )

        # Act
        result = AzureClientFactory.create_search_client(search_config, "test-index")

        # Assert
        mock_builder_class.assert_called_once_with(search_config, "test-index")
        mock_builder.build.assert_called_once()
        assert result == mock_client

    def test_create_search_client_without_search_package(self):
        """Test creating search client when package is not available."""
        if not HAS_SEARCH:
            search_config = AzureSearchSettings(
                service="test-search",
                endpoint="https://test-search.search.windows.net",
                key="test-key",
                client_id="",
                client_secret="",
                tenant_id="",
                authentication_method=AuthenticationMethod.DEFAULT_CREDENTIAL,
            )

            with pytest.raises(ImportError, match="azure-search-documents is required"):
                AzureClientFactory.create_search_client(search_config, "test-index")

    @pytest.mark.skipif(not HAS_SEARCH, reason="azure-search-documents not available")
    @patch(
        "ingenious.client.azure.azure_client_builder_factory.AzureSearchClientBuilder"
    )
    def test_create_search_client_from_params_with_search_package(
        self, mock_builder_class
    ):
        """Test creating search client from parameters when package is available."""
        # Arrange
        mock_builder = Mock()
        mock_client = Mock()
        mock_builder.build.return_value = mock_client
        mock_builder_class.return_value = mock_builder

        # Act
        result = AzureClientFactory.create_search_client_from_params(
            endpoint="https://test-search.search.windows.net",
            index_name="test-index",
            api_key="test-key",
            service="test-search",
            authentication_method=AuthenticationMethod.TOKEN,
        )

        # Assert
        mock_builder_class.assert_called_once()
        mock_builder.build.assert_called_once()
        assert result == mock_client

    def test_create_search_client_from_params_without_search_package(self):
        """Test creating search client from parameters when package is not available."""
        if not HAS_SEARCH:
            with pytest.raises(ImportError, match="azure-search-documents is required"):
                AzureClientFactory.create_search_client_from_params(
                    endpoint="https://test-search.search.windows.net",
                    index_name="test-index",
                    api_key="test-key",
                    authentication_method=AuthenticationMethod.TOKEN,
                )

    # SQL Client Tests
    @patch("ingenious.client.azure.azure_client_builder_factory.AzureSqlClientBuilder")
    def test_create_sql_client(self, mock_builder_class):
        """Test creating SQL client."""
        # Arrange
        mock_builder = Mock()
        mock_connection = Mock()
        mock_builder.build.return_value = mock_connection
        mock_builder_class.return_value = mock_builder

        sql_config = AzureSqlSettings(
            database_name="test-db",
            table_name="test-table",
            database_connection_string="Server=test;Database=test;",
        )

        # Act
        result = AzureClientFactory.create_sql_client(sql_config)

        # Assert
        mock_builder_class.assert_called_once_with(sql_config)
        mock_builder.build.assert_called_once()
        assert result == mock_connection

    @patch("ingenious.client.azure.azure_client_builder_factory.AzureSqlClientBuilder")
    def test_create_sql_client_from_params(self, mock_builder_class):
        """Test creating SQL client from parameters."""
        # Arrange
        mock_builder = Mock()
        mock_connection = Mock()
        mock_builder.build.return_value = mock_connection
        mock_builder_class.return_value = mock_builder

        # Act
        result = AzureClientFactory.create_sql_client_from_params(
            database_name="test-db",
            connection_string="Server=test;Database=test;",
            table_name="test-table",
        )

        # Assert
        mock_builder_class.assert_called_once()
        mock_builder.build.assert_called_once()
        assert result == mock_connection

    @patch(
        "ingenious.client.azure.azure_client_builder_factory.AzureSqlClientBuilderWithAuth"
    )
    def test_create_sql_client_with_auth(self, mock_builder_class):
        """Test creating SQL client with authentication."""
        # Arrange
        mock_builder = Mock()
        mock_connection = Mock()
        mock_builder.build.return_value = mock_connection
        mock_builder_class.return_value = mock_builder

        # Act
        result = AzureClientFactory.create_sql_client_with_auth(
            server="test-server",
            database="test-db",
            authentication_method=AuthenticationMethod.CLIENT_ID_AND_SECRET,
            client_id="test-client-id",
            client_secret="test-secret",
            tenant_id="test-tenant",
        )

        # Assert
        mock_builder_class.assert_called_once_with(
            server="test-server",
            database="test-db",
            authentication_method=AuthenticationMethod.CLIENT_ID_AND_SECRET,
            username=None,
            password=None,
            client_id="test-client-id",
            client_secret="test-secret",
            tenant_id="test-tenant",
        )
        mock_builder.build.assert_called_once()
        assert result == mock_connection

    def test_create_sql_client_with_auth_from_params_is_alias(self):
        """Test that create_sql_client_with_auth_from_params is an alias."""
        with patch.object(
            AzureClientFactory, "create_sql_client_with_auth"
        ) as mock_method:
            mock_connection = Mock()
            mock_method.return_value = mock_connection

            # Act
            result = AzureClientFactory.create_sql_client_with_auth_from_params(
                server="test-server",
                database="test-db",
                authentication_method=AuthenticationMethod.DEFAULT_CREDENTIAL,
            )

            # Assert
            mock_method.assert_called_once_with(
                server="test-server",
                database="test-db",
                authentication_method=AuthenticationMethod.DEFAULT_CREDENTIAL,
                username=None,
                password=None,
                client_id=None,
                client_secret=None,
                tenant_id=None,
            )
            assert result == mock_connection

    # Error Handling Tests
    def test_optional_dependency_flags(self):
        """Test that optional dependency flags are properly set."""
        # These should be boolean values
        assert isinstance(HAS_COSMOS, bool)
        assert isinstance(HAS_SEARCH, bool)

    def test_factory_is_static_class(self):
        """Test that factory methods are static and class doesn't need instantiation."""
        # Should be able to call methods without instantiating
        with patch(
            "ingenious.client.azure.azure_client_builder_factory.AzureOpenAIClientBuilder"
        ):
            # This should work without creating an instance
            AzureClientFactory.create_openai_client_from_params(
                model="gpt-4",
                base_url="https://test.openai.azure.com/",
                api_version="2023-05-15",
            )

    def test_authentication_method_enum_usage(self):
        """Test that authentication methods are properly used."""
        # Test that we can use all authentication methods
        methods = [
            AuthenticationMethod.DEFAULT_CREDENTIAL,
            AuthenticationMethod.TOKEN,
            AuthenticationMethod.MSI,
            AuthenticationMethod.CLIENT_ID_AND_SECRET,
        ]

        for method in methods:
            with patch(
                "ingenious.client.azure.azure_client_builder_factory.AzureOpenAIClientBuilder"
            ):
                AzureClientFactory.create_openai_client_from_params(
                    model="gpt-4",
                    base_url="https://test.openai.azure.com/",
                    api_version="2023-05-15",
                    authentication_method=method,
                )


class TestAzureClientFactoryIntegration:
    """Integration tests for Azure Client Factory."""

    def test_model_settings_creation_with_defaults(self):
        """Test that ModelSettings are created correctly with default values."""
        with patch(
            "ingenious.client.azure.azure_client_builder_factory.AzureOpenAIClientBuilder"
        ) as mock_builder_class:
            mock_builder = Mock()
            mock_builder_class.return_value = mock_builder

            # Act
            AzureClientFactory.create_openai_client_from_params(
                model="gpt-4",
                base_url="https://test.openai.azure.com/",
                api_version="2023-05-15",
            )

            # Assert
            call_args = mock_builder_class.call_args[0][0]
            assert call_args.model == "gpt-4"
            assert call_args.deployment == "gpt-4"  # Should default to model name
            assert call_args.api_key == ""  # Should default to empty string
            assert (
                call_args.authentication_method
                == AuthenticationMethod.DEFAULT_CREDENTIAL
            )

    def test_file_storage_settings_creation(self):
        """Test that FileStorageContainerSettings are created correctly."""
        with patch(
            "ingenious.client.azure.azure_client_builder_factory.BlobServiceClientBuilder"
        ) as mock_builder_class:
            mock_builder = Mock()
            mock_builder_class.return_value = mock_builder

            # Act
            AzureClientFactory.create_blob_service_client_from_params(
                account_url="https://teststorage.blob.core.windows.net/",
                authentication_method=AuthenticationMethod.TOKEN,
                token="test-token",
                client_id="test-client-id",
            )

            # Assert
            call_args = mock_builder_class.call_args[0][0]
            assert call_args.enable is True
            assert call_args.storage_type == "azure"
            assert call_args.url == "https://teststorage.blob.core.windows.net/"
            assert call_args.token == "test-token"
            assert call_args.client_id == "test-client-id"
            assert call_args.authentication_method == AuthenticationMethod.TOKEN

    @pytest.mark.skipif(not HAS_SEARCH, reason="azure-search-documents not available")
    def test_azure_search_settings_creation(self):
        """Test that AzureSearchSettings are created correctly."""
        with patch(
            "ingenious.client.azure.azure_client_builder_factory.AzureSearchClientBuilder"
        ) as mock_builder_class:
            mock_builder = Mock()
            mock_builder_class.return_value = mock_builder

            # Act
            AzureClientFactory.create_search_client_from_params(
                endpoint="https://test-search.search.windows.net",
                index_name="test-index",
                api_key="test-key",
                service="test-search",
                authentication_method=AuthenticationMethod.TOKEN,
                client_id="test-client-id",
            )

            # Assert
            call_args = mock_builder_class.call_args[0][0]
            assert call_args.service == "test-search"
            assert call_args.endpoint == "https://test-search.search.windows.net"
            assert call_args.key == "test-key"
            assert call_args.client_id == "test-client-id"
            assert call_args.authentication_method == AuthenticationMethod.TOKEN

    def test_azure_sql_settings_creation(self):
        """Test that AzureSqlSettings are created correctly."""
        with patch(
            "ingenious.client.azure.azure_client_builder_factory.AzureSqlClientBuilder"
        ) as mock_builder_class:
            mock_builder = Mock()
            mock_builder_class.return_value = mock_builder

            # Act
            AzureClientFactory.create_sql_client_from_params(
                database_name="test-db",
                connection_string="Server=test;Database=test;",
                table_name="test-table",
            )

            # Assert
            call_args = mock_builder_class.call_args[0][0]
            assert call_args.database_name == "test-db"
            assert call_args.table_name == "test-table"
            assert call_args.database_connection_string == "Server=test;Database=test;"
