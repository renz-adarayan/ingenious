from typing import Optional, Union

import pyodbc
from azure.identity import get_bearer_token_provider

from ingenious.client.azure.builder.base import AzureClientBuilder
from ingenious.common.enums import AuthenticationMethod
from ingenious.config.models import AzureSqlSettings
from ingenious.models.config import AzureSqlConfig


class AzureSqlClientBuilder(AzureClientBuilder):
    """Builder for Azure SQL clients with multiple authentication methods."""

    def __init__(self, sql_config: Union[AzureSqlConfig, AzureSqlSettings]):
        # Extract authentication parameters from config
        auth_config = self._create_auth_config_from_sql_config(sql_config)
        super().__init__(auth_config=auth_config)
        self.sql_config = sql_config

    def _create_auth_config_from_sql_config(self, sql_config):
        """Create AzureAuthConfig from SQL configuration."""
        from ingenious.config.auth_config import AzureAuthConfig

        return AzureAuthConfig.from_config(sql_config)

    def build(self) -> pyodbc.Connection:
        """
        Build Azure SQL client based on SQL configuration.

        Returns:
            pyodbc.Connection: Configured Azure SQL connection
        """
        # Check if connection string is provided
        connection_string = getattr(self.sql_config, "database_connection_string", None)
        if connection_string:
            # Use connection string directly
            return pyodbc.connect(connection_string)

        # If no connection string, we need to construct it
        # This requires additional configuration that's not in the current AzureSqlConfig
        raise ValueError(
            "Connection string is required for Azure SQL. "
            "Either provide database_connection_string or extend AzureSqlConfig "
            "to include server, database, and authentication details."
        )


class AzureSqlClientBuilderWithAuth(AzureClientBuilder):
    """Builder for Azure SQL clients with explicit authentication configuration."""

    def __init__(
        self,
        server: str,
        database: str,
        authentication_method: AuthenticationMethod = AuthenticationMethod.DEFAULT_CREDENTIAL,
        username: Optional[str] = None,
        password: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ):
        super().__init__()
        self.server = server
        self.database = database
        self.authentication_method = authentication_method
        self.username = username
        self.password = password
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id

    def build(self) -> pyodbc.Connection:
        """
        Build Azure SQL client based on configuration with authentication.

        Returns:
            pyodbc.Connection: Configured Azure SQL connection
        """
        # Base connection string components
        connection_parts = [
            "DRIVER={ODBC Driver 18 for SQL Server}",
            f"SERVER={self.server}",
            f"DATABASE={self.database}",
            "Encrypt=yes",
            "TrustServerCertificate=no",
            "Connection Timeout=30",
        ]

        if self.authentication_method == AuthenticationMethod.TOKEN:
            # SQL authentication with username/password
            if not self.username or not self.password:
                raise ValueError(
                    "Username and password are required for TOKEN authentication"
                )

            connection_parts.extend([f"UID={self.username}", f"PWD={self.password}"])

        elif self.authentication_method == AuthenticationMethod.DEFAULT_CREDENTIAL:
            # Azure AD authentication using default credentials
            connection_parts.append("Authentication=ActiveDirectoryDefault")

        elif self.authentication_method == AuthenticationMethod.MSI:
            # Managed Identity authentication
            if self.client_id:
                connection_parts.append(
                    f"Authentication=ActiveDirectoryMsi;UID={self.client_id}"
                )
            else:
                connection_parts.append("Authentication=ActiveDirectoryMsi")

        elif self.authentication_method == AuthenticationMethod.CLIENT_ID_AND_SECRET:
            # Service Principal authentication
            if not all([self.client_id, self.client_secret, self.tenant_id]):
                raise ValueError(
                    "client_id, client_secret, and tenant_id are required "
                    "for CLIENT_ID_AND_SECRET authentication method"
                )

            # For service principal, we need to get access token
            token_provider = get_bearer_token_provider(
                self.token_credential, "https://database.windows.net/.default"
            )
            access_token = token_provider()

            # Use access token in connection string
            connection_parts.append(f"AccessToken={access_token}")

        else:
            raise ValueError(
                f"Unsupported authentication method: {self.authentication_method}"
            )

        connection_string = ";".join(connection_parts)
        return pyodbc.connect(connection_string)
