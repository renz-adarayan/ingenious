from typing import List, Optional

from pydantic import BaseModel, Field

from ingenious.common.enums import AuthenticationMethod
from ingenious.config.auth_config import AzureAuthConfig
from ingenious.models import config as config_models
from ingenious.models import config_ns as config_ns_models
from ingenious.models import profile as profile_models


class ChatHistoryConfig(config_ns_models.ChatHistoryConfig, AzureAuthConfig):
    database_connection_string: str = Field(
        "", description="Connection string for the database. Only used for azuresql"
    )

    def __init__(
        self,
        config: config_ns_models.ChatHistoryConfig,
        profile: profile_models.ChatHistoryConfig,
    ):
        super().__init__(
            database_type=config.database_type,
            database_path=config.database_path,
            database_connection_string=profile.database_connection_string,
            database_name=config.database_name,
            memory_path=config.memory_path,
        )

        # Initialize the AzureAuthConfig part
        AzureAuthConfig.__init__(
            self,
            authentication_method=profile.authentication_method,
            client_id=profile.client_id if profile.client_id else None,
            client_secret=profile.client_secret if profile.client_secret else None,
            tenant_id=profile.tenant_id if profile.tenant_id else None,
            api_key=profile.api_key if profile.api_key else None,
        )


class ModelConfig(config_ns_models.ModelConfig, AzureAuthConfig):
    base_url: str = ""

    def __init__(
        self, config: config_ns_models.ModelConfig, profile: profile_models.ModelConfig
    ):
        # Get deployment and api_version from profile if available, otherwise from config
        deployment = profile.deployment or config.deployment
        api_version = profile.api_version or config.api_version

        # Initialize the config_ns_models.ModelConfig part
        config_ns_models.ModelConfig.__init__(
            self,
            model=config.model,
            api_type=config.api_type,
            api_version=api_version,
            deployment=deployment,
        )

        # Initialize the AzureAuthConfig part
        AzureAuthConfig.__init__(
            self,
            authentication_method=profile.authentication_method,
            client_id=profile.client_id if profile.client_id else None,
            client_secret=profile.client_secret if profile.client_secret else None,
            tenant_id=profile.tenant_id if profile.tenant_id else None,
            api_key=profile.api_key if profile.api_key else None,
        )

        # Set additional fields from profile that aren't in either parent class
        self.base_url = profile.base_url


class ChatServiceConfig(config_ns_models.ChatServiceConfig):
    def __init__(
        self,
        config: config_ns_models.ChatServiceConfig,
        profile: profile_models.ChatServiceConfig,
    ):
        super().__init__(type=config.type)


class ToolServiceConfig(config_ns_models.ToolServiceConfig):
    def __init__(
        self,
        config: config_ns_models.ToolServiceConfig,
        profile: profile_models.ToolServiceConfig,
    ):
        super().__init__(enable=config.enable)


class LoggingConfig(config_ns_models.LoggingConfig):
    def __init__(
        self,
        config: config_ns_models.LoggingConfig,
        profile: profile_models.LoggingConfig,
    ):
        super().__init__(
            root_log_level=config.root_log_level, log_level=config.log_level
        )


class AzureSearchConfig(config_ns_models.AzureSearchConfig, AzureAuthConfig):
    key: str = ""

    def __init__(
        self,
        config: config_ns_models.AzureSearchConfig,
        profile: profile_models.AzureSearchConfig,
    ):
        # Initialize the config_ns_models.AzureSearchConfig part
        config_ns_models.AzureSearchConfig.__init__(
            self, service=config.service, endpoint=config.endpoint
        )

        # Initialize the AzureAuthConfig part
        AzureAuthConfig.__init__(
            self,
            authentication_method=AuthenticationMethod.TOKEN,  # Search services typically use API keys
            client_id=None,
            client_secret=None,
            tenant_id=None,
            api_key=profile.key if profile.key else None,
        )

        # Set additional fields from profile
        self.key = profile.key


class AzureSqlConfig(config_ns_models.AzureSqlConfig, AzureAuthConfig):
    database_connection_string: str = Field(
        "", description="azure SQL Connection string"
    )

    def __init__(
        self,
        config: config_ns_models.AzureSqlConfig,
        profile: profile_models.AzureSqlConfig,
    ):
        # Initialize the config_ns_models.AzureSqlConfig part
        config_ns_models.AzureSqlConfig.__init__(
            self,
            table_name=config.table_name,
            database_name=config.database_name,
            database_connection_string=profile.database_connection_string,
        )

        # Initialize the AzureAuthConfig part
        AzureAuthConfig.__init__(
            self,
            authentication_method=AuthenticationMethod.DEFAULT_CREDENTIAL,  # SQL typically uses connection strings or default credential
            client_id=None,
            client_secret=None,
            tenant_id=None,
            api_key=None,
        )


class CosmosConfig(config_ns_models.CosmosConfig, AzureAuthConfig):
    def __init__(
        self,
        config: config_ns_models.CosmosConfig,
        profile: profile_models.CosmosConfig,
    ):
        # Initialize the config_ns_models.CosmosConfig part
        config_ns_models.CosmosConfig.__init__(
            self, uri=config.uri, database_name=config.database_name
        )

        # Initialize the AzureAuthConfig part
        AzureAuthConfig.__init__(
            self,
            authentication_method=profile.authentication_method,
            client_id=profile.client_id if profile.client_id else None,
            client_secret=profile.client_secret if profile.client_secret else None,
            tenant_id=profile.tenant_id if profile.tenant_id else None,
            api_key=profile.api_key if profile.api_key else None,
        )


class ReceiverConfig(profile_models.ReceiverConfig):
    def __init__(self, profile: profile_models.ReceiverConfig):
        super().__init__(
            enable=profile.enable,
            api_url=profile.api_url,
            api_key=profile.api_key,
        )


class WebConfig(config_ns_models.WebConfig):
    authentication: profile_models.WebAuthConfig = Field(
        default_factory=profile_models.WebAuthConfig
    )

    def __init__(
        self, config: config_ns_models.WebConfig, profile: profile_models.WebConfig
    ):
        super().__init__(
            ip_address=config.ip_address,
            port=config.port,
            type=config.type,
            asynchronous=config.asynchronous,
        )

        # Set additional fields from profile
        self.authentication = profile.authentication


class LocaldbConfig(config_ns_models.LocaldbConfig):
    def __init__(self, config: config_ns_models.LocaldbConfig):
        super().__init__(
            database_path=config.database_path,
            sample_csv_path=config.sample_csv_path,
            sample_database_name=config.sample_database_name,
        )


class FileStorageContainer(config_ns_models.FileStorageContainer, AzureAuthConfig):
    url: str = Field("", description="File Storage SAS URL")
    token: str = Field("", description="File Storage SAS Token")

    def __init__(
        self,
        config: config_ns_models.FileStorageContainer,
        profile: profile_models.FileStorageContainer,
    ):
        # Initialize the config_ns_models.FileStorageContainer part
        config_ns_models.FileStorageContainer.__init__(
            self,
            enable=config.enable,
            storage_type=config.storage_type,
            container_name=config.container_name,
            path=config.path,
            add_sub_folders=config.add_sub_folders,
        )

        # Initialize the AzureAuthConfig part
        AzureAuthConfig.__init__(
            self,
            authentication_method=profile.authentication_method,
            client_id=profile.client_id if profile.client_id else None,
            client_secret=None,  # File storage doesn't typically use client secrets
            tenant_id=None,  # File storage doesn't typically use tenant ID
            api_key=profile.token if profile.token else None,
        )

        # Set additional fields
        self.url = profile.url
        self.token = profile.token


class FileStorage(config_ns_models.FileStorage):
    pass

    def __init__(
        self, config: config_ns_models.FileStorage, profile: profile_models.FileStorage
    ):
        super().__init__(
            revisions=FileStorageContainer(config.revisions, profile.revisions),
            data=FileStorageContainer(config.data, profile.data),
        )


class Config(BaseModel):
    """
    This is the constructor class that brings together the configuration and profile classes.
    """

    chat_history: ChatHistoryConfig
    models: List[ModelConfig]
    logging: LoggingConfig
    tool_service: ToolServiceConfig
    chat_service: ChatServiceConfig
    azure_search_services: Optional[List[AzureSearchConfig]] = Field(
        default=None, description="Azure Search services configuration (optional)"
    )
    web_configuration: WebConfig
    receiver_configuration: ReceiverConfig
    local_sql_db: LocaldbConfig
    azure_sql_services: Optional[AzureSqlConfig] = Field(
        default=None, description="Azure SQL services configuration (optional)"
    )
    file_storage: FileStorage

    def __init__(
        self, config: config_ns_models.Config, profile: profile_models.Profile
    ):
        super().__init__(
            chat_history=ChatHistoryConfig(config.chat_history, profile.chat_history),
            models=[],
            logging=LoggingConfig(config.logging, profile.logging),
            tool_service=ToolServiceConfig(config.tool_service, profile.tool_service),
            chat_service=ChatServiceConfig(config.chat_service, profile.chat_service),
            azure_search_services=[],
            web_configuration=WebConfig(
                config.web_configuration, profile.web_configuration
            ),
            receiver_configuration=ReceiverConfig(profile.receiver_configuration),
            local_sql_db=LocaldbConfig(config.local_sql_db),
            azure_sql_services=AzureSqlConfig(
                config.azure_sql_services, profile.azure_sql_services
            )
            if config.azure_sql_services and profile.azure_sql_services
            else None,
            file_storage=FileStorage(config.file_storage, profile.file_storage),
        )

        models: List[config_models.ModelConfig] = []
        for config_model in config.models:
            for profile_model in profile.models:
                if config_model.model == profile_model.model:
                    models.append(
                        config_models.ModelConfig(config_model, profile_model)
                    )
        self.models = models

        self.azure_search_services = []
        if config.azure_search_services and profile.azure_search_services:
            for as_config in config.azure_search_services:
                for profile_as_config in profile.azure_search_services:
                    if as_config.service == profile_as_config.service:
                        self.azure_search_services.append(
                            config_models.AzureSearchConfig(
                                as_config, profile_as_config
                            )
                        )
