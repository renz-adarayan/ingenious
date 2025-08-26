"""
Typed protocols for dynamically imported classes in Ingenious.

This module defines protocols (interfaces) for classes that are commonly
imported dynamically, providing type safety and clear contracts.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ingenious.models.chat import ChatRequest, ChatResponse


@runtime_checkable
class WorkflowProtocol(Protocol):
    """Protocol for conversation flow/workflow classes."""

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the workflow."""
        ...

    @abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the workflow."""
        ...


@runtime_checkable
class ConversationFlowProtocol(Protocol):
    """Protocol for conversation flow classes in multi-agent systems."""

    @abstractmethod
    def __init__(
        self, chat_history_repository: Any, config: Any, **kwargs: Any
    ) -> None:
        """Initialize the conversation flow."""
        ...

    @abstractmethod
    async def get_chat_response(self, chat_request: ChatRequest) -> ChatResponse:
        """Get response from the conversation flow."""
        ...


@runtime_checkable
class ChatServiceProtocol(Protocol):
    """Protocol for chat service implementations."""

    @abstractmethod
    async def get_chat_response(self, chat_request: ChatRequest) -> ChatResponse:
        """Get chat response from the service."""
        ...


@runtime_checkable
class ExtractorProtocol(Protocol):
    """Protocol for document extractor classes."""

    @abstractmethod
    def extract(self, document_path: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract structured data from a document."""
        ...

    @abstractmethod
    def supports_format(self, file_extension: str) -> bool:
        """Check if the extractor supports a file format."""
        ...


@runtime_checkable
class RepositoryProtocol(Protocol):
    """Protocol for repository classes."""

    @abstractmethod
    def save(self, entity: Any) -> Any:
        """Save an entity to the repository."""
        ...

    @abstractmethod
    def get_by_id(self, entity_id: str) -> Optional[Any]:
        """Get an entity by ID."""
        ...


@runtime_checkable
class FileStorageProtocol(Protocol):
    """Protocol for file storage implementations."""

    @abstractmethod
    def read_file(self, file_path: str, **kwargs: Any) -> str:
        """Read file content."""
        ...

    @abstractmethod
    def write_file(self, file_path: str, content: str, **kwargs: Any) -> None:
        """Write file content."""
        ...

    @abstractmethod
    def list_files(self, directory_path: str, **kwargs: Any) -> List[str]:
        """List files in a directory."""
        ...


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol for AI agent implementations."""

    @abstractmethod
    async def process(self, input_data: Any, **kwargs: Any) -> Any:
        """Process input data and return result."""
        ...

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities."""
        ...


@runtime_checkable
class ToolProtocol(Protocol):
    """Protocol for tool implementations in multi-agent systems."""

    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        ...

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool's parameter schema."""
        ...


@runtime_checkable
class ValidatorProtocol(Protocol):
    """Protocol for validator classes."""

    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Validate data and return True if valid."""
        ...

    @abstractmethod
    def get_errors(self) -> List[str]:
        """Get validation error messages."""
        ...


@runtime_checkable
class ProcessorProtocol(Protocol):
    """Protocol for data processor classes."""

    @abstractmethod
    def process(self, input_data: Any, **kwargs: Any) -> Any:
        """Process input data and return result."""
        ...


@runtime_checkable
class ConfigurableProtocol(Protocol):
    """Protocol for classes that can be configured."""

    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the object with given settings."""
        ...

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        ...


@runtime_checkable
class ExtensionProtocol(Protocol):
    """Protocol for extension modules."""

    @abstractmethod
    def initialize(self, app: Any, config: Any) -> None:
        """Initialize the extension with app and config."""
        ...

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get extension metadata."""
        ...


# Protocol validation helpers
def validate_protocol_compliance(obj: Any, protocol: type) -> bool:
    """
    Check if an object complies with a protocol.

    Args:
        obj: Object to check
        protocol: Protocol to validate against

    Returns:
        True if object implements the protocol
    """
    try:
        return isinstance(obj, protocol)
    except Exception:
        return False


# Registry for protocol mappings
PROTOCOL_REGISTRY = {
    "workflow": WorkflowProtocol,
    "conversation_flow": ConversationFlowProtocol,
    "chat_service": ChatServiceProtocol,
    "extractor": ExtractorProtocol,
    "repository": RepositoryProtocol,
    "file_storage": FileStorageProtocol,
    "agent": AgentProtocol,
    "tool": ToolProtocol,
    "validator": ValidatorProtocol,
    "processor": ProcessorProtocol,
    "configurable": ConfigurableProtocol,
    "extension": ExtensionProtocol,
}


def get_protocol_by_name(name: str) -> Optional[type]:
    """Get a protocol by its registered name."""
    return PROTOCOL_REGISTRY.get(name.lower())
