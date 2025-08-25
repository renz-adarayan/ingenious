import enum
from typing import Any, List, Optional, Protocol

# Define the DatabaseClientType enum


class DatabaseClientType(enum.Enum):
    SQLITE = "sqlite"
    AZURESQL = "azuresql"
    COSMOS = "cosmos"


# Define an interface or base class for the database client
class DatabaseClient(Protocol):
    """Protocol for database client implementations."""

    def connect(self) -> None:
        """Establish connection to the database."""
        ...

    def execute_query(self, query: str, params: Optional[List[Any]] = None) -> None:
        """Execute a query without expecting results."""
        ...

    def fetch_all(self, query: str, params: Optional[List[Any]] = None) -> List[Any]:
        """Execute a query and fetch all results."""
        ...

    def fetch_one(
        self, query: str, params: Optional[List[Any]] = None
    ) -> Optional[Any]:
        """Execute a query and fetch one result."""
        ...
