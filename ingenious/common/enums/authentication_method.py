"""
Authentication method enumeration for Azure services.

This module defines the authentication methods available for connecting
to Azure services, particularly Azure OpenAI services.
"""

from enum import Enum
from typing import Any, Optional


class AuthenticationMethod(str, Enum):
    """Authentication methods for Azure services."""

    MSI = "msi"
    CLIENT_ID_AND_SECRET = "client_id_and_secret"
    DEFAULT_CREDENTIAL = "default_credential"
    TOKEN = "token"

    @classmethod
    def _missing_(cls, value: Any) -> Optional["AuthenticationMethod"]:
        """Handle case-insensitive enum lookups."""
        if isinstance(value, str):
            # Try to find a match with case-insensitive comparison
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        # If no match found, return None (will raise ValueError)
        return None
