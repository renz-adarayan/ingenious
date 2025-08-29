"""
Authentication-related dependency injection.

This module provides FastAPI dependency injection functions
for authentication and authorization services.
"""

import secrets
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBasic,
    HTTPBasicCredentials,
    HTTPBearer,
)
from typing_extensions import Annotated

from ingenious.auth.jwt import get_username_from_token
from ingenious.config.main_settings import IngeniousSettings
from ingenious.core.structured_logging import get_logger
from ingenious.services.fastapi_dependencies import get_config as _get_config

logger = get_logger(__name__)
security = HTTPBasic()
bearer_security = HTTPBearer()


def get_security_service(
    token: Annotated[HTTPAuthorizationCredentials, Depends(bearer_security)]
    | None = None,
    credentials: Annotated[HTTPBasicCredentials, Depends(security)] | None = None,
    config: IngeniousSettings = Depends(_get_config),
) -> str:
    """Security service with JWT and Basic Auth support."""
    if not config.web_configuration.authentication.enable:
        logger.warning(
            "Authentication is disabled. This is not recommended for production use."
        )
        return "anonymous"

    # Try JWT token first
    if token and token.credentials:
        try:
            username = get_username_from_token(token.credentials)
            return username
        except HTTPException:
            pass

    # Fall back to basic auth
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return _validate_basic_auth_credentials(credentials, config)


def get_security_service_optional(
    credentials: Optional[HTTPBasicCredentials] = None,
    config: IngeniousSettings = Depends(_get_config),
) -> Optional[str]:
    """Optional security service that doesn't require credentials when auth is disabled."""
    if not config.web_configuration.authentication.enable:
        logger.warning(
            "Authentication is disabled. This is not recommended for production use."
        )
        return None

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )

    return _validate_basic_auth_credentials(credentials, config)


def get_auth_user(
    request: Request, config: IngeniousSettings = Depends(_get_config)
) -> str:
    """Get authenticated user - supports both JWT and Basic Auth."""
    if not config.web_configuration.authentication.enable:
        logger.warning(
            "Authentication is disabled. This is not recommended for production use."
        )
        return "anonymous"

    auth_header = request.headers.get("Authorization", "")

    # Try JWT Bearer token first
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        try:
            username = get_username_from_token(token)
            return username
        except HTTPException:
            pass

    # Fall back to Basic Auth
    if auth_header.startswith("Basic "):
        return _handle_basic_auth_header(auth_header, config)

    # No valid authentication method provided
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_conditional_security(request: Request) -> str:
    """Get authenticated user - wrapper around get_auth_user for compatibility."""
    return get_auth_user(request)


def _validate_basic_auth_credentials(
    credentials: HTTPBasicCredentials, config: IngeniousSettings
) -> str:
    """Validate basic auth credentials against configuration."""
    current_username_bytes = credentials.username.encode("utf8")
    correct_username_bytes = config.web_configuration.authentication.username.encode(
        "utf-8"
    )
    is_correct_username = secrets.compare_digest(
        current_username_bytes, correct_username_bytes
    )
    current_password_bytes = credentials.password.encode("utf8")
    correct_password_bytes = config.web_configuration.authentication.password.encode(
        "utf-8"
    )
    is_correct_password = secrets.compare_digest(
        current_password_bytes, correct_password_bytes
    )
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.username


def _handle_basic_auth_header(auth_header: str, config: IngeniousSettings) -> str:
    """Handle Basic Auth header parsing and validation."""
    import base64

    try:
        credentials_str = base64.b64decode(auth_header[6:]).decode("utf-8")
        username, password = credentials_str.split(":", 1)
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication format",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate credentials
    current_username_bytes = username.encode("utf8")
    correct_username_bytes = config.web_configuration.authentication.username.encode(
        "utf-8"
    )
    is_correct_username = secrets.compare_digest(
        current_username_bytes, correct_username_bytes
    )

    current_password_bytes = password.encode("utf8")
    correct_password_bytes = config.web_configuration.authentication.password.encode(
        "utf-8"
    )
    is_correct_password = secrets.compare_digest(
        current_password_bytes, correct_password_bytes
    )

    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return username
