import secrets
from typing import Annotated, Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from ingenious.auth.jwt import (
    create_access_token,
    create_refresh_token,
    get_username_from_token,
    verify_token,
)
from ingenious.core.structured_logging import get_logger
from ingenious.services.fastapi_dependencies import get_config

logger = get_logger(__name__)
router = APIRouter()
security = HTTPBearer()


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/login", response_model=TokenResponse)
async def login(login_data: LoginRequest) -> TokenResponse:
    """Login endpoint that returns JWT tokens"""
    config = get_config()

    # Check if authentication is disabled
    if not config.web_configuration.authentication.enable:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Authentication is disabled. JWT login not available.",
        )

    # Validate credentials using the same logic as basic auth
    current_username_bytes = login_data.username.encode("utf8")
    correct_username_bytes = config.web_configuration.authentication.username.encode(
        "utf-8"
    )
    is_correct_username = secrets.compare_digest(
        current_username_bytes, correct_username_bytes
    )

    current_password_bytes = login_data.password.encode("utf8")
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
        )

    # Create tokens
    access_token = create_access_token(data={"sub": login_data.username})
    refresh_token = create_refresh_token(data={"sub": login_data.username})

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_access_token(refresh_data: RefreshRequest) -> TokenResponse:
    """Refresh access token using refresh token"""
    try:
        # Verify refresh token
        payload = verify_token(refresh_data.refresh_token, token_type="refresh")
        username = payload.get("sub")

        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )

        # Create new tokens
        access_token = create_access_token(data={"sub": username})
        new_refresh_token = create_refresh_token(data={"sub": username})

        return TokenResponse(
            access_token=access_token,
            refresh_token=new_refresh_token,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error refreshing token", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not refresh token",
        )


@router.get("/verify")
async def verify_token_endpoint(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> Dict[str, Any]:
    """Verify if a token is valid"""
    try:
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

        # Extract token from bearer scheme
        if not credentials or not credentials.credentials:
            raise credentials_exception

        username = get_username_from_token(credentials.credentials)
        return {"username": username, "valid": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error verifying token", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
