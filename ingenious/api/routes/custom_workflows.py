from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

from ingenious.core.structured_logging import get_logger
from ingenious.utils.namespace_utils import discover_custom_workflows

router = APIRouter()
logger = get_logger(__name__)


@router.get("/custom-workflows/list", response_model=Dict[str, Any])
async def list_available_custom_workflows() -> Dict[str, Any]:
    """
    Lists all available custom workflows using the dedicated discovery function
    that validates custom workflows.
    """
    try:
        discovery_result = discover_custom_workflows()

        return {
            "custom_workflows": discovery_result["workflows"],
            "count": len(discovery_result["workflows"]),
            "discovered_from": discovery_result["discovered_from"],  # Moved to root
        }
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during workflow discovery: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while listing custom workflows.",
        )
