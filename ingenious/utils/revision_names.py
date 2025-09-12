"""
Revision ID generation utilities.

This module provides utilities for generating revision IDs, including:
- GitHub Codespaces-style funny names with UUIDs
- User-supplied name conflict resolution with incremental numbering
- Name validation and normalization
"""

import random
import re
import uuid
from typing import List, Optional

from ingenious.core.structured_logging import get_logger

logger = get_logger(__name__)

# GitHub Codespaces-style word lists
ADJECTIVES = [
    "animated",
    "bouncy",
    "clever",
    "cosmic",
    "curvy",
    "dazzling",
    "electric",
    "fluffy",
    "fuzzy",
    "glowing",
    "happy",
    "jazzy",
    "kinetic",
    "lovely",
    "magical",
    "nimble",
    "quirky",
    "radiant",
    "shiny",
    "stellar",
    "turbo",
    "vibrant",
    "witty",
    "zesty",
]

NOUNS = [
    "disco",
    "ninja",
    "palm",
    "quantum",
    "rainbow",
    "space",
    "unicorn",
    "velvet",
    "wizard",
    "crystal",
    "golden",
    "silver",
    "emerald",
    "diamond",
    "ruby",
    "sapphire",
    "amber",
    "pearl",
    "jade",
    "onyx",
    "copper",
    "bronze",
    "platinum",
    "titanium",
]


def generate_funny_revision_id() -> str:
    """
    Generate a funny revision ID in GitHub Codespaces style.
    
    Format: <adjective>-<noun>-<short-uuid>
    Example: "cosmic-ninja-a1b2c3d4"
    
    Returns:
        str: A randomly generated funny revision ID with UUID suffix
    """
    adjective = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)
    # Use first 8 characters of UUID for brevity
    short_uuid = str(uuid.uuid4()).replace("-", "")[:8]
    
    funny_id = f"{adjective}-{noun}-{short_uuid}"
    
    logger.info(
        "Generated funny revision ID",
        revision_id=funny_id,
        adjective=adjective,
        noun=noun,
        uuid_suffix=short_uuid,
    )
    
    return funny_id


def resolve_user_revision_id(revision_id: str, existing_revision_ids: List[str]) -> str:
    """
    Resolve user-supplied revision ID conflicts by adding incremental numbers.
    
    If the revision_id already exists, appends -1, -2, -3, etc. until finding
    an available ID.
    
    Args:
        revision_id: The desired revision ID from the user
        existing_revision_ids: List of existing revision IDs to check against
        
    Returns:
        str: The resolved revision ID (either original or with number suffix)
        
    Examples:
        resolve_user_revision_id("my-workflow", ["my-workflow"]) -> "my-workflow-1"
        resolve_user_revision_id("my-workflow", ["my-workflow", "my-workflow-1"]) -> "my-workflow-2"
    """
    if not revision_id:
        raise ValueError("Revision ID cannot be empty")
    
    # Normalize the revision ID (lowercase, replace underscores with hyphens)
    normalized_id = normalize_revision_id(revision_id)
    
    # If the ID doesn't conflict, use it as-is
    if normalized_id not in existing_revision_ids:
        logger.info(
            "User revision ID available",
            original_revision_id=revision_id,
            resolved_id=normalized_id,
        )
        return normalized_id
    
    # Find the highest numbered version that exists
    pattern = re.compile(rf"^{re.escape(normalized_id)}-(\d+)$")
    highest_number = 0
    
    for existing_id in existing_revision_ids:
        match = pattern.match(existing_id)
        if match:
            number = int(match.group(1))
            highest_number = max(highest_number, number)
    
    # Generate the next available number
    next_number = highest_number + 1
    resolved_id = f"{normalized_id}-{next_number}"
    
    logger.info(
        "Resolved user revision ID conflict",
        original_revision_id=revision_id,
        normalized_id=normalized_id,
        resolved_id=resolved_id,
        highest_existing_number=highest_number,
        next_number=next_number,
    )
    
    return resolved_id


def normalize_revision_id(name: str) -> str:
    """
    Normalize a revision ID to follow consistent naming conventions.
    
    Rules:
    - Convert to lowercase
    - Replace underscores with hyphens
    - Remove any invalid characters (keep only alphanumeric, hyphens)
    - Ensure it doesn't start or end with a hyphen
    
    Args:
        name: The revision ID to normalize
        
    Returns:
        str: The normalized revision ID
        
    Raises:
        ValueError: If the name is empty or becomes empty after normalization
    """
    if not name or not name.strip():
        raise ValueError("Revision ID cannot be empty")
    
    # Convert to lowercase and replace underscores with hyphens
    normalized = name.lower().replace("_", "-")
    
    # Keep only alphanumeric characters and hyphens
    normalized = re.sub(r"[^a-z0-9-]", "", normalized)
    
    # Remove leading/trailing hyphens and collapse multiple hyphens
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    
    if not normalized:
        raise ValueError(f"Revision ID '{name}' contains no valid characters")
    
    logger.debug(
        "Normalized revision ID",
        original_name=name,
        normalized_id=normalized,
    )
    
    return normalized


def validate_revision_id(revision_id: str) -> bool:
    """
    Validate that a revision ID follows the expected format.
    
    Args:
        revision_id: The revision ID to validate
        
    Returns:
        bool: True if the ID is valid, False otherwise
    """
    if not revision_id:
        return False
    
    # Must contain only lowercase letters, numbers, and hyphens
    # Must not start or end with a hyphen
    # Must be between 1 and 50 characters
    pattern = r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$"
    
    is_valid = (
        1 <= len(revision_id) <= 50 and 
        re.match(pattern, revision_id) is not None
    )
    
    logger.debug(
        "Validated revision ID",
        revision_id=revision_id,
        is_valid=is_valid,
    )
    
    return is_valid


def generate_revision_id(revision_id: Optional[str], existing_revision_ids: List[str]) -> str:
    """
    Generate a revision ID based on user input or create a funny name.
    
    Args:
        revision_id: Optional user-supplied revision ID
        existing_revision_ids: List of existing revision IDs to check conflicts
        
    Returns:
        str: The final revision ID to use
        
    Raises:
        ValueError: If revision_id is provided but invalid
    """
    if revision_id:
        # User provided an ID - resolve any conflicts
        return resolve_user_revision_id(revision_id, existing_revision_ids)
    else:
        # Generate a funny name (UUID makes collisions virtually impossible)
        return generate_funny_revision_id()
