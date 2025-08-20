# ingenious/__init__.py
"""
Ingenious - AI-powered conversation and workflow engine.

This package provides a modular system for building AI-powered applications
with support for dynamic workflows, chat services, and extensible architectures.
"""

# Explicit imports for better IDE support and clearer dependencies
# Avoid importing services to prevent circular dependencies with container
from . import config, models, utils

__version__ = "0.2.4"
__all__ = ["config", "models", "utils"]
