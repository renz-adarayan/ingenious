from __future__ import annotations

import importlib
from typing import Any

__all__ = ("AzureClientFactory", "builder")


def __getattr__(name: str) -> Any:
    # Lazy export of the factory to avoid importing heavy/optional deps at package import time.
    if name == "AzureClientFactory":
        mod = importlib.import_module(".azure_client_builder_factory", __name__)
        return getattr(mod, "AzureClientFactory")
    if name == "builder":
        # Return the builder package module itself; its __init__ is also lightweight/lazy.
        return importlib.import_module(".builder", __name__)
    raise AttributeError(name)
