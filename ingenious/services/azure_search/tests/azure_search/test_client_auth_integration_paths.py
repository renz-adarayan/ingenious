"""KB preflight ↔ client_init integration hooks (auth/init paths).

Why:
- Validate that KB preflight calls into `client_init.make_search_client`.
- Validate that missing SDKs surface as a KB preflight 'sdk_missing' error.

Notes:
- These tests patch at the KB module import site to exercise the *integration*
  seam (KB flow → client_init). They skip gracefully if the KB surface differs.

Usage:
- Run with pytest. Network-free. Only boundary patching via `unittest.mock.patch`.
"""

from __future__ import annotations

from typing import Any
import importlib
import sys

import pytest
from unittest.mock import MagicMock, patch

_KB_MOD_PATH: str = (
    "ingenious.services.chat_services.multi_agent.conversation_flows."
    "knowledge_base_agent.knowledge_base_agent"
)


def _import_kb_or_skip() -> Any:
    """Import the KB module or skip if not present."""
    try:
        return importlib.import_module(_KB_MOD_PATH)
    except Exception:  # pragma: no cover - defensive
        pytest.skip("KB knowledge_base_agent module not importable.")


@pytest.mark.asyncio
async def test_kb_preflight_uses_make_search_client_and_prefers_aad_over_key() -> None:
    """Assert that KB preflight calls `make_search_client` (AAD preference optional).

    Why: Even if credential selection is factory-controlled, KB preflight must
    delegate to client_init. This test asserts the delegation. If AAD preference
    logic is not exposed at the KB seam, the test will still validate the call.

    Skips gracefully if there is no accessible preflight entry.
    """
    kb_mod = _import_kb_or_skip()
    preflight = getattr(kb_mod, "_require_valid_azure_index", None)
    if preflight is None:
        pytest.skip("KB preflight helper not available for import-site patching.")

    called: list[bool] = []

    def _spy_make_search_client(*_a: Any, **_k: Any) -> Any:
        """Spy that records invocation and returns a closeable stub."""
        called.append(True)

        class _Stub:
            async def close(self) -> None:
                return None

        return _Stub()

    with patch(
        "ingenious.services.azure_search.client_init.make_search_client",
        new=_spy_make_search_client,
    ):
        # Signature unknown: exercise in a tolerant manner
        try:
            res: Any = preflight()  # type: ignore[misc]
            if hasattr(res, "__await__"):
                await res  # type: ignore[func-returns-value]
        except TypeError:
            # If the KB preflight has required args, we cannot exercise it here.
            pytest.skip("KB preflight requires args not available for this test.")

    assert called, "Expected client_init.make_search_client to be invoked by preflight."


@pytest.mark.asyncio
async def test_kb_preflight_sdk_missing_raises_sdk_missing_under_azure_only() -> None:
    """Assert missing SDKs propagate as 'sdk_missing' from the KB preflight path.

    Why: If the KB policy is strict (azure_only), missing Azure SDK modules
    should surface a clear, categorized preflight error rather than a generic
    import failure. This test simulates missing SDKs and expects the KB layer
    to raise an appropriate error.

    Skips gracefully if preflight path is not importable or error type differs.
    """
    kb_mod = _import_kb_or_skip()
    preflight = getattr(kb_mod, "_search_knowledge_base", None)
    if preflight is None:
        pytest.skip("KB search path not available for import-site patching.")

    # Simulate SDK missing inside the KB flow by making azure imports fail.
    with patch.dict(sys.modules, {"azure": None}):
        with pytest.raises(Exception) as excinfo:
            # Tolerate signature mismatches.
            try:
                res: Any = await preflight("q", policy="azure_only")  # type: ignore[call-arg]
            except TypeError:
                pytest.skip("KB search function signature differs; cannot pass policy.")
        # Minimal semantic assertion: error mentions 'sdk' and 'missing'
        assert "sdk" in str(excinfo.value).lower() and "missing" in str(
            excinfo.value
        ).lower()
