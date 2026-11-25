"""Which MCP tools the model may see, by the MCP Apps ``visibility`` rule.

The spec (2026-01-26, ``_meta.ui.visibility``) says a host MUST keep a tool marked for
the server's own page only off the model's tool list. The rule here is deliberately
narrow: a tool is held back only when the marking is a list that names ``"app"`` and not
``"model"``. Anything else (no marking, an empty list, a string, unknown values, a
property that raises) leaves the tool with the model, the spec's default, so a server
that gets the field wrong can't make its tools disappear.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _marking(tool: Any) -> Any:
    try:
        # ADK's McpTool reads ``_meta.ui.visibility`` into this public property.
        return getattr(tool, "visibility", None)
    except Exception:  # noqa: BLE001 - a malformed definition must not cost the listing
        return None


def for_the_model(tool: Any) -> bool:
    """False only for a tool a server marked for its page alone."""
    marking = _marking(tool)
    if not isinstance(marking, (list, tuple)):
        return True
    values = {value for value in marking if isinstance(value, str)}
    return not ("app" in values and "model" not in values)


def model_tools(tools: list[Any], *, server: str) -> list[Any]:
    """The tools the model may see, logging how many were the page's alone."""
    kept = [tool for tool in tools if for_the_model(tool)]
    if len(kept) != len(tools):
        logger.info("kept %d tools for its page only out of the model's list: %s", len(tools) - len(kept), server)
    return kept
