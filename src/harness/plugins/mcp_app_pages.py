"""Keeps the page a server attaches to a tool call, and tells the client (MCP Apps).

ADK 2.14 notes a page on the call's event (``actions.render_ui_widgets``, one per call
id) when the MCP tool declares ``_meta.ui.resourceUri``. After such a call this plugin
fetches the page on the tool's own connection (tools/mcp_apps/page.py), keeps it with
the conversation (store.py), and puts its reference on the call's event under
``wire_names.PAGES``, keyed by call id. A page never costs the call: anything that
fails leaves the call as an ordinary tool call. ADK callbacks only.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from .. import wire_names
from ..tools.mcp_apps.limits import PageLimits
from ..tools.mcp_apps.page import PageSource, fetch_page
from ..tools.mcp_apps.store import PageRef, keep_page

logger = logging.getLogger(__name__)

MCP_PROVIDER = "mcp"


def page_uri_of(tool_context: ToolContext) -> Optional[str]:
    """The ``ui://`` page ADK noted for this call, if any."""
    widgets = getattr(tool_context.actions, "render_ui_widgets", None) or []
    for widget in widgets:
        payload = getattr(widget, "payload", None) or {}
        if getattr(widget, "id", None) == tool_context.function_call_id and getattr(widget, "provider", None) == MCP_PROVIDER:
            uri = payload.get("resource_uri") if isinstance(payload, dict) else None
            return uri if isinstance(uri, str) and uri.startswith("ui://") else None
    return None


class McpAppPagesPlugin(BasePlugin):
    def __init__(self, limits: PageLimits, shows_pages: Callable[[str], bool]) -> None:
        super().__init__(name="mcp_app_pages")
        self._limits = limits
        # Whether a server (its mount's name) may show pages: the product's rule.
        self._shows_pages = shows_pages
        self._kept: dict[str, dict[str, PageRef]] = {}

    async def after_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext, result: Any
    ) -> Optional[dict]:
        try:
            await self._keep(tool, tool_context)
        except Exception as error:  # noqa: BLE001 - a page never costs the call
            logger.warning("MCP App page skipped: %s", type(error).__name__)
        # The result stands as it is.
        return None

    async def _keep(self, tool: BaseTool, tool_context: ToolContext) -> None:
        source = getattr(tool, "page_source", None)
        if not isinstance(source, PageSource) or not self._shows_pages(source.server):
            return
        uri = page_uri_of(tool_context)
        if uri is None:
            return
        page = await fetch_page(source, uri, self._limits)
        if isinstance(page, str):
            logger.info("no MCP App page from %s: %s", source.server, page)
            return
        ref = await keep_page(tool_context, page, server=source.server, tool=tool.name)
        if ref is not None and tool_context.function_call_id:
            self._kept.setdefault(tool_context.invocation_id, {})[tool_context.function_call_id] = ref

    async def on_event_callback(self, *, invocation_context: InvocationContext, event: Event) -> Optional[Event]:
        kept = self._kept.get(invocation_context.invocation_id)
        if not kept:
            return None
        ids = {response.id for response in event.get_function_responses() if response.id}
        pages = {call_id: kept.pop(call_id).as_wire() for call_id in list(kept) if call_id in ids}
        if pages:
            event.custom_metadata = {**(event.custom_metadata or {}), wire_names.PAGES: pages}
        # In place, returning nothing: a returned event would end the chain here.
        return None

    async def after_run_callback(self, *, invocation_context: InvocationContext) -> None:
        self._kept.pop(invocation_context.invocation_id, None)
