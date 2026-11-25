"""MCP toolsets built from the web app's descriptors, kept per connection."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.mcp_tool.mcp_session_manager import (
    SseConnectionParams,
    StreamableHTTPConnectionParams,
)
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from mcp.shared._httpx_utils import create_mcp_http_client

from ..mcp_apps.advertise import AdvertisingTransport
from ..mcp_apps.page import PageSource
from ..mcp_apps.visibility import model_tools
from .mcp_address import HttpsOnly, PublicPeersOnly
from .mcp_limits import McpLimits
from .mount_rules import DEFAULT_RULES, MountRules, rules_of
from .timed_tool import CallsInFlight, TimedTool
from .unavailable import UnavailableTools

logger = logging.getLogger(__name__)

# Model providers refuse a whole request over one tool name outside this.
MAX_TOOL_NAME = 64
_VALID_TOOL_NAME = re.compile(rf"[a-zA-Z0-9_-]{{1,{MAX_TOOL_NAME}}}")


def _client_without_redirects(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
    *,
    allow_local: bool = False,
    advertise_apps: bool = False,
) -> httpx.AsyncClient:
    """The MCP client, with no redirects, https only, and only public servers at the socket."""
    client = create_mcp_http_client(headers=headers, timeout=timeout, auth=auth)
    client.follow_redirects = False
    # httpx offers no public way to give its default transport a network backend.
    pool = httpx.AsyncHTTPTransport()
    pool._pool._network_backend = PublicPeersOnly(allow_local=allow_local)
    transport: httpx.AsyncBaseTransport = HttpsOnly(pool, allow_local=allow_local)
    # Only the initialize request changes, by one field (MCP Apps, tools/mcp_apps/advertise.py).
    client._transport = AdvertisingTransport(transport) if advertise_apps else transport
    # A proxy from the environment would replace that transport, and its socket
    # says nothing about the server behind it.
    client._mounts = {}
    return client


@dataclass(frozen=True)
class UsableTools:
    tools: list[BaseTool]
    # The names the server listed that a model provider would refuse, left out.
    left_out: tuple[str, ...]


async def close_within(toolset: McpToolset, seconds: float, name: str) -> None:
    """Closes a toolset, giving up after this long: a server that won't hang up costs
    its connection, never the caller's wait."""
    try:
        await asyncio.wait_for(toolset.close(), timeout=seconds)
    except TimeoutError:
        logger.warning("closing an MCP toolset timed out: %s", name)
    except Exception as error:  # noqa: BLE001 - nothing more is wanted from it
        logger.warning("closing an MCP toolset failed: %s (%s)", name, type(error).__name__)


class TimedMcpToolset(McpToolset):
    """An McpToolset whose tools can't outlast the call limit."""

    def __init__(
        self,
        *,
        app: str,
        unavailable: UnavailableTools,
        call_timeout_seconds: float,
        rules: MountRules = DEFAULT_RULES,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._app = app
        self._rules = rules
        self._call_timeout_seconds = call_timeout_seconds
        self._unavailable = unavailable
        # Its tools' calls running now: a cache never closes it under one.
        self.calls = CallsInFlight()
        self._left_out: tuple[str, ...] = ()

    async def get_tools(self, readonly_context: Optional[ReadonlyContext] = None) -> list[BaseTool]:
        """The server's tools a model may be given, each under the call limit.

        ADK prefixes what this returns (``get_tools_with_prefix``, which is final), so the
        name check here is on the name the model will see after that."""
        # A tool a server made for its own page only never reaches the model (MCP Apps).
        tools = model_tools(await super().get_tools(readonly_context), server=self._app)
        usable = [tool for tool in tools if _VALID_TOOL_NAME.fullmatch(self._shown_name(tool))]
        self._left_out = tuple(
            self._shown_name(tool) for tool in tools if not _VALID_TOOL_NAME.fullmatch(self._shown_name(tool))
        )
        if self._left_out:
            logger.warning(
                "left out %d tools with invalid names from: %s",
                len(self._left_out),
                self._app,
            )
            if readonly_context is not None:
                self._unavailable.add(readonly_context.invocation_id, self._app)
        return [
            TimedTool(
                tool,
                self._app,
                self._unavailable,
                self.calls,
                self._rules,
                call_timeout_seconds=self._call_timeout_seconds,
                page_source=PageSource(connection=self.connection_params, server=self._app),
            )
            for tool in usable
        ]

    async def usable_tools(
        self, readonly_context: Optional[ReadonlyContext] = None
    ) -> UsableTools:
        """The tools a model may be given, prefixed, and the names left out of them."""
        tools = await self.get_tools_with_prefix(readonly_context)
        # Set by the get_tools behind these tools: ADK reuses its list only for the same
        # invocation as the last call, and nothing awaits between that call and this read.
        return UsableTools(tools, self._left_out)

    def _shown_name(self, tool: BaseTool) -> str:
        # As BaseToolset.get_tools_with_prefix names it.
        return f"{self.tool_name_prefix}_{tool.name}" if self.tool_name_prefix else tool.name


def build_mcp_toolset(
    server: dict[str, Any], unavailable: UnavailableTools, limits: McpLimits
) -> TimedMcpToolset:
    """Build an McpToolset from an MCP server descriptor the web app sent."""
    params_cls = (
        SseConnectionParams
        if server.get("transport") == "sse"
        else StreamableHTTPConnectionParams
    )
    tools = server.get("tools") or None
    rules = rules_of(server)
    return TimedMcpToolset(
        app=str(server["slug"]),
        unavailable=unavailable,
        call_timeout_seconds=limits.call_timeout_seconds,
        rules=rules,
        connection_params=params_cls(
            url=server["url"],
            headers=server.get("headers") or {},
            timeout=limits.connect_timeout_seconds,
            sse_read_timeout=limits.read_timeout_seconds,
            **(
                {}
                if rules.own_host_check
                else {
                    "httpx_client_factory": functools.partial(
                        _client_without_redirects,
                        allow_local=limits.allow_local_servers,
                        advertise_apps=limits.advertise_apps,
                    )
                }
            ),
        ),
        tool_filter=tools,
        # ADK joins "<prefix>_<name>", giving mcp_<slug>__<tool> names.
        tool_name_prefix=f"mcp_{server['slug']}_",
        tool_list_cache_ttl_seconds=limits.tool_list_ttl_seconds,
    )


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def _in_use(toolset: TimedMcpToolset) -> bool:
    # A stand-in toolset in a test may not count its calls.
    calls = getattr(toolset, "calls", None)
    return isinstance(calls, CallsInFlight) and calls.count > 0


class McpToolsetCache:
    """Small LRU of McpToolsets; evicted toolsets are closed in the background.

    A toolset with a tool call running is never the one evicted: closing it would end
    the session under the call. ADK's tool reconnects on its next call, so closing an
    idle one costs a new session, not a failure. When every toolset is busy the cache
    holds more than its limit until one is free.
    """

    def __init__(self, unavailable: UnavailableTools, limits: McpLimits) -> None:
        self._unavailable = unavailable
        self._items: OrderedDict[str, TimedMcpToolset] = OrderedDict()
        self._limits = limits

    def toolset_for(
        self, server: dict[str, Any], owner: str | None = None
    ) -> TimedMcpToolset:
        """The toolset for this exact descriptor, headers included.

        ``owner`` keeps one person's MCP session apart from another's when the
        descriptor alone doesn't (a plugin's connector has the same headers for
        everyone, and a server may keep state per session).
        """
        key = _fingerprint({"server": server, "owner": owner} if owner is not None else server)
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        toolset = build_mcp_toolset(server, self._unavailable, self._limits)
        self._items[key] = toolset
        self._evict()
        return toolset

    def _evict(self) -> None:
        """Closes the least recently used idle toolsets past the limit, oldest first."""
        for key in list(self._items):
            if len(self._items) <= self._limits.cache_limit:
                return
            toolset = self._items[key]
            if _in_use(toolset):
                continue
            del self._items[key]
            asyncio.ensure_future(close_within(toolset, self._limits.close_timeout_seconds, "evicted"))

    async def close(self) -> None:
        for toolset in list(self._items.values()):
            await close_within(toolset, self._limits.close_timeout_seconds, "cached")
