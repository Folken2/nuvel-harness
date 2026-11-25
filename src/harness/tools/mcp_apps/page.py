"""Fetching and checking the page a server attached to a tool call (MCP Apps).

The page is read with ``resources/read`` on its own short MCP session, opened with the
call's connection settings (address, headers, the hardened client), so the same address
rules hold and the turn's session is never touched. Only the MCP SDK's public client is
used: ADK reads resources by name through ``resources/list``, and a server may leave a
page out of that list.

``check_page`` is pure: it takes what the server answered and gives a ``Page`` or the
reason there is none. Narrow on doubt: a wrong MIME type, a second document, bad base64
or text, an oversized page, or a CSP origin that isn't a plain https/wss origin is
refused or dropped, never widened.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import httpx
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams, StreamableHTTPConnectionParams
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client

from .limits import PageLimits

logger = logging.getLogger(__name__)

PAGE_MIME_TYPE = "text/html;profile=mcp-app"
CSP_LISTS = ("connectDomains", "resourceDomains", "frameDomains", "baseUriDomains")
# A plain origin: scheme, host (a leading "*." for subdomains), optional port. No path.
_ORIGIN = re.compile(r"^(https|wss)://(\*\.)?[a-z0-9-]+(\.[a-z0-9-]+)+(:\d{1,5})?$")

Connection = Union[StreamableHTTPConnectionParams, SseConnectionParams]


@dataclass(frozen=True)
class PageSource:
    """Where a tool's page can be read: its server's connection, and the server's name."""

    connection: Connection
    server: str


@dataclass(frozen=True)
class Page:
    uri: str
    html: str
    # Each of CSP_LISTS present, with only the origins that passed.
    csp: dict[str, list[str]] = field(default_factory=dict)
    prefers_border: Optional[bool] = None
    sha256: str = ""


def _same_mime(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parts = [part.strip().lower() for part in value.split(";")]
    return parts[0] == "text/html" and "profile=mcp-app" in parts[1:]


def _origins(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    kept: list[str] = []
    for item in value:
        if isinstance(item, str) and _ORIGIN.match(item.strip().lower()) and item.strip().lower() not in kept:
            kept.append(item.strip().lower())
        if len(kept) >= limit:
            break
    return kept


def _field(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def check_page(uri: str, contents: Any, limits: PageLimits) -> Union[Page, str]:
    """The page in a ``resources/read`` answer, or why there is none."""
    if not isinstance(uri, str) or not uri.startswith("ui://"):
        return "not a ui:// address"
    items = list(contents) if isinstance(contents, (list, tuple)) else []
    matching = [item for item in items if _field(item, "uri") is not None and str(_field(item, "uri")) == uri]
    if len(matching) != 1:
        return "no single document at that address"
    item = matching[0]
    if not _same_mime(_field(item, "mimeType")):
        return "not an MCP App page (MIME type)"
    text, blob = _field(item, "text"), _field(item, "blob")
    if isinstance(text, str):
        raw = text.encode("utf-8")
    elif isinstance(blob, str):
        try:
            raw = base64.b64decode(blob, validate=True)
        except (binascii.Error, ValueError):
            return "the page's base64 doesn't decode"
    else:
        return "the page has no content"
    if len(raw) > limits.max_page_bytes:
        return f"the page is over {limits.max_page_bytes} bytes"
    try:
        html = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "the page isn't UTF-8"
    if not html.strip():
        return "the page is empty"
    meta = _field(item, "meta") or _field(item, "_meta") or {}
    ui = meta.get("ui") if isinstance(meta, dict) else None
    ui = ui if isinstance(ui, dict) else {}
    csp_meta = ui.get("csp") if isinstance(ui.get("csp"), dict) else {}
    csp = {name: _origins(csp_meta.get(name), limits.max_domains) for name in CSP_LISTS if name in csp_meta}
    border = ui.get("prefersBorder")
    return Page(
        uri=uri,
        html=html,
        csp=csp,
        prefers_border=border if isinstance(border, bool) else None,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


async def _read(source: PageSource, uri: str) -> Any:
    params = source.connection
    headers = dict(params.headers or {})
    if isinstance(params, SseConnectionParams):
        transport = sse_client(
            params.url,
            headers=headers,
            timeout=params.timeout,
            sse_read_timeout=params.sse_read_timeout,
            httpx_client_factory=params.httpx_client_factory,
        )
        async with transport as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return (await session.read_resource(uri)).contents
    client = params.httpx_client_factory(headers=headers, timeout=httpx.Timeout(params.timeout, read=params.sse_read_timeout))
    async with client:
        async with streamable_http_client(params.url, http_client=client) as (read, write, _session_id):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return (await session.read_resource(uri)).contents


async def fetch_page(source: PageSource, uri: str, limits: PageLimits) -> Union[Page, str]:
    """The checked page, or why there is none. Never raises."""
    if not isinstance(uri, str) or not uri.startswith("ui://"):
        return "not a ui:// address"
    try:
        contents = await asyncio.wait_for(_read(source, uri), timeout=limits.fetch_timeout_seconds)
    except TimeoutError:
        return "the page didn't come in time"
    except Exception as error:  # noqa: BLE001 - a page never costs the call
        return f"the page couldn't be read ({type(error).__name__})"
    return check_page(uri, contents, limits)
