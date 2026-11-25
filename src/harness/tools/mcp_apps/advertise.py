"""Telling an MCP server this host can show MCP Apps pages.

The spec has a host advertise ``capabilities.extensions["io.modelcontextprotocol/ui"]``
in its ``initialize`` request, and servers SHOULD check it before attaching pages. The
MCP SDK in use builds that request's capabilities itself, with no field for extensions,
so this adds the one field at the HTTP layer the harness already owns
(user_tools/mcp_toolsets.py builds the client): an ``httpx`` transport, public API only.

Only a single JSON-RPC ``initialize`` request is changed, and only by adding that entry
where none is: any other request, a batch, a body that isn't JSON or an extension already
there passes through byte for byte. Off unless the product says on.
"""

from __future__ import annotations

import json
from typing import Optional

import httpx

from .page import PAGE_MIME_TYPE

EXTENSION = "io.modelcontextprotocol/ui"


def advertised(body: bytes) -> Optional[bytes]:
    """The ``initialize`` request with the extension added; None to send the body as it is."""
    try:
        message = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(message, dict) or message.get("method") != "initialize":
        return None
    params = message.get("params")
    if not isinstance(params, dict):
        return None
    capabilities = params.setdefault("capabilities", {})
    if not isinstance(capabilities, dict):
        return None
    extensions = capabilities.setdefault("extensions", {})
    if not isinstance(extensions, dict) or EXTENSION in extensions:
        return None
    extensions[EXTENSION] = {"mimeTypes": [PAGE_MIME_TYPE]}
    return json.dumps(message).encode("utf-8")


class AdvertisingTransport(httpx.AsyncBaseTransport):
    """Sends every request through ``inner``, the ``initialize`` one with the extension."""

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            body = await request.aread()
            changed = advertised(body)
            if changed is not None:
                headers = [(k, v) for k, v in request.headers.multi_items() if k.lower() != "content-length"]
                request = httpx.Request(request.method, request.url, headers=headers, content=changed, extensions=request.extensions)
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()
