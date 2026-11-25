"""A person's MCP server is reached over https only, whoever built its toolset.

``check_server_address`` refuses plain http, but a product has to call it. The client
every toolset is built with refuses it too, before a connection is made, so the
server's headers never go out in the clear. Nothing here leaves this machine: the
network layer is replaced by one that records where it was asked to connect.
"""

import asyncio
import socket

import httpcore
import httpx
import pytest

from harness.tools.user_tools import McpLimits, McpToolsetCache, UnavailableTools
from harness.tools.user_tools import mcp_address
from harness.tools.user_tools.mcp_toolsets import _client_without_redirects

PUBLIC = "93.184.215.14"  # an address on the public internet, never dialled here
HEADERS = {"authorization": "Bearer secret"}


@pytest.fixture
def dialled(monkeypatch):
    """Where the client tried to connect. Each try fails, so nothing is sent."""
    seen: list[tuple[str, int]] = []

    async def connect_tcp(self, host, port, **_):
        seen.append((host, port))
        raise httpcore.ConnectError("not dialled in tests")

    monkeypatch.setattr(httpcore.AnyIOBackend, "connect_tcp", connect_tcp)
    return seen


def _get(url: str, *, allow_local: bool = False) -> Exception:
    async def go():
        async with _client_without_redirects(headers=HEADERS, allow_local=allow_local) as client:
            await client.get(url)

    with pytest.raises(httpx.ConnectError) as refused:
        asyncio.run(go())
    return refused.value


def test_plain_http_to_a_public_host_is_refused_before_any_connection(dialled):
    refused = _get(f"http://{PUBLIC}/mcp")
    assert "not an https address" in str(refused)
    assert dialled == []


def test_https_to_a_public_host_goes_on_to_connect(dialled):
    _get(f"https://{PUBLIC}/mcp")
    assert dialled == [(PUBLIC, 443)]


def test_allow_local_lets_plain_http_through(dialled):
    _get("http://127.0.0.1:9/mcp", allow_local=True)
    assert dialled == [("127.0.0.1", 9)]


def test_a_private_address_is_still_refused_at_the_socket():
    listener = socket.create_server(("127.0.0.1", 0))
    try:
        refused = _get(f"https://127.0.0.1:{listener.getsockname()[1]}/mcp")
    finally:
        listener.close()
    assert "server address isn't public" in str(refused)


def test_a_toolset_from_the_cache_never_reaches_a_plain_http_server(dialled):
    cache = McpToolsetCache(UnavailableTools(), McpLimits(attach_timeout_seconds=1, call_timeout_seconds=1, tool_list_ttl_seconds=60, cache_limit=4))
    toolset = cache.toolset_for({"slug": "docs", "url": f"http://{PUBLIC}/mcp", "headers": HEADERS})

    async def go():
        try:
            await toolset.get_tools()
        finally:
            await toolset.close()

    with pytest.raises(Exception):
        asyncio.run(go())
    assert dialled == []


def test_the_explicit_check_still_refuses_plain_http():
    with pytest.raises(mcp_address.ServerAddressRefused, match="not an https address"):
        asyncio.run(mcp_address.check_server_address(f"http://{PUBLIC}/mcp"))
