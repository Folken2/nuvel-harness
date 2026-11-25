"""MCP Apps: the host tells a server it can show pages, by one field of the
initialize request, and changes nothing else it sends."""

import asyncio
import json

import httpx

from harness.tools.mcp_apps.advertise import EXTENSION, AdvertisingTransport, advertised
from harness.tools.user_tools.mcp_toolsets import _client_without_redirects  # noqa: F401 - the client the factory builds


def _init(capabilities=None) -> bytes:
    params = {"protocolVersion": "2025-06-18", "clientInfo": {"name": "x", "version": "1"}}
    if capabilities is not None:
        params["capabilities"] = capabilities
    return json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": params}).encode()


def test_initialize_gets_the_extension_and_keeps_everything_it_had():
    body = json.loads(advertised(_init({"sampling": {}, "extensions": {"other/ext": {"a": 1}}})))
    capabilities = body["params"]["capabilities"]
    assert capabilities["sampling"] == {}
    assert capabilities["extensions"] == {"other/ext": {"a": 1}, EXTENSION: {"mimeTypes": ["text/html;profile=mcp-app"]}}
    assert json.loads(advertised(_init()))["params"]["capabilities"]["extensions"][EXTENSION]


def test_anything_else_passes_through_untouched():
    for body in [
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}).encode(),
        json.dumps([{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}]).encode(),
        _init({"extensions": {EXTENSION: {"mimeTypes": ["x"]}}}),
        _init("not a dict"),
        b"not json",
        b"\xff\xfe",
    ]:
        assert advertised(body) is None


def test_the_transport_sends_the_changed_initialize_and_the_rest_as_they_were():
    seen: list[tuple[str, bytes, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.content, request.headers.get("content-length", "")))
        return httpx.Response(200, json={})

    async def scenario():
        async with httpx.AsyncClient(transport=AdvertisingTransport(httpx.MockTransport(handler))) as client:
            await client.post("https://mcp.example.test/mcp", content=_init({}), headers={"content-type": "application/json"})
            await client.post("https://mcp.example.test/mcp", content=b'{"jsonrpc":"2.0","id":2,"method":"tools/list"}')
            await client.get("https://mcp.example.test/mcp")

    asyncio.run(scenario())
    (m1, body1, length1), (m2, body2, _), (m3, _, _) = seen
    assert EXTENSION in json.loads(body1)["params"]["capabilities"]["extensions"]
    assert int(length1) == len(body1)
    assert body2 == b'{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
    assert (m1, m2, m3) == ("POST", "POST", "GET")


def test_the_hardened_client_advertises_only_when_told():
    on = _client_without_redirects(advertise_apps=True)
    off = _client_without_redirects()
    try:
        assert isinstance(on._transport, AdvertisingTransport)
        assert not isinstance(off._transport, AdvertisingTransport)
    finally:
        asyncio.run(on.aclose())
        asyncio.run(off.aclose())
