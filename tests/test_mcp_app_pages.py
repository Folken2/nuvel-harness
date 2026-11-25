"""MCP Apps: the page a server attaches to a call is fetched on the call's own
connection, kept once with the conversation, and named on the call's event. Against a
real local MCP App server (tests/mcp_app_server.py); a page never costs the call."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from types import SimpleNamespace

import pytest
import uvicorn
from google.adk.events import Event
from google.adk.events.ui_widget import UiWidget
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from google.genai import types

from harness import wire_names
from harness.plugins.mcp_app_pages import McpAppPagesPlugin
from harness.tools.mcp_apps.limits import PageLimits
from harness.tools.mcp_apps.page import Page, PageSource, fetch_page
from harness.tools.user_tools.mcp_limits import McpLimits
from harness.tools.user_tools.mcp_toolsets import build_mcp_toolset
from harness.tools.user_tools.unavailable import UnavailableTools

import mcp_app_server

LIMITS = PageLimits(fetch_timeout_seconds=5, max_page_bytes=200_000)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server_url():
    port = _free_port()
    app = mcp_app_server.build(port).streamable_http_app()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/mcp"
    server.should_exit = True
    thread.join(timeout=5)


def _toolset(url: str):
    return build_mcp_toolset(
        {"slug": "pipeline", "url": url},
        UnavailableTools(),
        McpLimits(attach_timeout_seconds=5, call_timeout_seconds=10, tool_list_ttl_seconds=1, cache_limit=2, allow_local_servers=True),
    )


def test_the_model_gets_the_tool_with_a_page_and_never_the_pages_own(server_url):
    async def scenario():
        toolset = _toolset(server_url)
        try:
            tools = await toolset.get_tools_with_prefix()
        finally:
            await toolset.close()
        assert [tool.name for tool in tools] == ["mcp_pipeline__show_pipeline"]
        source = tools[0].page_source
        assert isinstance(source, PageSource) and source.server == "pipeline"
        return source

    asyncio.run(scenario())


def test_the_page_is_read_on_the_calls_own_connection(server_url):
    async def scenario():
        toolset = _toolset(server_url)
        source = PageSource(connection=toolset.connection_params, server="pipeline")
        page = await fetch_page(source, mcp_app_server.PAGE_URI, LIMITS)
        await toolset.close()
        return page

    page = asyncio.run(scenario())
    assert isinstance(page, Page)
    assert "Pipeline board" in page.html and page.prefers_border is True and page.csp == {}


def test_a_missing_slow_or_unreachable_page_is_a_reason_never_an_error(server_url):
    async def scenario():
        toolset = _toolset(server_url)
        source = PageSource(connection=toolset.connection_params, server="pipeline")
        missing = await fetch_page(source, "ui://pipeline/nothing", LIMITS)
        slow = await fetch_page(source, mcp_app_server.PAGE_URI, PageLimits(fetch_timeout_seconds=0.000001))
        await toolset.close()
        gone = PageSource(connection=StreamableHTTPConnectionParams(url=f"http://127.0.0.1:{_free_port()}/mcp", timeout=1), server="x")
        unreachable = await fetch_page(gone, mcp_app_server.PAGE_URI, LIMITS)
        not_ui = await fetch_page(source, "https://example.test/page", LIMITS)
        return missing, slow, unreachable, not_ui

    for reason in asyncio.run(scenario()):
        assert isinstance(reason, str)


class _Context:
    """The parts of ToolContext the plugin reads, with an artifact store in memory."""

    def __init__(self, uri: str | None, call_id: str = "call-1", kept: dict | None = None):
        self.function_call_id = call_id
        self.invocation_id = "inv-1"
        self.state = {}
        self.kept = kept if kept is not None else {}
        self.saves = 0
        widgets = [UiWidget(id=call_id, provider="mcp", payload={"resource_uri": uri})] if uri else []
        self.actions = SimpleNamespace(render_ui_widgets=widgets)

    async def load_artifact(self, name):
        return self.kept.get(name)

    async def save_artifact(self, name, part):
        self.saves += 1
        self.kept[name] = part
        return 0


def _response_event(call_id: str) -> Event:
    part = types.Part(function_response=types.FunctionResponse(id=call_id, name="mcp_pipeline__show_pipeline", response={"ok": True}))
    return Event(author="agent", invocation_id="inv-1", content=types.Content(role="user", parts=[part]))


def test_the_plugin_keeps_the_page_once_and_names_it_on_the_calls_event(server_url, monkeypatch):
    monkeypatch.setattr("harness.tools.mcp_apps.store.keeps_files", lambda state: True)

    async def scenario():
        toolset = _toolset(server_url)
        tools = await toolset.get_tools_with_prefix()
        plugin = McpAppPagesPlugin(LIMITS, shows_pages=lambda server: True)
        context = _Context(mcp_app_server.PAGE_URI)
        await plugin.after_tool_callback(tool=tools[0], tool_args={}, tool_context=context, result={})
        # A replay of the same call keeps nothing new.
        await plugin.after_tool_callback(tool=tools[0], tool_args={}, tool_context=context, result={})
        other = _response_event("call-2")
        await plugin.on_event_callback(invocation_context=SimpleNamespace(invocation_id="inv-1"), event=other)
        event = _response_event("call-1")
        await plugin.on_event_callback(invocation_context=SimpleNamespace(invocation_id="inv-1"), event=event)
        await toolset.close()
        return context, event, other

    context, event, other = asyncio.run(scenario())
    assert context.saves == 1
    [(name, part)] = context.kept.items()
    assert name == "app-pages/call-1.json"
    record = json.loads(part.text)
    assert record["uri"] == mcp_app_server.PAGE_URI and record["server"] == "pipeline" and "Pipeline board" in record["html"]
    assert event.custom_metadata[wire_names.PAGES] == {"call-1": {"name": name, "sha256": record["sha256"]}}
    assert not (other.custom_metadata or {}).get(wire_names.PAGES)


def test_no_page_for_a_server_the_product_leaves_out_a_call_without_a_note_or_a_plain_tool(server_url, monkeypatch):
    monkeypatch.setattr("harness.tools.mcp_apps.store.keeps_files", lambda state: True)

    async def scenario():
        toolset = _toolset(server_url)
        tools = await toolset.get_tools_with_prefix()
        refused = McpAppPagesPlugin(LIMITS, shows_pages=lambda server: False)
        allowed = McpAppPagesPlugin(LIMITS, shows_pages=lambda server: True)
        a, b, c = _Context(mcp_app_server.PAGE_URI), _Context(None), _Context(mcp_app_server.PAGE_URI)
        await refused.after_tool_callback(tool=tools[0], tool_args={}, tool_context=a, result={})
        await allowed.after_tool_callback(tool=tools[0], tool_args={}, tool_context=b, result={})
        await allowed.after_tool_callback(tool=SimpleNamespace(name="plain"), tool_args={}, tool_context=c, result={})
        await toolset.close()
        return a, b, c

    for context in asyncio.run(scenario()):
        assert context.kept == {}


def test_a_failing_store_costs_nothing(server_url, monkeypatch):
    monkeypatch.setattr("harness.tools.mcp_apps.store.keeps_files", lambda state: True)

    async def scenario():
        toolset = _toolset(server_url)
        tools = await toolset.get_tools_with_prefix()
        plugin = McpAppPagesPlugin(LIMITS, shows_pages=lambda server: True)
        context = _Context(mcp_app_server.PAGE_URI)

        async def broken(name, part):
            raise RuntimeError("store down")

        context.save_artifact = broken
        result = await plugin.after_tool_callback(tool=tools[0], tool_args={}, tool_context=context, result={"ok": True})
        event = _response_event("call-1")
        await plugin.on_event_callback(invocation_context=SimpleNamespace(invocation_id="inv-1"), event=event)
        await toolset.close()
        return result, event

    result, event = asyncio.run(scenario())
    # The call's result stands, and its event names no page.
    assert result is None
    assert not (event.custom_metadata or {}).get(wire_names.PAGES)
