"""MCP Apps: a tool a server marks for its page only never reaches the model,
and a marking a server got wrong never hides a tool.

The MCP Apps spec says a host MUST keep a tool whose ``_meta.ui.visibility`` leaves out
"model" off the model's tool list: such a tool exists for the server's own page to call.
"""

import asyncio

from google.adk.tools.mcp_tool.mcp_tool import McpTool
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from mcp import types

from harness.tools.mcp_apps.visibility import for_the_model
from harness.tools.user_tools.mcp_limits import McpLimits
from harness.tools.user_tools.mcp_toolsets import build_mcp_toolset
from harness.tools.user_tools.unavailable import UnavailableTools


def _tool(name: str, meta: dict | None = None) -> McpTool:
    definition = types.Tool(name=name, description=name, inputSchema={"type": "object"}, _meta=meta)
    return McpTool(mcp_tool=definition, mcp_session_manager=object())  # type: ignore[arg-type]


def _offered(monkeypatch, listed: list[McpTool]) -> list[str]:
    async def listing(self, readonly_context=None):
        return listed

    # Stands in for the server's tools/list.
    monkeypatch.setattr(McpToolset, "get_tools", listing)
    toolset = build_mcp_toolset(
        {"slug": "crm", "url": "https://crm.example.test/mcp"}, UnavailableTools(),
        McpLimits(attach_timeout_seconds=5, call_timeout_seconds=30, tool_list_ttl_seconds=60, cache_limit=4),
    )
    offered = asyncio.run(toolset.get_tools_with_prefix())
    return [tool.name.removeprefix(f"{toolset.tool_name_prefix}_") for tool in offered]


def test_a_tool_for_the_page_only_is_left_out_and_every_other_tool_stays(monkeypatch):
    listed = [
        _tool("show_pipeline", {"ui": {"resourceUri": "ui://crm/pipeline", "visibility": ["model", "app"]}}),
        _tool("move_deal", {"ui": {"visibility": ["app"]}}),
        _tool("find_deal", {"ui": {"visibility": ["model"]}}),
        _tool("plain"),
        _tool("default_both", {"ui": {"resourceUri": "ui://crm/deal"}}),
    ]
    assert _offered(monkeypatch, listed) == ["show_pipeline", "find_deal", "plain", "default_both"]


class _Marked:
    def __init__(self, visibility):
        self._visibility = visibility

    @property
    def visibility(self):
        if isinstance(self._visibility, Exception):
            raise self._visibility
        return self._visibility


def test_only_a_list_naming_app_and_not_model_holds_a_tool_back():
    assert for_the_model(_Marked(["app"])) is False
    assert for_the_model(_Marked(("app",))) is False
    assert for_the_model(_Marked(["app", "model"])) is True
    assert for_the_model(_Marked(["model"])) is True


def test_a_marking_a_server_got_wrong_leaves_the_tool_with_the_model():
    for wrong in ("app", "models", {"app": True}, [], [1, None], ["App"], ["application"], None, RuntimeError("bad meta")):
        assert for_the_model(_Marked(wrong)) is True, wrong
    assert for_the_model(object()) is True
