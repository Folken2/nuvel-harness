"""The two places the harness depends on how ADK is built inside, pinned so an ADK upgrade
that moves either one fails here rather than quietly in a run."""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import pytest
from google.adk.agents import LlmAgent
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.genai import types

from harness.agent_without_identity import IDENTITY_STEP, LlmAgentWithoutIdentity
from harness.tools.user_tools.mcp_toolsets import MAX_TOOL_NAME, TimedMcpToolset
from harness.tools.user_tools.mount_rules import MountRules
from harness.tools.user_tools.timed_tool import TimedTool
from harness.tools.user_tools.unavailable import UnavailableTools


def _steps(agent: LlmAgent) -> list[str]:
    return [getattr(step, "name", type(step).__name__) for step in agent._llm_flow.request_processors]


def test_adk_still_names_its_identity_step():
    # The override finds the step by this name. If ADK renames it, the line comes back.
    assert IDENTITY_STEP in _steps(LlmAgent(name="plain", model="any-model"))


def test_the_identity_step_is_the_only_one_dropped():
    plain = _steps(LlmAgent(name="plain", model="any-model"))
    without = _steps(LlmAgentWithoutIdentity(name="without", model="any-model"))
    assert without == [step for step in plain if step != IDENTITY_STEP]


class _ServerTool(BaseTool):
    def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
        return types.FunctionDeclaration(name=self.name, description=self.description)


@pytest.fixture
def toolset(monkeypatch: pytest.MonkeyPatch):
    listed = [
        _ServerTool(name="search", description="finds things"),
        _ServerTool(name="x" * MAX_TOOL_NAME, description="fits alone, not with the prefix"),
    ]

    async def server_lists(self: McpToolset, readonly_context: Any = None) -> list[BaseTool]:
        return listed

    # Stands in for the server's tools/list: the test has no MCP server.
    monkeypatch.setattr(McpToolset, "get_tools", server_lists)
    return TimedMcpToolset(
        app="notes",
        unavailable=UnavailableTools(),
        call_timeout_seconds=5,
        rules=MountRules(many_apps=True),
        connection_params=StreamableHTTPConnectionParams(url="https://mcp.example.com/mcp"),
        tool_name_prefix="mcp_notes",
    )


def test_tools_reach_the_model_prefixed_and_timed(toolset: TimedMcpToolset):
    usable = asyncio.run(toolset.usable_tools())
    assert [tool.name for tool in usable.tools] == ["mcp_notes_search"]
    (tool,) = usable.tools
    assert isinstance(tool, TimedTool)
    assert tool._get_declaration().name == "mcp_notes_search"
    # A refusal is kept under the name the model calls.
    assert tool._refused_key == "mcp_notes_search"


def test_a_name_too_long_once_prefixed_is_left_out(toolset: TimedMcpToolset):
    usable = asyncio.run(toolset.usable_tools())
    assert usable.left_out == ("mcp_notes_" + "x" * MAX_TOOL_NAME,)
