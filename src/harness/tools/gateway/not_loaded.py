"""The answer to a call of a deferred tool that isn't in this step's tools.

ADK finds no tool by that name, lets the before-tool callbacks answer first, and would
otherwise reply with every tool name it has. This says what to do instead, and comes
before the approval gate, so nobody is asked about a call that can't run.
"""

from __future__ import annotations

from typing import Any, Optional

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from .deferred import DeferredTools, loaded_names


def not_loaded_answer(deferred: DeferredTools, tool: BaseTool, tool_context: ToolContext) -> Optional[dict[str, Any]]:
    """None unless the call is to a deferred tool this step doesn't have."""
    held = deferred.of(tool_context.invocation_id).get(tool.name)
    if held is None or held is tool:
        return None
    if tool.name in loaded_names(tool_context.state):
        # Loaded by a find_tools call in the same step: callable from the next one.
        return {"status": "not_loaded_yet", "note": f"{tool.name} is loaded now. Call it again."}
    return {
        "status": "not_loaded",
        "note": f'{tool.name} isn\'t loaded. Load it with find_tools ("select:{tool.name}"), then call it.',
    }
