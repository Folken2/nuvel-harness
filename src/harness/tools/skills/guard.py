"""A skill tool whose failure is a result the model reads, never the end of the run.

ADK's skill tools answer most problems with an error result already. Some still raise:
a skill whose text reads session state that isn't there, a resource that won't decode.
A tool that raises ends the run unless a plugin answers for it, so each skill tool is
wrapped: anything it raises comes back as ``SKILL_ERROR`` and the model decides what
to do. A cancellation (a Stop) still goes through.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from google.adk.models import LlmRequest
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

logger = logging.getLogger(__name__)

ERROR_CODE = "SKILL_ERROR"
_MAX_DETAIL = 300


def failed(tool_name: str, error: Exception) -> dict[str, str]:
    detail = str(error)
    if len(detail) > _MAX_DETAIL:
        detail = detail[:_MAX_DETAIL] + "..."
    return {
        "error": (
            f"{tool_name} failed: {type(error).__name__}: {detail}. The skill may be broken; "
            "tell the person and carry on without it."
        ),
        "error_code": ERROR_CODE,
    }


class GuardedSkillTool(BaseTool):
    def __init__(self, tool: BaseTool) -> None:
        super().__init__(
            name=tool.name,
            description=tool.description,
            is_long_running=tool.is_long_running,
            custom_metadata=tool.custom_metadata,
        )
        self._tool = tool

    def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
        return self._tool._get_declaration()

    async def process_llm_request(self, *, tool_context: ToolContext, llm_request: LlmRequest) -> None:
        # The tool adds itself (and, for a resource, what it read); calls come here.
        await self._tool.process_llm_request(tool_context=tool_context, llm_request=llm_request)
        llm_request.tools_dict[self.name] = self

    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        try:
            return await self._tool.run_async(args=args, tool_context=tool_context)
        except Exception as error:  # noqa: BLE001 - the model gets it as a result
            logger.exception("skill tool %s raised", self.name)
            return failed(self.name, error)
