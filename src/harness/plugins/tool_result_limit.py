"""Shapes every tool result before the model sees it.

First the product's own pass makes it smaller, if it gave one (here the connected
account's profile is compacted, `context/user_info.py`), then the result is held to the size limit: its start and its end are kept
(`context/tool_results.py`), and its whole is kept as a file of the chat
(`files/tool_result_copy.py`) that the note names. Both live in this one plugin
because a plugin that returns a value ends the chain for that callback. For the same
reason the assembly (agent.py) puts it last: any plugin after it would miss every
result it changes.

The reader, the tool that reads those copies a chunk at a time, is named by the
assembly (agent.py). Its own results are cut but never kept: keeping them again would
only make another copy to read.

Limit: the product sets it (default 80,000 characters, about 20k
tokens); 0 turns trimming off.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from ..context.tool_results import fits, leaves_out, result_size, result_text, trim_tool_result
from ..files.tool_result_copy import keep_whole

logger = logging.getLogger(__name__)

Compact = Callable[[Any], Optional[dict[str, Any]]]

class ToolResultLimitPlugin(BasePlugin):
    def __init__(self, max_chars: int, reader: str, compact: Optional[Compact] = None) -> None:
        """``compact`` is the product's first pass over a result: the result made smaller,
        or None when it has nothing to take out."""
        super().__init__(name="tool_result_limit")
        self._compact = compact
        self.max_chars = max_chars
        self.reader = reader

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        compacted = self._compact(result) if self._compact else None
        if compacted is not None:
            logger.info(
                "compacted account profile in %s result from %s to %s chars",
                tool.name,
                f"{result_size(result):,}",
                f"{result_size(compacted):,}",
            )
            result = compacted
        if fits(result, self.max_chars):
            return compacted
        saved_as = None
        text = result_text(result)
        if tool.name != self.reader and leaves_out(text, self.max_chars):
            saved_as = await keep_whole(tool_context, tool.name, text)
        trimmed = trim_tool_result(result, self.max_chars, saved_as, reader=self.reader)
        logger.info(
            "trimmed %s result from %s to %s chars, %s",
            tool.name,
            f"{result_size(result):,}",
            f"{self.max_chars:,}",
            f"kept whole as {saved_as}" if saved_as else "not kept whole",
        )
        return trimmed
