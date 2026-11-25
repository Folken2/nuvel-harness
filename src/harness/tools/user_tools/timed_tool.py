"""A connected app's tool whose calls can't outlast the call limit."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from .mount_rules import DEFAULT_RULES, MountRules
from .unavailable import UnavailableTools


if TYPE_CHECKING:
    from ..mcp_apps.page import PageSource

logger = logging.getLogger(__name__)


def _no_answer(name: str, refused: str, seconds: float) -> dict[str, Any]:
    """A failed MCP result, which the web app shows as a failed tool."""
    # The person reads this, and the model repeats it.
    text = (
        f"A call to {name} got no answer within {seconds:.0f} seconds, "
        f"so {refused} won't be called again this turn. That call may still have gone "
        "through on its side, so don't assume either way. Say so to the person "
        "and carry on with what you can."
    )
    return {"isError": True, "content": [{"type": "text", "text": text}]}


class CallsInFlight:
    """How many of a toolset's tool calls are running, so it isn't closed under one."""

    def __init__(self) -> None:
        self.count = 0

    @contextmanager
    def running(self) -> Iterator[None]:
        self.count += 1
        try:
            yield
        finally:
            self.count -= 1


class TimedTool(BaseTool):
    """Runs a tool under the call limit; past it, the app is refused for the turn.

    The app's tools stay listed after a timeout, so a model that calls one again
    gets the same answer at once rather than a missing tool or another full wait.
    A mount that carries many apps (mount_rules.py) has only the tool refused.
    """

    def __init__(
        self,
        tool: BaseTool,
        app: str,
        unavailable: UnavailableTools,
        calls: CallsInFlight | None = None,
        rules: MountRules = DEFAULT_RULES,
        *,
        call_timeout_seconds: float,
        page_source: Optional["PageSource"] = None,
    ) -> None:
        super().__init__(
            name=tool.name,
            description=tool.description,
            is_long_running=tool.is_long_running,
            custom_metadata=tool.custom_metadata,
        )
        self._tool = tool
        self._app = app
        self._unavailable = unavailable
        self._calls = calls if calls is not None else CallsInFlight()
        # Past this, a call gets a failed result and the turn goes on.
        self._call_timeout_seconds = call_timeout_seconds
        self._shown_as = rules.shown_as or app
        self._refused_key = tool.name if rules.many_apps else app
        self._refused_words = "that tool" if rules.many_apps else "it"
        # Where a page this tool's server attaches can be read (MCP Apps); None for none.
        self.page_source = page_source

    @property
    def effect_hints(self) -> frozenset[str]:
        """The MCP hints the app's server gives this tool that say what it does
        (`readOnlyHint`, `destructiveHint`), for asking before it runs (approval/)."""
        raw = getattr(self._tool, "raw_mcp_tool", None)
        annotations = getattr(raw, "annotations", None)
        return frozenset(
            hint
            for hint in ("readOnlyHint", "destructiveHint")
            if getattr(annotations, hint, None) is True
        )

    def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
        return self._tool._get_declaration()

    def _no_answer(self) -> dict[str, Any]:
        return _no_answer(self._shown_as, self._refused_words, self._call_timeout_seconds)

    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> Any:
        invocation_id = tool_context.invocation_id
        if self._unavailable.left_unanswered(invocation_id, self._refused_key):
            return self._no_answer()
        try:
            with self._calls.running():
                return await asyncio.wait_for(
                    self._tool.run_async(args=args, tool_context=tool_context),
                    timeout=self._call_timeout_seconds,
                )
        except TimeoutError:
            logger.warning("mcp tool call timed out: %s", self.name)
            self._unavailable.add_unanswered(
                invocation_id, self._refused_key, self._app
            )
            return self._no_answer()
