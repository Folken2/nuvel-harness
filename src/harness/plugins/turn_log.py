"""One log line per tool call and per model failure, at its real level.

Names, durations and error types only: a tool's arguments and results are the
person's data, and this service serves many people.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

from google.adk.plugins.base_plugin import BasePlugin

logger = logging.getLogger(__name__)


def _as_it_is(model: str) -> str:
    return model


class TurnLogPlugin(BasePlugin):
    def __init__(self, model_name: Callable[[str], str] = _as_it_is) -> None:
        """``model_name`` is how a failed call's model id is written in the log: the
        product may take a router's mark off it."""
        super().__init__(name="turn_log")
        self._model_name = model_name
        self._started: dict[str, float] = {}

    def open_calls(self) -> int:
        return len(self._started)

    def _elapsed_ms(self, tool_context: Any) -> int:
        started = self._started.pop(tool_context.function_call_id, None)
        return int((time.monotonic() - started) * 1000) if started is not None else 0

    async def before_tool_callback(
        self, *, tool: Any, tool_args: dict[str, Any], tool_context: Any
    ) -> Optional[dict]:
        self._started[tool_context.function_call_id] = time.monotonic()
        return None

    async def after_tool_callback(
        self, *, tool: Any, tool_args: dict[str, Any], tool_context: Any, result: dict
    ) -> Optional[dict]:
        logger.info("tool done: %s %dms", tool.name, self._elapsed_ms(tool_context))
        return None

    async def on_tool_error_callback(
        self, *, tool: Any, tool_args: dict[str, Any], tool_context: Any, error: Exception
    ) -> Optional[dict]:
        logger.error(
            "tool failed: %s %dms %s",
            tool.name,
            self._elapsed_ms(tool_context),
            type(error).__name__,
        )
        return None

    async def on_model_error_callback(
        self, *, callback_context: Any, llm_request: Any, error: Exception
    ) -> Optional[Any]:
        model = self._model_name(llm_request.model or "unknown")
        logger.error("model call failed: %s %s", model, type(error).__name__)
        return None
