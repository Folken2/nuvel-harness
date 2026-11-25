"""One log line per turn on where time to first token goes.

``setup_ms`` is run start to the first model request (session load, tool
attach, request build); ``model_first_token_ms`` is that request to the first
streamed text. Cached prompt tokens show whether prompt caching hit.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.models import LlmRequest
from google.adk.plugins.base_plugin import BasePlugin

logger = logging.getLogger(__name__)


def _tool_sizes(llm_request: Any) -> list[tuple[str, int]]:
    """(name, schema size in characters) of each tool sent to the model."""
    config = getattr(llm_request, "config", None)
    sizes = []
    for tool in (getattr(config, "tools", None) or []):
        for declaration in getattr(tool, "function_declarations", None) or []:
            sizes.append(
                (declaration.name, len(declaration.model_dump_json(exclude_none=True)))
            )
    return sorted(sizes, key=lambda item: -item[1])


class TurnTimingPlugin(BasePlugin):
    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        super().__init__(name="turn_timing")
        self._clock = clock
        self._turns: dict[str, dict[str, Any]] = {}

    def _turn(self, invocation_id: str) -> dict[str, Any]:
        # A run that is cut off never reaches after_run: don't let them pile up.
        if invocation_id not in self._turns and len(self._turns) >= 256:
            self._turns.clear()
        return self._turns.setdefault(
            invocation_id,
            {"start": self._clock(), "model_calls": 0, "cached": 0, "prompt": 0},
        )

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        self._turn(invocation_context.invocation_id)
        return None

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        turn = self._turn(callback_context.invocation_id)
        turn["model_calls"] += 1
        if "first_request" not in turn:
            turn["first_request"] = self._clock()
            turn["tools"] = _tool_sizes(llm_request)
        return None

    async def on_event_callback(
        self, *, invocation_context: InvocationContext, event: Event
    ) -> Optional[Event]:
        turn = self._turn(invocation_context.invocation_id)
        parts = event.content.parts if event.content and event.content.parts else []
        if "first_text" not in turn and any(part.text for part in parts):
            turn["first_text"] = self._clock()
        usage = event.usage_metadata
        if usage and not event.partial:
            turn["prompt"] += usage.prompt_token_count or 0
            turn["cached"] += usage.cached_content_token_count or 0
        return None

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        turn = self._turns.pop(invocation_context.invocation_id, None)
        if not turn:
            return None

        def ms(start: str, end: str) -> Optional[int]:
            if start not in turn or end not in turn:
                return None
            return round((turn[end] - turn[start]) * 1000)

        turn["end"] = self._clock()
        logger.info(
            "turn timing: setup_ms=%s model_first_token_ms=%s first_token_ms=%s"
            " total_ms=%s model_calls=%d prompt_tokens=%d cached_tokens=%d",
            ms("start", "first_request"),
            ms("first_request", "first_text"),
            ms("start", "first_text"),
            ms("start", "end"),
            turn["model_calls"],
            turn["prompt"],
            turn["cached"],
        )
        tools = turn.get("tools") or []
        if tools:
            logger.info(
                "turn tools: count=%d schema_chars=%d by_size=%s",
                len(tools),
                sum(size for _, size in tools),
                ", ".join(f"{name}:{size}" for name, size in tools),
            )
        return None
