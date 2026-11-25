"""Gives each run the compaction threshold of the model it is on (context/window.py).

ADK's compaction config holds one threshold for the whole app, but it reads the config
from the invocation, so a run can carry its own copy: before the run starts, this plugin
puts one there with the threshold worked out from the window of the turn's model. ADK
checks it before every model call of the run and once after it, so a turn that moves to
a smaller model is rolled up before its first request.

How a turn says its model's window is the product's: the plugin is handed the reader.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any, Optional

from google.adk.agents.invocation_context import InvocationContext
from google.adk.plugins.base_plugin import BasePlugin

from ..context.window import WindowSettings, threshold_for

logger = logging.getLogger(__name__)


class ContextBudgetPlugin(BasePlugin):
    """Sets the run's own compaction threshold. With compaction off, or no token
    threshold configured, it does nothing."""

    def __init__(
        self,
        window_of_turn: Callable[[Mapping[str, Any]], Optional[int]],
        settings: WindowSettings,
    ) -> None:
        super().__init__(name="context_budget")
        self._window_of_turn = window_of_turn
        self._settings = settings

    async def before_run_callback(self, *, invocation_context: InvocationContext) -> None:
        config = invocation_context.events_compaction_config
        if config is None or config.token_threshold is None:
            return None
        window = self._window_of_turn(invocation_context.session.state)
        threshold = threshold_for(window, self._settings, config.token_threshold)
        if threshold != config.token_threshold:
            # A copy: the app's config is shared by every run.
            invocation_context.events_compaction_config = config.model_copy(
                update={"token_threshold": threshold}
            )
        logger.info(
            "run %s: history is rolled up at %s tokens (window %s)",
            invocation_context.invocation_id,
            f"{threshold:,}",
            f"{window:,}" if window else "unknown",
        )
        return None
