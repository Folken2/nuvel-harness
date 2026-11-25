"""Sends older tool results as previews once the prompt nears its limit (context/old_results.py).

Before each model call: when the run's last prompt reached the trigger and a batch of
older results is worth masking, the mark moves and each result of the batch is saved
whole as a file of the conversation. Then every result up to the mark goes out as its
preview and a note naming that file, at this call and at every later one. Only the
request changes: the session keeps each result as the tool returned it.

A batch whose results can't all be saved isn't masked: nothing the model can't read
back is taken out of a conversation that keeps files. A run that keeps none (a
background task) is masked with a note that says so.

The size history is rolled up at is the run's own (``plugins/context_budget.py``), read
from the invocation; ``fallback_threshold`` stands in when compaction is off.
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.plugins.base_plugin import BasePlugin

from ..context.old_results import MARK_KEY, MaskSettings, Result, masked, next_mark, results_of, to_mask
from ..files.conversation_files import keeps_files
from ..files.tool_result_copy import keep_masked, masked_copy_name

logger = logging.getLogger(__name__)


def _opening(result: Result) -> str:
    return (
        "[Earlier tool result shortened to keep the conversation within the model's "
        f"context: it was {len(result.text):,} characters, and its start is shown."
    )


class OldResultsPlugin(BasePlugin):
    """``reader`` is the tool that reads a saved copy a chunk at a time; its own results
    are masked without a copy, since the file they read is still there. ``never`` names
    tools whose results always stay whole."""

    def __init__(
        self,
        settings: MaskSettings,
        reader: str,
        fallback_threshold: int,
        never: Collection[str] = (),
    ) -> None:
        super().__init__(name="old_results")
        self._settings = settings
        self._reader = reader
        self._fallback_threshold = fallback_threshold
        self._never = frozenset(never)

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        results = results_of(llm_request.contents, self._never)
        if not results:
            return None
        state = callback_context.state
        mark = state.get(MARK_KEY)
        keeps = keeps_files(state)
        moved = next_mark(
            results, mark, self._last_prompt_tokens(callback_context), self._threshold(callback_context), self._settings
        )
        if moved is not None:
            batch = to_mask(results, moved, after=mark)
            if not keeps or await self._all_saved(callback_context, batch):
                state[MARK_KEY] = mark = moved
                logger.info(
                    "run %s: masked %d older tool results, about %s tokens",
                    callback_context.invocation_id,
                    len(batch),
                    f"{sum(result.saving for result in batch):,}",
                )
        going = to_mask(results, mark)
        if going:
            llm_request.contents = masked(llm_request.contents, going, lambda result: self._note(result, keeps))
        return None

    async def _all_saved(self, context: CallbackContext, batch: list[Result]) -> bool:
        for result in batch:
            if result.name == self._reader:
                continue
            if await keep_masked(context, result.name, result.key, result.text) is None:
                return False
        return True

    def _note(self, result: Result, keeps: bool) -> str:
        if result.name == self._reader:
            return f"{_opening(result)} It was a read of a saved file: read it again with {self._reader} if you need it.]"
        if keeps:
            name = masked_copy_name(result.name, result.key)
            return (
                f"{_opening(result)} The whole result is saved in this conversation as {name!r}. "
                f"To read it, call {self._reader} with name {name!r}, with start=0 or with query to "
                "search it. Read the saved copy rather than calling the tool again.]"
            )
        return (
            f"{_opening(result)} The rest wasn't kept. Work from what is shown; if you need what was "
            "cut, call the tool again only when doing so changes nothing (a search, a read).]"
        )

    def _threshold(self, context: CallbackContext) -> int:
        config = context.get_invocation_context().events_compaction_config
        return (config.token_threshold if config else None) or self._fallback_threshold

    @staticmethod
    def _last_prompt_tokens(context: CallbackContext) -> Optional[int]:
        """What the session's last model call was sent, as the provider counted it. A
        summary's own call doesn't count: its prompt isn't the conversation's."""
        for event in reversed(context.session.events):
            usage = event.usage_metadata
            if event.partial or (event.actions and event.actions.compaction):
                continue
            if usage and usage.prompt_token_count:
                return usage.prompt_token_count
        return None
