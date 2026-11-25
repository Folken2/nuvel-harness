"""Logs, for each model call, whether its prompt starts like the last one's (context/prefix_watch.py).

One line per call: the digests of the instruction and the tools, how many of the previous
call's messages are sent again unchanged, and where the first change is. A change other
than ``none`` means the provider's cached prefix was lost from there on. It only reads
the request, after every plugin that changes it, so its place is last among the
before-model callbacks.

The last request of each session is remembered in this process only, a few hundred
sessions at most: a restart, or a session not seen for a while, logs ``first`` again.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.plugins.base_plugin import BasePlugin

from ..context.prefix_watch import Fingerprint, compare, fingerprint

logger = logging.getLogger(__name__)

REMEMBERED_SESSIONS = 512


class PrefixWatchPlugin(BasePlugin):
    def __init__(self) -> None:
        super().__init__(name="prefix_watch")
        self._last: OrderedDict[tuple[str, str, str], Fingerprint] = OrderedDict()

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        session = callback_context.session
        key = (session.app_name, session.user_id, session.id)
        current = fingerprint(llm_request)
        found = compare(self._last.get(key), current)
        self._last[key] = current
        self._last.move_to_end(key)
        while len(self._last) > REMEMBERED_SESSIONS:
            self._last.popitem(last=False)
        logger.info(
            "prompt prefix: run=%s change=%s kept=%d/%d messages=%d instruction=%s tools=%s",
            callback_context.invocation_id,
            found.change,
            found.kept,
            found.before,
            found.now,
            current.instruction,
            current.tools,
        )
        return None
