"""Before each model call, the messages the person sent to the running turn go in.

The logic is in runs/steering.py. This history was read from the session just before
the callback, so a message written now goes in both: the session, for the steps and
turns after, and this call's request.

After the limits plugin, so a call it refuses doesn't use the message up.
"""

from __future__ import annotations

from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.plugins.base_plugin import BasePlugin

from ..runs.steering import current_inbox, write_steers


class SteeringPlugin(BasePlugin):
    def __init__(self) -> None:
        super().__init__(name="steering")

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        inbox = current_inbox()
        if inbox is None:
            return None
        # The run's own session, the object the runner writes to, and its store. The
        # context ADK hands back is a copy, but of the same session and the same store.
        llm_request.contents.extend(
            await write_steers(
                inbox,
                callback_context.session,
                callback_context.get_invocation_context().session_service,
                callback_context.invocation_id,
            )
        )
        return None
