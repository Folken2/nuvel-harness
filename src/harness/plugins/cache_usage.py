"""Reports prompt-cache writes to the web app, which bills them at their own rate."""

from __future__ import annotations

from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin

from .. import wire_names


class CacheUsagePlugin(BasePlugin):
    """Copies the cache-write count into the response's custom metadata.

    ADK attaches the count to the usage object as an extra attribute that
    doesn't survive event serialization (cache reads travel in the usage itself).
    """

    def __init__(self) -> None:
        super().__init__(name="cache_usage")

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        usage = llm_response.usage_metadata
        written = getattr(usage, "cache_creation_input_tokens", None) if usage else None
        if not written:
            return None
        llm_response.custom_metadata = {
            **(llm_response.custom_metadata or {}),
            wire_names.CACHE_WRITE_TOKENS: int(written),
        }
        return llm_response
