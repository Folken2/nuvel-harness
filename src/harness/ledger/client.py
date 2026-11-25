"""The LiteLLM client every provider request goes through, so none skips the ledger.

Each request is one attempt: its operation is
written as started right before it goes out, and settled here when it raises,
a streamed one's first chunk included (``first_chunk.py``).
A request that returns is settled from its final response (``LedgerPlugin``, or
the compaction summarizer). ``routing/client.py`` builds on this one.

What a provider reports about a whole answer (its cost, the model that ran) is read by
a function handed in: how it is reported is the provider's business, and reading it is
the product's (``billing/litellm_hook.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import litellm
from google.adk.models.lite_llm import LiteLLMClient

from . import meter
from .first_chunk import primed


class MeteredLiteLLMClient(LiteLLMClient):
    def __init__(self, on_whole_answer: Optional[Callable[[Any], None]] = None) -> None:
        """``on_whole_answer`` gets each answer that came whole, not streamed, so what
        the provider reported on it can be read. None: nothing is read from it."""
        super().__init__()
        self._on_whole_answer = on_whole_answer

    async def acompletion(self, model: Any, messages: Any, tools: Any, **kwargs: Any) -> Any:
        meter.begin_attempt(model)
        try:
            response = await super().acompletion(model, messages, tools, **kwargs)
            if kwargs.get("stream"):
                # A lazy stream's refusal comes with its first chunk: still this attempt.
                response = await primed(response)
        except litellm.UnsupportedParamsError as error:
            # LiteLLM refused it before anything was sent: nothing to pay.
            meter.attempt_failed(error, contacted=False)
            raise
        except BaseException as error:
            meter.attempt_failed(error)
            raise
        if self._on_whole_answer is not None and not kwargs.get("stream"):
            # A whole answer carries its cost; a streamed one, on its last chunk.
            self._on_whole_answer(response)
        return response
