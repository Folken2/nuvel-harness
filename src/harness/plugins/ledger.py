"""Records each model call of a run in the usage ledger (logic in ``ledger/``)."""

from __future__ import annotations

from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.plugins.base_plugin import BasePlugin

from .. import wire_names
from ..ledger import meter
from ..ledger.context import (
    RunLedger,
    StartReader,
    calls_before,
    current_run_ledger,
    set_run_ledger,
)
from ..ledger.writer import current_writer
from ..runs.usage_frames import current_announcer


class LedgerPlugin(BasePlugin):
    """Starts the run's ledger state and settles each model call from its final response.

    Goes after ``ProviderCostPlugin``, whose reported cost it reads, and before
    ``CacheUsagePlugin``, which returns the response and so ends the chain.
    """

    def __init__(self, read_start: StartReader) -> None:
        """``read_start`` reads the turn's state: whose run this is, the model's rates
        and the model asked for. The state's keys and shape are the product's."""
        super().__init__(name="ledger")
        self._read_start = read_start

    async def before_run_callback(self, *, invocation_context: InvocationContext) -> None:
        # With no writer the state isn't read at all.
        start = (
            self._read_start(invocation_context.session.state)
            if current_writer() is not None
            else None
        )
        if start is None:
            set_run_ledger(None)
            return None
        set_run_ledger(
            RunLedger(
                context=start.context,
                rates=start.rates,
                requested_model=start.requested_model,
                list_rates=start.list_rates,
                next_index=calls_before(
                    invocation_context.session.events, invocation_context.invocation_id
                ),
                announce_recorded=current_announcer(),
            )
        )
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        if llm_response.partial or not (llm_response.usage_metadata or llm_response.error_code):
            return None
        metadata = llm_response.custom_metadata or {}
        meter.finish(
            llm_response,
            reported_cost=metadata.get(wire_names.COST_USD),
            routed_model=metadata.get(wire_names.ROUTED_MODEL),
            response_id=metadata.get(wire_names.RESPONSE_ID),
            web_searches=metadata.get(wire_names.WEB_SEARCHES),
        )
        return None

    async def on_model_error_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest, error: Exception
    ) -> Optional[LlmResponse]:
        meter.attempt_failed(error)
        return None

    async def after_run_callback(self, *, invocation_context: InvocationContext) -> None:
        run, writer = current_run_ledger(), current_writer()
        if run is not None and writer is not None:
            # Post-run compaction still writes after this; it waits for its own writes.
            await writer.drain(run)
        return None
