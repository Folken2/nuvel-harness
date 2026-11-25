"""A provider call a tool makes itself, as one ledger operation (UG-1).

The turn's model calls reach a provider through the LiteLLM client, which opens their
operation (``ledger/client.py``). A built-in ability calls an endpoint that client can't
(an image, a completion with a server tool), so its tool opens the operation here and
settles it with ``meter.finish``, or ``meter.attempt_failed`` when the call fails. The
operation is keyed by the tool call, so a replay writes the same row (UG-3).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from . import meter
from .context import current_run_ledger
from .entries import RequestedModel
from .registry import REGISTRY

__all__ = ["direct_call"]


@asynccontextmanager
async def one_call_at_a_time() -> AsyncIterator[None]:
    """Holds the run's one metered call that isn't the turn's own model call.

    The run's ledger state holds a single open call (``call``, ``open_operation``), and
    ADK runs the tool calls of one response together. Every tool that meters a provider
    call of its own (an ability here, the advisor in ``tools/advisor.py``) takes this,
    or one could settle the other's row. Outside a run with a ledger it holds nothing.
    """
    run = current_run_ledger()
    if run is None:
        yield
        return
    async with run.direct_calls:
        yield


@asynccontextmanager
async def direct_call(kind: str, *, call_id: Optional[str], model: str) -> AsyncIterator[None]:
    """Opens ``kind``'s operation for a call to ``model`` (the router's own id for it) made inside.

    Whatever escapes settles it as failed, a Stop as cancelled. Outside a run with a
    ledger nothing is written. One at a time in a run: the run's ledger state holds a
    single open call, and the tool calls of one response run together.
    """
    if current_run_ledger() is None:
        yield
        return
    async with one_call_at_a_time():
        # Not the turn's model: the row names the one the ability asked for.
        with meter.metering_as(kind, call_id=meter.call_key(call_id), requested=RequestedModel(None, model)):
            meter.new_call()
            meter.begin_attempt(REGISTRY.model_ids.through_router(model))
            try:
                yield
            except BaseException as error:
                meter.attempt_failed(error)
                raise
            finally:
                meter.end_call()
