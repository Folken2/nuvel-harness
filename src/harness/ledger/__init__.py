"""The usage ledger, from the agent's side: every model and compaction call it makes.

``connect`` gives it somewhere to write and somewhere to price from (``stores.py``);
with nothing connected, nothing is recorded and runs go on as before. ``sweeping``
runs the stale-work sweep beside a service, closing work left open
(``stale.py``). Which database, and how long a run may take, are the product's settings:
this service's start-up is ``from_env.py``.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import AsyncIterator, Optional

from .client import MeteredLiteLLMClient
from .context import LedgerContext, RunStart, context_from_state
from .direct_call import one_call_at_a_time
from .entries import RequestedModel, settled_entry
from .registry import COMPACTION, IMAGE, Kind, LedgerRegistry, MODEL, ModelIds, REGISTRY, TOOL
from .stale import sweep_forever
from .stores import INVOCATION_KINDS, Prices, Store
from .task_entry import open_task_state
from .writer import LedgerWriter, current_writer, install


def connect(store: Optional[Store], catalogue: Optional[Prices] = None) -> None:
    """Where the ledger's rows go and where prices come from. None: nothing is recorded."""
    install(LedgerWriter(store, catalogue) if store is not None else None)


@contextlib.asynccontextmanager
async def sweeping(ceiling_seconds: float) -> AsyncIterator[None]:
    """The stale-work sweep while the block runs; nothing without a ledger.
    ``ceiling_seconds`` is the longest a run may take: work older than that is stale."""
    writer = current_writer()
    task = (
        asyncio.create_task(
            sweep_forever(writer.store, writer.catalogue, ceiling_seconds=ceiling_seconds)
        )
        if writer is not None
        else None
    )
    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


__all__ = [
    "COMPACTION",
    "connect",
    "context_from_state",
    "direct_call",
    "IMAGE",
    "INVOCATION_KINDS",
    "Kind",
    "LedgerContext",
    "LedgerRegistry",
    "meter",
    "MeteredLiteLLMClient",
    "MODEL",
    "ModelIds",
    "one_call_at_a_time",
    "open_task_state",
    "overage",
    "REGISTRY",
    "RequestedModel",
    "RunStart",
    "settled_entry",
    "sweeping",
    "TOOL",
]
