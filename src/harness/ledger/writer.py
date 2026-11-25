"""Ledger writes that never hold a run up (limits.md LM-23).

Each write runs in its own task with three retries (1, 2 and 4 seconds), and
a failure is logged with the run id, never raised into the run. A settlement
waits for its own start, so the two land in order. The run waits for what it
wrote only when it ends, so the web app closing the turn reads final totals.
"""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Any, Awaitable, Callable, Optional

from .context import RunLedger
from .stores import Prices, Store

logger = logging.getLogger(__name__)

RETRY_DELAYS = (1.0, 2.0, 4.0)
DRAIN_TIMEOUT_SECONDS = 15.0


async def _with_retries(what: str, run_id: str, write: Callable[[], Awaitable[Any]]) -> Any:
    for attempt, delay in enumerate((0.0, *RETRY_DELAYS)):
        if delay:
            await asyncio.sleep(delay)
        try:
            return await write()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - recording must never stop a run
            last = attempt == len(RETRY_DELAYS)
            logger.error(
                "ledger %s failed for run %s (attempt %d)%s",
                what,
                run_id,
                attempt + 1,
                "; giving up" if last else "",
                exc_info=last,
            )
    return None


class LedgerWriter:
    def __init__(self, store: Store, catalogue: Optional[Prices] = None) -> None:
        """Where the rows go and where prices come from (stores.py). ``catalogue`` is
        None only where nothing is priced from it (a test of the writes alone)."""
        self.store = store
        self.catalogue = catalogue

    def start(self, run: RunLedger, key: str, values: dict[str, Any]) -> None:
        task = asyncio.ensure_future(
            _with_retries("start", run.context.run_id, lambda: self.store.start(values))
        )
        run.starts[key] = task
        run.pending.append(task)

    def settle(
        self,
        run: RunLedger,
        key: str,
        values: Callable[[], Awaitable[dict[str, Any]]],
        on_recorded: Optional[Callable[[], None]] = None,
        announce_free: bool = True,
    ) -> None:
        """``values`` is built in the task: pricing may read the catalogue.

        ``on_recorded`` is called once the settlement has committed, never when it
        failed, so what it announces is what the ledger holds (limits.md LM-37).
        With ``announce_free`` false it isn't called for an operation that cost
        nothing: a free tool call changes no meter, so there's nothing to read again.
        """
        started = run.starts.pop(key, None)

        async def write() -> Any:
            if started is not None and not await started:
                # The interaction had closed, or the start never landed: nothing to settle.
                return None

            async def once() -> Any:
                return await self.store.settle(key, await values())

            stored = await _with_retries("settle", run.context.run_id, once)
            if stored is not None and on_recorded is not None and (announce_free or Decimal(str(stored)) > 0):
                try:
                    on_recorded()
                except Exception:  # noqa: BLE001 - a meter must never stop a run
                    logger.exception("ledger: couldn't announce a recorded call for run %s", run.context.run_id)
            return stored

        run.pending.append(asyncio.ensure_future(write()))

    async def drain(self, run: RunLedger, timeout: float = DRAIN_TIMEOUT_SECONDS) -> None:
        """Waits for the run's writes, at most ``timeout``; never raises."""
        pending = [t for t in run.pending if not t.done()]
        run.pending.clear()
        if not pending:
            return
        done, late = await asyncio.wait(pending, timeout=timeout)
        if late:
            logger.error("ledger: %d writes still pending for run %s", len(late), run.context.run_id)


_writer: Optional[LedgerWriter] = None


def install(writer: Optional[LedgerWriter]) -> None:
    """Set once at startup (``ledger.connect``); None turns recording off."""
    global _writer
    _writer = writer


def current_writer() -> Optional[LedgerWriter]:
    return _writer
