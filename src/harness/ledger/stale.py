"""Work left ``started`` gets closed.

A crash or a lost connection between an operation's start and its settlement
leaves a row ``started`` for good. Once it is older than the safety ceiling's
time value plus an hour, no run can still own it, so a periodic sweep settles
it as an aborted call: status ``failed``, error code ``ABANDONED``,
the estimate stored when it started (``entries.started_values``) priced at the
model's catalogue rates. Then the interactions left open with nothing pending
under them are closed the same way.

Settling goes through ``LedgerStore.settle``, so the person's buckets move in
the same transaction and a run settling the same row at the same time wins or
loses cleanly: whichever settles first, the other changes nothing.

``abandoned_values`` is pure; ``sweep`` and ``sweep_forever`` do the I/O.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional

from ..billing.pricing import (
    CUT_OFF,
    NO_ESTIMATE,
    FallbackRates,
    Rates,
    estimate_tokens,
    price_aborted,
)
from .entries import IMAGE, settled_values
from .stores import INVOCATION_KINDS, Prices, Store

logger = logging.getLogger(__name__)

ABANDONED = "ABANDONED"
# Stale after the ceiling's time value plus this, so a long run is never closed while alive.
STALE_MARGIN_SECONDS = 3600
SWEEP_INTERVAL_SECONDS = 600
BATCH_SIZE = 500


def stale_cutoff(now: datetime, ceiling_seconds: float) -> datetime:
    """Rows started before this are abandoned."""
    return now - timedelta(seconds=max(0.0, ceiling_seconds) + STALE_MARGIN_SECONDS)


def _stored_estimate(details: Mapping[str, Any]) -> tuple[int, int]:
    """The estimate written at the start, else the default estimate for a cut-off call."""
    stored = details.get("estimate") if isinstance(details, Mapping) else None
    if isinstance(stored, Mapping):
        return estimate_tokens(
            prompt_estimate=stored.get("input_tokens"), max_output=stored.get("output_tokens")
        )
    return estimate_tokens()


def abandoned_values(
    row: Mapping[str, Any], rates: Optional[Rates], fallback: FallbackRates
) -> dict[str, Any]:
    """What settling an abandoned operation writes: it is closed as an aborted call.

    An invocation counts as contacted and is charged its stored estimate,
    ``estimated_on_abort``. A tool has nothing to estimate: cost 0, source
    ``unknown``, pricing missing, so it shows as unresolved rather than free.
    """
    details = row.get("details") or {}
    if row.get("operation_kind") in INVOCATION_KINDS:
        priced = price_aborted(
            outcome=CUT_OFF, rates=rates, fallback=fallback, estimate=_stored_estimate(details)
        )
    else:
        priced = price_aborted(outcome=NO_ESTIMATE, rates=None, fallback=fallback)
    values = settled_values(
        priced=priced,
        status="failed",
        usage=None,
        requested_model=row.get("model_name"),
        error_code=ABANDONED,
    )
    # Keep the row's own identity: a model it never reported stays what was asked for.
    values["provider"] = row.get("provider") or values["provider"]
    values["model_name"] = row.get("model_name") or values["model_name"]
    values.pop("duration_ms", None)
    return values


@dataclass(frozen=True)
class SweepResult:
    operations: int
    interactions: int


async def sweep(
    store: Store, catalogue: Prices, *, ceiling_seconds: float, now: Optional[datetime] = None
) -> SweepResult:
    """One pass: settle the abandoned operations, then close their interactions."""
    cutoff = stale_cutoff(now or datetime.now(timezone.utc), ceiling_seconds)
    settled = 0
    fallback = await catalogue.fallback_rates()
    for row in await store.stale_operations(cutoff, BATCH_SIZE):
        try:
            model = row.get("model_name")
            images = row.get("operation_kind") == IMAGE
            rates = await catalogue.rates(model, images=images) if model else None
            if await store.settle(row["idempotency_key"], abandoned_values(row, rates, fallback)) is not None:
                settled += 1
        except Exception:  # noqa: BLE001 - one bad row mustn't keep the rest open
            logger.exception("ledger: can't close abandoned operation %s", row.get("idempotency_key"))
    closed = await store.close_stale_interactions(cutoff, ABANDONED, BATCH_SIZE)
    if settled or closed:
        logger.info("ledger: closed %d abandoned operations and %d interactions", settled, closed)
    return SweepResult(settled, closed)


async def sweep_forever(
    store: Store, catalogue: Prices, *, ceiling_seconds: float, interval: float = SWEEP_INTERVAL_SECONDS
) -> None:
    """The periodic job, until cancelled. A failed pass is logged and the next one tries again."""
    while True:
        try:
            await sweep(store, catalogue, ceiling_seconds=ceiling_seconds)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the sweep never stops the service
            logger.exception("ledger: the stale-work sweep failed")
        await asyncio.sleep(interval)
