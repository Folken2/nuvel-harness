"""What the ledger writes to and prices from, as the code that meters a run sees it.

The meters, the writer and the stale sweep (meter.py, writer.py, stale.py) work against
these two shapes. Where the rows are kept and where prices come from is the product's:
this service's are Postgres over the web app's tables (store.py, catalogue.py).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Optional, Protocol, runtime_checkable

from ..billing.pricing import FallbackRates, Rates

# The operation kinds that are calls to a model, which is what gets estimated when one
# is left open (stale.py).
INVOCATION_KINDS = ("model", "image", "embedding", "compaction", "title")


@runtime_checkable
class Store(Protocol):
    """The usage ledger's rows: operations under an interaction."""

    async def start(self, values: dict[str, Any]) -> bool:
        """Writes a started operation; False when its interaction is closed."""
        ...

    async def settle(self, key: str, values: dict[str, Any]) -> Optional[Decimal]:
        """Settles an operation once; its stored cost, or None when there was nothing to settle."""
        ...

    async def stale_operations(self, cutoff: datetime, limit: int) -> list[dict[str, Any]]:
        """Operations still started that began before ``cutoff``, oldest first."""
        ...

    async def close_stale_interactions(self, cutoff: datetime, error_code: str, limit: int) -> int:
        """Closes interactions left open from before ``cutoff``; how many."""
        ...

    async def open_background(self, values: dict[str, Any]) -> Optional[str]:
        """An interaction opened by the agent itself (a background task); its id."""
        ...

    async def close_background(self, interaction_id: str, status: str) -> None: ...

    async def units_used(self, *, sku: str, subject: str, since: datetime) -> Decimal:
        """Units a subject's settled calls of ``sku`` used since ``since``."""
        ...


@runtime_checkable
class Prices(Protocol):
    """What a call costs when the turn brought no rates for it."""

    async def rates(self, litellm_model: str, *, images: bool = False) -> Optional[Rates]:
        """A model's rates; None when it isn't listed."""
        ...

    async def fallback_rates(self) -> FallbackRates:
        """The rates for a call whose own are missing."""
        ...

    async def web_search_rate(self, model: str) -> Optional[Decimal]:
        """The price of one hosted web search on ``model``; None when unlisted."""
        ...

    async def highest_web_search_rate(self) -> Optional[Decimal]:
        """The highest listed price of a hosted web search."""
        ...
