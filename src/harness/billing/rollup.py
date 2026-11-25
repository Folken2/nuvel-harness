"""How sure each ledger amount is, and an interaction's totals from its operations.

Pure, so the ledger writer and the reports agree on one set of labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable

from .pricing import (
    ESTIMATED_ON_ABORT,
    FALLBACK_PRICING,
    FREE,
    PROVIDER_REPORTED,
    UNKNOWN,
    ZERO,
    money,
)

MIXED = "mixed"

# Cost quality.
UNPRICED = "unpriced"
ESTIMATED = "estimated"
CALCULATED = "calculated"


@dataclass(frozen=True)
class CostSplit:
    """The four parts of an entry's cost: provider reported, calculated, estimated and
    unresolved. They add up to the entry's cost, none negative."""

    provider: Decimal = ZERO
    calculated: Decimal = ZERO
    estimated: Decimal = ZERO
    unresolved: Decimal = ZERO

    def total(self) -> Decimal:
        return self.provider + self.calculated + self.estimated + self.unresolved

    def __add__(self, other: "CostSplit") -> "CostSplit":
        return CostSplit(
            provider=self.provider + other.provider,
            calculated=self.calculated + other.calculated,
            estimated=self.estimated + other.estimated,
            unresolved=self.unresolved + other.unresolved,
        )


@dataclass(frozen=True)
class Settled:
    """What a settled operation contributes to its interaction."""

    cost_usd: Decimal
    source: str
    pricing_missing: bool = False
    status: str = "completed"
    provider_contacted: bool = True


def quality(cost_usd: Decimal, source: str, pricing_missing: bool) -> str:
    """The simpler label: how sure the amount is (its cost quality)."""
    if source == PROVIDER_REPORTED:
        return PROVIDER_REPORTED
    if source == ESTIMATED_ON_ABORT:
        return ESTIMATED
    if pricing_missing:
        return ESTIMATED if cost_usd > 0 else UNPRICED
    return CALCULATED


def is_unresolved(entry: Settled) -> bool:
    """Whether the entry counts among the unresolved ones."""
    if entry.source == PROVIDER_REPORTED:
        return False
    return (
        entry.source in (UNKNOWN, FALLBACK_PRICING)
        or entry.pricing_missing
        or (entry.source == FREE and entry.status == "partial" and entry.provider_contacted)
    )


def split(entry: Settled) -> CostSplit:
    """Which of the four parts the operation's cost belongs to."""
    cost = money(entry.cost_usd)
    if entry.source == PROVIDER_REPORTED:
        return CostSplit(provider=cost)
    if entry.source == ESTIMATED_ON_ABORT:
        return CostSplit(estimated=cost)
    if is_unresolved(entry):
        return CostSplit(unresolved=cost)
    return CostSplit(calculated=cost)


def _paid_sources(entries: list[Settled]) -> set[str]:
    return {e.source for e in entries if e.cost_usd > 0}


def interaction_source(entries: Iterable[Settled]) -> str:
    """The interaction's cost source from its settled operations."""
    entries = list(entries)
    cost = sum((e.cost_usd for e in entries), ZERO)
    paid = _paid_sources(entries)
    for unresolved in (FALLBACK_PRICING, UNKNOWN):
        if any(e.source == unresolved for e in entries):
            others = paid - {unresolved}
            return MIXED if others else unresolved
    if cost > 0:
        if len(paid) > 1 or MIXED in paid:
            return MIXED
        return next(iter(paid))
    if any(e.source == PROVIDER_REPORTED for e in entries):
        return PROVIDER_REPORTED
    return FREE


@dataclass(frozen=True)
class Totals:
    cost_usd: Decimal
    source: str
    split: CostSplit
    pricing_missing: bool
    unresolved_entries: int


def totals(entries: Iterable[Settled]) -> Totals:
    """An interaction's cost, source and split: the sum of its leaves."""
    entries = list(entries)
    parts = sum((split(e) for e in entries), CostSplit())
    return Totals(
        cost_usd=money(sum((e.cost_usd for e in entries), ZERO)),
        source=interaction_source(entries),
        split=parts,
        pricing_missing=any(e.pricing_missing for e in entries),
        unresolved_entries=sum(1 for e in entries if is_unresolved(e)),
    )
