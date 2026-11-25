"""Tool price rules: what one tool call costs (usage-and-analytics.md UG-26). Pure.

A rule meters a sku for a billing subject (the person, or the platform) over a
daily or monthly period. A call's cost is the increase in the subject's bill it
causes: billable units are the units used so far in the period, minus the free
allowance, never below 0.

The only rule this product can reach today is web search through the router,
priced per call at the rate the catalogue lists for the model that ran it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from .pricing import ZERO, money

PER_CALL = "per_call"
PER_UNIT_PRORATED = "per_unit_prorated"
PER_UNIT_CEIL = "per_unit_ceil"
METERS = (PER_CALL, PER_UNIT_PRORATED, PER_UNIT_CEIL)

DAILY = "daily"
MONTHLY = "monthly"

TOOL_PRICING_RULE = "tool_pricing_rule"


@dataclass(frozen=True)
class ToolRule:
    sku: str
    rate_usd: Decimal
    meter: str = PER_CALL
    period: str = MONTHLY
    unit_size: Decimal = Decimal(1)
    free_allowance: Decimal = ZERO
    effective_from: Optional[datetime] = None
    effective_to: Optional[datetime] = None

    def applies(self, at: datetime) -> bool:
        """Inside its effective dates: from inclusive, to exclusive."""
        at = _utc(at)
        if self.effective_from is not None and at < _utc(self.effective_from):
            return False
        return self.effective_to is None or at < _utc(self.effective_to)

    def needs_history(self) -> bool:
        """Whether a call's cost depends on the units already used this period."""
        return self.free_allowance > 0 or self.meter == PER_UNIT_CEIL


@dataclass(frozen=True)
class ToolCharge:
    cost_usd: Decimal
    # Section 2, "settlement": the formula and the units before and after.
    settlement: dict[str, Any]


def _utc(at: datetime) -> datetime:
    return at.astimezone(timezone.utc) if at.tzinfo else at.replace(tzinfo=timezone.utc)


def period_start(rule: ToolRule, at: datetime) -> datetime:
    """Where the subject's counter restarts: the UTC day, or the UTC month."""
    day = _utc(at).replace(hour=0, minute=0, second=0, microsecond=0)
    return day if rule.period == DAILY else day.replace(day=1)


def _billable(rule: ToolRule, used: Decimal) -> Decimal:
    return max(ZERO, used - rule.free_allowance)


def charge(rule: ToolRule, *, units_before: Decimal, units: Decimal = Decimal(1)) -> ToolCharge:
    """The cost of ``units`` more for a subject that used ``units_before`` this period."""
    before = _billable(rule, units_before)
    after = _billable(rule, units_before + units)
    if rule.meter == PER_UNIT_CEIL:
        size = rule.unit_size if rule.unit_size > 0 else Decimal(1)
        blocks = Decimal(math.ceil(after / size)) - Decimal(math.ceil(before / size))
        cost = blocks * rule.rate_usd
    elif rule.meter == PER_UNIT_PRORATED:
        size = rule.unit_size if rule.unit_size > 0 else Decimal(1)
        cost = (after - before) * rule.rate_usd / size
    else:
        cost = (after - before) * rule.rate_usd
    return ToolCharge(
        cost_usd=money(cost),
        settlement={
            "rule": rule.sku,
            "formula": rule.meter,
            "rate_usd": str(rule.rate_usd),
            "unit_size": str(rule.unit_size),
            "free_allowance": str(rule.free_allowance),
            "period": rule.period,
            "units_before": str(units_before),
            "units_after": str(units_before + units),
        },
    )


def once_per_period_rule(sku: str, rate_usd: Decimal, period: str = MONTHLY) -> ToolRule:
    """Billed once per subject per period, such as a code session: every later call is 0.

    One block so large no subject fills it: the first unit opens it, the rest fall inside.
    """
    return ToolRule(sku=sku, rate_usd=rate_usd, meter=PER_UNIT_CEIL, period=period, unit_size=Decimal(10) ** 18)


def router_web_search_sku(engine: str) -> str:
    """The sku names the search engine: the model's own (``native``) or the one used."""
    return f"router:web_search:{engine or 'native'}"


# A search on a model with no listed price, when the catalogue lists none at all:
# high, so an unpriced search can't run unbounded (UG-20, as FallbackRates does for tokens).
DEFAULT_WEB_SEARCH_FALLBACK_USD = Decimal("0.05")


def fallback_web_search_rate(highest_listed: Optional[Decimal]) -> Decimal:
    """The conservative per-call rate: the catalogue's highest, else the default."""
    if highest_listed is not None and highest_listed > 0:
        return highest_listed
    return DEFAULT_WEB_SEARCH_FALLBACK_USD


def router_web_search_rule(engine: str, rate_usd: Optional[Decimal]) -> Optional[ToolRule]:
    """Per call at the catalogue's rate for the model; None when the catalogue has none."""
    if rate_usd is None:
        return None
    return ToolRule(sku=router_web_search_sku(engine), rate_usd=rate_usd)
