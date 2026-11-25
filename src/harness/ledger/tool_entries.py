"""What one tool operation's row holds, and what the tool cost. Pure: no I/O.

A tool with no price rule costs 0, source ``free``. Hosted web search is the
one tool with a rule today: when the model call it ran in reported a cost
covering the whole request, the search is already inside that cost, so the tool
stores 0 and says how much was included. Otherwise the rule prices it.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

from ..billing.pricing import FALLBACK_PRICING, FREE, PROVIDER_REPORTED, ZERO, PricedCall, money
from ..billing.tool_rules import DEFAULT_WEB_SEARCH_FALLBACK_USD, TOOL_PRICING_RULE, ToolRule, charge
from .context import LedgerContext
from .entries import ERROR_MESSAGE_MAX, TOOL, ServedModel, accounting, base_values, operation_key
from .tool_identity import ToolIdentity

# Provider reported cost scope: the amount covers the whole request.
REQUEST_SCOPE = "request"
# The meter named when a priced tool has no rate in the catalogue.
MISSING_TOOL_RATE = "toolRate"


def tool_key(run_id: str, call: str) -> str:
    """One key per tool call, so a replay writes the same row."""
    return operation_key(run_id, TOOL, call)


def hosted_search_key(run_id: str, model_call_key: str) -> str:
    """The searches of one model call: one row, keyed by that call's key."""
    call = model_call_key.removeprefix(f"{run_id}:").replace(":", ".")
    return tool_key(run_id, f"web_search@{call}")


def billing_subject(context: LedgerContext) -> str:
    """Whom a rule meters: the person, or the platform for work nobody caused."""
    return f"user:{context.user_id}" if context.user_id and context.quota_scope == "user" else "platform"


@dataclass(frozen=True)
class ToolPriced:
    cost_usd: Decimal
    source: str
    pricing_missing: bool = False
    missing: tuple[str, ...] = ()
    # Settlement: the rule's formula and the units before and after.
    settlement: Optional[dict[str, Any]] = None
    # Included in provider usage.
    included: Optional[dict[str, Any]] = None
    provider_contacted: bool = False


def price_tool(rule: Optional[ToolRule], *, units_before: Decimal = ZERO, units: Decimal = Decimal(1)) -> ToolPriced:
    """A tool call by its rule; with none, 0 and ``free``."""
    if rule is None:
        return ToolPriced(cost_usd=ZERO, source=FREE)
    priced = charge(rule, units_before=units_before, units=units)
    return ToolPriced(
        cost_usd=priced.cost_usd,
        source=TOOL_PRICING_RULE,
        settlement=priced.settlement,
        provider_contacted=True,
    )


def price_hosted_search(
    *,
    searches: int,
    model_reported: bool,
    rate_usd: Optional[Decimal],
    rule: Optional[ToolRule],
    model_call_key: str,
    response_id: Optional[str] = None,
    units_before: Decimal = ZERO,
    fallback_rate_usd: Decimal = DEFAULT_WEB_SEARCH_FALLBACK_USD,
) -> ToolPriced:
    """Hosted searches: inside the model call's reported cost, else priced by the rule.

    ``model_reported`` is whether every model call of the response reported its
    cost for the whole request; here a response is one model call. With no rule
    (no catalogue rate), the search is charged at ``fallback_rate_usd`` and
    flagged: a missing price is never free.
    """
    if model_reported:
        included: dict[str, Any] = {
            "amount_usd": str(money(rate_usd * searches)) if rate_usd is not None else None,
            "model_call_key": model_call_key,
        }
        if response_id:
            included["provider_response_id"] = response_id
        return ToolPriced(
            cost_usd=ZERO, source=PROVIDER_REPORTED, included=included, provider_contacted=True
        )
    if rule is None:
        fallback = price_tool(ToolRule(sku="fallback", rate_usd=fallback_rate_usd), units=Decimal(searches))
        return ToolPriced(
            cost_usd=fallback.cost_usd,
            source=FALLBACK_PRICING,
            pricing_missing=True,
            missing=(MISSING_TOOL_RATE,),
            settlement=fallback.settlement,
            provider_contacted=True,
        )
    return price_tool(rule, units_before=units_before, units=Decimal(searches))


def tool_started_values(
    context: LedgerContext,
    *,
    identity: ToolIdentity,
    key: str,
    tool_call_id: Optional[str],
    model: ServedModel,
    is_ai: bool = False,
) -> dict[str, Any]:
    """The row written as the tool starts.

    A tool has no model of its own: it carries the one it is grouped under.
    """
    details: dict[str, Any] = {}
    if context.teams:
        details["teams"] = list(context.teams)
    return {
        **base_values(context, TOOL),
        "idempotency_key": key,
        "event_type": identity.event_type,
        "feature": identity.feature,
        "is_ai": is_ai,
        "status": "started",
        "provider": model.provider,
        "model_name": model.model_name,
        "tool_call_id": tool_call_id,
        "tool_name": identity.name,
        "tool_source": identity.source,
        "tool_namespace": identity.namespace,
        "toolkit_slug": identity.toolkit_slug,
        "tool_display_name": identity.display_name,
        "billing_sku": identity.billing_sku,
        "billing_subject": billing_subject(context),
        "details": details,
    }


def tool_settled_values(
    *,
    priced: ToolPriced,
    status: str,
    duration_ms: Optional[int],
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
    billing_units: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """What settling a tool operation writes. Never the tool's arguments or result."""
    as_call = PricedCall(
        cost_usd=priced.cost_usd,
        source=priced.source,
        pricing_missing=priced.pricing_missing,
        missing=priced.missing,
        breakdown={"tool": priced.cost_usd},
        pricing_mode="unit" if priced.settlement else "none",
        provider_contacted=priced.provider_contacted,
    )
    details: dict[str, Any] = {
        "provider_contacted": priced.provider_contacted,
        "pricing_missing": priced.pricing_missing,
        "missing_pricing": list(priced.missing),
        "pricing_mode": as_call.pricing_mode,
        "cost_accounting": accounting(as_call, status),
    }
    if priced.settlement:
        details["settlement"] = priced.settlement
    if priced.included:
        details["included_in_provider_usage"] = priced.included
        details["provider_reported_cost_scope"] = REQUEST_SCOPE
    if billing_units:
        details["billing_units"] = billing_units
    if error_code:
        details["error_code"] = error_code
    if error_message:
        details["error_message"] = error_message[:ERROR_MESSAGE_MAX]
    return {
        "status": status,
        "cost_usd": priced.cost_usd,
        "cost_source": priced.source,
        "duration_ms": max(0, duration_ms) if duration_ms is not None else None,
        "details": details,
    }
