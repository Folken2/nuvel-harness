"""What one ledger row holds, built from a priced call. Pure: no I/O.

Column names are the ledger table's, as the product's database defines them.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

from ..billing.evidence import provider_id, sanitise_provider_usage
from ..billing.pricing import PROVIDER_REPORTED, PricedCall
from ..billing.rollup import Settled, quality, split
from ..billing.tokens import TokenUsage
from .context import LedgerContext
from .registry import COMPACTION, IMAGE, MODEL, REGISTRY, TOOL, ModelIds

# The ledger's own kinds, for the modules that take them from here.
__all__ = ["COMPACTION", "IMAGE", "MODEL", "TOOL"]


ERROR_MESSAGE_MAX = 500


def operation_key(run_id: str, kind: str, index: int | str, attempt: int = 0) -> str:
    """A replay writes the same key, a retry the next attempt."""
    return f"{run_id}:{kind}:{index}:{attempt}"


@dataclass(frozen=True)
class RequestedModel:
    """The catalogue model a call asked for, when it isn't the turn's."""

    id: Optional[str]
    name: Optional[str]


@dataclass(frozen=True)
class ServedModel:
    provider: Optional[str]
    model_name: Optional[str]
    router: Optional[str]
    reported: bool


def name_served(ids: ModelIds, routed: object, requested: Optional[str]) -> ServedModel:
    """The model that served the call, normalised, under a product's id scheme.

    A router's mark goes, so an id through a router and the maker's own are one model,
    and the provider is the maker the id's prefix stands for, as the catalogue names
    it. With no report from the provider the requested model stands, and that is noted.
    """
    reported = isinstance(routed, str) and bool(routed)
    raw = str(routed) if reported else (requested or "")
    router = ids.router_of(raw) or ids.router_of(requested or "")
    raw = ids.without_router(raw)
    if not raw:
        return ServedModel(None, None, router, reported)
    prefix, _, name = raw.partition("/")
    if not name:
        return ServedModel(None, raw, router, reported)
    maker = ids.maker_of(prefix)
    return ServedModel(maker, f"{maker}/{name}", router, reported)


def served_model(routed: object, requested: Optional[str]) -> ServedModel:
    """``name_served`` under the scheme the product registered (registry.py)."""
    return name_served(REGISTRY.model_ids, routed, requested)


def base_values(context: LedgerContext, kind: str) -> dict[str, Any]:
    """What every operation of ``kind`` carries. What the kind is called is the product's (registry.py)."""
    names = REGISTRY.kind(kind)
    feature = context.model_feature if kind == MODEL and context.model_feature else names.feature
    return {
        "level": "operation",
        "root_id": context.interaction_id,
        "parent_id": context.interaction_id,
        "run_id": context.run_id,
        "event_type": names.event_type,
        "operation_kind": names.operation_kind,
        "feature": feature,
        "use_case": names.use_case,
        "is_ai": True,
        "quota_scope": context.quota_scope,
        "user_id": context.user_id,
        "actor_name": context.actor_name,
        "actor_email": context.actor_email,
        "team_id": context.team_id,
        "team_name": context.team_name,
        "conversation_id": context.conversation_id,
        "message_id": context.message_id,
        "requested_model_id": context.requested_model_id,
        "requested_model_name": context.requested_model_name,
    }


# A background task (runs/tasks.py): an interaction of its own, opened by the agent,
# with its model calls under it. It counts toward the person's spend, not as a request.
TASK_EVENT_TYPE = "agent.task"


def task_interaction_key(task_id: str) -> str:
    """The task's id is its run id, so a resumed task finds its entry."""
    return f"{task_id}:{TASK_EVENT_TYPE}"


def task_interaction_values(context: LedgerContext, task_id: str) -> dict[str, Any]:
    """The task's interaction, on behalf of the person whose turn started it."""
    details: dict[str, Any] = {"teams": list(context.teams)} if context.teams else {}
    return {
        "level": "interaction",
        "idempotency_key": task_interaction_key(task_id),
        "run_id": task_id,
        "event_type": TASK_EVENT_TYPE,
        "feature": "task",
        "use_case": "assistant_response",
        "is_ai": True,
        "status": "started",
        "quota_scope": context.quota_scope,
        "user_id": context.user_id,
        "actor_name": context.actor_name,
        "actor_email": context.actor_email,
        "team_id": context.team_id,
        "team_name": context.team_name,
        "conversation_id": context.conversation_id,
        "message_id": context.message_id,
        "requested_model_id": context.requested_model_id,
        "requested_model_name": context.requested_model_name,
        "details": details,
    }


def started_values(
    context: LedgerContext,
    *,
    kind: str,
    key: str,
    requested_model: Optional[str],
    estimate: tuple[int, int],
    requested: Optional["RequestedModel"] = None,
) -> dict[str, Any]:
    """The row written before the provider is called.

    The estimate is stored now: the stale-work job has no prompt to read.
    ``requested`` names a model other than the turn's.
    """
    model = served_model(None, requested_model)
    details: dict[str, Any] = {
        "provider_contacted": True,
        "estimate": {"input_tokens": estimate[0], "output_tokens": estimate[1]},
    }
    if context.teams:
        details["teams"] = list(context.teams)
    return {
        **base_values(context, kind),
        **(
            {"requested_model_id": requested.id, "requested_model_name": requested.name}
            if requested is not None
            else {}
        ),
        "idempotency_key": key,
        "status": "started",
        "provider": model.provider,
        "model_name": model.model_name,
        "details": details,
    }


def _usage_details(usage: TokenUsage) -> dict[str, Any]:
    return {
        "version": 1,
        "cache_read": usage.cache_read,
        "cache_write": usage.cache_write,
        "cache_write_1h": usage.cache_write_1h,
        "cache_write_30m": usage.cache_write_30m,
        "cache_write_12h": usage.cache_write_12h,
        "ordinary_input": usage.ordinary_input,
        "total": usage.total,
        "input_image": usage.input_image,
        "input_audio": usage.input_audio,
        "output_image": usage.output_image,
        "output_audio": usage.output_audio,
    }


def accounting(priced: PricedCall, status: str) -> dict[str, Any]:
    parts = split(
        Settled(priced.cost_usd, priced.source, priced.pricing_missing, status, priced.provider_contacted)
    )
    return {
        "version": 1,
        "quality": quality(priced.cost_usd, priced.source, priced.pricing_missing),
        "source": priced.source,
        "split": {
            "provider": str(parts.provider),
            "calculated": str(parts.calculated),
            "estimated": str(parts.estimated),
            "unresolved": str(parts.unresolved),
        },
        "token_cost": {name: str(value) for name, value in priced.breakdown.items()},
        "pricing_snapshot": priced.snapshot,
    }


def settled_values(
    *,
    priced: PricedCall,
    status: str,
    usage: Optional[TokenUsage],
    routed_model: object = None,
    requested_model: Optional[str] = None,
    raw_usage: object = None,
    response_id: object = None,
    duration_ms: Optional[int] = None,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
    output_count: Optional[int] = None,
    list_cost: Optional[Decimal] = None,
) -> dict[str, Any]:
    """What settling an operation writes: final status, tokens, cost and evidence.

    ``output_count`` is the images an image generation returned.
    ``list_cost`` is what the call would have cost at the model's own rates when
    someone else paid for it; it is kept beside the cost and never counted in it.
    """
    model = served_model(routed_model, requested_model)
    details: dict[str, Any] = {
        "provider_contacted": priced.provider_contacted,
        "pricing_missing": priced.pricing_missing,
        "missing_pricing": list(priced.missing),
        "pricing_mode": priced.pricing_mode,
        "cost_accounting": accounting(priced, status),
    }
    if usage is not None:
        details["usage"] = _usage_details(usage)
    evidence = sanitise_provider_usage(raw_usage)
    if evidence:
        details["provider_usage"] = evidence
    if priced.source == PROVIDER_REPORTED:
        # The router's reported cost covers the whole request, search fees included.
        details["provider_reported_cost_scope"] = "request"
    if (rid := provider_id(response_id)) is not None:
        details["provider_response_id"] = rid
    if model.router:
        details["router"] = model.router
    if not model.reported:
        details["model_not_reported"] = True
    if priced.billing_units:
        details["billing_units"] = priced.billing_units
    if output_count is not None:
        details["output_count"] = output_count
    if list_cost is not None:
        details["list_cost_usd"] = str(list_cost)
    if error_code:
        details["error_code"] = error_code
    if error_message:
        details["error_message"] = error_message[:ERROR_MESSAGE_MAX]
    return {
        "status": status,
        "provider": model.provider,
        "model_name": model.model_name,
        "tokens_in": usage.input_total if usage else 0,
        "tokens_out": usage.output_total if usage else 0,
        "reasoning_tokens": usage.reasoning if usage else 0,
        "cost_usd": priced.cost_usd,
        "cost_source": priced.source,
        "duration_ms": max(0, duration_ms) if duration_ms is not None else None,
        "details": details,
    }


def settled_entry(row: dict[str, Any]) -> Settled:
    """A stored operation, as the rollup reads it."""
    details = row.get("details") or {}
    return Settled(
        cost_usd=Decimal(str(row.get("cost_usd") or 0)),
        source=row.get("cost_source") or "unknown",
        pricing_missing=bool(details.get("pricing_missing")),
        status=row.get("status") or "completed",
        provider_contacted=bool(details.get("provider_contacted", True)),
    )
