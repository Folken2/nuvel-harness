"""One ledger operation per tool call.

``ToolLedgerPlugin`` calls ``begin`` as a tool is about to run, and ``finish`` or
``failed`` when it ends. The model meter calls ``hosted_searches`` for the web
searches the provider ran inside a model call, which reach no tool callback.
Outside a run that sent ledger state, or with no ledger database, nothing here
writes.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from ..billing.pricing import ZERO
from ..billing.tool_rules import ToolRule, fallback_web_search_rate, period_start, router_web_search_rule
from .context import OpenTool, RunLedger, current_run_ledger
from .entries import ServedModel, served_model
from .registry import REGISTRY
from .tool_entries import (
    ToolPriced,
    billing_subject,
    hosted_search_key,
    price_hosted_search,
    price_tool,
    tool_key,
    tool_settled_values,
    tool_started_values,
)
from .tool_identity import ToolIdentity, hosted_web_search, identify, namespace_of
from .writer import LedgerWriter, current_writer

TOOL_ERROR = "TOOL_ERROR"
UNKNOWN_TOOL = "UNKNOWN_TOOL"


def _active() -> Optional[tuple[RunLedger, LedgerWriter]]:
    run, writer = current_run_ledger(), current_writer()
    return (run, writer) if run is not None and writer is not None else None


def _call_part(run: RunLedger, call_id: Optional[str]) -> str:
    """A tool call's key part: its id, with a suffix when this process already used it.

    Some models number their calls (``tool_0``) afresh in each response, so one id
    can come back in a run; a replay in a new process gets the same key back.
    """
    base = call_id or f"t{len(run.tool_ids)}"
    part, n = base, 1
    while part in run.tool_ids:
        n += 1
        part = f"{base}~{n}"
    run.tool_ids.add(part)
    return part


def _grouping_model(run: RunLedger) -> ServedModel:
    """A tool has no model of its own: it goes under the turn's selected model."""
    return served_model(None, run.requested_model)


def rule_for(identity: ToolIdentity) -> Optional[ToolRule]:
    """The price rule of a tool the agent called. None of them has one today."""
    return None


def begin(tool: Any, call_id: Optional[str]) -> None:
    """Writes the ``started`` operation for a tool about to run."""
    if (active := _active()) is None or not call_id:
        return
    run, writer = active
    identity = identify(str(getattr(tool, "name", "") or ""), namespace=namespace_of(tool))
    key = tool_key(run.context.run_id, _call_part(run, call_id))
    writer.start(
        run,
        key,
        tool_started_values(
            run.context,
            identity=identity,
            key=key,
            tool_call_id=call_id,
            model=_grouping_model(run),
            # A tool that calls a model itself: its row says a model was involved.
            is_ai=REGISTRY.tool_calls_model(identity.name),
        ),
    )
    run.open_tools[call_id] = OpenTool(key=key, identity=identity, started=time.monotonic())


def _failed_result(result: Any) -> bool:
    """A result that says the call failed, in any of the three shapes tools use.

    MCP's ``isError``, ``status: error``, and a built-in tool's refusal: a non-empty
    ``error`` string. An empty or non-text ``error`` (a connected app's ``error: null``)
    is not one. A client that shows tool failures should read the same three.
    """
    if not isinstance(result, dict):
        return False
    refusal = result.get("error")
    return (
        result.get("isError") is True
        or result.get("status") == "error"
        or (isinstance(refusal, str) and bool(refusal.strip()))
    )


def finish(call_id: Optional[str], result: Any) -> None:
    """Settles the tool's operation from its result. Never stores the result."""
    failed = _failed_result(result)
    _settle(call_id, "failed" if failed else "completed", TOOL_ERROR if failed else None)


def failed(call_id: Optional[str], tool: Any, error: BaseException) -> None:
    """The tool raised, was cancelled, or wasn't found."""
    if isinstance(error, asyncio.CancelledError):
        _settle(call_id, "cancelled", None)
        return
    not_found = type(tool).__name__ == "BaseTool" and getattr(tool, "description", "") == "Tool not found"
    # The error's type, never its text: it can carry the person's data.
    _settle(call_id, "failed", UNKNOWN_TOOL if not_found else type(error).__name__.upper(), type(error).__name__)


def _settle(call_id: Optional[str], status: str, error_code: Optional[str], error_message: Optional[str] = None) -> None:
    if (active := _active()) is None or not call_id:
        return
    run, writer = active
    open_tool = run.open_tools.pop(call_id, None)
    if open_tool is None:
        return
    duration = int((time.monotonic() - open_tool.started) * 1000)
    rule = rule_for(open_tool.identity)
    subject = billing_subject(run.context)

    async def values() -> dict[str, Any]:
        before = await _units_before(writer, rule, subject)
        priced = price_tool(rule, units_before=before)
        return tool_settled_values(
            priced=priced,
            status=status,
            duration_ms=duration,
            error_code=error_code,
            error_message=error_message,
            billing_units={"units": "1"} if rule is not None else None,
        )

    # A model call moves the context meter and a hosted search the cost; a tool call moves
    # a meter only when it costs something.
    writer.settle(run, open_tool.key, values, on_recorded=run.announce_recorded, announce_free=False)


async def _units_before(writer: LedgerWriter, rule: Optional[ToolRule], subject: str) -> Decimal:
    """Units the subject already used this period, read only when the rule's cost depends on them."""
    if rule is None or not rule.needs_history():
        return ZERO
    return await writer.store.units_used(
        sku=rule.sku, subject=subject, since=period_start(rule, datetime.now(timezone.utc))
    )


def hosted_searches(
    run: RunLedger,
    writer: LedgerWriter,
    *,
    model_call_key: str,
    searches: int,
    model_reported: bool,
    served: ServedModel,
    response_id: Optional[str],
) -> None:
    """The web searches the provider ran inside one model call: one tool operation.

    Its model is the one that searched. Its cost is inside the model call's
    reported cost when that covers the whole request, else the router's per-call rule
    prices it.
    """
    identity = hosted_web_search()
    key = hosted_search_key(run.context.run_id, model_call_key)
    writer.start(
        run,
        key,
        tool_started_values(run.context, identity=identity, key=key, tool_call_id=None, model=served, is_ai=True),
    )
    subject = billing_subject(run.context)
    model = served.model_name or ""

    async def values() -> dict[str, Any]:
        rate = await writer.catalogue.web_search_rate(model) if model else None
        rule = router_web_search_rule("native", rate)
        fallback = (
            fallback_web_search_rate(await writer.catalogue.highest_web_search_rate())
            if rule is None and not model_reported
            else None
        )
        priced: ToolPriced = price_hosted_search(
            searches=searches,
            model_reported=model_reported,
            rate_usd=rate,
            rule=rule,
            model_call_key=model_call_key,
            response_id=response_id,
            units_before=await _units_before(writer, rule, subject),
            **({"fallback_rate_usd": fallback} if fallback is not None else {}),
        )
        return tool_settled_values(
            priced=priced,
            status="completed",
            duration_ms=None,
            billing_units={"units": str(searches)},
        )

    writer.settle(run, key, values, on_recorded=run.announce_recorded)
