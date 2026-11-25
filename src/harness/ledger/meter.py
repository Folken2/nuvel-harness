"""One ledger operation per provider attempt (usage-and-analytics.md UG-1 to UG-3).

``MeteredLiteLLMClient`` (``routing/client.py``) calls ``begin_attempt`` right
before each request goes to a provider and ``attempt_failed`` when one raises.
``LedgerPlugin`` calls ``finish`` with the call's final response, and the
compaction summarizer does the same for its own call. Outside a run that sent
ledger state, or with no ledger database configured, every function here does
nothing.
"""

from __future__ import annotations

import asyncio
import contextvars
import time
import uuid
from contextlib import contextmanager
from decimal import Decimal
from typing import Any, Iterator, Optional

from google.adk.models import LlmResponse

from ..billing.pricing import (
    CUT_OFF,
    FALLBACK_PRICING,
    NOT_CONTACTED,
    REFUSED,
    Rates,
    Units,
    estimate_tokens,
    price_aborted,
    price_call,
    to_decimal,
)
from ..billing.evidence import provider_id
from ..billing.tokens import TokenUsage, normalise
from . import tool_meter
from .context import RunLedger, current_run_ledger
from .entries import (
    COMPACTION,
    IMAGE,
    MODEL,
    RequestedModel,
    operation_key,
    served_model,
    settled_values,
    started_values,
)
from .writer import LedgerWriter, current_writer

__all__ = ["attempt_failed", "call_key", "finish", "metering_as", "metering_turn_call"]

_kind: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("ledger_kind", default=MODEL)
# A call keyed by something stable instead of a count, and the model it asked for.
_call: contextvars.ContextVar[Optional[tuple[str, Optional[RequestedModel]]]] = contextvars.ContextVar(
    "ledger_call", default=None
)


@contextmanager
def metering_as(
    kind: Optional[str], *, call_id: Optional[str] = None, requested: Optional[RequestedModel] = None
) -> Iterator[None]:
    """Calls inside are recorded as ``kind`` (compaction) instead of model calls;
    None records nothing. ``call_id`` keys them instead of a count, so a replay
    writes the same key (UG-3); ``requested`` is the model they asked for."""
    kind_token = _kind.set(kind)
    call_token = _call.set((call_id, requested) if call_id else None)
    try:
        yield
    finally:
        _call.reset(call_token)
        _kind.reset(kind_token)


def metering_turn_call() -> bool:
    """Whether the call going out is one of the turn's own model calls: not compaction,
    nor a call keyed apart (the advisor's), nor one that isn't recorded (a title)."""
    return _kind.get() == MODEL and _call.get() is None


def call_key(call_id: Optional[str]) -> str:
    """A stable key part for a call that isn't counted, such as the advisor's tool call.

    A replay after a restart is a new process and gets the same id back, so it writes
    the same row (UG-3). An id this process already keyed in the run is a second call,
    never the first one again (a tool call runs at most once, runs/resume.py), so it
    gets a suffix rather than losing its row. With no id, a fresh one.
    """
    base = call_id or uuid.uuid4().hex
    run = current_run_ledger()
    if run is None:
        return base
    key, n = base, 1
    while key in run.call_ids:
        n += 1
        key = f"{base}~{n}"
    run.call_ids.add(key)
    return key


def _active() -> Optional[tuple[RunLedger, LedgerWriter]]:
    if _kind.get() is None:
        return None
    run, writer = current_run_ledger(), current_writer()
    return (run, writer) if run is not None and writer is not None else None


def new_call() -> None:
    """A model call starts: its attempts share one index (UG-3)."""
    if (active := _active()) is not None:
        run = active[0]
        kind = _kind.get() or MODEL
        run.call = (kind, _take_index(run, kind), 0, time.monotonic())


def end_call() -> None:
    if (active := _active()) is not None:
        active[0].call = None


def _take_index(run: RunLedger, kind: str) -> int | str:
    if (fixed := _call.get()) is not None:
        return fixed[0]
    if kind == COMPACTION:
        run.compactions += 1
        return run.compactions - 1
    return run.take_index()


def begin_attempt(model: Any) -> Optional[str]:
    """Writes the ``started`` operation for the request about to go out (UG-2)."""
    if (active := _active()) is None:
        return None
    run, writer = active
    if run.call is None:
        new_call()
    kind, index, attempt, started = run.call  # type: ignore[misc]
    run.call = (kind, index, attempt + 1, started)
    key = operation_key(run.context.run_id, kind, index, attempt)
    estimate = estimate_tokens(last_prompt_size=run.last_prompt_tokens)
    writer.start(
        run,
        key,
        started_values(
            run.context,
            kind=kind,
            key=key,
            requested_model=str(model),
            estimate=estimate,
            requested=fixed[1] if (fixed := _call.get()) is not None else None,
        ),
    )
    run.open_operation = (key, kind, str(model), estimate, time.monotonic())
    return key


async def _rates(writer: LedgerWriter, run: RunLedger, kind: str, model: str) -> Optional[Rates]:
    if kind == MODEL and run.rates is not None:
        return run.rates
    return await writer.catalogue.rates(model, images=kind == IMAGE)


def _list_cost(run: RunLedger, kind: str, usage: Optional[TokenUsage], failed: bool) -> Optional[Decimal]:
    """What a model call someone else paid for would have cost at the model's own
    rates. Only for a call that ran, and only from rates that are all there: a
    stand-in rate would make the figure a guess."""
    if kind != MODEL or run.list_rates is None or usage is None or failed:
        return None
    listed = price_call(usage=usage, rates=run.list_rates)
    return None if listed.pricing_missing or listed.source == FALLBACK_PRICING else listed.cost_usd


def attempt_failed(error: BaseException, *, contacted: bool = True) -> None:
    """The request raised: refused with an error, or cut off with no usage (UG-23)."""
    if (active := _active()) is None or active[0].open_operation is None:
        return
    run, writer = active
    key, kind, model, estimate, started = run.open_operation
    run.open_operation = None
    status = "cancelled" if isinstance(error, asyncio.CancelledError) else "failed"
    outcome = NOT_CONTACTED if not contacted else REFUSED if _has_status(error) else CUT_OFF
    duration = int((time.monotonic() - started) * 1000)

    async def values() -> dict[str, Any]:
        priced = price_aborted(
            outcome=outcome,
            rates=await _rates(writer, run, kind, model),
            fallback=await writer.catalogue.fallback_rates(),
            estimate=estimate,
        )
        return settled_values(
            priced=priced,
            status=status,
            usage=None,
            requested_model=model,
            duration_ms=duration,
            error_code=_error_code(error),
            # Never the provider's text: it can carry the request's URL or a key.
            error_message=type(error).__name__,
        )

    writer.settle(run, key, values, on_recorded=run.announce_recorded)


def _has_status(error: BaseException) -> bool:
    status = getattr(error, "status_code", None)
    return isinstance(status, int) and 400 <= status < 600


def _error_code(error: BaseException) -> str:
    status = getattr(error, "status_code", None)
    if status == 429:
        return "RATE_LIMIT"
    if status in (401, 403):
        return "AUTH"
    if type(error).__name__ in ("Timeout", "TimeoutError", "APITimeoutError"):
        return "TIMEOUT"
    return type(error).__name__.upper()


def usage_of(llm_response: LlmResponse) -> Optional[TokenUsage]:
    """ADK's usage for a call through LiteLLM: input includes cached tokens and
    output includes thinking (checked against a live call, see the PR)."""
    usage = llm_response.usage_metadata
    if usage is None:
        return None
    return normalise(
        input_tokens=usage.prompt_token_count,
        cache_read=usage.cached_content_token_count,
        cache_write=getattr(usage, "cache_creation_input_tokens", None),
        output_tokens=usage.candidates_token_count,
        reasoning=usage.thoughts_token_count,
        total=usage.total_token_count,
    )


def finish(
    llm_response: LlmResponse,
    *,
    reported_cost: object = None,
    routed_model: object = None,
    response_id: object = None,
    web_searches: object = None,
    output_count: Optional[int] = None,
    units: Optional[Units] = None,
) -> None:
    """Settles the open operation with the call's final response, and records the
    hosted web searches the provider ran inside it (UG-25). ``output_count`` is the
    images an image generation returned. ``units`` are what a unit-priced call billed,
    such as its pictures (UG-22), for a model whose rates price by unit."""
    if (active := _active()) is None or active[0].open_operation is None:
        return
    run, writer = active
    key, kind, model, estimate, started = run.open_operation
    run.open_operation = None
    usage = usage_of(llm_response)
    if usage is not None and usage.input_total:
        run.last_prompt_tokens = usage.input_total
    duration = int((time.monotonic() - started) * 1000)
    raw = llm_response.usage_metadata.model_dump(exclude_none=True) if llm_response.usage_metadata else None
    failed = bool(llm_response.error_code)

    async def values() -> dict[str, Any]:
        rates = await _rates(writer, run, kind, model)
        fallback = await writer.catalogue.fallback_rates()
        if usage is None and reported_cost is None:
            priced = price_aborted(outcome=CUT_OFF, rates=rates, fallback=fallback, estimate=estimate)
        else:
            priced = price_call(
                usage=usage or TokenUsage(), rates=rates, fallback=fallback, reported_cost=reported_cost, units=units
            )
        list_cost = _list_cost(run, kind, usage, failed)
        return settled_values(
            priced=priced,
            status="failed" if failed else "completed",
            list_cost=list_cost,
            usage=usage,
            routed_model=routed_model,
            requested_model=model,
            raw_usage=raw,
            response_id=response_id,
            duration_ms=duration,
            error_code=str(llm_response.error_code) if failed else None,
            error_message=None,
            output_count=output_count,
        )

    # Every settled call moves the person's meters, so each one is announced (LM-37).
    writer.settle(run, key, values, on_recorded=run.announce_recorded)
    if isinstance(web_searches, int) and not isinstance(web_searches, bool) and web_searches > 0:
        tool_meter.hosted_searches(
            run,
            writer,
            model_call_key=key,
            searches=web_searches,
            model_reported=to_decimal(reported_cost) is not None,
            served=served_model(routed_model, model),
            response_id=provider_id(response_id),
        )


async def drain() -> None:
    """Waits for the run's pending writes (after the run, where nothing else will)."""
    if (active := _active()) is not None:
        await active[1].drain(active[0])
