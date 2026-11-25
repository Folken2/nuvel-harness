"""Whose run this is, for the ledger, and where the run's call count stands.

Whoever starts the turn opens its interaction and sends who and what it is in the
turn's state. How that state is spelled is the product's: a parser reads it into
``LedgerContext`` (this product's is ``ledger/web_app_state.py``), and nothing here
knows a key of it. ``LedgerPlugin`` reads it when the run starts and keeps it in a
context variable, so the model client and the compaction summarizer, which see no
callback context, find the run they are part of. Each run is driven in its own
task, so runs never share one.
"""

from __future__ import annotations

import asyncio
import contextvars
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Any, Callable, Mapping, Optional

from ..billing.pricing import Rates


@dataclass(frozen=True)
class LedgerContext:
    """The interaction a run's operations go under, and who caused them."""

    interaction_id: str
    run_id: str
    user_id: Optional[str] = None
    actor_name: Optional[str] = None
    actor_email: Optional[str] = None
    team_id: Optional[str] = None
    team_name: Optional[str] = None
    # Several teams: their ids and names, kept in details (UG-11).
    teams: tuple[dict[str, str], ...] = ()
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    requested_model_id: Optional[str] = None
    requested_model_name: Optional[str] = None
    quota_scope: str = "user"
    # The feature this run's model calls count under, when the app says one; else the
    # model kind's own (registry.py). An app built for one job names it, so its calls
    # read as that job, not as chat.
    model_feature: Optional[str] = None


# Reads the turn's ledger state, in whatever shape the product sends it.
StateParser = Callable[[Mapping[str, Any]], Optional[LedgerContext]]


def context_from_state(raw: object, parser: StateParser) -> Optional[LedgerContext]:
    """The turn's ledger state as ``parser`` reads it, or None when the turn sent none:
    no mapping, or one that names no interaction or no run."""
    if not isinstance(raw, Mapping):
        return None
    context = parser(raw)
    if context is None or not context.interaction_id or not context.run_id:
        return None
    return context


@dataclass(frozen=True)
class RunStart:
    """What a run's ledger state starts from, read from the turn's state."""

    context: LedgerContext
    # The turn model's rates, and the requested model's LiteLLM id.
    rates: Optional[Rates] = None
    requested_model: Optional[str] = None
    # The model's own rates when someone other than the platform pays for its calls
    # (``rates`` is then what the platform pays): what each call would have cost.
    list_rates: Optional[Rates] = None


# Reads a session's state at the start of a run; None when the turn isn't recorded.
StartReader = Callable[[Mapping[str, Any]], Optional[RunStart]]


@dataclass(frozen=True)
class OpenTool:
    """A tool call under way, for its settlement (ledger/tool_meter.py)."""

    key: str
    identity: Any
    started: float


@dataclass
class RunLedger:
    """One run's ledger state while it goes on."""

    context: LedgerContext
    # The turn model's rates, as the turn sent them (model-controls.md MC-19).
    rates: Optional[Rates]
    # The requested model's LiteLLM id ("provider/model").
    requested_model: Optional[str]
    # What a model call would have cost when someone else pays for it (RunStart).
    list_rates: Optional[Rates] = None
    # Calls already made in this run before a resume: indexes go on from here.
    next_index: int = 0
    # Prompt size of the last call that reported one, for a cut-off call's estimate.
    last_prompt_tokens: Optional[int] = None
    # The model call under way: (kind, index, next attempt, monotonic start).
    call: Optional[tuple[str, int | str, int, float]] = None
    # The attempt under way, for its settlement:
    # (key, kind, model, (estimated input, output), monotonic start).
    open_operation: Optional[tuple[str, str, str, tuple[int, int], float]] = None
    compactions: int = 0
    # Call ids already keyed in this process (meter.call_key), so two calls never share one.
    call_ids: set[str] = field(default_factory=set)
    # Tool calls under way by their call id, and the key parts tool calls already used.
    open_tools: dict[str, OpenTool] = field(default_factory=dict)
    tool_ids: set[str] = field(default_factory=set)
    # Write tasks: every one still running, and each start by its operation key.
    pending: list[Any] = field(default_factory=list)
    starts: dict[str, Any] = field(default_factory=dict)
    # Held by a tool's own provider call (ledger/direct_call.py): ``call`` and
    # ``open_operation`` hold one call, and tool calls of one response run together.
    direct_calls: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Says a settled call is in the ledger to whoever reads the run (runs/usage_frames.py).
    announce_recorded: Optional[Callable[[], None]] = None

    def take_index(self) -> int:
        index = self.next_index
        self.next_index += 1
        return index


_run: contextvars.ContextVar[Optional[RunLedger]] = contextvars.ContextVar(
    "run_ledger", default=None
)


def calls_before(events: Sequence[Any], run_id: str) -> int:
    """Model calls the run already made, on a resume: their keys are taken (UG-3).
    ``events`` are the session's; a call is one that finished, with its usage."""
    return sum(
        1
        for event in events
        if event.invocation_id == run_id
        and event.author != "user"
        and event.usage_metadata
        and not event.partial
    )


def current_run_ledger() -> Optional[RunLedger]:
    return _run.get()


def set_run_ledger(run: Optional[RunLedger]) -> None:
    _run.set(run)
