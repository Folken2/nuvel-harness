"""Rolls old turns into summaries so long conversations stay cheap.

Every model call re-sends the whole history, so cost grows with the length of
a conversation. ADK's event compaction replaces older turns with an LLM-written
summary, on two triggers:

- every ``interval`` user turns, with ``overlap`` turns shared between summaries;
- as soon as a call's prompt reaches ``token_threshold`` tokens, keeping the last
  ``retention`` events verbatim.

Summaries are written by a model of their own, not the user's chat model, so
compaction stays cheap. What they are asked for, and what they carry word for word (the
person's latest messages, the plan), is ``summary_prompt.py``. Which model, the four numbers and whether compaction is on
at all are the product's settings: this module is given them (``CompactionSettings``,
and the summarizer's model as a function that builds it) and reads none. This
service's are in ``context_settings.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events.event import Event
from google.adk.models import BaseLlm, LlmResponse
from google.genai import types

from ..ledger import meter
from ..ledger.entries import COMPACTION
from .summary_prompt import TEMPLATE, carried


@dataclass(frozen=True)
class CompactionSettings:
    """When history is rolled up: every ``interval`` user turns with ``overlap`` shared,
    and once a prompt reaches ``token_threshold`` tokens, keeping ``retention`` events."""

    interval: int
    overlap: int
    token_threshold: int
    retention: int


def _nothing_started() -> None:
    return None


def _nothing_reported() -> Optional[Mapping[str, Any]]:
    return None


@dataclass(frozen=True)
class ProviderReport:
    """How to read what the provider said about one call (its cost, the model that
    ran, its id): ``start`` before the call goes out, ``take`` once it is back, giving
    ``cost_usd``, ``routed_model`` and ``response_id`` where known. How a provider
    reports it is the product's business; the default reads nothing."""

    start: Callable[[], None] = _nothing_started
    take: Callable[[], Optional[Mapping[str, Any]]] = _nothing_reported


class MeteredSummarizer(LlmEventSummarizer):
    """Records each summary call in the usage ledger as a compaction operation
    of the run that triggered it, on the compaction model.

    ADK compacts after the run's last event and before its stream ends, so the
    web app hasn't closed the turn yet and the operation goes under it.
    """

    def __init__(
        self,
        llm: BaseLlm,
        report: Optional[ProviderReport] = None,
        plan_tool: Optional[str] = None,
    ) -> None:
        super().__init__(llm=llm, prompt_template=TEMPLATE)
        self._report = report or ProviderReport()
        self._plan_tool = plan_tool

    async def maybe_summarize_events(self, *, events: list[Event]) -> Optional[Event]:
        self._report.start()
        with meter.metering_as(COMPACTION):
            meter.new_call()
            try:
                event = await super().maybe_summarize_events(events=events)
            finally:
                meter.end_call()
            reported = self._report.take() or {}
            meter.finish(
                LlmResponse(usage_metadata=event.usage_metadata if event else None),
                reported_cost=reported.get("cost_usd"),
                routed_model=reported.get("routed_model"),
                response_id=reported.get("response_id"),
            )
            await meter.drain()
        _add_carried(event, carried(events, self._plan_tool).text())
        return event


def _add_carried(event: Optional[Event], block: str) -> None:
    """In place: the summary followed by what it carries as it was."""
    compaction = event.actions.compaction if event and event.actions else None
    content = compaction.compacted_content if compaction else None
    if not block or content is None:
        return
    content.parts = [*(content.parts or []), types.Part(text=f"\n\n{block}")]


def events_compaction(
    settings: CompactionSettings,
    summarizer_model: Callable[[], BaseLlm],
    report: Optional[ProviderReport] = None,
    plan_tool: Optional[str] = None,
) -> EventsCompactionConfig:
    """ADK's compaction config, with a metered summarizer on the model that
    ``summarizer_model`` builds (called once, here). ``plan_tool`` names the tool whose
    result is the agent's plan, which each summary then carries."""
    return EventsCompactionConfig(
        summarizer=MeteredSummarizer(llm=summarizer_model(), report=report, plan_tool=plan_tool),
        compaction_interval=settings.interval,
        overlap_size=settings.overlap,
        token_threshold=settings.token_threshold,
        event_retention_size=settings.retention,
    )
