"""The frame that tells the web app a call is in the ledger (limits.md LM-37, T-49)."""

import asyncio
import json

from harness.ledger import writer as writer_module
from harness.ledger.context import LedgerContext, RunLedger
from harness.ledger.writer import LedgerWriter
from harness.runs.usage_frames import current_announcer, reset_frame_sink, set_frame_sink, usage_frame
from harness.wire_names import USAGE_FRAME as USAGE_FRAME_KEY


def _run() -> RunLedger:
    return RunLedger(context=LedgerContext(interaction_id="i1", run_id="r1"), rates=None, requested_model=None)


class _Store:
    def __init__(self, results):
        self.results = list(results)

    async def settle(self, key, values):
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def test_the_frame_is_one_sse_event_naming_only_that_a_call_was_recorded():
    frame = usage_frame()
    assert frame.startswith("data: ") and frame.endswith("\n\n")
    assert json.loads(frame[len("data: "):]) == {USAGE_FRAME_KEY: {"recorded": True}}


def test_the_announcer_reaches_the_run_it_was_taken_in_and_none_outside_one():
    assert current_announcer() is None
    frames: list[str] = []
    token = set_frame_sink(frames.append)
    try:
        announce = current_announcer()
    finally:
        reset_frame_sink(token)
    assert announce is not None
    # A write finishing after the run's context is gone still reaches its own run.
    announce()
    assert frames == [usage_frame()]


def test_a_committed_settlement_is_announced_once_it_has_committed():
    async def go():
        order: list[str] = []

        class Store(_Store):
            async def settle(self, key, values):
                order.append("committed")
                return await super().settle(key, values)

        run = _run()
        writer = LedgerWriter(Store(["0.01"]))

        async def values():
            return {"cost_usd": "0.01"}

        writer.settle(run, "k", values, on_recorded=lambda: order.append("announced"))
        await writer.drain(run)
        return order

    assert asyncio.run(go()) == ["committed", "announced"]


def test_a_failed_write_announces_nothing_so_the_meter_waits_for_the_next_call(monkeypatch):
    monkeypatch.setattr(writer_module, "RETRY_DELAYS", (0.0, 0.0, 0.0))

    async def go():
        announced: list[int] = []
        run = _run()
        writer = LedgerWriter(_Store([RuntimeError("db down")] * 4 + ["0.02"]))

        async def values():
            return {"cost_usd": "0.02"}

        writer.settle(run, "call-2", values, on_recorded=lambda: announced.append(2))
        await writer.drain(run)
        assert announced == []
        writer.settle(run, "call-3", values, on_recorded=lambda: announced.append(3))
        await writer.drain(run)
        return announced

    assert asyncio.run(go()) == [3]


def test_an_announcement_that_raises_never_reaches_the_run():
    async def go():
        run = _run()
        writer = LedgerWriter(_Store(["0.01"]))

        async def values():
            return {"cost_usd": "0.01"}

        def broken():
            raise RuntimeError("reader gone")

        writer.settle(run, "k", values, on_recorded=broken)
        await writer.drain(run)
        return [t.result() for t in run.pending]

    assert asyncio.run(go()) == []


def test_a_settlement_that_cost_nothing_is_announced_only_when_asked_to_be():
    """A $0 tool call changes no meter, so it asks for no read (LM-37); a model call always does."""

    async def go():
        announced: list[str] = []
        run = _run()
        writer = LedgerWriter(_Store(["0", "0.000000000000", "0.01", "0"]))

        async def values():
            return {}

        writer.settle(run, "free-tool", values, on_recorded=lambda: announced.append("free-tool"), announce_free=False)
        writer.settle(run, "free-tool-2", values, on_recorded=lambda: announced.append("free-tool-2"), announce_free=False)
        writer.settle(run, "paid-tool", values, on_recorded=lambda: announced.append("paid-tool"), announce_free=False)
        writer.settle(run, "model-call", values, on_recorded=lambda: announced.append("model-call"))
        await writer.drain(run)
        return announced

    assert asyncio.run(go()) == ["paid-tool", "model-call"]
