"""Each model call logs whether its prompt starts like the last one's.

The comparison first (context/prefix_watch.py), then through a real ADK runner with the
plugins that rewrite a request: appending never counts as a change, and masking an old
tool result does, once.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import AsyncGenerator, ClassVar

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types

from harness.context.old_results import MaskSettings
from harness.context.prefix_watch import FIRST, HISTORY, INSTRUCTION, SAME, TOOLS, compare, fingerprint
from harness.plugins.old_results import OldResultsPlugin
from harness.plugins.prefix_watch import PrefixWatchPlugin


def _request(*texts: str, instruction: str = "Be brief.", tools: tuple[str, ...] = ("search",)) -> LlmRequest:
    declarations = [types.FunctionDeclaration(name=name, description=f"{name} tool") for name in tools]
    return LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text=text)]) for text in texts],
        config=types.GenerateContentConfig(
            system_instruction=instruction, tools=[types.Tool(function_declarations=declarations)]
        ),
    )


def test_a_first_call_has_nothing_to_compare_with():
    found = compare(None, fingerprint(_request("hi")))

    assert (found.change, found.broken) == (FIRST, False)


def test_new_messages_after_the_same_start_change_nothing():
    found = compare(fingerprint(_request("hi", "hello")), fingerprint(_request("hi", "hello", "more")))

    assert (found.change, found.kept, found.before, found.now, found.broken) == (SAME, 2, 2, 3, False)


def test_a_rewritten_or_missing_earlier_message_breaks_the_prefix_where_it_is():
    before = fingerprint(_request("one", "two", "three"))

    rewritten = compare(before, fingerprint(_request("one", "TWO", "three", "four")))
    shorter = compare(before, fingerprint(_request("one", "two")))

    assert (rewritten.change, rewritten.kept, rewritten.broken) == (HISTORY, 1, True)
    assert (shorter.change, shorter.kept) == (HISTORY, 2)


def test_a_changed_instruction_or_tool_list_breaks_everything():
    before = fingerprint(_request("hi"))

    assert compare(before, fingerprint(_request("hi", instruction="Be brief. Today is Tuesday."))).change == INSTRUCTION
    tools = compare(before, fingerprint(_request("hi", tools=("search", "send"))))
    assert (tools.change, tools.kept, tools.broken) == (TOOLS, 0, True)


# Through ADK

READS: list[int] = []


def read_page(page: int) -> dict:
    """Reads one long page."""
    READS.append(page)
    return {"page": page, "text": "lorem ipsum " * 700}


class Reader(BaseLlm):
    model: str = "reader"
    pages: ClassVar[int] = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        if len(READS) < Reader.pages:
            part = types.Part(function_call=types.FunctionCall(id=f"c{len(READS)}", name="read_page", args={"page": len(READS)}))
        else:
            part = types.Part(text="done")
        size = sum(len(str(content)) for content in llm_request.contents) // 4
        yield LlmResponse(
            content=types.Content(role="model", parts=[part]),
            usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=size, candidates_token_count=1),
        )


def _changes(pages: list[int], masking: bool, caplog) -> list[str]:
    """One turn per entry of ``pages``: the change each model call logged."""
    READS.clear()
    plugins = []
    if masking:
        plugins.append(
            OldResultsPlugin(MaskSettings(trigger=0.6, protect=0.2, min_saving=0.1), reader="read_document", fallback_threshold=10_000)
        )
    plugins.append(PrefixWatchPlugin())
    runner = InMemoryRunner(
        app=App(name="probe", root_agent=LlmAgent(name="probe", model=Reader(), tools=[FunctionTool(read_page)]), plugins=plugins)
    )

    async def go() -> None:
        session = await runner.session_service.create_session(app_name="probe", user_id="u")
        total = 0
        for count in pages:
            total += count
            Reader.pages = total
            async for _ in runner.run_async(
                user_id="u", session_id=session.id, new_message=types.Content(role="user", parts=[types.Part(text="read on")])
            ):
                pass

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="harness.plugins.prefix_watch"):
        asyncio.run(go())
    return [re.search(r"change=(\w+)", record.message).group(1) for record in caplog.records if "prompt prefix" in record.message]


def test_a_conversation_that_only_grows_never_breaks_its_prefix(caplog):
    changes = _changes([2, 1, 0], masking=False, caplog=caplog)

    # Three turns, six model calls: the first, then nothing but new messages.
    assert changes == [FIRST, SAME, SAME, SAME, SAME, SAME]


def test_masking_old_results_breaks_it_once_per_batch_and_the_log_says_so(caplog):
    changes = _changes([10], masking=True, caplog=caplog)

    assert changes[0] == FIRST and len(changes) == 11
    # Ten pages are masked in three batches (test_old_results.py): three breaks, no more.
    assert changes.count(HISTORY) == 3 and changes.count(SAME) == 7
