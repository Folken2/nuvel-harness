"""Telling a too-long request apart (context/overflow.py), and masking it smaller
(context/masking.py)."""

from __future__ import annotations

import pytest

from harness.context.masking import PREVIEW_CHARS, mask_old_tool_results
from harness.context.overflow import is_context_overflow


@pytest.mark.parametrize(
    "message",
    [
        # Anthropic
        "prompt is too long: 210000 tokens > 200000 maximum",
        '{"type":"error","error":{"type":"request_too_large"}}',
        # OpenAI, and the OpenAI-compatible APIs
        "This model's maximum context length is 128000 tokens. However, your messages resulted in 200000 tokens.",
        "Your input exceeds the context window of this model.",
        '{"code": "context_length_exceeded"}',
        # a router
        "This endpoint's maximum context length is 1000 tokens. However, you requested about 2000 tokens",
        "Input exceeds the maximum allowed input length of 32,000 tokens",
        # Gemini
        "The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).",
        # xAI, Mistral, Moonshot
        "This model's maximum prompt length is 131072 but the request contains 140000 tokens.",
        "Prompt contains 40000 tokens, too large for model with 32768 maximum context length",
        "exceeded model token limit: 262144",
        # LiteLLM's own check
        "litellm.ContextWindowExceededError: context window exceeded",
    ],
)
def test_overflow_wordings(message):
    assert is_context_overflow(message)


@pytest.mark.parametrize(
    "message",
    [
        "Rate limit exceeded",
        "ThrottlingException: Too many tokens, please wait before trying again.",
        "Too many tokens per day, please wait",
        "Too many requests",
        "max_tokens must be less than or equal to 8192",
        "invalid tool schema",
        "",
    ],
)
def test_not_overflow(message):
    assert not is_context_overflow(message)


def _tool(text, call_id):
    return {"role": "tool", "tool_call_id": call_id, "content": text}


CALL = {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]}
BIG = "x" * 5_000


def test_older_steps_keep_a_preview_and_a_note():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        CALL, _tool(BIG, "a"),
        CALL, _tool(BIG, "b"),
        CALL, _tool(BIG, "c"),
    ]
    masked = mask_old_tool_results(messages, keep_steps=1)
    assert masked is not None
    assert masked[:3] == messages[:3]
    for i in (3, 5):
        assert masked[i]["content"].startswith("x" * PREVIEW_CHARS)
        assert "5,000 characters" in masked[i]["content"]
        assert len(masked[i]["content"]) < 500
    assert masked[3]["tool_call_id"] == "a"
    assert masked[7]["content"] == BIG
    # The request's messages are a copy: the caller's list is untouched.
    assert messages[3]["content"] == BIG


def test_a_steps_parallel_results_stay_together():
    messages = [CALL, _tool(BIG, "a"), CALL, _tool(BIG, "b1"), _tool(BIG, "b2"), _tool(BIG, "b3")]
    masked = mask_old_tool_results(messages, keep_steps=1)
    assert masked is not None
    assert "shortened" in masked[1]["content"]
    assert [m["content"] for m in masked[3:]] == [BIG, BIG, BIG]


def test_the_note_never_asks_for_the_call_again():
    [_, masked] = mask_old_tool_results([CALL, _tool(BIG, "a"), CALL, _tool("z", "b")], keep_steps=1)[:2]
    note = masked["content"].lower()
    assert "don't call the tool again" in note
    assert "call the tool again if" not in note


def test_text_parts_are_masked_too():
    messages = [CALL, _tool([{"type": "text", "text": "y" * 3_000}], "a"), CALL, _tool("z", "b")]
    masked = mask_old_tool_results(messages, keep_steps=1)
    assert masked is not None and "3,000 characters" in masked[1]["content"]


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "x" * 10_000}],
        [CALL, _tool("short", "a"), CALL, _tool("x" * 10_000, "b")],
        [CALL, _tool("x" * 10_000, "a"), _tool("x" * 10_000, "b")],
    ],
)
def test_nothing_to_mask(messages):
    assert mask_old_tool_results(messages, keep_steps=1) is None
