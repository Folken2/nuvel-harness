"""A reply cut at the output limit: none of its tool calls runs.

When a reply stops at the output limit while it writes a tool call, the provider's
stream ends mid-arguments, and what is parsed from it can be a call whose arguments are
valid but incomplete: half a document to save, a message missing its end. Running it
would do something the model never asked for, and a tool call runs at most once. So
every call in such a reply is answered with a refusal instead, and the model, which
sees why, sends the calls again (pi, ``packages/agent/src/agent-loop.ts``).

Most often the arguments don't parse at all, and ADK's model client drops the call
(``dropped_call``). The reply gets a note (``agent_note.py``) saying the call was
dropped, once per run, so the model sends it again in smaller pieces instead of the turn
failing. Unstreamed, that is one reply cut at the limit with its text and no call. A
stream ends in two: the text alone, cut at the limit, then the dropped call with no
content. The text stands (``text_before_dropped_call``), since the turn goes on; what
tells the two apart from a text answer cut at the limit is the partial call the stream
sent while it wrote the arguments (``writes_call``).

A reply cut with no tool call ever in it is left as the provider gave it: its text is
the answer so far.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from google.adk.models import LlmResponse
from google.genai import types

_MAX_TOKENS = types.FinishReason.MAX_TOKENS
# How ADK's LiteLLM client says, on an unstreamed reply, that it dropped calls whose
# arguments don't parse; it is appended to the output limit's message
# (google/adk/models/lite_llm.py, _apply_provider_finish_reason). Pinned by a test
# through that client; the streamed path by a replay test.
_DROPPED = "were not a valid JSON object"

DROPPED_KIND = "cut_call_dropped"

DROPPED_NOTE = (
    "Your last reply reached the output limit while it was writing a tool call, so the "
    "call was cut off and not run. Send it again with complete arguments. If its "
    "content is long, split it across several smaller calls."
)


def cut_calls(response: LlmResponse) -> list[types.FunctionCall]:
    """The tool calls of a reply that stopped at the output limit; none for any other."""
    if response.partial or response.finish_reason != _MAX_TOKENS or not response.content:
        return []
    return [part.function_call for part in response.content.parts or [] if part.function_call]


def writes_call(response: LlmResponse) -> bool:
    """A partial response of a stream that is writing a tool call."""
    parts = response.content.parts if response.content else None
    return bool(response.partial) and any(part.function_call for part in parts or [])


def _cut_without_calls(response: LlmResponse) -> Optional[list[types.Part]]:
    """A finished reply cut at the output limit with no call in it: its parts; else None."""
    if response.partial or response.finish_reason != _MAX_TOKENS:
        return None
    parts = list(response.content.parts or []) if response.content else []
    return None if any(part.function_call for part in parts) else parts


def dropped_call(response: LlmResponse, call_streamed: bool) -> bool:
    """A finished reply cut at the output limit whose call ADK dropped for broken
    arguments. ``call_streamed``: this model call's stream was writing a tool call."""
    parts = _cut_without_calls(response)
    if parts is None:
        return False
    return _DROPPED in (response.error_message or "") or (call_streamed and not parts)


def text_before_dropped_call(response: LlmResponse, call_streamed: bool) -> bool:
    """The text a stream sends on its own, cut at the output limit, before the call it
    was writing is dropped."""
    parts = _cut_without_calls(response)
    return call_streamed and bool(parts)


def held_ids(calls: list[types.FunctionCall]) -> list[str]:
    """Each call's id, given one where it has none, so its refusal can find it."""
    for call in calls:
        if not call.id:
            call.id = f"call-{uuid.uuid4().hex}"
    return [call.id for call in calls if call.id]


def clear_limit_error(response: LlmResponse) -> None:
    """The reply stands, its calls answered: not an error that ends the turn."""
    if response.error_code == _MAX_TOKENS:
        response.error_code = None
        response.error_message = None


def refusal(tool_name: str) -> dict[str, Any]:
    return {
        "status": "error",
        "error": (
            f"Not run: your reply reached the output limit while it was writing this "
            f"{tool_name} call, so its arguments may be incomplete. Send the call again "
            "with complete arguments. If its content is long, split it across several "
            "smaller calls."
        ),
    }
