"""Whether a model call's prompt starts the way the last one did.

A provider's prompt cache holds a prefix: the instruction, the tools, then the
conversation in order. A call reads it from the cache only while that start is the same,
byte for byte, as an earlier call's. Anything that changes it (a date in the
instruction, a tool that comes and goes, an old message rewritten) makes the call pay
full price for everything after the change, and nothing says so.

This compares a request with the one before it in the same session and says where the
first difference is. It measures; it changes nothing. Pure: the plugin
(``plugins/prefix_watch.py``) remembers the last request of each session and logs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional

from google.adk.models import LlmRequest

# Where the prefix first differs from the previous call's.
SAME = "none"  # everything sent before is sent again unchanged; only new messages follow
INSTRUCTION = "instruction"
TOOLS = "tools"
HISTORY = "history"  # an earlier message changed or went: the cache holds up to it
FIRST = "first"  # the session's first call here: nothing to compare with


@dataclass(frozen=True)
class Fingerprint:
    instruction: str
    tools: str
    # One digest per message of the conversation, in order.
    messages: tuple[str, ...]


@dataclass(frozen=True)
class Comparison:
    change: str
    # How many leading messages are the same as the previous call's, and how many it had.
    kept: int
    before: int
    # How many messages this call sends.
    now: int

    @property
    def broken(self) -> bool:
        """Whether part of what was cached can't be read back."""
        return self.change in (INSTRUCTION, TOOLS, HISTORY)


def _digest(value: Any) -> str:
    text = json.dumps(value, default=str, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _dumped(value: Any) -> Any:
    return value.model_dump(mode="json", exclude_none=True) if hasattr(value, "model_dump") else value


def fingerprint(request: LlmRequest) -> Fingerprint:
    config = request.config
    instruction = _dumped(config.system_instruction) if config and config.system_instruction else ""
    tools = [_dumped(tool) for tool in (config.tools if config and config.tools else [])]
    return Fingerprint(
        instruction=_digest(instruction),
        tools=_digest(tools),
        messages=tuple(_digest(_dumped(content)) for content in request.contents),
    )


def compare(previous: Optional[Fingerprint], current: Fingerprint) -> Comparison:
    """Where ``current`` first differs from ``previous``, in the order a provider reads a
    prompt: the tools and the instruction, then the conversation."""
    now = len(current.messages)
    if previous is None:
        return Comparison(FIRST, 0, 0, now)
    before = len(previous.messages)
    kept = 0
    for old, new in zip(previous.messages, current.messages):
        if old != new:
            break
        kept += 1
    if previous.tools != current.tools:
        return Comparison(TOOLS, 0, before, now)
    if previous.instruction != current.instruction:
        return Comparison(INSTRUCTION, 0, before, now)
    return Comparison(SAME if kept == before else HISTORY, kept, before, now)
