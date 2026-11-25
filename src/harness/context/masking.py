"""Older tool results masked in a request, so it fits a smaller context.

Pure: it works on the messages LiteLLM sends (the OpenAI chat format, where a tool
result is a ``tool`` message after the assistant message that called it), never on the
session, so the history keeps every result and only the one request is smaller. The
results of the newest steps stay whole, every parallel result of a step together, since
the step the model is on needs them; an older one keeps its start and a note saying it
was cut.

The note never asks for the call again: a tool that sends or creates something would
do it twice.
"""

from __future__ import annotations

from typing import Any, Optional

# What an older result keeps of its start, so the model still sees what it was.
PREVIEW_CHARS = 200


def _note(size: int) -> str:
    return (
        f"[Earlier tool result shortened to fit the model's context: it was {size:,} "
        "characters. Work from what is shown. Don't call the tool again to get the rest, "
        "since that would do it again; if you need what was cut, ask the person.]"
    )


def _text_of(content: Any) -> Optional[str]:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [part.get("text") for part in content if isinstance(part, dict)]
        if texts and all(isinstance(text, str) for text in texts):
            return "".join(texts)
    return None


def _masked(text: str) -> str:
    return f"{text[:PREVIEW_CHARS]}\n{_note(len(text))}"


def _role(message: Any) -> Optional[str]:
    return message.get("role") if isinstance(message, dict) else None


def mask_old_tool_results(messages: list[Any], keep_steps: int) -> Optional[list[Any]]:
    """The messages with the tool results of all but the newest ``keep_steps`` steps
    shortened. A step is an assistant message and the tool results after it.

    None when nothing would get smaller: no older steps, or their results short already.
    """
    steps: list[list[int]] = []
    for i, message in enumerate(messages):
        if _role(message) == "assistant":
            steps.append([])
        elif _role(message) == "tool":
            if not steps:
                steps.append([])
            steps[-1].append(i)
    steps = [step for step in steps if step]
    older = [i for step in steps[: max(len(steps) - keep_steps, 0)] for i in step]
    masked = list(messages)
    changed = False
    for i in older:
        text = _text_of(messages[i].get("content"))
        if text is None:
            continue
        shorter = _masked(text)
        if len(shorter) >= len(text):
            continue
        masked[i] = {**messages[i], "content": shorter}
        changed = True
    return masked if changed else None
