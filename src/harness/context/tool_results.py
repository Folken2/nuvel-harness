"""Cuts oversized tool results before the model sees them.

Every later model call in a run re-sends earlier tool results, so one huge
result (a full wiki, a long email thread) multiplies the cost of the whole
turn. A result over the limit keeps its start and its end, with the cut in the
middle: the start usually says what the result is, the end how it finished
(a total, an error, the last page). The note between them says what was left
out and, when the whole result was kept as a file of the chat, its name, so
the model can read the rest instead of calling the tool again.

The kept text is ``result_text(result)``, the same text the saved file holds, so
the positions the note gives are positions in that file.

Pure: the plugin (plugins/tool_result_limit.py) saves the file.
"""

from __future__ import annotations

import json
from typing import Any, Optional


def result_size(value: Any) -> int:
    return len(json.dumps(value, default=str, ensure_ascii=False))


def fits(result: Any, max_chars: int) -> bool:
    """Whether the result goes to the model as it is (a limit of 0 or less: all do)."""
    return max_chars <= 0 or result_size(result) <= max_chars


def leaves_out(text: str, max_chars: int) -> bool:
    """Whether the cut of a result whose text is ``text`` leaves part of it out, so there
    is something to keep whole."""
    return len(text) > max_chars


def _is_mcp(result: Any) -> bool:
    return isinstance(result, dict) and isinstance(result.get("content"), list)


def result_text(result: Any) -> str:
    """The result as one text: an MCP result's items joined, text as it is, a picture or
    another binary item as a short mark (``_binary_mark``), anything else as JSON
    (structuredContent usually repeats the text, so it goes); a string as it is,
    anything else as JSON."""
    if _is_mcp(result):
        return "\n\n".join(_item_text(item) for item in result["content"])
    if isinstance(result, str):
        return result
    return json.dumps(result, default=str, ensure_ascii=False)


def _item_text(item: Any) -> str:
    if isinstance(item, dict) and isinstance(item.get("text"), str):
        return item["text"]
    mark = _binary_mark(item)
    if mark is not None:
        return mark
    return json.dumps(item, default=str, ensure_ascii=False)


def _binary_mark(item: Any) -> Optional[str]:
    """A picture, a sound or a blob resource, named instead of shown: its base64 means
    nothing as text, so neither the cut nor the kept copy holds it."""
    if not isinstance(item, dict):
        return None
    resource = item.get("resource") if isinstance(item.get("resource"), dict) else {}
    data = item.get("data") if isinstance(item.get("data"), str) else resource.get("blob")
    if not isinstance(data, str):
        return None
    kind = item.get("mimeType") or resource.get("mimeType") or item.get("type") or "binary"
    size = len(data) * 3 // 4
    return f"[{kind}, about {size:,} bytes, not shown]"


def _note(size: int, head: int, tail_start: int, length: int, saved_as: Optional[str], reader: str) -> str:
    shown = (
        f"[Cut: this tool result was {size:,} characters, too long to show whole. Shown "
        f"are its first {head:,} and its last {length - tail_start:,} characters; the "
        f"{tail_start - head:,} between them are left out here."
    )
    if saved_as is not None:
        return (
            f"{shown} The whole result is saved in this conversation as {saved_as!r}. "
            f"To read the part left out, call {reader} with name {saved_as!r} and "
            f"start={head}, or with query to search it. Read the saved copy rather than "
            "calling the tool again.]"
        )
    return (
        f"{shown} The whole result couldn't be kept. If you need the part left out, call "
        "the tool again with a narrower query or ask for a specific section.]"
    )


def trim_tool_result(
    result: Any, max_chars: int, saved_as: Optional[str] = None, reader: str = "the file reader"
) -> Optional[dict[str, Any]]:
    """The cut result, or None when it already fits. ``saved_as`` names the file that
    holds ``result_text(result)`` whole, and ``reader`` the tool that reads it a chunk
    at a time (by name, start and query); None says it wasn't kept."""
    if fits(result, max_chars):
        return None
    size = result_size(result)
    text = result_text(result)
    if not leaves_out(text, max_chars):
        # Over the limit only as JSON (escaping, a repeated structuredContent): the
        # text alone fits, so nothing is left out.
        return _shaped(result, [text])
    # Half each end.
    head = max_chars // 2
    tail_start = len(text) - (max_chars - head)
    note = _note(size, head, tail_start, len(text), saved_as, reader)
    return _shaped(result, [text[:head], note, text[tail_start:]], cut=True)


def _shaped(result: Any, texts: list[str], cut: bool = False) -> dict[str, Any]:
    """An MCP result keeps its other keys (isError above all) and its shape."""
    if _is_mcp(result):
        trimmed = {
            key: value
            for key, value in result.items()
            if key not in ("content", "structuredContent")
        }
        trimmed["content"] = [{"type": "text", "text": text} for text in texts]
        return trimmed
    if not cut:
        return {"result": texts[0]}
    start, note, end = texts
    return {"truncated": True, "start": start, "note": note, "end": end}
