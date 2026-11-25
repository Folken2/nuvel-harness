"""Finds a conversation file: the newest of the original the web app kept and the
artifact the agent saved under the same name."""

from __future__ import annotations

import logging
from typing import Callable, Optional

from google.adk.tools import ToolContext
from google.genai import types

from .layout import layout

logger = logging.getLogger(__name__)


async def load_original_or_named(
    tool_context: ToolContext,
    name: str,
    accept: Optional[Callable[[types.Part], bool]] = None,
) -> Optional[types.Part]:
    """The latest version of ``name``: of its kept original (``original/<name>``,
    where an upload and the person's edits of it live) and of the artifact the agent
    saved as ``<name>``, the one saved last wins. With no save times to compare, the
    original comes first.

    ``name`` may carry the ``original/`` prefix or not; both find the same file. A
    name under ``output/`` that matches nothing is looked up without it: that's where a
    run saved the file, and models name it so in a later call.

    A text version never stands in for a file kept as bytes: the text the model was
    shown of an upload is saved under ``<name>`` a moment after its original, so it
    is the newer of the two, but code asking for a workbook needs the workbook.

    ``accept`` skips a part the caller can't use, moving on to the next one.
    """
    places = layout()
    plain = places.plain(name)
    names = [plain]
    if (saved := places.saved_output(plain)) is not None:
        names.append(saved)
    for each in names:
        original = places.original_of(each)
        series = await _newest_first(tool_context, [original, each])
        text_copy: Optional[types.Part] = None
        for candidate in series:
            part = await tool_context.load_artifact(candidate)
            if part is None or (accept is not None and not accept(part)):
                continue
            if text_copy is not None:
                # The original, older than the text: the file itself unless it's text too.
                return text_copy if _is_text(part) else part
            if candidate != original and _is_text(part):
                text_copy = part
                continue
            return part
        if text_copy is not None:
            return text_copy
    return None


def _is_text(part: types.Part) -> bool:
    if getattr(part, "text", None) is not None:
        return True
    data = getattr(part, "inline_data", None)
    return data is not None and (data.mime_type or "").startswith("text/")


async def _newest_first(tool_context: ToolContext, candidates: list[str]) -> list[str]:
    """The candidates by save time, newest first; given order where a time is missing."""
    times = [await _saved_at(tool_context, candidate) for candidate in candidates]
    if any(saved is None for saved in times):
        # One of them doesn't exist or has no time: nothing to compare, so the order stands.
        return candidates
    return [c for _, _, c in sorted(zip(times, range(len(candidates)), candidates), key=lambda t: (-t[0], t[1]))]


async def _saved_at(tool_context: ToolContext, filename: str) -> Optional[float]:
    get_version = getattr(tool_context, "get_artifact_version", None)
    if get_version is None:
        return None
    try:
        version = await get_version(filename)
    except Exception:  # noqa: BLE001 - a store that can't say keeps the old order
        logger.warning("artifact version lookup failed: %s", filename, exc_info=True)
        return None
    return getattr(version, "create_time", None) if version is not None else None
