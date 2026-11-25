"""Keeps the whole of a tool result the model is shown cut, as a file of its conversation.

Two moments cut a result: its own size when the tool returns it (``keep_whole``), and its
age once the prompt is nearly full (``keep_masked``, context/old_results.py).

Saved through the run's artifact service like every file the agent saves, so it has the
same owner and the same end: the web app checks the person owns the conversation before
the save (/api/chat/adk-artifacts), keeps it in ``agent_artifacts`` under that
conversation, and deletes it with the conversation. It lives under ``markdown/``, where
read_document reads text a chunk at a time, and the web app shows nothing under there as
a file of the chat: it is for the agent to read, not a result for the person.

A save that fails returns None, and the model is told the result wasn't kept. It never
fails the call: the call has run, and runs at most once.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import ToolContext
from google.genai import types

from .conversation_files import keeps_files
from .layout import layout
from .names import safe_name

logger = logging.getLogger(__name__)


def copy_name(tool_name: str) -> str:
    """The name the model reads the copy by: unique, since one tool can be cut twice in a run."""
    return layout().tool_result(f"{safe_name(tool_name)}-{uuid.uuid4().hex[:12]}.txt")


def is_copy(artifact_name: str) -> bool:
    return layout().is_tool_result(artifact_name)


async def keep_whole(tool_context: ToolContext, tool_name: str, text: str) -> Optional[str]:
    """Saves ``text``; the name to read it by (with read_document), or None when it isn't kept."""
    if not keeps_files(tool_context.state):
        return None
    name = copy_name(tool_name)
    try:
        await tool_context.save_artifact(layout().reading_of(name), types.Part(text=text))
    except Exception as error:  # noqa: BLE001 - the cut result still reaches the model
        # The error's type, never its text: it can carry the result.
        logger.warning("couldn't keep the whole %s result: %s", tool_name, type(error).__name__)
        return None
    return name


def masked_copy_name(tool_name: str, key: str) -> str:
    """The name a masked result is read by. From the result's own key, so every later
    request names the same file without anything being remembered."""
    return layout().tool_result(f"{safe_name(tool_name)}-{key[:12]}.txt")


async def keep_masked(context: CallbackContext, tool_name: str, key: str, text: str) -> Optional[str]:
    """Saves ``text`` under the masked result's name; the name, or None when it isn't kept."""
    if not keeps_files(context.state):
        return None
    name = masked_copy_name(tool_name, key)
    try:
        await context.save_artifact(layout().reading_of(name), types.Part(text=text))
    except Exception as error:  # noqa: BLE001 - the result then stays whole in the request
        logger.warning("couldn't keep the masked %s result: %s", tool_name, type(error).__name__)
        return None
    return name
