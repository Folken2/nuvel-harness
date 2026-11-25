"""Keeping a checked page with its conversation (MCP Apps).

A page is kept as one of the conversation's files, under the layout's ``pages`` place
(files/layout.py), through ADK's public artifact API. So it goes wherever the
conversation's files go, under the same owner, and is removed with them. It is never
listed to the model (files/artifacts.py) nor shown as a file. Keeping is idempotent per
call: the same page for the same call is kept once.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

from google.adk.tools import ToolContext
from google.genai import types

from ...files.conversation_files import keeps_files
from ...files.layout import layout
from .page import Page

logger = logging.getLogger(__name__)

RECORD_VERSION = 1


@dataclass(frozen=True)
class PageRef:
    """What a client is told about a kept page: the file it's under, and its hash."""

    name: str
    sha256: str

    def as_wire(self) -> dict[str, str]:
        return {"name": self.name, "sha256": self.sha256}


def page_record(page: Page, *, server: str, tool: str) -> str:
    """The kept form: the page and what it may reach, as one JSON document."""
    return json.dumps(
        {
            "version": RECORD_VERSION,
            "uri": page.uri,
            "server": server,
            "tool": tool,
            "sha256": page.sha256,
            "csp": page.csp,
            "prefersBorder": page.prefers_border,
            "html": page.html,
        },
        ensure_ascii=False,
    )


def _kept_hash(part: Optional[types.Part]) -> Optional[str]:
    try:
        return json.loads(part.text or "").get("sha256") if part is not None else None
    except (ValueError, AttributeError):
        return None


async def keep_page(tool_context: ToolContext, page: Page, *, server: str, tool: str) -> Optional[PageRef]:
    """Keeps the call's page; its reference, or None when it isn't kept. Never raises."""
    call_id = tool_context.function_call_id
    if not call_id or not keeps_files(tool_context.state):
        return None
    name = layout().page_of(call_id)
    try:
        if _kept_hash(await tool_context.load_artifact(name)) != page.sha256:
            await tool_context.save_artifact(name, types.Part(text=page_record(page, server=server, tool=tool)))
    except Exception as error:  # noqa: BLE001 - a page never costs the call
        logger.warning("couldn't keep an MCP App page from %s: %s", server, type(error).__name__)
        return None
    return PageRef(name=name, sha256=page.sha256)
