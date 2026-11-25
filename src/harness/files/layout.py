"""Where a conversation's files live by name: a layout the product sets.

The file modules (lookup.py, names.py, tool_result_copy.py) and the tools that read a
file name four places: where an upload's original is kept, where a job saves what it
shares, where text read a chunk at a time lives, and under that, the whole of a cut
tool result. What each place is called belongs to whoever stores the files, so it is
data: a product calls ``set_layout`` once, and every name is built from ``layout()``
at the call, never at import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = ["layout"]


@dataclass(frozen=True)
class FileLayout:
    """The prefixes of a conversation's file names. Each ends with its separator."""

    # The kept original of an upload, where the person's edits of it live too.
    original: str
    # Where a job saves what it shares; the file is kept under its name without this.
    output: str
    # Text the agent reads a chunk at a time, never shown as a file of the conversation.
    reading: str
    # Under ``reading``: the whole of a tool result the model was shown cut.
    tool_results: str
    # A page a server attached to a tool call (MCP Apps, tools/mcp_apps/): never a file of
    # the conversation, never listed to the model.
    pages: str

    def plain(self, name: str) -> str:
        """``name`` without the original's prefix."""
        return name.removeprefix(self.original)

    def original_of(self, name: str) -> str:
        return f"{self.original}{name}"

    def saved_output(self, name: str) -> Optional[str]:
        """The name a job's output is kept under; None when ``name`` isn't under the output place."""
        if name.startswith(self.output) and name != self.output:
            return name.removeprefix(self.output)
        return None

    def read_name(self, name: str) -> str:
        """``name`` as the reader is given it: without the reading prefix."""
        return name.removeprefix(self.reading)

    def reading_of(self, name: str) -> str:
        """Where the text read under ``name`` is stored."""
        return f"{self.reading}{name}"

    def tool_result(self, name: str) -> str:
        """The name a kept tool result is read by."""
        return f"{self.tool_results}{name}"

    def is_tool_result(self, stored_name: str) -> bool:
        return stored_name.startswith(f"{self.reading}{self.tool_results}")

    def page_of(self, call_id: str) -> str:
        """Where the page attached to this tool call is kept."""
        return f"{self.pages}{call_id}.json"

    def is_page(self, stored_name: str) -> bool:
        return stored_name.startswith(self.pages)


_layout: Optional[FileLayout] = None


def set_layout(layout: Optional[FileLayout]) -> None:
    """Sets the layout every file name is built from. None unsets it."""
    global _layout
    _layout = layout


def layout() -> FileLayout:
    if _layout is None:
        raise LookupError("no file layout is set: a product calls files.layout.set_layout first")
    return _layout
