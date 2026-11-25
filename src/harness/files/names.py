"""File name rules shared by the tools that hand conversation files to a job."""

from __future__ import annotations

import re
from pathlib import PurePath

from .layout import layout

# Names the web app sends the extracted text under keep their extension; the text
# copy gets ".txt" appended so the program doesn't mistake it for the original.
_TEXT_SUFFIXES = {".txt", ".md", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml", ".html", ".htm"}
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9 ._()-]+")


def plain_name(name: str) -> str:
    return layout().plain(name)


def safe_name(name: str) -> str:
    cleaned = _UNSAFE_CHARS.sub("_", PurePath(name).name).strip(" ._") or "file"
    return cleaned[:150]


def text_copy_name(name: str) -> str:
    """The job name for a file's extracted text, given its safe name."""
    return name if PurePath(name).suffix.lower() in _TEXT_SUFFIXES else f"{name}.txt"
