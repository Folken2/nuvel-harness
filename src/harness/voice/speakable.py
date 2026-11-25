"""Reply text as it should be said: Markdown read aloud is noise."""

from __future__ import annotations

import re

__all__ = ["speakable"]

_FENCE = re.compile(r"```.*?(```|$)", re.DOTALL)
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
_QUOTE = re.compile(r"^\s*>\s?", re.MULTILINE)
_RULE = re.compile(r"^\s*(?:[-*_]\s*){3,}$", re.MULTILINE)
_TABLE_RULE = re.compile(r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*:?-*:?\s*$", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*\*|__|\*|_|~~|`)")


def speakable(text: str) -> str:
    """The words to say. Code is left out, since nobody wants it read character by character."""
    text = _FENCE.sub(" ", text)
    text = _IMAGE.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub("", text)
    text = _TABLE_RULE.sub("", text)
    text = _RULE.sub("", text)
    text = _HEADING.sub("", text)
    text = _BULLET.sub("", text)
    text = _QUOTE.sub("", text)
    text = text.replace("|", ", ")
    text = _EMPHASIS.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\s*\n\s*", "\n", text).strip()
