"""Function tools whose parameters are described in their schema.

ADK builds a function tool's schema from its signature with pydantic, and sends the whole
docstring as the tool's description. A docstring's ``Args:`` block so reaches the model as
prose, while each property carries only a type and a generated ``title``. A model reads a
parameter's ``description`` where it fills the parameter in, so this moves each ``Args:``
entry there, once, for every tool built with these classes.

Also, on the way, two things pydantic writes that a model never needs:
- ``title`` on every schema (``"Task Id"``, ``"check_taskParams"``);
- an optional parameter as ``anyOf`` the type or null with a null default: leaving it out
  means the same, so it is declared as its type alone.

A ``description`` written in the signature (``Annotated[int, Field(description=...)]``)
stands; the ``Args:`` entry for it is dropped. Limits and fixed sets go in the signature
too (``Literal``, ``Field(ge=..., le=...)``), and pydantic puts them in the schema.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Optional

from google.adk.tools import FunctionTool, LongRunningFunctionTool
from google.genai import types

_ARGS = re.compile(r"^(?P<indent>[ \t]*)Args:[ \t]*$", re.MULTILINE)
_ENTRY = re.compile(r"^(?P<name>\w+)(?:[ \t]*\([^)]*\))?:[ \t]*(?P<text>.*)$")


def split_args(doc: str) -> tuple[str, dict[str, str]]:
    """The description without its ``Args:`` block, and each entry's text by name.

    An entry is ``name: text`` at the block's first indent, its text running on over the
    lines indented deeper. The block ends at a line indented no deeper than ``Args:``
    (such as the note ADK appends to a long-running tool's description).
    """
    found = _ARGS.search(doc)
    if found is None:
        return doc, {}
    base = len(found.group("indent"))
    lines = doc[found.end() :].split("\n")[1:]
    entries: dict[str, list[str]] = {}
    current: Optional[list[str]] = None
    entry_indent: Optional[int] = None
    used = 0
    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if not stripped:
            used += 1
            continue
        if indent <= base:
            break
        if entry_indent is None:
            entry_indent = indent
        entry = _ENTRY.match(stripped) if indent == entry_indent else None
        if entry is not None:
            current = entries.setdefault(entry.group("name"), [])
            current.append(entry.group("text"))
        elif current is not None and indent > entry_indent:
            current.append(stripped)
        else:
            break
        used += 1
    rest = "\n".join(lines[used:]).strip("\n")
    before = doc[: found.start()].rstrip()
    description = f"{before}\n\n{rest}" if rest else before
    return description, {name: " ".join(part for part in parts if part) for name, parts in entries.items()}


def _plain(schema: Any) -> Any:
    """The schema without pydantic's titles and with optional types as the type alone."""
    if isinstance(schema, list):
        return [_plain(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "title" and isinstance(value, str):
            continue
        if key in ("properties", "$defs") and isinstance(value, dict):
            out[key] = {name: _plain(sub) for name, sub in value.items()}
        else:
            out[key] = _plain(value)
    variants = out.get("anyOf")
    if isinstance(variants, list) and "default" in out and out["default"] is None:
        kept = [v for v in variants if not (isinstance(v, dict) and v.get("type") == "null")]
        if len(kept) == 1 and len(kept) < len(variants) and isinstance(kept[0], dict):
            rest = {k: v for k, v in out.items() if k not in ("anyOf", "default")}
            out = {**kept[0], **rest}
    return out


def describe(declaration: types.FunctionDeclaration) -> types.FunctionDeclaration:
    """The declaration with its ``Args:`` entries in its properties, in place."""
    description, entries = split_args(declaration.description or "")
    schema = declaration.parameters_json_schema
    if isinstance(schema, dict):
        schema = _plain(copy.deepcopy(schema))
        for name, prop in (schema.get("properties") or {}).items():
            if name in entries and isinstance(prop, dict) and not prop.get("description"):
                prop["description"] = entries[name]
        declaration.parameters_json_schema = schema
    elif declaration.parameters is not None:
        for name, prop in (declaration.parameters.properties or {}).items():
            if name in entries and not prop.description:
                prop.description = entries[name]
    declaration.description = description or None
    return declaration


class DescribedTool(FunctionTool):
    """A function tool with its parameters described in its schema."""

    def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
        declaration = super()._get_declaration()
        return describe(declaration) if declaration is not None else None


class DescribedLongRunningTool(LongRunningFunctionTool):
    """The same for a tool whose answer comes later, such as the person's."""

    def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
        declaration = super()._get_declaration()
        return describe(declaration) if declaration is not None else None
