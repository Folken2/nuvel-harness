"""The parts of a plugin besides its skills, listed for the catalogue.

Commands (M6-8) are listed here; a turn that runs one makes it a skill
(``tools/skills/command.py``). Connectors are read by
``connectors.py`` and mounted per turn (``tools/plugin_connectors/``). Hooks and
subagents are listed with a warning, since this service never runs a plugin's shell
hook and has no subagents.
"""

from __future__ import annotations

import json
import posixpath
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import yaml

from .archive import ArchiveRefused
from .connectors import Connector, read_connectors
from .manifest import Manifest
from .skills import SKILL_FILE


@dataclass(frozen=True)
class Command:
    name: str
    description: str | None


@dataclass(frozen=True)
class Components:
    commands: tuple[Command, ...]
    connectors: tuple[Connector, ...]
    hook_events: tuple[str, ...]
    agents: tuple[str, ...]
    warnings: tuple[str, ...]


def read_components(files: Mapping[str, bytes], manifest: Manifest) -> Components:
    commands = _markdown_parts(files, manifest.command_dirs)
    agents = tuple(part.name for part in _markdown_parts(files, manifest.agent_dirs))
    connectors = read_connectors(_json_part(files, manifest.connectors, "mcpServers"))
    hook_events = _hook_events(_json_part(files, manifest.hooks, "hooks"))
    warnings = []
    if hook_events:
        warnings.append("Hooks are listed but never run here.")
    if agents:
        warnings.append("Subagents aren't supported yet.")
    if any(c.kind == "local" for c in connectors):
        warnings.append("Connectors that start a local command can't run here.")
    return Components(
        commands=commands,
        connectors=connectors,
        hook_events=hook_events,
        agents=agents,
        warnings=tuple(warnings),
    )


def _markdown_parts(files: Mapping[str, bytes], folders: tuple[str, ...]) -> tuple[Command, ...]:
    """``<folder>/<name>.md`` files directly in a folder, with their description."""
    found: dict[str, Command] = {}
    for folder in folders:
        for path in sorted(files):
            parent, base = posixpath.split(path)
            if parent != folder or not base.endswith(".md") or base == SKILL_FILE:
                continue
            name = base[: -len(".md")]
            found.setdefault(name, Command(name, _description(files[path])))
    return tuple(found[name] for name in sorted(found))


def _description(raw: bytes) -> str | None:
    text = raw.decode("utf-8", errors="replace")
    if not text.startswith("---"):
        return None
    head = text[3:].split("\n---", 1)[0]
    try:
        meta = yaml.safe_load(head)
    except yaml.YAMLError:
        return None
    value = meta.get("description") if isinstance(meta, dict) else None
    return value.strip() if isinstance(value, str) and value.strip() else None


def _json_part(files: Mapping[str, bytes], where: str | Mapping[str, Any], field: str) -> Mapping[str, Any]:
    if not isinstance(where, str):
        return where
    if where not in files:
        return {}
    try:
        value = json.loads(files[where].decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ArchiveRefused("bad_manifest", f"{where} isn't valid JSON.") from None
    if not isinstance(value, dict):
        raise ArchiveRefused("bad_manifest", f"{where} must be a JSON object.")
    # .mcp.json and hooks.json wrap their entries in one key; a bare object is allowed.
    inner = value.get(field)
    return inner if isinstance(inner, dict) else value


def _hook_events(hooks: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(name for name, value in hooks.items() if isinstance(value, list) and value))
