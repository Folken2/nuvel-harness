"""A plugin's three manifests read into one.

The plugins repo ships ``plugin.json`` (agent-plugins.org), ``.claude-plugin/plugin.json``
and ``.codex-plugin/plugin.json``, and keeps name and version the same in all three.
Any of them is enough; those present must agree. Where the parts live follows the
Claude plugin layout: a default folder per part, plus any paths the manifest names.
"""

from __future__ import annotations

import json
import posixpath
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .archive import MANIFEST_NAMES, ArchiveRefused

# The same rule as an ADK skill's name: lowercase kebab-case, at most 64 characters.
_NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*\Z")
_NAME_MAX = 64
# Semantic versioning 2.0: MAJOR.MINOR.PATCH, an optional pre-release and build.
_VERSION = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?\Z"
)
_VERSION_MAX = 64

DEFAULT_SKILLS = "skills"
DEFAULT_COMMANDS = "commands"
DEFAULT_AGENTS = "agents"
DEFAULT_HOOKS = "hooks/hooks.json"
DEFAULT_CONNECTORS = ".mcp.json"


@dataclass(frozen=True)
class Manifest:
    name: str
    version: str
    description: str
    display_name: str | None
    author: str | None
    homepage: str | None
    license: str | None
    keywords: tuple[str, ...]
    # Folders holding the parts, from the plugin root; the defaults come first.
    skill_dirs: tuple[str, ...]
    command_dirs: tuple[str, ...]
    agent_dirs: tuple[str, ...]
    # A path to a JSON file, or the manifest's own object.
    hooks: str | Mapping[str, Any]
    connectors: str | Mapping[str, Any]


def read_manifest(files: Mapping[str, bytes]) -> Manifest:
    present = {name: _parse(name, files[name]) for name in MANIFEST_NAMES if name in files}
    if not present:
        raise ArchiveRefused("no_manifest", "No plugin.json at the plugin's root.")
    for field in ("name", "version"):
        values = {str(manifest.get(field)) for manifest in present.values()}
        if len(values) > 1:
            raise ArchiveRefused(
                "manifests_disagree",
                f"The manifests give different {field}s: {', '.join(sorted(values))}.",
            )
    first = next(iter(present.values()))
    name, version = first.get("name"), first.get("version")
    if not isinstance(name, str) or len(name) > _NAME_MAX or not _NAME.match(name):
        raise ArchiveRefused("bad_name", "The plugin's name must be lowercase kebab-case, at most 64 characters.")
    if not isinstance(version, str) or len(version) > _VERSION_MAX or not _VERSION.match(version):
        raise ArchiveRefused("bad_version", "The plugin's version must be a semantic version such as 1.2.0.")

    # Where the parts are: the Claude manifest says it most fully, then the others.
    ordered = [present[n] for n in (".claude-plugin/plugin.json", "plugin.json", ".codex-plugin/plugin.json") if n in present]
    interface = present.get(".codex-plugin/plugin.json", {}).get("interface")
    return Manifest(
        name=name,
        version=version,
        description=_first_text(ordered, "description") or "",
        display_name=_text(interface.get("displayName")) if isinstance(interface, dict) else None,
        author=_author(ordered),
        homepage=_first_text(ordered, "homepage"),
        license=_first_text(ordered, "license"),
        keywords=_keywords(ordered),
        skill_dirs=_dirs(ordered, "skills", DEFAULT_SKILLS),
        command_dirs=_dirs(ordered, "commands", DEFAULT_COMMANDS),
        agent_dirs=_dirs(ordered, "agents", DEFAULT_AGENTS),
        hooks=_file_or_object(ordered, "hooks", DEFAULT_HOOKS),
        connectors=_file_or_object(ordered, "mcpServers", DEFAULT_CONNECTORS),
    )


def _parse(name: str, raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ArchiveRefused("bad_manifest", f"{name} isn't valid JSON.") from None
    if not isinstance(value, dict):
        raise ArchiveRefused("bad_manifest", f"{name} must be a JSON object.")
    return value


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _first_text(manifests: list[dict[str, Any]], field: str) -> str | None:
    for manifest in manifests:
        if text := _text(manifest.get(field)):
            return text
    return None


def _author(manifests: list[dict[str, Any]]) -> str | None:
    for manifest in manifests:
        author = manifest.get("author")
        if isinstance(author, dict) and (name := _text(author.get("name"))):
            return name
        if name := _text(author):
            return name
    return None


def _keywords(manifests: list[dict[str, Any]]) -> tuple[str, ...]:
    for manifest in manifests:
        keywords = manifest.get("keywords")
        if isinstance(keywords, list):
            return tuple(k for k in keywords if isinstance(k, str))
    return ()


def plugin_path(value: str, field: str) -> str:
    """A path the manifest gives, from the plugin root: ``./skills/`` is ``skills``."""
    if value.startswith("/") or "\\" in value:
        raise ArchiveRefused("bad_manifest", f"{field} must be a path inside the plugin.")
    path = posixpath.normpath(value)
    if path == ".." or path.startswith("../"):
        raise ArchiveRefused("bad_manifest", f"{field} must be a path inside the plugin.")
    return "" if path == "." else path


def _dirs(manifests: list[dict[str, Any]], field: str, default: str) -> tuple[str, ...]:
    """The default folder and any the manifest adds, as the Claude layout does."""
    found = [default]
    for manifest in manifests:
        value = manifest.get(field)
        listed = [value] if isinstance(value, str) else value if isinstance(value, list) else []
        for item in listed:
            if not isinstance(item, str):
                raise ArchiveRefused("bad_manifest", f"{field} must be a path or a list of paths.")
            found.append(plugin_path(item, field))
        if listed:
            break
    return tuple(dict.fromkeys(found))


def _file_or_object(manifests: list[dict[str, Any]], field: str, default: str) -> str | Mapping[str, Any]:
    for manifest in manifests:
        value = manifest.get(field)
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            return plugin_path(value, field)
    return default
