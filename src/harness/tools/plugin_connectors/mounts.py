"""The connectors a turn's plugins bring, as servers to mount, each under its plugin's name.

A connector's tools are named ``mcp_<plugin>_<connector>__<tool>``. The one ``_`` in
the middle is what keeps a plugin's connector apart from a person's own server, whose
name never holds one (the web app's ``slugFromName``), and from another plugin's
connector of the same name. A connector gets its own entry's address and headers and
nothing else: none of the person's servers' headers, no other connector's.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..plugin_source import Plugin
from ..user_tools.mcp_toolsets import MAX_TOOL_NAME
from .missing import Missing

SAME_NAME = "another connector of this plugin has the same name"
NAME_TOO_LONG = "its name and its plugin's are too long to leave room for its tools' names"
# Below this, most tools' names won't fit: an author is told when the plugin is checked.
ROOM_WORTH_A_WARNING = 20

_NOT_IN_A_NAME = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Mount:
    plugin: str
    connector: str
    # What its tools are prefixed with, and what a notice calls it.
    name: str
    # The descriptor user_tools' build_mcp_toolset takes.
    server: dict[str, Any]


def _part(name: str) -> str:
    return _NOT_IN_A_NAME.sub("-", name.strip().lower()).strip("-")


def mount_name(plugin: str, connector: str) -> str:
    return f"{_part(plugin)}_{_part(connector)}"


def room_for_tool_names(plugin: str, connector: str) -> int:
    """How long a tool's own name may be once ``mcp_<plugin>_<connector>__`` is in
    front of it. A longer one is left out, since a model provider refuses it."""
    return MAX_TOOL_NAME - len(f"mcp_{mount_name(plugin, connector)}__")


def name_warnings(plugin: Plugin) -> tuple[str, ...]:
    """What an author is told when a connector's name leaves its tools little room."""
    found = []
    for connector in plugin.components.connectors:
        room = room_for_tool_names(plugin.name, connector.name)
        if room < 1:
            found.append(f"Connector {connector.name}: {NAME_TOO_LONG}, so none of its tools can be used.")
        elif room < ROOM_WORTH_A_WARNING:
            found.append(
                f"Connector {connector.name}: with the plugin's name, its tools' names can be at most "
                f"{room} characters; a longer one is left out."
            )
    return tuple(found)


def connectors_of(plugins: Iterable[Plugin]) -> tuple[list[Mount], list[Missing]]:
    """Every connector of these plugins: one to try, or one the turn goes without."""
    mounts: list[Mount] = []
    missing: list[Missing] = []
    taken: set[str] = set()
    for plugin in plugins:
        for connector in plugin.components.connectors:
            if connector.problem is not None or connector.url is None:
                missing.append(Missing(plugin.name, connector.name, connector.problem or "it has no address"))
                continue
            name = mount_name(plugin.name, connector.name)
            if room_for_tool_names(plugin.name, connector.name) < 1:
                missing.append(Missing(plugin.name, connector.name, NAME_TOO_LONG))
                continue
            if name in taken:
                missing.append(Missing(plugin.name, connector.name, SAME_NAME))
                continue
            taken.add(name)
            server = {
                "slug": name,
                "url": connector.url,
                "transport": connector.transport,
                "headers": dict(connector.headers),
            }
            mounts.append(Mount(plugin.name, connector.name, name, server))
    return mounts, missing
