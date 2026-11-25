"""Where a turn's plugins come from, as the tools that serve them see it.

The skill toolset and the connector toolset (skills/, plugin_connectors/) are handed a
``PluginSource`` and read plugins through these shapes. They import nothing of the
library that loads, stores and checks plugins: that is the product's (library/), and
its loader and its ``Plugin`` fit these.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional, Protocol

from google.adk.skills import Skill

# The file that makes a folder a skill.
SKILL_FILE = "SKILL.md"


class Connector(Protocol):
    name: str
    url: Optional[str]
    transport: str
    headers: Mapping[str, str]
    # Why it can never run from this service, when it can't.
    problem: Optional[str]


class Parts(Protocol):
    connectors: Sequence[Connector]


class Layout(Protocol):
    """Where a plugin keeps its parts, from its root."""

    skill_dirs: Sequence[str]
    command_dirs: Sequence[str]
    agent_dirs: Sequence[str]
    # A path to a JSON file, or the hooks themselves.
    hooks: Any


class Plugin(Protocol):
    skills: Sequence[Skill]
    # Every file by path from the plugin root.
    files: Mapping[str, bytes]
    content_hash: str

    @property
    def name(self) -> str: ...

    @property
    def manifest(self) -> Layout: ...

    @property
    def components(self) -> Parts: ...


class TurnPlugins(Protocol):
    loaded: Sequence[Plugin]
    # The names of plugins the turn names but goes without.
    unavailable: Sequence[str]


class PluginSource(Protocol):
    def pinned_hashes(self, state: Mapping[str, Any]) -> tuple[str, ...]:
        """The content hash of each plugin version the turn names, read from its state
        alone: what a turn's toolsets are kept under."""
        ...

    async def for_turn(self, state: Mapping[str, Any]) -> TurnPlugins:
        """Those plugins, loaded."""
        ...
