"""A slash command as a skill (M6-8): the turn has it loaded before the model's first call.

What a command is comes in as a ``CommandRef``. The toolset is handed a ``CommandReader``
that finds the turn's in its state (this service's is ``library/command.py``, which
reads what the web app sent); nothing here knows where it came from. A ref is one of:

- ``source="own"``: a prompt the person saved. Its text comes with the ref (``instructions``)
  and becomes a skill here.
- ``source="plugin"``, ``kind="skill"``: one of the plugin's skills, by name.
- ``source="plugin"``, ``kind="command"``: the plugin's ``commands/<name>.md``, made into
  a skill of that name.

The toolset (``toolset.py``) puts the skill first in the turn's registry, activates it,
and at every model call of the turn says the person ran it, with its instructions. A
command this turn can't find (its plugin didn't load) is left out with a warning, and
the message runs as an ordinary one.
"""

from __future__ import annotations

import logging
import posixpath
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import yaml
from google.adk.skills import Frontmatter, Skill

from ..plugin_source import Plugin
from .registry import plugin_skills

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommandRef:
    source: str
    name: str
    plugin: str | None = None
    kind: str | None = None
    description: str = ""
    instructions: str = ""

    @property
    def key(self) -> tuple[str, ...]:
        """What tells two turns' commands apart, for the toolset's per-turn cache."""
        return (self.source, self.plugin or "", self.kind or "", self.name, self.instructions)


# Finds the turn's command in its state: None when the message didn't run one.
CommandReader = Callable[[Mapping[str, Any]], "CommandRef | None"]


def no_command(state: Mapping[str, Any]) -> CommandRef | None:
    """The reader of a toolset built with none: no turn runs a command."""
    return None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def command_skill(command: CommandRef, plugins: Iterable[Plugin]) -> Skill | None:
    """The skill the command runs, or None when this turn doesn't have it."""
    try:
        if command.source == "own":
            description = command.description or f"The person's own command /{command.name}."
            return Skill(frontmatter=Frontmatter(name=command.name, description=description), instructions=command.instructions)
        plugin = next((p for p in plugins if p.name == command.plugin), None)
        if plugin is None:
            logger.warning("command /%s left out: plugin %s isn't in this turn", command.name, command.plugin)
            return None
        if command.kind == "skill":
            found = next((s for s in plugin_skills(plugin) if s.name == command.name), None)
        else:
            found = _command_file_skill(plugin, command.name)
        if found is None:
            logger.warning("command /%s left out: plugin %s has no such %s", command.name, plugin.name, command.kind)
        return found
    except ValueError:
        # A name ADK won't take as a skill's: the web app's rule is the same, so this is a bad row.
        logger.warning("command /%s left out: its name isn't a skill name", command.name)
        return None


def _command_file_skill(plugin: Plugin, name: str) -> Skill | None:
    for folder in plugin.manifest.command_dirs:
        raw = plugin.files.get(posixpath.join(folder, f"{name}.md"))
        if raw is None:
            continue
        meta, body = _split_front_matter(raw.decode("utf-8", errors="replace"))
        description = _text(meta.get("description")) or f"The {plugin.name} command /{name}."
        return Skill(frontmatter=Frontmatter(name=name, description=description), instructions=body.strip())
    return None


def _split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    head, sep, body = text[3:].partition("\n---")
    if not sep:
        return {}, text
    try:
        meta = yaml.safe_load(head)
    except yaml.YAMLError:
        meta = None
    return (meta if isinstance(meta, dict) else {}), body.split("\n", 1)[1] if "\n" in body else ""


def command_instruction(command: CommandRef, skill: Skill) -> str:
    """What the model is told at each call of a turn that runs a command."""
    return (
        f"The person ran the command /{command.name} with this message. Its skill `{skill.name}` is"
        " already loaded: don't call load_skill for it. Follow its instructions below for this message,"
        " taking what the person wrote as its input. When they wrote nothing else, the message is just"
        f" /{command.name}: carry out the instructions as they stand.\n"
        f'<command_skill name="{skill.name}">\n{skill.instructions}\n</command_skill>'
    )
