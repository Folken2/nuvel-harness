"""A turn's skills: those of the plugins installed and pinned for this person.

ADK's ``SkillRegistry`` interface over the plugins the loader read for the turn
(``library/turn_plugins.py``). Each skill carries its plugin's shared files
(``shared_files.py``). Two plugins with a skill of the same name: the first one named
keeps it, and the other's is left out with a warning, since the model loads skills by
name.

A plugin's text never reads session state: ADK's ``adk_inject_state`` would fill
``{key}`` in a skill's body from the session when it's loaded, ``temp:`` keys and the
turn's provider keys included, so it is taken off every plugin skill and the body
reaches the model as written.
"""

from __future__ import annotations

import logging
import posixpath
from collections.abc import Iterable

from google.adk.skills import Frontmatter, Skill, SkillRegistry

from ..plugin_source import SKILL_FILE, Plugin
from .shared_files import shared_files, with_shared_files

logger = logging.getLogger(__name__)

# ADK metadata a plugin's skill may not set (see the module's docstring).
REFUSED_METADATA = ("adk_inject_state",)


class SkillNotInstalled(LookupError):
    pass


class PluginSkillRegistry(SkillRegistry):
    def __init__(self, plugins: Iterable[Plugin], first: Iterable[Skill] = ()) -> None:
        # `first`: the skill of the command the person ran (command.py), which keeps its
        # name over any plugin's.
        self._skills: dict[str, Skill] = {skill.name: skill for skill in first}
        for plugin in plugins:
            for skill in plugin_skills(plugin):
                if skill.name in self._skills:
                    logger.warning(
                        "skill %s of plugin %s left out: another plugin has one of that name",
                        skill.name,
                        plugin.name,
                    )
                    continue
                self._skills[skill.name] = skill

    def skills(self) -> list[Skill]:
        return list(self._skills.values())

    async def get_skill(self, *, name: str) -> Skill:
        skill = self._skills.get(name)
        if skill is None:
            raise SkillNotInstalled(f"No installed skill is named {name}.")
        return skill

    async def search_skills(self, *, query: str) -> list[Frontmatter]:
        words = [w for w in query.lower().split() if w]
        return [
            skill.frontmatter
            for skill in self._skills.values()
            if any(w in f"{skill.name} {skill.description}".lower() for w in words)
        ]


def plugin_skills(plugin: Plugin) -> list[Skill]:
    """A plugin's skills as a turn serves them: its shared files added, refused metadata taken off."""
    m = plugin.manifest
    hooks_dir = posixpath.dirname(m.hooks) if isinstance(m.hooks, str) else ""
    part_dirs = (*m.skill_dirs, *m.command_dirs, *m.agent_dirs, *((hooks_dir,) if hooks_dir else ()))
    shared = shared_files(plugin.files, part_dirs)
    found = []
    for skill in map(_without_refused_metadata, plugin.skills):
        skill_dir = next(
            (f"{d}/{skill.name}" for d in m.skill_dirs if f"{d}/{skill.name}/{SKILL_FILE}" in plugin.files),
            None,
        )
        found.append(skill if skill_dir is None else with_shared_files(skill, skill_dir, shared))
    return found


def _without_refused_metadata(skill: Skill) -> Skill:
    metadata = skill.frontmatter.metadata
    if not any(key in metadata for key in REFUSED_METADATA):
        return skill
    logger.warning("skill %s: %s is not allowed and was removed", skill.name, ", ".join(REFUSED_METADATA))
    kept = {k: v for k, v in metadata.items() if k not in REFUSED_METADATA}
    return skill.model_copy(update={"frontmatter": skill.frontmatter.model_copy(update={"metadata": kept})})
