"""One plugin version, loaded from its files: what the catalogue and a turn see."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from google.adk.skills import Skill

from .archive import content_hash
from .components import Components, read_components
from .manifest import Manifest, read_manifest
from .skills import read_skills


@dataclass(frozen=True)
class Plugin:
    manifest: Manifest
    skills: tuple[Skill, ...]
    components: Components
    # What an author should fix, though the plugin loads: the skills' then the parts'.
    warnings: tuple[str, ...]
    # Every file by path from the plugin root, shared references included.
    files: Mapping[str, bytes]
    content_hash: str

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def version(self) -> str:
        return self.manifest.version

    def summary(self) -> dict[str, Any]:
        """What the web app keeps as the version's manifest (plugin_versions.manifest)."""
        m, c = self.manifest, self.components
        return {
            "name": m.name,
            "version": m.version,
            "displayName": m.display_name,
            "description": m.description,
            "author": m.author,
            "homepage": m.homepage,
            "license": m.license,
            "keywords": list(m.keywords),
            "skills": [{"name": s.name, "description": s.description} for s in self.skills],
            "commands": [{"name": x.name, "description": x.description} for x in c.commands],
            "connectors": [{"name": x.name, "kind": x.kind} for x in c.connectors],
            "hooks": list(c.hook_events),
            "agents": list(c.agents),
            "warnings": list(self.warnings),
        }


def load_plugin(files: Mapping[str, bytes]) -> Plugin:
    """The plugin these files make, or ArchiveRefused naming what's wrong."""
    manifest = read_manifest(files)
    frozen = MappingProxyType(dict(files))
    skills = read_skills(frozen, manifest.skill_dirs)
    components = read_components(frozen, manifest)
    return Plugin(
        manifest=manifest,
        skills=skills.skills,
        components=components,
        warnings=skills.warnings + components.warnings,
        files=frozen,
        content_hash=content_hash(frozen),
    )
