"""The plugin-level files a skill shares with its siblings, as the skill's own resources.

A plugin's skills cite files outside their folder (``../../references/safety.md``,
``../../templates/README.md``). ADK serves a skill only its own ``references/``,
``assets/`` and ``scripts/``, so a cited file outside them is out of reach. Each shared
file is added to every skill under the name ADK can serve, and the skill's text is
rewritten to cite that name:

- ``references/x`` at the plugin root is the skill's ``references/x``,
- ``scripts/x`` is its ``scripts/x``,
- anything else, ``templates/x`` or ``assets/x``, is under ``assets/``, as
  ``assets/templates/x`` or ``assets/x``.

Shared means a file in a top-level folder that isn't hidden, a skills folder or another
part of the plugin (commands, agents, hooks). A skill's own file of the same name wins.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Mapping

from google.adk.skills import Resources, Script, Skill

# A path that climbs out of the skill's folder, as SKILL.md cites it.
_CLIMBING_PATH = re.compile(r"(?:\.\./)+[A-Za-z0-9_][A-Za-z0-9_./-]*[A-Za-z0-9_]")


def shared_files(files: Mapping[str, bytes], part_dirs: tuple[str, ...]) -> dict[str, bytes]:
    """The plugin's shared files by path from its root."""
    tops = {d.split("/", 1)[0] for d in part_dirs}
    return {
        path: content
        for path, content in files.items()
        if "/" in path and not path.startswith(".") and path.split("/", 1)[0] not in tops
    }


def resource_path(path: str) -> str:
    """The name a shared file has among a skill's resources."""
    top, rest = path.split("/", 1)
    if top in ("references", "scripts", "assets"):
        return path
    return f"assets/{top}/{rest}"


def with_shared_files(skill: Skill, skill_dir: str, shared: Mapping[str, bytes]) -> Skill:
    """The skill with the shared files among its resources, and its text citing them there."""
    own = skill.resources
    references = dict(own.references)
    assets = dict(own.assets)
    scripts = dict(own.scripts)
    served: dict[str, str] = {}
    for path, content in shared.items():
        name = resource_path(path)
        kind, rest = name.split("/", 1)
        if kind == "scripts":
            if rest in scripts:
                continue
            scripts[rest] = Script(src=content.decode("utf-8", errors="replace"))
        else:
            target = references if kind == "references" else assets
            if rest in target:
                continue
            target[rest] = _text_or_bytes(content)
        served[path] = name

    def cite(match: re.Match[str]) -> str:
        cited = match.group(0)
        path = posixpath.normpath(posixpath.join(skill_dir, cited))
        return served.get(path, cited)

    return skill.model_copy(
        update={
            "instructions": _CLIMBING_PATH.sub(cite, skill.instructions),
            "resources": Resources(references=references, assets=assets, scripts=scripts),
        }
    )


def _text_or_bytes(content: bytes) -> str | bytes:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content
