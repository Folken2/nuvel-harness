"""A plugin's skills, read into ADK's own ``Skill`` model.

Validation is ADK's ``Frontmatter``, the model a ``SkillToolset`` loads, so a plugin
that passes here is one ADK will take. A skill is a folder under a skills folder with
a ``SKILL.md`` whose name matches the folder, as ADK's own loader requires. Its
``references/``, ``assets/`` and ``scripts/`` become its resources; files the plugin
shares between skills (``../../references/x.md``) stay in the plugin's files.

Frontmatter is read as YAML. Claude also takes a top-level value that strict YAML
refuses, such as an unquoted description holding ": ", so that is read too, by quoting
the value, and the plugin gets a warning saying which file.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Mapping
from dataclasses import dataclass

import yaml
from google.adk.skills import Frontmatter, Resources, Script, Skill
from pydantic import ValidationError

from .archive import ArchiveRefused

from ..tools.plugin_source import SKILL_FILE  # noqa: E402,F401 - the tools own the name
_FRONTMATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n(.*))?\Z", re.DOTALL)
_RESOURCE_KINDS = ("references", "assets", "scripts")
# A top-level `key: value` line whose value is plain text, not quoted or a block.
_PLAIN_LINE = re.compile(r"^([A-Za-z0-9_-]+):[ \t]+([^'\"|>\[{&*!#\s].*)$")


@dataclass(frozen=True)
class ReadSkills:
    skills: tuple[Skill, ...]
    warnings: tuple[str, ...]


def read_skills(files: Mapping[str, bytes], skill_dirs: tuple[str, ...]) -> ReadSkills:
    skills: dict[str, Skill] = {}
    warnings: list[str] = []
    for folder in skill_dirs:
        for path in sorted(files):
            skill_dir = _skill_dir(path, folder)
            if skill_dir is None:
                continue
            skill, lenient = _read_skill(files, skill_dir)
            if skill.name in skills:
                raise ArchiveRefused("bad_skill", f"Two skills are named {skill.name}.")
            skills[skill.name] = skill
            if lenient:
                warnings.append(f"{skill_dir}/{SKILL_FILE} frontmatter isn't strict YAML; quote its values.")
    return ReadSkills(tuple(skills[name] for name in sorted(skills)), tuple(warnings))


def _skill_dir(path: str, folder: str) -> str | None:
    """``skills/brief`` for ``skills/brief/SKILL.md``; None for anything else."""
    parent, name = posixpath.split(path)
    if name != SKILL_FILE or posixpath.dirname(parent) != folder:
        return None
    return parent


def _read_skill(files: Mapping[str, bytes], skill_dir: str) -> tuple[Skill, bool]:
    """The skill, and whether its frontmatter needed the lenient reading."""
    path = f"{skill_dir}/{SKILL_FILE}"
    try:
        text = files[path].decode("utf-8")
    except UnicodeDecodeError:
        raise ArchiveRefused("bad_skill", f"{path} isn't UTF-8 text.") from None
    match = _FRONTMATTER.match(text.lstrip("﻿"))
    if match is None:
        raise ArchiveRefused("bad_skill", f"{path} must start with YAML frontmatter between --- lines.")
    meta, lenient = _frontmatter(match.group(1), path)
    if not isinstance(meta, dict):
        raise ArchiveRefused("bad_skill", f"{path} frontmatter must be a mapping.")
    try:
        frontmatter = Frontmatter.model_validate(meta)
    except ValidationError as error:
        problems = "; ".join(_problem(e) for e in error.errors())
        raise ArchiveRefused("bad_skill", f"{path}: {problems}") from None
    folder_name = posixpath.basename(skill_dir)
    if frontmatter.name != folder_name:
        raise ArchiveRefused("bad_skill", f"{path} is named {frontmatter.name}, not {folder_name} like its folder.")
    skill = Skill(
        frontmatter=frontmatter,
        instructions=(match.group(2) or "").strip(),
        resources=_resources(files, skill_dir),
    )
    return skill, lenient


def _frontmatter(text: str, path: str) -> tuple[object, bool]:
    try:
        return yaml.safe_load(text), False
    except yaml.YAMLError:
        pass
    quoted = "\n".join(_quoted(line) for line in text.splitlines())
    try:
        return yaml.safe_load(quoted), True
    except yaml.YAMLError:
        raise ArchiveRefused("bad_skill", f"{path} has frontmatter that isn't valid YAML.") from None


def _quoted(line: str) -> str:
    match = _PLAIN_LINE.match(line)
    if match is None:
        return line
    value = match.group(2).rstrip().replace("\\", "\\\\").replace('"', '\\"')
    return f'{match.group(1)}: "{value}"'


def _problem(error: dict) -> str:
    field = ".".join(str(part) for part in error.get("loc", ()))
    message = str(error.get("msg", "")).removeprefix("Value error, ")
    return f"{field}: {message}" if field else message


def _resources(files: Mapping[str, bytes], skill_dir: str) -> Resources:
    found: dict[str, dict[str, str | bytes]] = {kind: {} for kind in _RESOURCE_KINDS}
    for path, content in files.items():
        for kind in _RESOURCE_KINDS:
            prefix = f"{skill_dir}/{kind}/"
            if path.startswith(prefix):
                found[kind][path[len(prefix) :]] = _text_or_bytes(content)
    return Resources(
        references=found["references"],
        assets=found["assets"],
        scripts={name: Script(src=_as_text(src)) for name, src in found["scripts"].items()},
    )


def _text_or_bytes(content: bytes) -> str | bytes:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content


def _as_text(content: str | bytes) -> str:
    return content if isinstance(content, str) else content.decode("utf-8", errors="replace")
