"""A marketplace file: the plugins a repository lists, and where each one lives.

Claude's ``.claude-plugin/marketplace.json`` and Codex's ``.agents/plugins/marketplace.json``.
A plugin is a folder of the same repository (``"./plugins/x"`` or
``{"source": "local", "path": ...}``), or lives in another repository. Which kinds of
other repository a service can fetch, and how their addresses read, is the service's:
``read_marketplace`` hands every source that isn't a folder to the ``repo_source`` it
is given. A kind nobody reads is listed but can't be added.
"""

from __future__ import annotations

import json
import posixpath
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .archive import ArchiveRefused

__all__ = ["read_marketplace"]

MARKETPLACE_FILES = (".claude-plugin/marketplace.json", ".agents/plugins/marketplace.json")


@dataclass(frozen=True)
class LocalSource:
    # From the marketplace's folder.
    path: str


@dataclass(frozen=True)
class RepoSource:
    owner: str
    repo: str
    path: str
    ref: str | None
    commit: str | None


@dataclass(frozen=True)
class Listing:
    name: str
    description: str | None
    # None: a kind of source this service can't fetch.
    source: LocalSource | RepoSource | None


# A source that isn't a folder of the same repository, read the service's way: where it
# is, or None for a kind or an address the service can't fetch.
RepoSourceReader = Callable[[dict[str, Any]], "RepoSource | None"]


def read_marketplace(raw: bytes, repo_source: RepoSourceReader) -> list[Listing]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ArchiveRefused("bad_marketplace", "The repository's marketplace.json isn't valid JSON.") from None
    plugins = value.get("plugins") if isinstance(value, dict) else None
    if not isinstance(plugins, list):
        raise ArchiveRefused("bad_marketplace", "The repository's marketplace.json lists no plugins.")
    listings = []
    for entry in plugins:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            continue
        description = entry.get("description")
        listings.append(
            Listing(
                entry["name"],
                description if isinstance(description, str) else None,
                _source(entry.get("source"), repo_source),
            )
        )
    return listings


def _source(source: Any, repo_source: RepoSourceReader) -> LocalSource | RepoSource | None:
    if isinstance(source, str):
        return local_source(source)
    if not isinstance(source, dict):
        return None
    kind = source.get("source")
    if kind == "local" and isinstance(source.get("path"), str):
        return local_source(source["path"])
    return repo_source(source)


def local_source(path: str) -> LocalSource | None:
    """A folder under the marketplace's own, or None for a path that leaves it."""
    if path.startswith("/") or "://" in path:
        return None
    clean = posixpath.normpath(path)
    if clean == ".." or clean.startswith("../"):
        return None
    return LocalSource("" if clean == "." else clean)
