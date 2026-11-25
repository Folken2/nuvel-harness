"""The check a plugin passes before it is stored as a version: an uploaded zip, or
files fetched from somewhere else. How large an upload may be is the service's to say."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from ..tools.plugin_connectors.mounts import name_warnings
from .archive import ArchiveLimits, canonical_archive, read_archive
from .plugin import Plugin, load_plugin

__all__ = ["inspect_archive"]


@dataclass(frozen=True)
class Inspected:
    plugin: Plugin
    # The files as the canonical zip: the bytes that are kept, which the hash covers.
    archive: bytes


def inspect_archive(data: bytes, limits: ArchiveLimits) -> Inspected:
    """The plugin in an uploaded zip no larger than ``limits``, or ArchiveRefused."""
    return inspect_files(read_archive(data, limits))


def inspect_files(files: Mapping[str, bytes]) -> Inspected:
    """The plugin these files make, with its canonical zip, or ArchiveRefused.

    A connector whose name leaves its tools too little room is a warning: its tools'
    names only show when a turn connects, and a long one is left out then."""
    plugin = load_plugin(files)
    plugin = replace(plugin, warnings=plugin.warnings + name_warnings(plugin))
    return Inspected(plugin=plugin, archive=canonical_archive(plugin.files))
