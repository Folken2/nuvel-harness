"""A plugin archive read safely, and the one canonical form of its files.

A release is ``git archive --prefix=<plugin>/``, so the files sit under one top folder;
a zip of the folder's contents has them at its root. Both read the same. The canonical
archive and the content hash depend only on paths and bytes, never on the zip's
timestamps or order, so the same tree always gives the same hash.
"""

from __future__ import annotations

import hashlib
import io
import posixpath
import stat
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass

# What an operating system leaves in a zip it makes; never part of a plugin.
_JUNK_PARTS = frozenset({"__MACOSX", ".DS_Store", "Thumbs.db"})
# zip's earliest date: fixed, so the canonical archive is the same bytes every time.
_FIXED_DATE = (1980, 1, 1, 0, 0, 0)
MANIFEST_NAMES = ("plugin.json", ".claude-plugin/plugin.json", ".codex-plugin/plugin.json")


class ArchiveRefused(ValueError):
    """The upload isn't a plugin this service will keep. ``code`` is for the web app."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class ArchiveLimits:
    max_files: int
    max_unpacked_bytes: int


def read_archive(data: bytes, limits: ArchiveLimits) -> dict[str, bytes]:
    """The plugin's files by path from its root, or ArchiveRefused."""
    files = _read_members(data, limits, lambda path: path)
    if not files:
        raise ArchiveRefused("empty", "The archive holds no files.")
    return _from_plugin_root(files)


def read_repo_folder(data: bytes, folder: str, limits: ArchiveLimits) -> dict[str, bytes]:
    """One folder of a repository zip (GitHub's: everything under ``<repo>-<sha>/``),
    rebased to that folder. Only its files count against the limits."""
    prefix = f"{_repo_top(data)}/{folder}/" if folder else f"{_repo_top(data)}/"
    files = _read_members(data, limits, lambda path: path[len(prefix) :] if path.startswith(prefix) else None)
    if not files:
        raise ArchiveRefused("empty", f"The repository has no files under {folder or 'its root'}.")
    return files


def read_repo_file(data: bytes, path: str, max_bytes: int) -> bytes | None:
    """One file of a repository zip, or None when it isn't there."""
    name = f"{_repo_top(data)}/{path}"
    with _open(data) as archive:
        try:
            info = archive.getinfo(name)
        except KeyError:
            return None
        if info.file_size > max_bytes:
            raise ArchiveRefused("too_large", f"{path} is over {max_bytes} bytes.")
        with archive.open(info) as member:
            return member.read(max_bytes + 1)[:max_bytes]


def _open(data: bytes) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ArchiveRefused("not_a_zip", "The file isn't a zip archive.") from None


def _repo_top(data: bytes) -> str:
    with _open(data) as archive:
        names = archive.namelist()
    tops = {name.split("/", 1)[0] for name in names}
    if len(tops) != 1:
        raise ArchiveRefused("not_a_repository", "The archive isn't one repository folder.")
    return tops.pop()


def _read_members(data: bytes, limits: ArchiveLimits, rebase: Callable[[str], str | None]) -> dict[str, bytes]:
    """Every file whose path ``rebase`` keeps, under that new path, checked and capped."""
    files: dict[str, bytes] = {}
    unpacked = 0
    with _open(data) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            path = _safe_path(info.filename)
            if path is None:
                continue
            path = rebase(path)
            if not path:
                continue
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ArchiveRefused("unsafe_path", f"{path} is a link; a plugin holds only files.")
            if len(files) >= limits.max_files:
                raise ArchiveRefused("too_many_files", f"A plugin holds at most {limits.max_files} files.")
            # file_size is what the zip claims, so the read is capped as well.
            budget = limits.max_unpacked_bytes - unpacked
            if info.file_size > budget:
                raise ArchiveRefused("too_large", _too_large(limits))
            with archive.open(info) as member:
                content = member.read(budget + 1)
            if len(content) > budget:
                raise ArchiveRefused("too_large", _too_large(limits))
            if path in files:
                raise ArchiveRefused("unsafe_path", f"{path} appears twice in the archive.")
            unpacked += len(content)
            files[path] = content
    return files


def _too_large(limits: ArchiveLimits) -> str:
    return f"A plugin unpacks to at most {limits.max_unpacked_bytes // (1024 * 1024)} MB."


def _safe_path(name: str) -> str | None:
    """The member's path, None for operating-system junk, or ArchiveRefused."""
    if "\\" in name or name.startswith("/") or "\x00" in name:
        raise ArchiveRefused("unsafe_path", f"{name!r} isn't a relative path.")
    parts = [part for part in name.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        raise ArchiveRefused("unsafe_path", f"{name!r} points outside the plugin.")
    if not parts or any(part in _JUNK_PARTS or part.startswith("._") for part in parts):
        return None
    return posixpath.join(*parts)


def _from_plugin_root(files: dict[str, bytes]) -> dict[str, bytes]:
    """The files rebased so the manifests are at the root, when one folder holds all."""
    if any(name in files for name in MANIFEST_NAMES):
        return files
    tops = {path.split("/", 1)[0] for path in files}
    if len(tops) != 1 or any("/" not in path for path in files):
        return files
    prefix = tops.pop() + "/"
    return {path[len(prefix) :]: content for path, content in files.items()}


def content_hash(files: Mapping[str, bytes]) -> str:
    """sha256 over each path and its bytes' sha256, in path order."""
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.encode())
        digest.update(b"\x00")
        digest.update(hashlib.sha256(files[path]).hexdigest().encode())
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


def canonical_archive(files: Mapping[str, bytes]) -> bytes:
    """The files as a zip in path order with fixed dates: what the web app stores."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            info = zipfile.ZipInfo(path, date_time=_FIXED_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, files[path])
    return out.getvalue()
