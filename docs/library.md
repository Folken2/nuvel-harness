# Library

A product that lets people add plugins (skills, commands, connectors) has to take an uploaded
zip or a repository it fetched, refuse anything unsafe, and keep one exact version of what
passed. This package reads plugin archives and marketplace files into checked, hashed plugin
versions. It never fetches or connects to anything: getting the bytes is the product's job.
A product imports `from harness.library import ArchiveLimits, ArchiveRefused, inspect_files,
load_plugin, read_archive, PluginCache, ...`, plus `inspect.inspect_archive` and
`marketplace.read_marketplace` from the two submodules listed in `__all__`.

How a turn then uses a stored plugin's skills and connectors is in [tools](tools.md).

## What it guarantees

- **Every refusal is an `ArchiveRefused`** (a `ValueError`) with a `code` a product can map to
  a message and a `detail` in plain words. Codes: `not_a_zip`, `empty`, `unsafe_path`,
  `too_many_files`, `too_large`, `not_a_repository`, `no_manifest`, `manifests_disagree`,
  `bad_name`, `bad_version`, `bad_manifest`, `bad_skill`, `bad_marketplace`.
- **An archive is read safely.** `read_archive(data, limits)` refuses absolute paths,
  backslashes, NUL bytes, `..`, symbolic links and a path that appears twice. It counts files
  against `max_files` and bytes against `max_unpacked_bytes`; the size a zip declares is
  checked, and the read itself is capped too, so a zip that lies about its sizes is still
  stopped. Operating-system junk (`__MACOSX`, `.DS_Store`, `Thumbs.db`, `._*`) is skipped.
- **Two shapes read the same.** A zip with every file under one top folder (what
  `git archive --prefix=<plugin>/` makes) is rebased to that folder when no manifest sits at the
  root, so it reads like a zip of the folder's contents.
- **Manifests are checked.** Any of `MANIFEST_NAMES` (`plugin.json`,
  `.claude-plugin/plugin.json`, `.codex-plugin/plugin.json`) is enough. Those present must
  agree on name and version. The name is lowercase kebab-case, at most 64 characters (the same
  rule as an ADK skill name); the version is a semantic version, at most 64 characters.
- **Skills are ones ADK will load.** `load_plugin` reads each `<skills folder>/<name>/SKILL.md`
  into ADK's own `Skill`, validated with ADK's `Frontmatter`. The frontmatter name must match
  its folder, and two skills may not share a name. A skill's `references/`, `assets/` and
  `scripts/` become its resources. Frontmatter that strict YAML refuses but that has plain
  `key: value` lines (such as an unquoted description holding `: `) is read by quoting the
  values, and the plugin gets a warning naming the file.
- **The other parts are listed, with warnings, never run.** Commands are `<folder>/<name>.md`
  files with an optional frontmatter description. Hooks and subagents are listed with a
  warning. Each connector in `.mcp.json` (or the manifest's `mcpServers`) is kept with a
  `problem` when it starts a local command, has no address, has headers that aren't text, or
  needs a `${NAME}` value. `inspect_files` also warns when a connector's name leaves its tools
  little room for their own names.
- **The same tree always gives the same version.** `Plugin.content_hash` is
  `sha256:<hex>` over each path and the sha256 of its bytes, in path order. `Inspected.archive`
  is the canonical zip: files in path order, one fixed date, the same mode. Neither depends on
  the uploaded zip's order or timestamps.
- **`Plugin.summary()`** returns a JSON-ready dict: `name`, `version`, `displayName`,
  `description`, `author`, `homepage`, `license`, `keywords`, `skills`, `commands`,
  `connectors` (name and kind), `hooks`, `agents` and `warnings`.
- **Marketplaces.** `marketplace.read_marketplace(raw, repo_source)` reads one of
  `MARKETPLACE_FILES` (`.claude-plugin/marketplace.json`, `.agents/plugins/marketplace.json`)
  into `Listing`s. A string source or `{"source": "local", "path": ...}` becomes a
  `LocalSource` through `local_source`, which returns `None` for an absolute path, a URL or a
  path that leaves the marketplace's folder. Every other source is handed to the product's
  `repo_source`, which returns a `RepoSource` or `None` for a kind it can't fetch. An entry
  without a string `name` is skipped.
- **Repository zips.** `read_repo_folder(data, folder, limits)` reads one folder of a zip whose
  files sit under a single top folder (as a GitHub download does), rebased to that folder;
  only that folder's files count against the limits. `read_repo_file(data, path, max_bytes)`
  returns one file, `None` when it isn't there, and refuses one over `max_bytes`.

## What it costs

- No network, no model calls, no database writes. Everything works on bytes in memory.
- Memory: the unpacked files, up to `max_unpacked_bytes`, plus the canonical zip
  `inspect_files` builds from them.
- CPU: one sha256 per file for the content hash, and one deflate of every file for the
  canonical zip.
- `PluginCache(limit)` keeps up to `limit` loaded plugins, each with all its files, in the
  process. A version never changes, so an entry never goes stale; the oldest-used one is dropped
  past the limit.

## Configure it

`ArchiveLimits(max_files: int, max_unpacked_bytes: int)` has no defaults: how large an upload
may be is the product's call. `PluginCache(limit: int)` likewise.

```python
from harness.library import ArchiveLimits, ArchiveRefused, PluginCache, RepoSource
from harness.library.inspect import inspect_archive
from harness.library.marketplace import read_marketplace

LIMITS = ArchiveLimits(max_files=500, max_unpacked_bytes=20 * 1024 * 1024)
cache = PluginCache(limit=64)


def add_upload(data: bytes) -> dict:
    try:
        inspected = inspect_archive(data, LIMITS)
    except ArchiveRefused as refusal:
        return {"error": refusal.code, "detail": refusal.detail}
    plugin = inspected.plugin
    save_version(plugin.content_hash, inspected.archive, plugin.summary())  # the product's store
    cache.put(plugin)
    return plugin.summary()


def github_source(source: dict) -> RepoSource | None:
    if source.get("source") != "github" or not isinstance(source.get("repo"), str):
        return None
    owner, _, repo = source["repo"].partition("/")
    return RepoSource(owner=owner, repo=repo, path=source.get("path", ""),
                      ref=source.get("ref"), commit=source.get("sha"))


listings = read_marketplace(marketplace_bytes, github_source)
```

For files fetched some other way, `inspect_files(files)` takes a `{path: bytes}` mapping and
`load_plugin(files)` loads one without the canonical zip.

## Test code that uses it

Everything here is a pure function over bytes. Build a zip in the test with `zipfile` and an
`io.BytesIO`, pass it in, and assert on the `Plugin`, the `summary()` or the refusal's
`code`. `repo_source` is a plain callable, so a test passes its own. The harness has no test
file for this package yet.

## Reference

| Name | What it is |
| --- | --- |
| `ArchiveLimits` | `max_files` and `max_unpacked_bytes` for one read. |
| `ArchiveRefused` | Why an upload was refused: `code` and `detail`. |
| `read_archive` | A plugin zip's files by path from the plugin root. |
| `read_repo_folder` | One folder of a repository zip, rebased to it. |
| `read_repo_file` | One file of a repository zip, or `None`. |
| `MANIFEST_NAMES` | The three manifest paths a plugin may ship. |
| `load_plugin` | The `Plugin` these files make. |
| `Plugin` | One version: manifest, skills, components, warnings, files, content hash. |
| `inspect_files` | `load_plugin` plus connector-name warnings and the canonical zip. |
| `Inspected` | The plugin and its canonical zip bytes. |
| `inspect` | Submodule; `inspect.inspect_archive(data, limits)` reads a zip and inspects it. |
| `PluginCache` | Loaded plugins in this process by content hash, least recently used dropped first. |
| `marketplace` | Submodule; `marketplace.read_marketplace(raw, repo_source)` lists a marketplace's plugins. |
| `MARKETPLACE_FILES` | The two marketplace file paths. |
| `Listing` | A marketplace entry: name, description, source (or `None`). |
| `LocalSource` | A plugin folder inside the marketplace's repository. |
| `RepoSource` | A plugin in another repository: owner, repo, path, ref, commit. |
| `local_source` | A `LocalSource` for a path, or `None` when the path leaves the folder. |
