# AGENTS.md (nuvel-harness)

Public, MIT. Production machinery for agents on Google ADK. A product installs it from git
pinned by commit SHA. `CLAUDE.md` only imports this file. Edit here.

## Setup and tests

```bash
python3.13 -m venv .venv && .venv/bin/pip install -e '.[test]' && .venv/bin/pytest
```

`tests/conftest.py` stands in for a product: it sets a neutral wire namespace and file
layout, which a real product sets at its own start.

- Tests are plain functions; async code runs through `asyncio.run(...)` inside them.
- A fake MCP server patches `McpToolset.get_tools` (the server's `tools/list`) with the
  server's own tool names. ADK prefixes them afterwards in `get_tools_with_prefix`.
- Overrides of ADK's classes are checked with
  `.venv/bin/pip install -e '.[types]' && .venv/bin/pyright --pythonpath .venv/bin/python`
  (CI job "ADK overrides"). It fails if it can't import ADK, rather than checking nothing.

## Rules

- **The public API is each package's `__all__`.** A product imports only those names. A
  package imports its public names from its own modules; `harness/__init__.py` imports
  nothing. A function named like its module is reached through the module, never
  re-exported over it. Every public function and method is fully typed, and every module
  imports first in a fresh process. `tests/test_public_surface.py` enforces this. Renaming
  or removing a public name breaks products: say so in the PR.
- **The product passes settings in.** Configuration arrives as arguments or setters
  (`wire_names.use_namespace`, `files.set_layout`); `tests/test_no_environment.py` keeps
  every module free of environment reads.
- **One job per module.** A module you'd describe with "and" is two modules.
- **Private names stay private.** A `_name` another module needs becomes public, or moves
  where both can reach it.
- **ADK is extended through the hooks it gives subclasses**: `BasePlugin` callbacks,
  `BaseToolset.get_tools`, `BaseTool._get_declaration`. Pyright fails on an override of a
  method ADK marks final. The two places that need ADK's internals are pinned in
  `tests/test_adk_hooks.py`; where the harness goes beyond an ADK built-in is in
  `docs/adk.md`, which a new such piece updates.
- **Plugin order is behaviour.** Which plugin may return a value from a callback, and which
  pairs must run in order, is checked in a product's assembly tests: run them before
  changing what a plugin returns.
- **Moves and refactors go in separate PRs**, so `git log --follow` keeps working.
- **Dependencies are ranges** in `pyproject.toml`. A product pins exact hashed versions
  and runs `pip check` against these ranges.
- **Public repo: write for outsiders.** Commits, PRs, issues and docstrings describe the
  harness itself. A product's file paths, issue numbers, settings names, customers and the
  vendors it uses stay in that product's repo; name a provider only for its API's behaviour
  (an error wording, a field), next to the others that behave alike.
  `tests/test_no_outside_references.py` fails on rule and ticket ids from outside documents.
- **Code taken from another project is credited in the same PR.** Its module docstring
  names the source file, and `NOTICE` gets the project, its copyright line and its license.
  Code written from scratch on another project's idea names the source in its docstring
  only.

## Shipping a change

Feature branch, PR, merge to `main`. A product picks it up when it moves its pin to the
merged SHA and runs its own tests on that bump.
