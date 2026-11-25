# nuvel-harness

The production layer around an agent built on [Google's Agent Development Kit](https://github.com/google/adk-python).
ADK gives you the agent loop. This package adds what a real product needs around it. It was
extracted from a production agent platform, where it runs every turn.

- **Runs that survive.** A run belongs to the service, not to the connection: it keeps going
  when the browser drops, resumes after a restart from the durable session, and can be steered
  mid-run ("send now") or stopped.
- **Model routing and retry.** A per-turn route (the person's own key, a fallback router), a
  retry policy that tells a rate limit from a quota from a bad request, and a context-overflow
  path that masks old tool results and tries once more.
- **Context management.** Compaction, masking of old tool results, prompt-cache hints and a
  window budget, each a plugin with its own settings.
- **Usage metering.** Every model, tool and compaction call is priced and written to a ledger,
  including the attempts a retry or a fallback made.
- **Approval.** Tools with effects pause the run and ask the person; a policy and a classifier
  decide what can run unasked, and the gate stops asking after a few refusals in a row.
- **Tools and plugins.** A tool gateway that lists large toolsets in a catalog and loads them on
  demand, background tasks that report back, `ask_person`, a plan the agent keeps, MCP Apps
  pages, conversation files, loop guards and a run ceiling.
- **Voice.** Endpointing, clip transcription and sentence-by-sentence speech, with usage tallied.

## Install

Python 3.13 or later.

```bash
pip install "nuvel-harness @ git+https://github.com/Folken2/nuvel-harness"
```

## Use

A product names itself and its file layout once, at start-up, then imports what it needs from
each package. Every package lists its public names in `__all__`; those are the API.

```python
from harness import wire_names
from harness.files import FileLayout, set_layout
from harness.plugins import ApprovalPlugin, LedgerPlugin, LoopGuardPlugin

wire_names.use_namespace("myproduct")  # prefixes the keys it puts on events
set_layout(FileLayout(original="original/", output="output/", reading="markdown/",
                      tool_results="tool-results/", pages="app-pages/"))
```

For a whole product in two files, an agent and the service that runs it, see
[`examples/`](examples/).

The harness never reads the environment: settings are passed in, so one process can hold two
configurations and a test never depends on the shell it runs in.

## Design rules

The test suite enforces the first three:

- **The public API is each package's `__all__`.** A module inside a package can move without a
  product noticing. Every public function and method is fully typed.
- **Every module imports first, alone, in a fresh process.** An eager re-export can't close an
  import cycle that only one import order would hit.
- **No environment reads** anywhere in the package.
- **ADK through the hooks it gives subclasses**, never a patch, so an ADK upgrade stays a
  version bump. The two places that rely on ADK's internals are pinned by tests. Where the
  harness ships its own piece beside an ADK built-in, and why, is in [docs/adk.md](docs/adk.md).
- **Plugin order is behaviour.** A plugin that returns a value from a callback replaces what
  runs next, so which plugins may return one, and which pairs must run in order, is a
  deliberate list; every other plugin returns nothing and can stand anywhere.

## Development

```bash
python3.13 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest
```

## License

MIT. See `LICENSE`, and `NOTICE` for the libraries it builds on.
