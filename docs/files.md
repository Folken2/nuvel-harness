# Files

A conversation collects files: the person's uploads, their edits of them, and what the agent
saves. This package says where each kind lives by name, finds the right version of a file when
a model names it, and lists the files to the model without breaking the provider's prompt
cache. A product imports `from harness.files import ConversationFilesTool, FileLayout,
FilesRule, set_layout, set_rule, ...`.

Files are stored in ADK's artifact service. The harness adds no store of its own.

## What it guarantees

- **One layout, read at each call.** `set_layout(FileLayout(...))` sets the five name prefixes
  every file name is built from. Names are built from `layout()` when they are needed, never at
  import. With no layout set, `layout()` raises `LookupError`; `set_layout(None)` unsets it.
- **The newest version wins.** `load_original_or_named(tool_context, name)` compares the kept
  original (`<original><name>`, where an upload and the person's edits of it live) with the
  artifact the agent saved as `<name>`, and returns the one saved last, by the artifact
  version's `create_time`. When either has no save time, or the store fails to say, the original
  comes first (the failure is logged).
- **Any spelling of the name finds the file.** `name` may carry the original prefix or not. A
  name under the output prefix that matches nothing is tried again without it, since that is
  how models name a file a run saved.
- **A text copy never hides a binary original.** If the newer candidate is text and the
  original is not, the original is returned: code that asks for a workbook gets the workbook,
  not the text the model was shown of it. `accept` skips any part the caller can't use.
- **The file list stays out of the system instruction.** `ConversationFilesTool` is ADK's
  `LoadArtifactsTool` with the list moved into a message of its own in the session history,
  written only when the list changes (including when it becomes empty). A new file no longer
  changes the start of every later prompt. See [adk](adk.md) for how this differs from ADK's
  tool.
- **Not everything is listed.** The tool leaves out files this run made (recorded with
  `note_made`), the kept copies of cut tool results, and pages a server attached to a tool call
  (the layout's `pages` prefix). A file made this turn is listed from the next turn on.
- **No rule, no kept files.** `keeps_files(state)` is `rule.enabled() and
  rule.in_conversation(state)` for the rule passed to `set_rule`. With no rule set it is always
  false. The harness asks it before keeping the whole of a cut tool result (see
  [plugins](plugins.md) and [context](context.md)) and before keeping an MCP Apps page (see
  [tools](tools.md)). A failed save of a cut result is logged by the error's type only and
  never fails the call.
- **Safe names.** `safe_name` keeps the base name, replaces every character outside
  `A-Za-z0-9 ._()-` with `_`, strips leading and trailing spaces, dots and underscores, falls
  back to `file`, and cuts to 150 characters. `text_copy_name` appends `.txt` unless the name
  already ends in a text extension (`.txt .md .csv .tsv .json .xml .yaml .yml .html .htm`).
  `plain_name` drops the original prefix.

## What it costs

- `ConversationFilesTool`: one appended session event, and one extra message in each later
  request, each time the list changes. Nothing when it doesn't. A loaded file's content goes
  into the request for the next step only (ADK's behavior).
- `load_original_or_named`: for each name it tries (at most two), up to two artifact version
  lookups and two artifact loads.
- A kept cut tool result: one artifact save, under the reading prefix, named
  `<tool>-<12 hex characters>.txt`.
- `note_made`: a list in the run's `temp:` state, which lasts the invocation and is never
  saved.
- `FileLayout`, the name helpers and `keeps_files`: nothing beyond the call.

## Configure it

`FileLayout` fields, all required, each ending with its separator:

| Field | What lives there |
| --- | --- |
| `original` | An upload's kept original and the person's edits of it. |
| `output` | Where a job saves what it shares; the file is kept under its name without this prefix. |
| `reading` | Text the agent reads a chunk at a time, never shown as a file of the conversation. |
| `tool_results` | Under `reading`: the whole of a tool result the model was shown cut. |
| `pages` | A page a server attached to a tool call. Never listed to the model. |

`FilesRule` has two callables: `enabled() -> bool`, asked at each call so a changed setting
takes effect at once, and `in_conversation(state) -> bool`, which says whether this run belongs
to a person's conversation (a background task in its own session does not).

```python
from google.adk.agents import LlmAgent

from harness.files import ConversationFilesTool, FileLayout, FilesRule, set_layout, set_rule

set_layout(FileLayout(original="original/", output="output/", reading="markdown/",
                      tool_results="tool-results/", pages="app-pages/"))


def is_conversation(state) -> bool:
    return state.get("myproduct:task_id") is None


set_rule(FilesRule(enabled=lambda: settings.files_on, in_conversation=is_conversation))

agent = LlmAgent(name="assistant", model="...", tools=[ConversationFilesTool()])
```

A tool that saves a file the model already knows about records it, so it isn't listed back:

```python
from google.adk.tools import ToolContext
from google.genai import types

from harness.files import load_original_or_named, note_made, safe_name


async def make_chart(tool_context: ToolContext, title: str) -> dict:
    name = f"{safe_name(title)}.png"
    await tool_context.save_artifact(name, types.Part.from_bytes(data=render(title), mime_type="image/png"))
    note_made(tool_context, name)
    return {"saved": name}


async def read_upload(tool_context: ToolContext, name: str) -> dict:
    part = await load_original_or_named(tool_context, name)
    return {"found": part is not None}
```

The runner needs an artifact service (any ADK one) for any of this to have somewhere to store.

## Test code that uses it

- Set a layout once for the test session, as `tests/conftest.py` does with neutral prefixes.
  `set_layout(None)` clears it.
- `load_original_or_named` calls only `tool_context.load_artifact(name)` and, if present,
  `tool_context.get_artifact_version(name)`. A small object with those methods over ADK's
  `InMemoryArtifactService` stands in for a `ToolContext`; the in-memory service records each
  version's save time. See `tests/test_file_lookup.py` and `tests/test_file_lookup_order.py`.
- `set_rule(FilesRule(enabled=lambda: True, in_conversation=lambda state: True))` turns kept
  copies on for a test; the default keeps none.
- For `ConversationFilesTool` end to end, run the agent with ADK's `Runner`, its
  `InMemorySessionService` and `InMemoryArtifactService`, and a scripted model such as
  `ScriptedLlm` in `examples/test_assistant.py`, then read the list message from the request
  the model was given.

## Reference

| Name | What it is |
| --- | --- |
| `ConversationFilesTool` | ADK's `load_artifacts` tool, with the list sent as its own message and the run's own files left out. |
| `FileLayout` | The five name prefixes, with helpers that build and test names. |
| `set_layout` | Sets the layout every name is built from; `None` unsets it. |
| `layout` | The submodule; `layout.layout()` returns the current layout or raises `LookupError`. |
| `FilesRule` | `enabled` and `in_conversation`: when a run keeps files. |
| `set_rule` | Sets the rule. The default keeps none. |
| `keeps_files` | Whether the run with this state keeps files. |
| `load_original_or_named` | The newest of a file's kept original and the agent's saved copy. |
| `note_made` | Records a file this run made, so it isn't listed back this turn. |
| `plain_name` | A name without the original prefix. |
| `safe_name` | A name safe to hand a job, at most 150 characters. |
| `text_copy_name` | The name for a file's extracted text. |
