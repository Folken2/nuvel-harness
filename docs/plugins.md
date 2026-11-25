# Plugins

Everything the harness does during a run reaches ADK as a `BasePlugin`: a plugin, not an
agent callback, so it holds for every agent and every tool, the person's connected apps
included. A product picks the plugins it wants and lists them on its `App`. It imports them
`from harness.plugins import ...`; the names are below. Most plugins hold only the
callbacks: the logic lives in another package, linked from each entry.

## Order is behavior

ADK calls each callback on the plugins in list order, and stops at the first plugin that
returns a value: that value replaces what would run next (the model call, the tool call, the
response or the result), and plugins later in the list never see that callback. So:

- **Plugins that may return a value:** `ApprovalPlugin`, `BadRepliesPlugin`,
  `CacheUsagePlugin`, `LoopGuardPlugin`, `LoopPausePlugin`, `RunLimits`,
  `ToolResultLimitPlugin`, and the tool gateway's plugin (`ToolGateway.plugin()`, see
  [tools](tools.md)).
- **Every other plugin returns nothing** from every callback and changes things in place, so
  it can stand anywhere the constraints below allow.

The constraints, each stated in the plugin's source:

| Plugin | Must come |
| --- | --- |
| `LoopPausePlugin` | right after `RunLimits`, and before anything that prepares the next model call (such as `SteeringPlugin`) |
| `SteeringPlugin` | after `RunLimits`, so a call the ceiling refuses doesn't use up the person's message |
| `BadRepliesPlugin` | right after `PlanFieldPlugin`; before `ApprovalPlugin`; before the ledger, cache-usage and log plugins, so they see the repaired reply |
| `LoopGuardPlugin` | after `BadRepliesPlugin`, and before `ApprovalPlugin` (a call that won't run is never put to the person) |
| tool gateway plugin | before `ApprovalPlugin` |
| `ToolLedgerPlugin` | right after `ApprovalPlugin` (a held or denied call is not recorded as run), and before `ToolResultLimitPlugin` |
| `LedgerPlugin` | before `CacheUsagePlugin`, and after any plugin of the product's that puts the provider's reported cost on the response's `custom_metadata` |
| `PrefixWatchPlugin` | last among the before-model callbacks, after every plugin that changes the request |
| `ToolResultLimitPlugin` | last: any after-tool callback after it would miss every result it changes |

One order that meets all of them:

```python
from harness.plugins import (
    ApprovalPlugin,
    BadRepliesPlugin,
    CacheUsagePlugin,
    LoopGuardPlugin,
    LoopPausePlugin,
    PlanFieldPlugin,
    PrefixWatchPlugin,
    RunLimits,
    SteeringPlugin,
    ToolResultLimitPlugin,
    TurnLogPlugin,
)
from harness.runs import LoopSettings, loop_guard


def plugins(approval: ApprovalPlugin) -> list:
    loop = LoopSettings(note_at=3, stop_at=5)
    return [
        RunLimits(max_seconds=600, max_cost_usd=2.0, max_steps=60, last_call_tokens=400),
        LoopPausePlugin(loop),
        SteeringPlugin(),
        PlanFieldPlugin(),
        BadRepliesPlugin(notes={loop_guard.KIND: loop_guard.NOTE}),
        LoopGuardPlugin(loop, ask_tool="ask_person"),
        approval,
        # LedgerPlugin and ToolLedgerPlugin go here (ledger.md), ToolLedgerPlugin first.
        CacheUsagePlugin(),
        TurnLogPlugin(),
        PrefixWatchPlugin(),
        ToolResultLimitPlugin(max_chars=80_000, reader="read_document"),
    ]
```

## The plugins

Each entry: what it does, the callbacks it implements, and whether it may return a value.

### `ApprovalPlugin(gate)`

Asks the person before a tool with effects runs. Its `before_tool_callback` returns the
gate's answer in place of a call it holds or denies. See [approval](approval.md).

### `BadRepliesPlugin(notes=None)`

Repairs tool-call arguments the schema settles, holds every call of a reply cut at the
output limit, and sends one note per run for an empty reply or a dropped cut call. `notes`
maps further note kinds to their text, for a product plugin that adds its own note
(`agent_note.add_note`). `LoopGuardPlugin` is one: pass
`notes={loop_guard.KIND: loop_guard.NOTE}` with it, or its note call is answered with the
empty-reply note. Callbacks: `before_model_callback` (rewrites the request so the
provider never sees a note call), `after_model_callback` (in place), `before_tool_callback`.
Returns a value: from `before_tool_callback` only, to answer a note call or refuse a held
call. See [replies](replies.md).

### `CacheUsagePlugin()`

Copies the provider's prompt-cache write count (`cache_creation_input_tokens` on the usage
object, which doesn't survive event serialization) into the response's `custom_metadata`
under `<namespace>_cache_write_tokens`, so it is saved with the event. Callback:
`after_model_callback`. Returns a value: yes, the response, whenever there is a write count
to copy; so it ends that callback's chain and goes after the plugins that read the response.

### `ContextBudgetPlugin(window_of_turn, settings)`

Before each run, sets the run's own compaction threshold from the window of the turn's
model. `window_of_turn` reads the window from the session state (`window_of` helps);
`settings` is a `WindowSettings`. With compaction off, or no token threshold, it does
nothing. Callback: `before_run_callback`. Returns nothing. See [context](context.md).

### `LedgerPlugin(read_start)`

Starts the run's ledger state from the turn's state (`read_start`), settles each model call
from its final response, records a failed attempt, and waits for the run's ledger writes at
the end. With no ledger writer set, it reads nothing. Callbacks: `before_run_callback`,
`after_model_callback`, `on_model_error_callback`, `after_run_callback`. Returns nothing.
See [ledger](ledger.md).

### `LoopGuardPlugin(settings, exempt=(), ask_tool=None)`

Counts repeated tool calls (the same call, or a cycle of up to three) from the run's events.
At `note_at` repeats the reply gets a note for the model; at `stop_at` the call is held and
its answer tells the model to ask the person with `ask_tool`. With no `ask_tool`, or in a
background task, the call is refused instead. `exempt` names tools whose repeats never
count; the note call and `ask_tool` never count. Callbacks: `after_model_callback` (in
place), `before_tool_callback`. Returns a value: from `before_tool_callback`, for a held or
refused call. Keeps no state between callbacks. See [runs](runs.md).

### `LoopPausePlugin(settings)`

After a repeated call was refused where nobody could be asked, ends the run paused, with
reason `loop` in `<namespace>_paused`, the way a run at its ceiling ends. Callback:
`before_model_callback`. Returns a value: yes, the paused response, in place of the model
call. See [runs](runs.md).

### `McpAppPagesPlugin(limits, shows_pages)`

After an MCP tool call whose server attached a `ui://` page, fetches the page on the tool's
own connection, keeps it with the conversation and puts its reference on the call's event
under `<namespace>_pages`, keyed by call id. `shows_pages(server)` is the product's rule for
which servers may show pages; `limits` is a `PageLimits`. Anything that fails leaves the call
as an ordinary tool call. Callbacks: `after_tool_callback`, `on_event_callback`,
`after_run_callback`. Returns nothing. See [tools](tools.md).

### `OldResultsPlugin(settings, reader, fallback_threshold, never=())`

Before each model call, once the prompt nears the compaction threshold, sends older tool
results as a 200-character preview and a note, in batches, saving each whole as a
conversation file first. Only the request changes. `reader` is the tool that reads a saved
copy; `never` names tools whose results always stay whole. Callback: `before_model_callback`.
Returns nothing. See [context](context.md).

### `PlanFieldPlugin()`

Lets the agent's own tool calls carry the plan: adds a plan field to their declarations,
and lifts it into a `todo_list` call after the reply. Callbacks: `before_model_callback`,
`after_model_callback` (in place). Returns nothing. Goes right before `BadRepliesPlugin`.
See [tools](tools.md).

### `PrefixWatchPlugin()`

Logs, for each model call, where its prompt first differs from the previous call's in the
same session: `none`, `instruction`, `tools`, `history`, or `first`. Anything but `none` and
`first` means the provider's cached prefix was lost from there on. It changes nothing.
Remembers the last request's digests for 512 sessions (`REMEMBERED_SESSIONS`) in process
memory; a restart logs `first` again. Callback: `before_model_callback`. Returns nothing.
See [context](context.md) and `tests/test_prefix_watch.py`.

### `RunLimits(max_seconds, max_cost_usd, max_steps, retry_budget_seconds=math.inf, costs=RunCosts(), last_call_tokens=0)`

The run's safety ceiling. Each model call and each tool call is a step; each call's cost (as
`costs` prices it) is added. Before each model call it checks steps, then cost, then time,
and a ceiling is reached at "equal or above" (60 steps allows exactly 60). At the ceiling:

- with `last_call_tokens > 0`, the run gets one last model call with tool calls turned off,
  its reply capped at `last_call_tokens`, and a note asking the model to say what was done
  and what is left. Any tool call in that reply is dropped, and the reply carries
  `<namespace>_paused` (`{"reason", "reached", "ceiling"}`, reason `steps`, `cost` or
  `time`);
- any call after that, or every call at the ceiling when `last_call_tokens` is 0, isn't made:
  the run ends with an empty response that carries the pause. Either way it ends as a
  finished run, not an error.

The ceiling is the one the turn's state carries under `costs.limits_key`
(`{"ceiling": {"steps", "costUsd", "seconds"}}`; a missing or non-positive value falls back),
else the values given here. On a resume, what the run already used is counted again from its
saved events, so a crash loop can't escape the ceiling. Before each model call it also hands
the call a retry window (the time left to the ceiling and `retry_budget_seconds` across the
run), which `KeyedLiteLLMClient` spends from; past the ceiling, no retries (see
[routing](routing.md)). Callbacks: `before_run_callback`, `before_model_callback`,
`after_model_callback`, `after_tool_callback`, `after_run_callback`. Returns a value: from
`before_model_callback`, the paused response.

### `RunCosts(limits_key=None, pricing_key=None, cost_of_call=..., cost_of_tool=...)`

How `RunLimits` prices a run. `limits_key` and `pricing_key` are the state keys where a
turn's ceiling and its model's prices arrive. `cost_of_call(llm_response, pricing)` and
`cost_of_tool(tool_name, result)` return dollars; both default to 0, so with nothing set
only steps and time can pause a run.

### `SteeringPlugin()`

Before each model call, writes the messages the person sent to the running turn into the
session and into this call's request. Callback: `before_model_callback`. Returns nothing.
See [runs](runs.md).

### `ToolLedgerPlugin()`

Records one ledger operation per tool call that runs, and a failed one when the tool raises.
Callbacks: `before_tool_callback`, `after_tool_callback`, `on_tool_error_callback`. Returns
nothing. See [ledger](ledger.md).

### `ToolResultLimitPlugin(max_chars, reader, compact=None)`

Shapes every tool result before the model sees it. First `compact(result)`, the product's own
pass, if given (it returns a smaller result or `None`). Then a result over `max_chars` (as
JSON) keeps its start and its end, with a note, and its whole text is saved as a conversation
file the note names when the run keeps files. The `reader` tool's own results are cut but
never saved again. A `max_chars` of 0 or less turns cutting off. Callback:
`after_tool_callback`. Returns a value: the compacted or cut result, which is why it goes last.
See [context](context.md).

### `TurnLogPlugin(model_name=...)`

One log line per tool call (`tool done: <name> <ms>ms`, or `tool failed: <name> <ms>ms
<ErrorType>` at error level) and per failed model call (`model call failed: <model>
<ErrorType>`). Names, durations and error types only: never arguments, results or the
provider's text. `model_name` rewrites the model id for the log. `open_calls()` says how many
tool calls are under way. Callbacks: `before_tool_callback`, `after_tool_callback`,
`on_tool_error_callback`, `on_model_error_callback`. Returns nothing.

### `TurnTimingPlugin(clock=time.perf_counter)`

One log line per turn: `setup_ms` (run start to first model request), `model_first_token_ms`
(that request to the first streamed text), `first_token_ms`, `total_ms`, the number of model
calls and the prompt and cached prompt tokens; then one line with each tool schema's size in
characters. Holds at most 256 turns in memory: when a new turn starts with 256 held (turns
cut off before their end), it clears them all. Callbacks: `before_run_callback`, `before_model_callback`,
`on_event_callback`, `after_run_callback`. Returns nothing.

### `UnavailableToolsPlugin(registry)`

Puts the names of the connected apps that couldn't be loaded for a turn on the next event,
under `<namespace>_unavailable_tools`, so a client can warn the person. `registry` is the
`UnavailableTools` the toolsets report to. Callbacks: `on_event_callback`,
`after_run_callback`. Returns nothing. See [tools](tools.md).

## What it costs

Most plugins add no model call, no network and no storage. The exceptions:

- `BadRepliesPlugin`: at most one extra model call per run for an empty reply, and one for a
  dropped cut call.
- `RunLimits`: one last model call, capped at `last_call_tokens`, when a run reaches its
  ceiling and `last_call_tokens > 0`.
- `LoopGuardPlugin`: one extra model step when a note is sent; it reads the run's events at
  each reply and each tool call.
- `OldResultsPlugin`, `ToolResultLimitPlugin`: one conversation file saved per masked or cut
  result when the run keeps files ([context](context.md)).
- `McpAppPagesPlugin`: one page fetch per call with a page, on the tool's connection, held
  to `PageLimits` (8 seconds and 2,000,000 bytes by default), and the page kept with the
  conversation ([tools](tools.md)).
- `LedgerPlugin`, `ToolLedgerPlugin`: ledger writes ([ledger](ledger.md)).
- `SteeringPlugin`: one session event per message the person sent to the running turn.
- In memory: `PrefixWatchPlugin` up to 512 sessions of digests, `TurnTimingPlugin` up to 256
  turns, `RunLimits`, `TurnLogPlugin` and `McpAppPagesPlugin` a small entry per open run or
  call, dropped when it ends.

## Configure it

Every setting is a constructor argument, listed in each entry above. The settings objects
they take are `WindowSettings` and `MaskSettings` ([context](context.md)), `LoopSettings`
([runs](runs.md)), `PageLimits` ([tools](tools.md)), `RunCosts` (above) and the ledger's
start reader ([ledger](ledger.md)). The order snippet above shows a realistic set.

## Test code that uses it

- **ADK's in-memory runner and a scripted model.** Build the `App` with the plugins under
  test and run it with `InMemoryRunner`, with a `BaseLlm` subclass that yields chosen
  replies. `tests/test_prefix_watch.py` does this for `OldResultsPlugin` and
  `PrefixWatchPlugin`; `examples/test_assistant.py` runs `PlanFieldPlugin`,
  `BadRepliesPlugin`, `LoopGuardPlugin` and `TurnLogPlugin` in a whole app through the
  service.
- **Logs are the output** for `PrefixWatchPlugin`, `TurnLogPlugin` and `TurnTimingPlugin`:
  read them with pytest's `caplog` at `INFO`, as `tests/test_prefix_watch.py` does.
  `TurnTimingPlugin` takes a `clock`, so a test can step time by hand.
- **Plugins driven directly.** `tests/test_mcp_app_pages.py` builds `McpAppPagesPlugin` with
  `shows_pages=lambda server: True` (or `False`) and calls its callbacks against a local MCP
  server (`tests/mcp_app_server.py`).
- **Seams in the constructors.** `RunCosts` takes plain functions, so a test prices calls as
  it likes; `ContextBudgetPlugin` takes the window reader; `LedgerPlugin` reads nothing when
  no ledger writer is set; with no files rule set, no run keeps files and nothing is saved.

## Reference

| Name | One line |
| --- | --- |
| `ApprovalPlugin` | Asks the person before a tool with effects runs ([approval](approval.md)). |
| `BadRepliesPlugin` | Repairs arguments, holds a cut reply's calls, notes an empty reply ([replies](replies.md)). |
| `CacheUsagePlugin` | Copies the prompt-cache write count onto the response's metadata. |
| `ContextBudgetPlugin` | Sets each run's compaction threshold from its model's window ([context](context.md)). |
| `LedgerPlugin` | Records each model call of a run in the usage ledger ([ledger](ledger.md)). |
| `LoopGuardPlugin` | Notes repeated calls, then holds them for the person ([runs](runs.md)). |
| `LoopPausePlugin` | Ends the run paused after a repeated call nobody could be asked about. |
| `McpAppPagesPlugin` | Keeps the page an MCP server attaches to a call ([tools](tools.md)). |
| `OldResultsPlugin` | Sends older tool results as previews once the prompt nears its limit ([context](context.md)). |
| `PlanFieldPlugin` | Lets the agent's tool calls carry the plan ([tools](tools.md)). |
| `PrefixWatchPlugin` | Logs where each call's prompt first differs from the last one's. |
| `RunCosts` | How `RunLimits` prices calls and finds a turn's ceiling. |
| `RunLimits` | Pauses a run at its step, cost or time ceiling, after one last summing-up call. |
| `SteeringPlugin` | Adds the person's mid-run messages before each model call ([runs](runs.md)). |
| `ToolLedgerPlugin` | Records each tool call that runs in the usage ledger ([ledger](ledger.md)). |
| `ToolResultLimitPlugin` | Compacts and cuts oversized tool results, keeping the whole as a file ([context](context.md)). |
| `TurnLogPlugin` | Logs tool calls and model failures by name, duration and error type. |
| `TurnTimingPlugin` | Logs where a turn's time to first token went. |
| `UnavailableToolsPlugin` | Tells the client which connected apps couldn't be loaded ([tools](tools.md)). |
