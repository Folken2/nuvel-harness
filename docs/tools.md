# Tools

The tools an ADK agent gets beyond its own functions: a person's MCP servers under time
limits and address rules, a gateway that keeps large toolsets out of the prompt until the
model loads them, MCP Apps pages, plugin skills and connectors, a question to the person,
a plan, background tasks and a stronger advisor model. Each piece is an ADK `BaseTool` or
`BaseToolset`, so it goes in an agent's `tools` like ADK's own.

A product imports from `harness.tools` (`ask_tools`, `plan_tools`, `TaskToolset`,
`AdvisorToolset`, `DescribedTool` and the rest) and from its subpackages
`harness.tools.user_tools`, `harness.tools.gateway`, `harness.tools.mcp_apps`,
`harness.tools.skills` and `harness.tools.plugin_connectors`. Several pieces need a plugin
from [plugins](plugins.md) to finish their job; each section names it.

## What it guarantees

### MCP servers (`user_tools`)

- **A call can't outlast its limit.** `McpToolsetCache.toolset_for` builds a
  `TimedMcpToolset` whose tools run under `McpLimits.call_timeout_seconds`. Past it the call
  gets a failed MCP result (`isError: true`) whose text says the call may still have gone
  through, and the app is refused for the rest of the run. Its tools stay listed, so a
  second call gets the same answer at once instead of another wait. With
  `MountRules(many_apps=True)` only that one tool is refused.
- **Only public servers.** `check_server_address` raises `ServerAddressRefused` unless the
  URL is `https` and every address its host resolves to is public (loopback, private,
  link-local, multicast, site-local and IPv6 forms that carry an IPv4 address are refused).
  Independently, the HTTP client every toolset built by `toolset_for` uses refuses any
  request that isn't https before it connects, so a server's headers never go out over
  plain http. It also checks the peer of each new socket and refuses one that isn't
  public, follows no redirects and ignores proxies from the environment.
  `allow_local_servers=True` turns all of these off, for development. Calling
  `check_server_address` before you mount a person's server is still worth it: it
  refuses a bad address up front with `ServerAddressRefused`, where the client's refusal
  shows up as the server being unavailable when its tools are listed. A mount whose
  `MountRules(own_host_check=True)` keeps its own client and its own host check.
- **A bad tool name costs only that tool.** A tool whose name, once prefixed
  `mcp_<slug>__`, falls outside `[a-zA-Z0-9_-]{1,64}` is left out and logged, and the
  server is reported as skipped, rather than the whole model request being refused
  (`tests/test_adk_hooks.py`).
- **Toolsets are shared safely.** `McpToolsetCache` keys a toolset on the whole descriptor
  (headers included) plus `owner`, keeps at most `cache_limit`, evicts the least recently
  used idle one, never closes one with a call running, and gives a closing server
  `attach_timeout_seconds` to hang up.
- **Skipped apps are reported once.** `UnavailableTools` collects the apps a run skipped,
  and `UnavailableToolsPlugin` puts their names on the next event's `custom_metadata`
  under `wire_names.UNAVAILABLE_TOOLS`, each name once per run.
- **Effect hints reach the approval gate.** Each timed tool exposes the server's
  `readOnlyHint` and `destructiveHint` as `effect_hints`, which the gate reads (see
  [approval](approval.md)).

### Tool gateway (`gateway`)

- **Large servers leave the prompt.** A toolset wrapped with `ToolGateway.defer` groups
  its tools by server (the part of a name before `__`). A group whose declarations add up
  to more than `threshold_tokens` is held back, except names in `pinned` (a tool's name,
  or a server's, which pins all its tools). Held-back tools appear as one line each
  (first sentence, cut at 160 characters) in a "Tools to load" section of the
  instruction, inside a tag the instruction names as data, and a description can't close
  that tag.
- **Loading is mid-turn and lasts the conversation.** `find_tools` takes
  `select:name_a,name_b` or keywords (`+word` must match; a word in the name scores 3, in
  the description 1; at most 5 tools). A loaded tool is callable from the model's next
  step, and the loaded names live in session state (`agent_loaded_tools`), so a new turn
  or a resumed run keeps them.
- **The cached prefix moves once per load.** Loaded tools' declarations go last, after
  `find_tools`, in the order they were loaded. On a step with nothing held back the
  gateway adds nothing to the request.
- **A call to a tool not loaded is answered, not run.** `ToolGateway.plugin()` returns a
  plugin that answers such a call with `not_loaded` (or `not_loaded_yet` when it was
  loaded in the same step) before the approval gate, so nobody is asked about it.

### MCP Apps pages (`mcp_apps`)

- **Page-only tools never reach the model.** A tool whose `_meta.ui.visibility` is a list
  naming `"app"` and not `"model"` is left off the model's list; any other marking,
  malformed ones included, keeps the tool (`tests/test_mcp_app_visibility.py`).
- **A page never costs the call.** `McpAppPagesPlugin` reads a call's `ui://` page on the
  tool's own connection settings, within `PageLimits.fetch_timeout_seconds`. A slow,
  missing, oversized (over `max_page_bytes`), non-UTF-8 or wrongly typed page is logged
  and the call shows as an ordinary one (`tests/test_mcp_app_pages.py`,
  `tests/test_mcp_app_page.py`).
- **CSP lists are narrowed, never widened.** Only plain `https://` or `wss://` origins are
  kept, at most `max_domains` per list.
- **A kept page goes with the conversation.** When the conversation keeps files, it is
  saved once per call as a
  conversation file under the layout's `pages` place, never listed to the model, and its
  name and SHA-256 go on the call's event under `wire_names.PAGES` (see
  [files](files.md)).
- **Advertising is opt-in.** With `McpLimits(advertise_apps=True)` the `initialize`
  request carries `capabilities.extensions["io.modelcontextprotocol/ui"]`; every other
  request passes through byte for byte (`tests/test_mcp_app_advertise.py`).

### Plugin skills (`skills`)

- **Only names and descriptions are in the prompt.** `PluginSkillToolset` serves the
  turn's plugin skills as ADK's `SkillToolset` in eager mode; bodies, resources and
  scripts reach the model only through `load_skill` and the other skill tools. A turn
  with no pinned plugins and no command gets no tools and no catalog.
- **Plugin text can't read session state.** `adk_inject_state` is removed from every
  plugin skill's metadata, so `{key}` in a body is sent as written.
- **Shared files are reachable.** A file in a plugin's top-level folder (not a part
  folder) is added to each skill's resources, and a `../../path` citation in the skill's
  text is rewritten to the served name. A skill's own file of the same name wins.
- **A broken skill doesn't end the run.** Anything a skill tool raises comes back as a
  result with `error_code: "SKILL_ERROR"`. Two plugins with the same skill name: the first
  keeps it, the other is left out with a warning.
- **A slash command is loaded before the first call.** When `read_command` returns a
  `CommandRef`, its skill goes first in the registry and its instructions are added at
  every model call of the turn. A command whose plugin didn't load is left out with a
  warning and the message runs as an ordinary one. Building the skills can fail without
  failing the turn.

### Plugin connectors (`plugin_connectors`)

- **One broken connector costs only its own tools.** `PluginConnectorToolset` mounts each
  connector by itself, in parallel, under the same address check and attach timeout as a
  person's server. A connector that fails is told to the model in the instruction (the
  list ends with the `closing` paragraph you give) and to the person through
  `UnavailableTools`. One that hung isn't tried again in the same run.
- **Sessions stay per person.** Each connector's toolset is cached per user id, so
  server-side session state never crosses people. Its tools are named
  `mcp_<plugin>_<connector>__<tool>`, and a connector gets only its own headers.
- **No secrets in logs.** Failures log the mount name and the error type, not the
  traceback, since an address can carry a key.
- **`ConnectorCheck` answers what a turn would.** It runs the same attach, keeps each
  plugin version's answer for `ttl_seconds`, runs at most `concurrency` attaches at once
  across all callers, shares one attach between concurrent callers, and closes every
  toolset it opened. A 401 is reported as needing a sign-in and remembered by address for
  `lasting_ttl_seconds`.

### `ask_person` (`ask_tools`)

- **A valid question pauses the run.** On a resumable ADK app, `ask_person` returns
  nothing, so the call stays unanswered and the run pauses. The answer, sent as this
  call's function response, resumes the same invocation (`tests/test_ask.py`).
- **A bad question is refused at once** with words the model can act on: 2 to 4 options,
  a question of at most 300 characters, labels of at most 80 and consequences of at most
  120, at most one `recommended`, no two labels alike. A second question in the same step
  is refused: one at a time.

### The plan (`plan_tools`, `todo_list`)

- **The tool keeps nothing.** `todo_list` checks the whole list (at most 20 items, each at
  most 200 characters, status `pending`, `in_progress`, `completed` or `cancelled`) and
  returns it with ids `"1"`, `"2"`, ... as the snapshot a client reads. A resumed run
  can't repeat anything by writing it again.
- **The plan rides on other calls.** With `PlanFieldPlugin`, each of the agent's own
  function tools (and tools whose class sets `takes_plan_field = True`) gets an optional
  `todos` field. Before the calls run it is lifted into a `todo_list` call placed just
  before the call that carried it. When every other tool takes the field, `todo_list` is
  hidden from the model. MCP and skill tools keep their schemas, and a tool with its own
  `todos` argument is left alone.

### Background tasks (`TaskToolset`, `task_tools`)

- **A task belongs to the conversation that started it.** The tools reach tasks only
  through the run they are part of; outside one they answer
  `{"error": "Tasks can't be started here."}`.
- **Tasks don't start tasks.** `TaskToolset` lists nothing when `enabled()` is false or
  the run's state carries `wire_names.TASK_MARK`. `NotInTaskToolset` hides any other tools
  (such as `ask_person`) on the same condition.
- **Inputs and caps are checked with words the model can act on:** a description of 1 to
  80 characters, a prompt of 1 to 20,000, and the board's caps on running tasks and tasks
  per turn. `wait_for_tasks` is cut to `wait_max_seconds()`.
- `WAITING_TOOLS` (`check_task`, `wait_for_tasks`) is the set to pass to
  `LoopGuardPlugin` as `exempt`, so polling a slow task isn't taken for a loop.

### Advisor (`AdvisorToolset`)

- **Offered only when the turn has an advisor.** `AdvisorToolset` asks
  `ConsultHooks.advisor_of(state)` each step and lists nothing on `None`.
- **Built on ADK's `ModelConsultTool`**, which keeps the per-turn (`max_uses`) and
  per-conversation (`session_max_uses`) budgets and never raises into the loop. The
  harness adds the route around the call, the metering, and its own wording: consult
  rarely, on hard steps.
- **Every call is metered.** The call runs inside `meter(...).alone()` and
  `recorded()`, and an answered consultation's result is merged with what `Settle`
  returns (its cost, under your keys). An unreachable route returns an error result
  telling the model to go on without it. `usage_metadata(result)` turns a result's
  `usage` into ADK's usage shape, or `None` when there is none.
- More on why the harness wraps ADK's advisor: [adk](adk.md).

### Described tools (`DescribedTool`)

- `DescribedTool` is ADK's `FunctionTool` with each docstring `Args:` entry moved into
  its parameter's `description`, pydantic's `title`s removed, and an optional `X | None =
  None` parameter declared as its type alone. A `description` written in the signature
  wins over the docstring.

## What it costs

- **MCP calls:** no extra model calls. One `tools/list` per server per
  `tool_list_ttl_seconds`; each new socket's peer is checked with no extra network.
  `check_server_address` is one DNS lookup. A hung call costs `call_timeout_seconds`
  once per run, then the app answers at once. Memory: up to `cache_limit` open toolsets (more while
  every one has a call running).
- **Gateway:** no model calls of its own; a load costs the model one `find_tools` step.
  Each tool declaration's tokens are counted once (cached for 4,096 distinct
  declarations). The loaded set is one list in session state. Per-run memory is dropped
  when the run ends, and at most 64 runs are kept.
- **MCP Apps:** per call with a page, one short extra MCP session (`initialize` plus
  `resources/read`), bounded by `fetch_timeout_seconds`, and one conversation-file write
  of at most `max_page_bytes` of HTML plus its record. The same page for the same call is
  written once.
- **Skills:** the catalog of names and descriptions in every model call's instruction;
  a command's full instructions in every call of its turn. Reading the turn's plugins once
  per turn (cached for 64 turns). Scripts run in your code executor, up to
  `script_timeout` seconds each.
- **Connectors:** one attach per connector each time ADK lists the turn's tools, in parallel, bounded by
  `attach_timeout_seconds`; a connector that hung costs that once per run. `ConnectorCheck` makes real attaches (it counts them in
  `attaches`), bounded by `concurrency` and its caches.
- **`ask_person`, `todo_list`:** nothing beyond the tool call. With `PlanFieldPlugin`, a
  plan update rides on a call the model was making anyway, which saves a model round trip
  per update; the field adds its schema to each carrier's declaration.
- **Tasks:** each task is a full run of its own, with a run's budget (see
  [runs](runs.md)), bounded by `max_running` and `max_per_turn`.
- **Advisor:** one advisor model call per consultation, with up to `max_context_chars`
  of the conversation and `max_output_tokens` out, capped by `max_uses` per turn and
  `session_max_uses` per conversation. Recorded through your `meter` hook.
- **`DescribedTool`:** nothing at run time; the schema is rewritten when declared.

## Configure it

The harness reads no environment: every limit is passed in.

### `McpLimits` and `MountRules`

`McpLimits` (frozen dataclass): `attach_timeout_seconds`, `call_timeout_seconds`,
`tool_list_ttl_seconds`, `cache_limit` (all required), `allow_local_servers=False`,
`advertise_apps=False`. ADK's own connect and read waits are set 5 seconds past the
attach and call limits, so the harness's limit decides.

A server descriptor is a dict: `slug`, `url`, optional `transport` (`"sse"`, otherwise
streamable HTTP), `headers`, `tools` (a filter of tool names) and `RULES_KEY`. Under
`RULES_KEY` a descriptor built in code may carry `MountRules(own_host_check=False,
many_apps=False, shown_as=None)`; one that came as JSON always gets the defaults.
`own_host_check=True` skips the hardened client, for a mount whose builder checks the
host itself.

```python
from harness.plugins import UnavailableToolsPlugin
from harness.tools.user_tools import (
    McpLimits, McpToolsetCache, ServerAddressRefused, UnavailableTools, check_server_address,
)

limits = McpLimits(attach_timeout_seconds=10, call_timeout_seconds=60,
                   tool_list_ttl_seconds=300, cache_limit=32)
unavailable = UnavailableTools()
cache = McpToolsetCache(unavailable, limits)
plugin = UnavailableToolsPlugin(unavailable)

async def notes_toolset(user_id: str):
    server = {"slug": "notes", "url": "https://mcp.example.com/mcp", "headers": {}}
    try:
        await check_server_address(server["url"], allow_local=limits.allow_local_servers)
    except ServerAddressRefused:
        return None
    return cache.toolset_for(server, owner=user_id)
```

### `ToolGateway`

`ToolGateway(threshold_tokens=..., pinned=[...])`, both required. Wrap each toolset whose
servers may be large in `defer`, put `gateway.toolset` last in the agent's tools, and put
`gateway.plugin()` before the approval plugin.

```python
from google.adk.agents import LlmAgent
from harness.tools.gateway import ToolGateway

gateway = ToolGateway(threshold_tokens=4_000, pinned=["mcp_search"])
server = {"slug": "notes", "url": "https://mcp.example.com/mcp"}
agent = LlmAgent(
    name="assistant",
    model="any-model",
    tools=[gateway.defer(cache.toolset_for(server, owner="u1")), gateway.toolset],
)
plugins = [gateway.plugin()]  # then the approval plugin
```

The deferring toolset is meant to be the outermost wrapper of what it defers: a toolset
around it would be kept for the whole invocation again.

### `PageLimits`

`PageLimits(fetch_timeout_seconds=8.0, max_page_bytes=2_000_000, max_domains=20)`. Pass it
to `McpAppPagesPlugin(limits, shows_pages)`, where `shows_pages(server_slug)` is your rule
for which servers may show pages, and set `advertise_apps=True` on `McpLimits` to tell
servers.

```python
from harness.plugins import McpAppPagesPlugin
from harness.tools.mcp_apps import PageLimits

pages = McpAppPagesPlugin(PageLimits(), shows_pages=lambda server: server != "untrusted")
```

### Plugins: `PluginSource`, `PluginSkillToolset`, `PluginConnectorToolset`, `ConnectorCheck`

`PluginSource` is a protocol your plugin loader implements: `pinned_hashes(state)` (the
content hash of each plugin version the turn names, from state alone) and
`for_turn(state)`, which returns a `TurnPlugins` (`loaded` plugins and the names of
`unavailable` ones). A plugin exposes `name`, `skills`, `files`, `content_hash`, a
`manifest` with `skill_dirs`, `command_dirs`, `agent_dirs` and `hooks`, and `components`
with its `connectors` (`name`, `url`, `transport`, `headers`, `problem`).

- `PluginSkillToolset(loader, *, code_executor=None, script_timeout=300,
  read_command=...)`. With no executor, scripts can't run. `read_command(state)` returns
  the turn's `CommandRef` or `None` (the default reader always returns `None`).
- `CommandRef(source, name, plugin=None, kind=None, description="", instructions="")`:
  `source="own"` with `instructions` for a saved prompt; `source="plugin"` with
  `kind="skill"` for a plugin's skill or `kind="command"` for its `commands/<name>.md`.
- `PluginConnectorToolset(unavailable, loader, limits, *, closing)`. It keeps its own
  `McpToolsetCache` under `limits`, one entry per person and connector, so give it a
  `cache_limit` of its own.
- `ConnectorCheck(ttl_seconds, concurrency, *, limits, close_timeout_seconds,
  lasting_ttl_seconds=0)`, where `limits` is a callable read at each check.
  `await check.check(turn_plugins)` returns one `PluginStatus(name, problem, connectors)`
  per plugin.

```python
from harness.tools.plugin_connectors import ConnectorCheck, PluginConnectorToolset
from harness.tools.skills import PluginSkillToolset

# loader: your PluginSource; unavailable and limits as in the MCP snippet
skills = PluginSkillToolset(loader, read_command=lambda state: None)
connectors = PluginConnectorToolset(
    unavailable, loader, limits, closing="Tell the person which connector the task needed."
)
check = ConnectorCheck(ttl_seconds=60, concurrency=4, limits=lambda: limits,
                       close_timeout_seconds=5, lasting_ttl_seconds=3_600)
```

### `ask_tools`, `plan_tools`, `DescribedTool`

No settings. The limits above are constants in the code. `ask_person` needs a resumable
app (`App(..., resumability_config=ResumabilityConfig(is_resumable=True))`). The tool's
name is `harness.tools.ask.TOOL_NAME`.

```python
from google.adk.agents import LlmAgent
from harness.plugins import PlanFieldPlugin
from harness.tools import DescribedTool, ask_tools, plan_tools

def word_count(text: str) -> dict:
    """Counts the words in a text.

    Args:
      text: The text to count.
    """
    return {"words": len(text.split())}

agent = LlmAgent(name="assistant", model="any-model",
                 tools=[DescribedTool(word_count), *plan_tools(), *ask_tools()])
plugins = [PlanFieldPlugin()]
```

### `TaskToolLimits`

`TaskToolLimits(enabled, max_running, max_per_turn, wait_max_seconds)`, each a callable
read when a tool runs, so a changed setting is seen without a rebuild. `max_running` and
`max_per_turn` only word the refusals: the caps themselves are the `TaskBoard`'s (see
[runs](runs.md)). `task_tools(limits)` returns the three functions if you build your own
toolset.

The tools tell a task's own run by `wire_names.TASK_MARK` in its state, while
`TaskBoard` writes its marker under `TaskSettings.marker_key`. Pass
`TaskSettings(marker_key=wire_names.TASK_MARK)` so the two agree.

```python
from harness import wire_names
from harness.runs import TaskSettings
from harness.tools import NotInTaskToolset, TaskToolLimits, TaskToolset, ask_tools

task_limits = TaskToolLimits(enabled=lambda: True, max_running=lambda: 3,
                             max_per_turn=lambda: 5, wait_max_seconds=lambda: 120.0)
tools = [TaskToolset(task_limits), NotInTaskToolset(ask_tools())]
task_settings = TaskSettings(marker_key=wire_names.TASK_MARK)
```

### `ConsultSettings`, `ConsultHooks`, `Advisor`

- `ConsultSettings(max_uses, session_max_uses, max_output_tokens, max_context_chars,
  timeout_seconds)`, all required, read once when the toolset builds a tool.
- `Advisor(model, pricing=None, id=None, name=None)`: `model` is the id the call goes out
  under; `pricing` is in your own shape and only your hooks read it.
- `ConsultHooks(name, advisor_of, client, route, meter)`: the tool's name, the turn's
  advisor from state, a factory for ADK's `LiteLLMClient`, a route with
  `target(advisor, state)` and `open(advisor, state)` (a context manager, or `None` when
  unreachable), and `meter(advisor, call_id)` returning an object with `alone()` (async
  context manager) and `recorded()` (a context manager yielding a `Settle`).

```python
from harness.tools import AdvisorToolset, ConsultSettings

settings = ConsultSettings(max_uses=2, session_max_uses=5, max_output_tokens=2_000,
                           max_context_chars=200_000, timeout_seconds=60)
advisor_tools = AdvisorToolset(settings, hooks)  # hooks: your ConsultHooks
```

## Test code that uses it

Nothing here needs a network or a real model in a test.

- **Run the real agent loop with a scripted model.** Subclass ADK's `BaseLlm` to return
  the calls you want, then run with ADK's `Runner` and `InMemorySessionService`.
  `tests/test_ask.py` does this for `ask_person`: it pauses on the question, then sends
  the answer as a function response and checks the same invocation resumes.
  `examples/test_assistant.py` does the same for `todo_list` and a product tool through
  the service's routes.
- **Call the plain tools directly.** `todo_list(...)` is a pure function that returns the
  snapshot or an `{"error": ...}`. `DescribedTool(fn)._get_declaration()` shows the schema
  the model gets, as `tests/test_ask.py` checks for `ask_person`.
- **MCP servers in-process.** Set `McpLimits(allow_local_servers=True)` and run a small
  MCP server on `127.0.0.1`; `tests/mcp_app_server.py` is one, used by
  `tests/test_mcp_app_pages.py`. For no server at all, monkeypatch
  `McpToolset.get_tools` to return stand-in tools, as `tests/test_adk_hooks.py` and
  `tests/test_mcp_app_visibility.py` do. `check_server_address(url, allow_local=True)`
  returns at once.
- **Plugins without a library.** `PluginSource` and `TurnPlugins` are protocols: a small
  class with `pinned_hashes` and `for_turn` stands in for your loader. A source whose
  `pinned_hashes` returns `()` gives a turn with no skills and no connectors.
  `ConnectorCheck.attaches` counts the connections it made.
- **The advisor.** Every outside effect is a `ConsultHooks` field: give `client` a fake
  `LiteLLMClient` (ADK's seam, with `acompletion`), and `route` and `meter` stand-ins.
- **Tasks.** Outside a run the service drives, the task tools answer with an error and
  start nothing. Testing a task end to end goes through `TaskBoard` and the service (see
  [runs](runs.md)).

The harness has no tests of its own yet for the gateway, the skill and connector
toolsets, the task tools or the advisor; the guarantees above for those come from the
source.

## Reference

`harness.tools`:

| Name | What it is |
| --- | --- |
| `ask` | Module of `ask_person`; exports `TOOL_NAME` (`"ask_person"`). |
| `ask_tools()` | `[ask_person]`, a long-running tool that pauses the run for the person's answer. |
| `plan_tools()` | `[todo_list]` as a `DescribedTool`. |
| `todo_list` | The plan tool's function: checks and returns the whole list. |
| `TaskToolLimits` | Callables for whether tasks are on, the two caps and the longest wait. |
| `task_tools(limits)` | `start_task`, `check_task`, `wait_for_tasks` as functions. |
| `TaskToolset` | The three task tools, hidden in a task's own run or when disabled. |
| `WAITING_TOOLS` | `{"check_task", "wait_for_tasks"}`, for the loop guard's `exempt`. |
| `NotInTaskToolset` | Wraps tools or a toolset so a background task doesn't get them. |
| `Advisor` | The advisor a turn may consult: model id, pricing, catalog id and name. |
| `ConsultSettings` | Budgets and limits for consultations. |
| `ConsultHooks` | Tool name, advisor lookup, model client, route and metering. |
| `Settle` | Type of the function that records a consultation and returns its extra keys. |
| `AdvisorToolset` | Offers the consult tool on turns with an advisor. |
| `usage_metadata(result)` | A consultation's usage as ADK's usage metadata, or `None`. |
| `DescribedTool` | `FunctionTool` with docstring `Args:` moved into the parameter schema. |
| `PluginSource` | Protocol: where a turn's plugins come from. |
| `TurnPlugins` | Protocol: a turn's loaded plugins and the names it goes without. |
| `user_tools` | MCP servers: limits, cache, address check, mount rules, skipped apps. |
| `gateway` | `ToolGateway`. |
| `mcp_apps` | `PageLimits`. |
| `skills` | `PluginSkillToolset`, `CommandRef`. |
| `plugin_connectors` | `PluginConnectorToolset`, `ConnectorCheck`, `PluginStatus`. |

Subpackages:

| Name | What it is |
| --- | --- |
| `user_tools.McpLimits` | Timeouts, list cache, toolset cache size, local servers, MCP Apps advertising. |
| `user_tools.McpToolsetCache` | `toolset_for(server, owner=None)`: one timed toolset per descriptor and owner. |
| `user_tools.check_server_address` | Raises unless the URL is https and its host is public. |
| `user_tools.ServerAddressRefused` | Raised by `check_server_address`. |
| `user_tools.MountRules` | Per-mount rules: own host check, many apps, display name. |
| `user_tools.RULES_KEY` | `"rules"`, the descriptor key for `MountRules`. |
| `user_tools.UnavailableTools` | Apps a run skipped, reported once each by `UnavailableToolsPlugin`. |
| `gateway.ToolGateway` | `defer(toolset)`, `toolset`, `plugin()`, `forget(invocation_id)`. |
| `mcp_apps.PageLimits` | Fetch timeout, largest page, CSP origins per list. |
| `skills.PluginSkillToolset` | The turn's plugin skills and slash command. |
| `skills.CommandRef` | The slash command a message runs. |
| `plugin_connectors.PluginConnectorToolset` | The turn's plugin connectors, each mounted alone. |
| `plugin_connectors.ConnectorCheck` | Whether each plugin's connectors work now. |
| `plugin_connectors.PluginStatus` | A plugin's name, problem and connectors' statuses. |
