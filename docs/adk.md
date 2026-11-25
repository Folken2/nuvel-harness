# The harness and ADK

The harness is built on [Google's Agent Development Kit](https://github.com/google/adk-python)
(ADK). It never patches ADK. Every piece attaches the way ADK's own code does: a plugin is a
`BasePlugin`, a toolset is a `BaseToolset`, a tool is a `BaseTool` or `FunctionTool`, and the
one model-side change is a subclass of ADK's `LlmEventSummarizer`. So you can mix harness pieces
with ADK's, and an ADK upgrade changes nothing the harness didn't choose to depend on.

Where ADK already has a built-in for a job, the harness sometimes ships its own anyway, because
it does more or behaves differently in ways production runs need. This page lists those pieces,
says what each adds over ADK's, what ADK's has that the harness lacks, and whether to run both.

## Two places the harness relies on ADK's internals

ADK offers no public hook for these. Both are pinned by `tests/test_adk_hooks.py`, so an ADK
upgrade that moves either one fails the tests rather than a run.

- **`LlmAgentWithoutIdentity`** overrides `LlmAgent._llm_flow` to drop the request step ADK
  names `identity`. That step adds `You are an agent. Your internal name is "<name>"` and the
  agent's description to every model call. An agent's name is an app id, not something the
  model should know or repeat, and the line costs tokens on every call. The agent's own
  instruction says who it is.
- **`TimedMcpToolset`** overrides `get_tools`, the hook ADK means subclasses to use. ADK then
  prefixes the names in `get_tools_with_prefix`, which is final, by the rule
  `f"{prefix}_{name}"`. The harness checks names by that same rule, so the check is on the name
  the model will see.

## Retry and repair

ADK: `ReflectAndRetryToolPlugin`, `ReflectAndRetryModelPlugin`.
Harness: `routing/retry_policy.py`, `replies/` (argument repair, empty and cut replies), wired by
`BadRepliesPlugin`.

These complement each other: they fire on different signals. ADK's plugins react to a tool that
raised, and to a model reply marked as a malformed function call. The harness reacts to a
request that failed and to a reply that finished but is unusable.

What the harness adds:

- It sorts a failed request into overflow, quota, refused, rate limited or transient. A spent
  quota is told apart from a per-minute limit, and is never resent on the same route: it falls
  back to the next one.
- It waits as long as the server asks (`retry-after`, `retry-after-ms`, or a delay in the error
  body), capped by what the run may still spend, with jittered backoff otherwise.
- A request that's too long is shrunk once, by masking old tool results, and sent again.
- Tool-call arguments are repaired where the tool's JSON schema leaves one reading: a number or
  flag sent as text, an object or list sent as JSON text, `{}` for an optional field. Every
  repair is logged, without the values.
- An empty reply gets one note asking the model again, once per run. A reply cut off at the
  output limit has its tool calls refused instead of run with half-written arguments.

What ADK's has that the harness lacks: a per-tool retry count with a configurable scope, a hook
to read an error out of a tool's result, and the reflection loop for malformed calls.

Running both: possible, with one caution. ADK's model plugin adds a synthetic function call to
the history, and some providers reject a history with calls they didn't make when extended
thinking is on. The harness only ever removes the notes it added itself.

## Context trimming

ADK: `ContextFilterPlugin`.
Harness: `OldResultsPlugin`, `ToolResultLimitPlugin`, `ContextBudgetPlugin` (`context/`).

The harness replaces this one. ADK's plugin keeps the last N invocations and drops older ones
whole, the person's messages included.

What the harness adds:

- No turn is ever dropped. Only older tool results are sent as a 200-character preview with a
  note, and the session keeps them whole.
- The full result is saved as a conversation file the note names, so the model can read it
  again. A batch that can't be saved isn't masked.
- Results are masked in batches from a saved mark, so the provider's cached prompt prefix breaks
  once per batch rather than on every call.
- The results of the step the model is on are never masked.
- An oversized result is cut where it arrives, keeping its head and tail, with the full copy
  saved. Binary content is replaced by a size mark.
- The point at which history is summarized is set per run from the model's context window.

What ADK's has that the harness lacks: a plain "keep the last N turns" cap and a custom filter
function.

Running both: it works mechanically, but `ContextFilterPlugin` would drop the turns the harness
keeps and break the cached prefix on every cut. Pick one.

## Compaction

ADK: `LlmEventSummarizer` and `EventsCompactionConfig`.
Harness: `MeteredSummarizer` and `events_compaction()` (`context/compaction.py`).

The harness extends ADK's here: same config, same triggers, a subclass of ADK's summarizer.

What the harness adds:

- Each summary call is recorded in the usage ledger, with its cost and the model that served it.
- The person's last five messages and the latest plan are written after the summary word for
  word, and carried into later summaries, so a long run never loses what was asked for.
- The summarizer is told the history is text to describe, so an instruction inside it is
  recorded, never followed.
- The prompt asks for fixed sections, exact names and figures, and the person's own words.

## MCP servers

ADK: `McpToolset`.
Harness: `TimedMcpToolset`, `TimedTool`, `McpToolsetCache` (`tools/user_tools/`).

The harness extends ADK's: it calls through to `McpToolset` for the connection and the tool list.

What the harness adds:

- A per-call time limit. ADK waits as long as its read timeout, five minutes by default. Past
  the limit, the app is refused for the rest of the turn but stays listed, so a model that
  calls it again gets the same answer at once.
- Only public servers: https only, every resolved address must be public, checked again on each
  connection at the socket. Redirects and proxies from the environment are off.
- A tool whose name model providers would refuse (outside `[a-zA-Z0-9_-]{1,64}`) is left out
  and reported, rather than failing the whole request.
- Toolsets are cached per server and owner, never closed while one of their calls is running,
  and closing one is time boxed.
- The MCP `readOnlyHint` and `destructiveHint` feed the approval gate.

## Logging

ADK: `LoggingPlugin`.
Harness: `TurnLogPlugin`.

ADK's plugin prints message content, tool arguments and tool results. The harness logs tool
names, durations and error types only, so a person's data stays out of the logs. Running both
puts that data back in.

## Smaller overlaps

- **Conversation files.** ADK's `LoadArtifactsTool` lists every artifact. The harness's
  `ConversationFilesTool` leaves out files the run made itself and saved copies of cut results,
  and sends the list as its own message rather than in the system instruction, which keeps the
  cached prefix stable.
- **Advisor.** The harness's `AdvisorConsultTool` picks the advisor model per turn, routes the
  call and records it in the ledger. ADK calls its advisor outside the plugins, so nothing would
  record it.
- **Asking the person.** ADK's `get_user_choice_tool` and the harness's `ask_person` pause the
  run the same way, as a long-running tool. `ask_person` also checks the question. Use one.

The harness uses these ADK pieces as they are: `SkillToolset`, `SaveFilesAsArtifactsPlugin`,
`EventsCompactionConfig`, `ResumabilityConfig`, the session services and the runner.
