# Runs

ADK runs an agent for as long as the HTTP request that asked for it stays open. A product
needs more: a run that keeps going when the browser drops, that a second tab can watch, that
a person can stop or send a message to mid-run, and that picks up safely after a restart.
This package holds those runs, the background tasks an agent hands work to, and the guard
against a run that keeps repeating itself. A product imports
`from harness.runs import LoopSettings, RunRegistry, TaskBoard, TaskSettings`, and
`harness.runs.tasks` and `harness.runs.loop_guard` for a few more names.

Most of it reaches a product through `create_service` (see [service](service.md)), which
adds the run routes below to ADK's API server and returns the `RunRegistry` they use.

## What it guarantees

### Runs (`RunRegistry` and the run routes)

- **One run per session.** `RunRegistry.reserve` refuses a session whose run is still
  reserved or running, and the route answers `409 {"error": "run_active"}`. Two runs at
  once would interleave one history.
- **A run belongs to the service, not the connection.** The run is driven in-process
  against ADK's own `/run_sse`, so request parsing, the session check and event formatting
  stay ADK's. A viewer leaving changes nothing. Only a Stop disconnects it, through the
  same `http.disconnect` a dropped client would send.
- **Any number of viewers, from any point.** Every frame is kept in memory. `GET .../events`
  streams the frames past `?after=N`, live until the run ends, and reading changes nothing,
  so a viewer that reconnects reads the same run again. The last frame names how the run
  ended (`{"<namespace>Run": {"status": "finished"}}`), so a reader can tell an ended run
  from a dropped connection. After 15 quiet seconds a `: keepalive` comment is sent.
- **Stop.** A reserved run ends at once. A running one ends once ADK has unwound, with
  status `stopped`. A Stop on a conversation also stops its background tasks. Model streams
  the run left open are closed when ADK is done, so a stopped call does not keep reading
  (and paying for) the provider's answer.
- **No run is left holding a session.** A reserved run that never starts lapses after
  180 seconds and ends `failed`. An error between a run becoming `running` and its driver
  taking it (a cancelled request included) ends it `failed` (`RunRegistry.until_driven`).
- **Resume never runs a tool call twice.** After a restart, `POST .../resume` rebuilds the
  run from the session store. A call with no saved result is answered as interrupted
  ("Its outcome is unknown ... Check before repeating it"), so the model decides. A tool
  named in `ServiceOptions.safe_to_repeat` is answered with a note asking the model to call
  it again. The call itself is never run by the resume. A call the person had approved is
  treated the same way. A call waiting on the person (a question, an approval) is a pause
  and is left alone.
- **Resume is not fooled.** A run that had already ended is not run again: its frames are
  replayed and it ends `finished`. A run with no new steps since its last resume
  (`afterSteps`) is a crash loop and is refused with `no_progress`.
- **Send now.** `POST .../steer` puts a message in the run's inbox, once per message id.
  Before each model call `SteeringPlugin` writes waiting messages into the history as the
  person's, and the run sends a steer frame for each one that went in. Accepted is not
  taken: a message still waiting when the run ends never went in.
- **Regenerate rewinds instead of duplicating.** A start whose body carries the regenerate
  flag and a message id rewinds the last turn with ADK's own rewind first, but only when
  that turn was stamped with the same message id. If the rewind fails the run does not
  start (`502`).
- **Erasure.** `RunRegistry.stop_all_of` stops every run of a person or a conversation
  (its tasks' sessions included) and drops their frames as soon as each has ended.

### Background tasks (`TaskBoard`)

- **A task is an ordinary run**, on a session of its own named `<parent>.task-<id>`, so it
  keeps going unwatched, stops with a Stop and resumes like any run. It is started
  in-process as an internal call, so it does not depend on the parent's token.
- **Caps.** `max_running` bounds the tasks one person has running, counting starts still
  under way, so calls made together in one step cannot all pass it. `max_per_run` bounds
  the tasks one parent run may start (default: `max_running`). A start that fails is not
  counted.
- **A task belongs to its conversation.** `TaskBoard.get` returns a task only to the parent
  that started it.
- **The report comes back as data.** When a task ends, its report (its last message that
  called no tool, cut at 20,000 characters) joins the parent's run as a message. It is
  fenced between markers with a random nonce, marker lookalikes in the text are removed, and
  the parent is told to treat it as information, not instructions. A task that finished
  without a report reads as `failed`. A wait that is collecting the report, or a Stop, keeps
  it out of the inbox.
- **After a restart** the resume route re-adopts a task's run from its session's marker, so
  its report still reaches the parent.

### Loop guard (`LoopSettings`, used by `LoopGuardPlugin` and `LoopPausePlugin`)

- Counts the same call (exact name and arguments) with the same result, and short cycles
  of up to three calls (A, B, A, B) with the same results each round. A different result is
  progress and resets the count.
- At `note_at` repeats the model gets one note (`loop_guard.NOTE`). At `stop_at` the call is
  held and the model is told to ask the person whether to go on. Their answer starts the
  count again. Where nobody can be asked (no ask tool, or a background task) the call is
  refused and `LoopPausePlugin` ends the run paused with reason `loop`.
- Nothing is kept between callbacks: the count is read from the run's own events, so a
  resume counts the same.

## What it costs

- **Memory, one process.** The registry is a dict in memory. Every frame of a run is held
  while it runs and for 600 seconds after it ends. A restart loses the frames but not the
  completed steps, which ADK keeps in the session store. A second instance would neither see
  nor stop another's runs: run one instance per deployment.
- **No storage of its own.** The registry and the board write nothing. A resume reads the
  session once and appends one event when it has interrupted calls to answer. A steered
  message is one event in the session.
- **Regenerate:** two in-process calls (read the session, then rewind), only when the body
  carries the regenerate flag. Other bodies are not even parsed for it.
- **Tasks:** each task is a full run with its own model calls and its own run budget, plus
  one in-process call to create its session. The parent's context grows by each report (at
  most 20,000 characters), and the parent's viewers get a task frame each time a task's
  steps or activity change.
- **Loop guard:** no model calls. Before each tool call, and after each reply with calls, it
  walks the run's events and hashes each call and result. A note is one extra tool turn in
  the history, sent to the provider as plain user text.
- **Limits on input:** a steered message is at most 32,000 characters. Only the
  `authorization` header of the caller is forwarded to `/run_sse`.

## Configure it

`RunRegistry()` takes no settings: `create_service` makes one and returns it as
`service.runs`. Background tasks are off unless the product passes a board:

```python
TaskBoard(registry, max_running, max_per_run=None, *, settings=None)

@dataclass(frozen=True)
class TaskSettings:
    marker_key: str = "temp:task"   # state key the task's session is marked under
    inherited: frozenset[str] = frozenset()   # parent state keys a task starts with
    own_state: OwnState | None = None   # async (parent_state, task_id) -> extra state

@dataclass(frozen=True)
class LoopSettings:
    note_at: int
    stop_at: int
```

Set `marker_key` to `wire_names.TASK_MARK`. The rest of the harness (the task tools, the
loop guard, `NotInTaskToolset`) recognizes a task's run by that key, and ADK does not save
`temp:` keys, so with the default the mark is not kept and a resumed task is not re-adopted.

```python
from collections.abc import Mapping

from harness import wire_names
from harness.plugins import BadRepliesPlugin, LoopGuardPlugin, LoopPausePlugin
from harness.runs import LoopSettings, RunRegistry, TaskBoard, TaskSettings, loop_guard
from harness.service import ServiceOptions, Stores, create_service

wire_names.use_namespace("myproduct")


async def own_state(parent_state: Mapping[str, object], task_id: str) -> Mapping[str, object]:
    return {"task_ledger_id": task_id}


def task_board(runs: RunRegistry) -> TaskBoard:
    return TaskBoard(
        runs,
        max_running=3,
        max_per_run=2,
        settings=TaskSettings(
            marker_key=wire_names.TASK_MARK,
            inherited=frozenset({"user:locale"}),
            own_state=own_state,
        ),
    )


service = create_service(
    ServiceOptions(agents_dir="agents", safe_to_repeat=("search_docs",)),
    Stores(session_uri=None, connect_args=lambda uri: {}),
    tasks=task_board,
)

# In the agent's App: the loop guard, and the note text it stands for.
loop = LoopSettings(note_at=3, stop_at=5)
plugins = [
    BadRepliesPlugin(notes={loop_guard.KIND: loop_guard.NOTE}),
    LoopGuardPlugin(loop, ask_tool="ask_person"),
    LoopPausePlugin(loop),
]
```

`BadRepliesPlugin` answers the loop guard's note call. Without `notes={loop_guard.KIND:
loop_guard.NOTE}` it answers with its empty-reply note instead. Where each plugin goes in
the order is on the [plugins](plugins.md) page. The task tools the model calls
(`TaskToolset`, `TaskToolLimits`) are on the [tools](tools.md) page.

`harness.runs.tasks.inherited_state(parent_state, inherited)` returns the part of a
parent's state a task starts with: the keys in `inherited`, nothing else.

### The run routes

All under `/runs/apps/{app_name}/users/{user_id}/sessions/{session_id}`:

| Call | Does | Answers |
| --- | --- | --- |
| `POST` | Reserves the session's run, so a Stop has a target. | `201` with `runId`, `409 run_active` |
| `POST /start?runId=` | Starts it. The body is ADK's `/run_sse` body. | `202`, `409 run_stopped`, `/run_sse`'s own refusal, `502` on a failed rewind |
| `POST /resume?since=&afterSteps=` | Picks up the run started at `since` after a restart. Body: the run's original body. | `202` or `200` (already ended) with `steps`, `409` with `run_active`, `nothing_to_resume` or `no_progress` |
| `POST /replay?since=&status=` | Puts back what a run did, without running more. `status` is `stopped` or `failed`. | `200`, `400 invalid_status` |
| `GET` | The run's status. | `200`, `404 no_run` |
| `GET /events?after=` | The run's frames as server-sent events. | stream, `404 no_run` |
| `POST /steer` | Body `{"id", "text"}`: a message for the running turn. | `202`, `400 invalid_body`, `409 no_run` |
| `POST /stop` | Stops the run and its tasks. | `{"stopped": true}` or `false` |

A run's status is one of `reserved`, `running`, `finished`, `stopped`, `failed`.

## Test code that uses it

Everything here runs in memory, so a test needs no network and no model key:

- **The whole service.** Build it with `Stores(session_uri=None, connect_args=lambda uri:
  {})` (ADK's in-memory sessions), put a scripted `BaseLlm` on the agent, and call the run
  routes with FastAPI's `TestClient`: create the session, reserve, start, read `/events`.
  `examples/test_assistant.py` does exactly this, including reading the run a second time
  from the start.
- **One namespace per process.** `wire_names.use_namespace` refuses a second namespace, so a
  product's tests set theirs once (as `tests/conftest.py` does) or run in a fresh process
  (as `tests/test_examples.py` runs the example).
- **Resume and replay** read the sessions through `Stores.run_sessions()`. With in-memory
  sessions that is a store of its own, so `/resume` answers `nothing_to_resume`. To test a
  resume, give the test a session database. `tests/test_ask.py` shows how to build a run's
  history with ADK's `Runner` and `InMemorySessionService` and read it back.
- **The registry on its own** is a plain object with no I/O: `RunRegistry()` in a test, then
  `reserve`, `begin`, `stop` and `get` on a key `(app, user, session)`.
- **The loop guard** needs only a scripted model that makes the same call each time. With
  `LoopSettings(note_at=2, stop_at=3)` and `BadRepliesPlugin(notes={loop_guard.KIND:
  loop_guard.NOTE})`, the second identical call puts `loop_guard.NOTE` in the next request
  as user text, and the third is held (or refused, with no `ask_tool`).

The harness has no test of its own for the registry, the board or the loop guard yet.

## Reference

| Name | What it is |
| --- | --- |
| `RunRegistry` | The runs in progress, one per session, and their frames. In memory. |
| `TaskBoard` | Starts, follows and caps background tasks, and delivers their reports. |
| `TaskSettings` | Which state key marks a task, which parent keys it inherits, its own state. |
| `LoopSettings` | `note_at` and `stop_at` for the loop guard plugins. |
| `tasks` | Submodule. Public: `inherited_state`. |
| `tasks.inherited_state` | The parent's state restricted to the inherited keys. |
| `loop_guard` | Submodule. Public: `KIND`, `NOTE`. |
| `loop_guard.KIND` | The note kind the loop guard adds (`"repeated_calls"`). |
| `loop_guard.NOTE` | The text the model reads at `note_at` repeats. |
