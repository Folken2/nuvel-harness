# Service and start-up

A product that runs an ADK agent behind HTTP needs the same few things at start-up: ADK's
API server over a durable session store, routes that let a run outlive its connection, a
way to delete what the agent keeps about a person, sane database time limits, and names for
the keys it puts on events. These top-level modules do that. A product imports
`from harness.service import ...`, `from harness import wire_names, run_state, engine_limits`,
`from harness.erasure import ...` and `from harness.agent_without_identity import
LlmAgentWithoutIdentity`.

None of them reads the environment or names a product: the product's start-up reads its own
settings and passes them in.

## What it guarantees

### `service`

- **`create_service`** builds ADK's own API server (`get_fast_api_app`, with ADK's web UI
  and A2A off) over the stores it is given, adds `GET /health` (`{"status": "ok"}`) and the
  run routes (see [runs](runs.md)), and returns `Service(app, runs)`: the FastAPI app and
  its `RunRegistry`.
- **Sessions in a database make runs durable.** With `Stores.session_uri` set, ADK saves
  every completed event, which is what a resume after a restart rebuilds from. With it
  `None`, sessions are in memory and nothing can be resumed.
- **`add_erasure_routes`** adds `DELETE /erasure/apps/{app}/users/{user}/sessions/{session}`
  and `DELETE /erasure/apps/{app}/users/{user}`. Each stops the runs it covers first, then
  erases, and answers with the counts (`{"sessions", "events", "user_states", "files"}`).
- **`asyncpg_uri`** turns a `postgres://` or `postgresql://` URI into
  `postgresql+asyncpg://` and drops `sslmode`, which asyncpg does not take.
- **`session_db_kwargs`** checks a pooled connection before reusing it
  (`pool_pre_ping=True`) and replaces it after 300 seconds (`pool_recycle=300`), so a
  connection a database restart or a proxy dropped does not fail the next turn. A database
  that is not on this machine (not `localhost`, `127.0.0.1` or a socket path) is reached
  with `ssl="require"`. With `max_connections` (from `Stores.max_connections`) one engine
  pools that many connections and opens none beyond them (`pool_size=n`,
  `max_overflow=0`); a checkout past them waits SQLAlchemy's `pool_timeout`. Left out,
  SQLAlchemy's default pool applies: 5, plus 10 overflow.

### `erasure`

- **`AgentDataEraser.erase_conversation`** deletes a conversation's files, then its
  `sessions` and `events` rows and those of its background tasks, in one transaction.
  The person's own files (ADK's `user:` prefix) outlive the conversation and are kept.
- **`AgentDataEraser.erase_user`** deletes all of a person's files, sessions, events and
  their `user_states` row. `app_states` belongs to the app and is never touched.
- **Safe to repeat.** A second run finds nothing and returns zeros, so a delete that failed
  partway is retried by running it again. Files go first, so a stop partway still leaves the
  session row that says what to retry.
- With no database engine (in-memory sessions) it deletes no rows and logs a warning.

### `engine_limits`

- **`connect_args`** gives asyncpg's `connect_args` for one URL: `command_timeout` (the
  client's own limit) and `server_settings` (`statement_timeout` and
  `idle_in_transaction_session_timeout`, in milliseconds). A limit of 0 is left out. For a
  transaction pooler's URL the server settings are left out, since a pooler refuses startup
  parameters it can't track, and `server_limits` is not called.

### `wire_names`

- **One namespace per process.** `use_namespace` refuses an empty one (`ValueError`) and a
  second, different one (`RuntimeError`); saying the same one again is fine. Reading a name
  before it is set raises `RuntimeError`.
- **Names are spelled when read, not at import**, so any harness module can be imported
  before the product says its namespace. A key on event metadata is `<namespace>_<name>`
  (`wire_names.COST_USD` is `"myproduct_cost_usd"`). A frame's key is `<namespace>Name`
  (`"myproduct" + "Run"`).
- These are a contract with every client and every stored event: changing the namespace
  changes all of them.

### `run_state`

- **`run_state.ALL`** is every state key the harness keeps for itself during one
  invocation. All are ADK `temp:` keys, so none is ever saved, and a client neither sends
  nor reads them.

### `agent_without_identity`

- **`LlmAgentWithoutIdentity`** is an `LlmAgent` whose model calls leave out ADK's
  `identity` step (`You are an agent. Your internal name is "<name>"` plus the agent's
  description). Every other request step stays, in order. It overrides `_llm_flow`, an ADK
  internal, which `tests/test_adk_hooks.py` pins. See [the harness and ADK](adk.md).

## What it costs

- **Database connections.** With a session URI there can be three engines: ADK's own,
  the one `Stores.run_sessions()` builds for resume reads, and the one `Stores.eraser()`
  builds (it connects on first use). So at most three times `Stores.max_connections` (45
  with SQLAlchemy's defaults); add the product's own engines when checking the total
  against the database's connection limit. Each pooled checkout costs one ping
  (`pool_pre_ping`).
- **Erasure:** in the artifact store, one listing and one delete per file. In the database,
  one transaction of two `DELETE` statements for a conversation, three for a person, plus
  one `SELECT` of a person's session ids. The SQL is Postgres (`starts_with`,
  `to_regclass`). Before ADK has created its tables, nothing is queried past that check.
- **`LlmAgentWithoutIdentity`:** saves the identity line's tokens on every model call.
- **`wire_names`, `run_state`, `engine_limits`:** nothing at run time.

## Configure it

```python
@dataclass(frozen=True)
class ServiceOptions:
    agents_dir: str
    reload_agents: bool = False
    extra_plugins: Sequence[str] = ()
    max_llm_calls: Optional[int] = None
    lifespan: Optional[Callable[..., Any]] = None   # runs beside the service
    safe_to_repeat: Collection[str] = ()   # read-only tools a resume asks the model to call again

@dataclass(frozen=True)
class Stores:
    session_uri: Optional[str]   # None: sessions in memory
    connect_args: Callable[[str], dict[str, Any]]   # the engine's connect_args for a URI
    artifact_uri: Optional[str] = None
    max_connections: Optional[int] = None   # per engine; None: SQLAlchemy's default pool

@dataclass(frozen=True)
class ServerLimits:
    statement_timeout_seconds: float = 0
    idle_in_transaction_timeout_seconds: float = 0

create_service(options, stores, *, tasks)   # tasks: (RunRegistry) -> TaskBoard | None
connect_args(url, *, command_timeout_seconds, server_limits, is_transaction_pooler)
AgentDataEraser(engine, files_to_clear)
```

A start-up, with the namespace said first:

```python
from harness import engine_limits, wire_names
from harness.engine_limits import ServerLimits
from harness.service import ServiceOptions, Stores, add_erasure_routes, create_service

wire_names.use_namespace("myproduct")


def db_limits(url: str) -> dict:
    return engine_limits.connect_args(
        url,
        command_timeout_seconds=30,
        server_limits=lambda: ServerLimits(
            statement_timeout_seconds=25, idle_in_transaction_timeout_seconds=60
        ),
        is_transaction_pooler=lambda url: False,  # your rule for which hosts are poolers
    )


# session_uri and artifact_uri come from the product's own settings.
stores = Stores(session_uri=session_uri, connect_args=db_limits, artifact_uri=artifact_uri)
service = create_service(
    ServiceOptions(agents_dir="agents", safe_to_repeat=("search_docs",)),
    stores,
    tasks=lambda runs: None,  # no background tasks; see runs.md for a TaskBoard
)
# Erase files too. Pass `lambda files: None` when another system owns the files.
add_erasure_routes(service.app, service.runs, stores.eraser(lambda files: files))
app = service.app
```

`Stores.eraser` takes a function that picks which artifact store an erasure clears, given
the one built from `artifact_uri`. In the agent, use `LlmAgentWithoutIdentity` wherever
`LlmAgent` would go; it takes the same arguments.

## Test code that uses it

- **The service:** `Stores(session_uri=None, connect_args=lambda uri: {})` gives ADK's
  in-memory sessions and files, with no database. Drive it with FastAPI's `TestClient` and a
  scripted `BaseLlm` on the agent, as `examples/test_assistant.py` and `examples/serve.py`
  do.
- **Erasure without a database:** `AgentDataEraser(None, InMemoryArtifactService())`
  erases files from ADK's in-memory artifact store and returns the counts, so a test can
  check which files go and which (`user:`) stay. The row deletes need Postgres.
- **`connect_args`, `asyncpg_uri`, `session_db_kwargs`** are pure functions: call them with
  a URL and compare the dicts.
- **`wire_names`:** one namespace per process. Set it once in a `conftest.py`, as
  `tests/conftest.py` does, or run a test in a fresh process, as `tests/test_examples.py`
  runs the example.
- **`LlmAgentWithoutIdentity`:** `tests/test_adk_hooks.py` builds one and checks its request
  steps against a plain `LlmAgent`'s.

## Reference

| Module | Name | What it is |
| --- | --- | --- |
| `service` | `create_service` | ADK's API server over the stores, with `/health` and the run routes. |
| | `ServiceOptions` | What ADK's app is built with, besides the stores. |
| | `Stores` | Session and artifact store URIs, and the engine's `connect_args`. |
| | `add_erasure_routes` | The two `DELETE /erasure/...` routes. |
| | `asyncpg_uri` | A Postgres URI as `postgresql+asyncpg://`, without `sslmode`. |
| | `session_db_kwargs` | SQLAlchemy engine options for the session database. |
| `erasure` | `AgentDataEraser` | Deletes a conversation's or a person's sessions, events and files. |
| | `Erased` | The counts an erasure returns, with `as_dict()`. |
| `engine_limits` | `connect_args` | asyncpg `connect_args` carrying the time limits. |
| | `ServerLimits` | The server's statement and idle-in-transaction limits, in seconds. |
| `wire_names` | `use_namespace` | Sets the process's namespace, once. |
| | `COST_USD` | Event metadata key: a model call's reported cost. Clients and the ledger read it. |
| | `ROUTED_MODEL` | Event metadata key: the model that served the call. Clients and the ledger read it. |
| | `RESPONSE_ID` | Event metadata key the ledger reads: the call's response id. |
| | `WEB_SEARCHES` | Event metadata key the ledger reads: the web searches a call made. |
| | `TASK_MARK` | State key that marks a background task's session. |
| `run_state` | `ALL` | Every `temp:` state key the harness keeps for itself. |
| `agent_without_identity` | `LlmAgentWithoutIdentity` | An `LlmAgent` without ADK's identity step. |
