"""An agent's HTTP service: ADK's API server with the harness's runs and erasure on it.

A product's start-up reads its settings and calls these, then adds what is its own
(its routes, its token gate). Nothing here reads a setting or names a product:

- ``create_service`` builds ADK's app over the stores it is told, adds the health check
  and the run routes (runs/routes.py), and returns the app with its run registry.
- ``add_erasure_routes`` adds the two routes that delete what the agent keeps for a
  conversation or a person (erasure.py).
- ``asyncpg_uri`` and ``session_db_kwargs`` say how the session database is reached.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from fastapi import FastAPI
from google.adk.artifacts import BaseArtifactService
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.cli.service_registry import get_service_registry
from google.adk.sessions import BaseSessionService, DatabaseSessionService, InMemorySessionService
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from .erasure import AgentDataEraser
from .runs.registry import RunRegistry
from .runs.routes import add_run_routes
from .runs.tasks import TaskBoard

__all__ = ["ServiceOptions", "Stores", "add_erasure_routes", "asyncpg_uri", "create_service", "session_db_kwargs"]


def asyncpg_uri(uri: str) -> str:
    """postgres(ql):// -> postgresql+asyncpg://, dropping libpq-only params."""
    parts = urlsplit(uri)
    scheme = "postgresql+asyncpg"
    query = [(k, v) for k, v in parse_qsl(parts.query) if k not in ("sslmode",)]
    return urlunsplit((scheme, parts.netloc, parts.path, urlencode(query), ""))


def is_local_db(uri: str) -> bool:
    parts = urlsplit(uri)
    socket_host = dict(parse_qsl(parts.query)).get("host", "")
    return parts.hostname in (None, "", "localhost", "127.0.0.1") or socket_host.startswith("/")


def session_db_kwargs(uri: str, connect_args: dict[str, Any]) -> dict[str, Any]:
    """SQLAlchemy engine options for the session database.

    pool_pre_ping checks a pooled connection before reusing it and
    pool_recycle replaces it after 5 minutes, so a connection dropped by a
    database restart or proxy timeout doesn't fail the user's next turn.
    ``connect_args`` are the service's own, its time limits among them; a database
    that isn't on this machine is reached over TLS.
    """
    kwargs: dict[str, Any] = {"pool_pre_ping": True, "pool_recycle": 300}
    connect_args = dict(connect_args)
    if not is_local_db(uri):
        connect_args["ssl"] = "require"
    if connect_args:
        kwargs["connect_args"] = connect_args
    return kwargs


@dataclass(frozen=True)
class Stores:
    """Where the service keeps sessions and files.

    ``session_uri`` None means sessions in memory: nothing survives a restart, so nothing
    can be resumed. Whether that is allowed is the product's rule, checked before this.
    """

    session_uri: Optional[str]
    # The session engine's connect_args for a URI: the service's time limits.
    connect_args: Callable[[str], dict[str, Any]]
    artifact_uri: Optional[str] = None

    def db_kwargs(self) -> dict[str, Any]:
        assert self.session_uri is not None
        return session_db_kwargs(self.session_uri, self.connect_args(self.session_uri))

    def run_sessions(self) -> BaseSessionService:
        """The store a resume reads: the same database ADK was given."""
        if not self.session_uri:
            return InMemorySessionService()
        return DatabaseSessionService(asyncpg_uri(self.session_uri), **self.db_kwargs())

    def eraser(self, files_to_clear: FilesToClear) -> AgentDataEraser:
        """Reaches the same stores ADK was given. Connects on first use."""
        engine: AsyncEngine | None = (
            create_async_engine(asyncpg_uri(self.session_uri), **self.db_kwargs())
            if self.session_uri
            else None
        )
        files = (
            get_service_registry().create_artifact_service(self.artifact_uri)
            if self.artifact_uri
            else None
        )
        return AgentDataEraser(engine, files_to_clear(files))


# Which of the artifact stores an erasure clears: a product leaves out one that isn't
# the agent's to clear.
FilesToClear = Callable[[Optional[BaseArtifactService]], Optional[BaseArtifactService]]


@dataclass(frozen=True)
class ServiceOptions:
    """What ADK's app is built with, besides the stores."""

    agents_dir: str
    reload_agents: bool = False
    extra_plugins: Sequence[str] = ()
    max_llm_calls: Optional[int] = None
    # Runs beside the service for as long as it does.
    lifespan: Optional[Callable[..., Any]] = None
    # The tools that only read: after a restart the model is asked to call one again,
    # where any other interrupted call is answered as unknown (runs/resume.py).
    safe_to_repeat: Collection[str] = ()


@dataclass(frozen=True)
class Service:
    app: FastAPI
    # Runs that outlive the connection that started them. One registry, so one instance
    # of the service per deployment: see runs/registry.py.
    runs: RunRegistry


def create_service(
    options: ServiceOptions,
    stores: Stores,
    *,
    tasks: Callable[[RunRegistry], Optional[TaskBoard]],
) -> Service:
    """ADK's API server over ``stores``, with the health check and the run routes.

    Sessions in a database make every completed event durable, which is what lets an
    interrupted run resume.
    """
    common: dict[str, Any] = dict(
        agents_dir=options.agents_dir,
        web=False,
        a2a=False,
        host="",
        url_prefix=None,
        reload_agents=options.reload_agents,
        extra_plugins=list(options.extra_plugins),
        max_llm_calls=options.max_llm_calls,
        artifact_service_uri=stores.artifact_uri,
        lifespan=options.lifespan,
    )
    if stores.session_uri:
        app = get_fast_api_app(
            session_service_uri=asyncpg_uri(stores.session_uri),
            session_db_kwargs=stores.db_kwargs(),
            **common,
        )
    else:
        app = get_fast_api_app(session_service_uri=None, use_local_storage=False, **common)

    @app.get("/health")
    async def health():  # no return type: FastAPI would make it a response model
        return {"status": "ok"}

    runs = RunRegistry()
    add_run_routes(app, runs, stores.run_sessions(), tasks(runs), options.safe_to_repeat)
    return Service(app, runs)


def add_erasure_routes(app: FastAPI, runs: RunRegistry, eraser: AgentDataEraser) -> None:
    """Deleting a conversation or a person: their runs stop, then what is kept goes."""

    @app.delete("/erasure/apps/{app_name}/users/{user_id}/sessions/{session_id}")
    async def erase_conversation(app_name: str, user_id: str, session_id: str):
        runs.stop_all_of(app_name, user_id, session_id)
        return (await eraser.erase_conversation(app_name, user_id, session_id)).as_dict()

    @app.delete("/erasure/apps/{app_name}/users/{user_id}")
    async def erase_user(app_name: str, user_id: str):
        runs.stop_all_of(app_name, user_id)
        return (await eraser.erase_user(app_name, user_id)).as_dict()
