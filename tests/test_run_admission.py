"""How many runs a service admits at once, and what a caller is told past that."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.adk.sessions import InMemorySessionService

from harness.runs.registry import TASK_SESSION_MARK, RunRegistry, RunsFull, RunStatus
from harness.runs.routes import add_run_routes


def _key(session: str) -> tuple[str, str, str]:
    return ("app", "ada", session)


def test_no_cap_by_default():
    registry = RunRegistry()
    for n in range(50):
        registry.reserve(_key(f"s{n}"))
    assert registry.live_runs() == 50


def test_a_reserve_past_the_cap_is_refused_until_a_run_ends():
    registry = RunRegistry(max_live_runs=2)
    first = registry.reserve(_key("a"))
    registry.reserve(_key("b"))
    with pytest.raises(RunsFull):
        registry.reserve(_key("c"))

    first.end(RunStatus.FINISHED)
    registry.reserve(_key("c"))
    assert registry.live_runs() == 2


def test_background_tasks_neither_count_nor_are_refused():
    registry = RunRegistry(max_live_runs=1)
    registry.reserve(_key("a"))
    registry.reserve(_key(f"a{TASK_SESSION_MARK}t1"))
    registry.reserve(_key(f"a{TASK_SESSION_MARK}t2"))
    assert registry.live_runs() == 1


def test_a_run_back_after_a_restart_counts_and_is_never_refused():
    registry = RunRegistry(max_live_runs=1)
    registry.restore(_key("a"), started_at=0.0)
    registry.restore(_key("b"), started_at=0.0)
    assert registry.live_runs() == 2
    with pytest.raises(RunsFull):
        registry.reserve(_key("c"))


def test_a_cap_below_one_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        RunRegistry(max_live_runs=0)


def test_the_reserve_route_says_busy_with_a_retry_after():
    app = FastAPI()
    add_run_routes(app, RunRegistry(max_live_runs=1), InMemorySessionService())
    client = TestClient(app)

    assert client.post("/runs/apps/app/users/ada/sessions/a").status_code == 201
    busy = client.post("/runs/apps/app/users/bob/sessions/b")
    assert busy.status_code == 503
    assert busy.json() == {"error": "busy"}
    assert busy.headers["retry-after"] == "10"
    # The same session again is still "already running", not "busy".
    assert client.post("/runs/apps/app/users/ada/sessions/a").status_code == 409
