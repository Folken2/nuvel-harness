"""How many connections each engine over the session database may open."""

from __future__ import annotations

import pytest

from harness import service

# Nothing here runs a query, so no connection is opened.
LOCAL = "postgresql://agent@localhost/agent"
REMOTE = "postgresql://agent@db.example.com/agent"


def test_left_unset_keeps_sqlalchemys_pool():
    kwargs = service.session_db_kwargs(LOCAL, {})
    assert "pool_size" not in kwargs and "max_overflow" not in kwargs


def test_a_cap_pools_that_many_and_opens_none_beyond():
    kwargs = service.session_db_kwargs(REMOTE, {"command_timeout": 5}, max_connections=8)
    assert kwargs["pool_size"] == 8
    assert kwargs["max_overflow"] == 0
    assert kwargs["connect_args"] == {"command_timeout": 5, "ssl": "require"}


def test_a_cap_below_one_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        service.session_db_kwargs(LOCAL, {}, max_connections=0)


def test_every_engine_the_stores_build_gets_the_cap(monkeypatch):
    built: list[dict] = []

    def capture(uri, **kwargs):
        built.append(kwargs)
        return object()

    # The builders the stores call; the harness's tests install no Postgres driver.
    monkeypatch.setattr(service, "DatabaseSessionService", capture)
    monkeypatch.setattr(service, "create_async_engine", capture)
    stores = service.Stores(session_uri=LOCAL, connect_args=lambda _uri: {}, max_connections=3)

    stores.run_sessions()
    stores.eraser(lambda files: files)
    assert stores.db_kwargs()["pool_size"] == 3  # what create_service hands ADK's own
    assert [(each["pool_size"], each["max_overflow"]) for each in built] == [(3, 0), (3, 0)]
