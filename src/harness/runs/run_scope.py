"""Which run a piece of code is part of, for the tools that act on runs (tools/tasks.py).

Set by the routes around a run they drive, like the inbox (./steering.py): the tasks
ADK starts for the run copy the context, so a tool call finds its own run here.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .registry import Run, RunKey

_scope: contextvars.ContextVar[RunScope | None] = contextvars.ContextVar(
    "run_scope", default=None
)


class ScopedTask(Protocol):
    """A background task, as the tool that started or asked about it reads it."""

    id: str
    run: Run

    @property
    def status(self) -> str: ...

    def view(self) -> dict[str, object]: ...


class ScopedTasks(Protocol):
    """What a run's tools do with background tasks. ./tasks.py has the board that does it."""

    async def start(
        self, parent: RunKey, description: str, prompt: str, state: Mapping[str, object]
    ) -> ScopedTask: ...

    def get(self, parent: RunKey, task_id: str) -> ScopedTask | None: ...

    async def wait(self, tasks: list[Any], timeout: float) -> None: ...


@dataclass(frozen=True)
class RunScope:
    key: RunKey
    tasks: ScopedTasks


def current_run_scope() -> RunScope | None:
    """The run this code is part of; None outside a run the routes drive."""
    return _scope.get()


def set_run_scope(scope: RunScope) -> contextvars.Token[RunScope | None]:
    return _scope.set(scope)


def reset_run_scope(token: contextvars.Token[RunScope | None]) -> None:
    _scope.reset(token)
