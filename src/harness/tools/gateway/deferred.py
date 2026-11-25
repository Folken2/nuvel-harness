"""Which tools are deferred on a run's latest step, and which the conversation has loaded.

The deferred tools are this process's memory, written by each deferring toolset when ADK
asks it for its tools and read later in the same step (the catalog, ``find_tools``, the
answer to a call before loading). ADK asks every toolset first, so a step's are all in by
then.

The loaded set is session state, not ``temp:``: it lasts the conversation, a new turn and
a resumed run (``runs/resume.py``) read it back, and its order is the order of loading.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

from google.adk.tools.base_tool import BaseTool

LOADED = "agent_loaded_tools"
# Runs remembered at once; past this the oldest is dropped (each is forgotten when it ends).
_RUNS_KEPT = 64


def loaded_names(state: Any) -> list[str]:
    names = state.get(LOADED) if state is not None else None
    return [name for name in names if isinstance(name, str)] if isinstance(names, list) else []


def with_loaded(state: Any, names: Iterable[str]) -> list[str]:
    """The loaded set with these added at its end, written back."""
    merged = list(dict.fromkeys([*loaded_names(state), *names]))
    state[LOADED] = merged
    return merged


class DeferredTools:
    """Per run, the tools each deferring toolset held back on the latest step."""

    def __init__(self) -> None:
        self._runs: OrderedDict[str, dict[int, dict[str, BaseTool]]] = OrderedDict()

    def put(self, invocation_id: str, source: int, tools: dict[str, BaseTool]) -> None:
        sources = self._runs.setdefault(invocation_id, {})
        sources[source] = tools
        self._runs.move_to_end(invocation_id)
        while len(self._runs) > _RUNS_KEPT:
            self._runs.popitem(last=False)

    def of(self, invocation_id: str) -> dict[str, BaseTool]:
        merged: dict[str, BaseTool] = {}
        for tools in self._runs.get(invocation_id, {}).values():
            merged.update(tools)
        return merged

    def forget(self, invocation_id: str) -> None:
        self._runs.pop(invocation_id, None)
