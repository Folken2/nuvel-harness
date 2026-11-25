"""The connected apps a turn had to skip, waiting to be reported."""

from __future__ import annotations


class UnavailableTools:
    """Skipped apps per invocation. Each name is reported once per invocation.

    An app that hung is also remembered for the invocation, so it costs its
    timeout once rather than on every model step. An app that failed quickly
    isn't: trying it again is cheap, and it may be back. An app that left a call
    unanswered keeps its tools but is refused for the invocation.
    """

    def __init__(self) -> None:
        self._pending: dict[str, list[str]] = {}
        self._reported: dict[str, set[str]] = {}
        self._hung: dict[str, set[str]] = {}
        self._unanswered: dict[str, set[str]] = {}

    def add(self, invocation_id: str, name: str) -> None:
        if name in self._reported.get(invocation_id, set()):
            return
        pending = self._pending.setdefault(invocation_id, [])
        if name not in pending:
            pending.append(name)

    def add_hung(self, invocation_id: str, name: str) -> None:
        self._hung.setdefault(invocation_id, set()).add(name)
        self.add(invocation_id, name)

    def is_hung(self, invocation_id: str, name: str) -> bool:
        return name in self._hung.get(invocation_id, set())

    def add_unanswered(self, invocation_id: str, key: str, name: str) -> None:
        """`key` is what's refused from now on, `name` what the notice calls it."""
        self._unanswered.setdefault(invocation_id, set()).add(key)
        self.add(invocation_id, name)

    def left_unanswered(self, invocation_id: str, key: str) -> bool:
        return key in self._unanswered.get(invocation_id, set())

    def take(self, invocation_id: str) -> list[str]:
        names = self._pending.pop(invocation_id, [])
        self._reported.setdefault(invocation_id, set()).update(names)
        return names

    def forget(self, invocation_id: str) -> None:
        self._pending.pop(invocation_id, None)
        self._reported.pop(invocation_id, None)
        self._hung.pop(invocation_id, None)
        self._unanswered.pop(invocation_id, None)
