"""Every name of a service's that something outside a module reads: the keys on an
event's metadata, the keys of the frames a run sends, and the mark on a call the service
makes to itself, and the state key that marks a background task's session.

They all sit under one namespace, which is the product's: it calls ``use_namespace`` once,
before anything reads a name (wire_namespace.py). A client reads most of them, so a value
is a contract: change the namespace and every stored event and every client changes with it.

A name is read when it is used (``wire_names.PAUSED``), never copied at import, so a
module here can be imported before the product has said its namespace.
"""

from __future__ import annotations

from typing import Optional

__all__ = ["COST_USD", "RESPONSE_ID", "ROUTED_MODEL", "TASK_MARK", "WEB_SEARCHES", "use_namespace"]

_namespace: Optional[str] = None

_METADATA = "metadata"
_FRAME = "frame"
_SCOPE = "scope"

_NAMES: dict[str, tuple[str, str]] = {
    # On a model call's event. A client reads the first two; the ledger reads the others.
    "COST_USD": (_METADATA, "cost_usd"),
    "ROUTED_MODEL": (_METADATA, "routed_model"),
    "RESPONSE_ID": (_METADATA, "response_id"),
    "WEB_SEARCHES": (_METADATA, "web_searches"),
    "CACHE_WRITE_TOKENS": (_METADATA, "cache_write_tokens"),
    # On the event after a connected app was skipped (plugins/unavailable.py).
    "UNAVAILABLE_TOOLS": (_METADATA, "unavailable_tools"),
    # On the event a run paused at its ceiling ends with (runs/ceiling.py).
    "PAUSED": (_METADATA, "paused"),
    # On the answer written for a call a restart interrupted (runs/resume.py).
    "INTERRUPTED": (_METADATA, "interrupted"),
    # On the person's message: its row's id, and that it repeats one (runs/regenerate.py).
    "MESSAGE_ID": (_METADATA, "message_id"),
    "REGENERATE": (_METADATA, "regenerate"),
    # On a tool call's event: the page its server attached, kept per call id
    # (tools/mcp_apps/, plugins/mcp_app_pages.py).
    "PAGES": (_METADATA, "pages"),
    # On a message that joined a running turn: the client's id for it (runs/steering.py).
    "STEER_ID": (_METADATA, "steer_id"),
    # On a user-role message the agent wrote into the history itself, such as the
    # conversation's file list (files/file_list.py): not the person's words.
    "OWN_MESSAGE": (_METADATA, "own_message"),
    # In a confirmation's payload when the person answered something else (approval/).
    "NOT_DECIDED": (_METADATA, "not_decided"),
    # Frames: how a run ended, a message it took, a background task's progress, and that
    # a call's usage is in the ledger.
    "RUN_FRAME": (_FRAME, "run"),
    "STEER_FRAME": (_FRAME, "steer"),
    "TASK_FRAME": (_FRAME, "task"),
    "USAGE_FRAME": (_FRAME, "usage"),
    # On the ASGI scope of a call the service makes to itself (auth/middleware.py).
    "INTERNAL_CALL": (_SCOPE, "internal_call"),
    # A key of the session's state, set only on a background task's own session
    # (runs/tasks.py) and saved with it, so a resume knows the session is a task.
    "TASK_MARK": (_METADATA, "task"),
}


def use_namespace(namespace: str) -> None:
    """Say the namespace, once. Saying another one is refused: names already read, and
    events already stored, were spelled with the first."""
    global _namespace
    if not namespace:
        raise ValueError("a namespace can't be empty")
    if _namespace is not None and _namespace != namespace:
        raise RuntimeError(f"the wire namespace is already {_namespace!r}")
    _namespace = namespace


def namespace() -> str:
    if _namespace is None:
        raise RuntimeError("no wire namespace: the product calls wire_names.use_namespace first")
    return _namespace


def metadata_key(name: str) -> str:
    """A key on an event's ``custom_metadata``: namespace, underscore, name."""
    return f"{namespace()}_{name}"


def frame_key(name: str) -> str:
    """The one key of a frame a run sends beside ADK's events: namespace, then Name."""
    return f"{namespace()}{name[:1].upper()}{name[1:]}"


def scope_key(name: str) -> str:
    """A key on an ASGI scope: namespace, dot, name."""
    return f"{namespace()}.{name}"


_SPELL = {_METADATA: metadata_key, _FRAME: frame_key, _SCOPE: scope_key}


def all_names() -> dict[str, str]:
    """Each name and its value under the namespace."""
    return {name: _SPELL[kind](short) for name, (kind, short) in _NAMES.items()}


def __getattr__(name: str) -> str:
    if name == "NAMESPACE":
        return namespace()
    if name in _NAMES:
        kind, short = _NAMES[name]
        return _SPELL[kind](short)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
