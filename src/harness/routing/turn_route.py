"""The key this model call uses, from the turn's state to the LiteLLM client.

A context variable, because concurrent turns share this process and LiteLlm
has no per-request field for a key. ``model.apply_model_override`` sets it
before every call, so a call never reads the one before it.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True)
class TurnRoute:
    api_key: str
    # Where the call goes if the maker refuses it: the router's model, with
    # the stored router key, or None for the service's own.
    fallback_model: Optional[str] = None
    fallback_key: Optional[str] = None


_route: contextvars.ContextVar[Optional[TurnRoute]] = contextvars.ContextVar(
    "turn_route", default=None
)


def set_route(route: Optional[TurnRoute]) -> None:
    _route.set(route)


def current_route() -> Optional[TurnRoute]:
    return _route.get()


@contextmanager
def routed(route: Optional[TurnRoute]) -> Iterator[None]:
    """Calls inside go on ``route``; the turn's own route comes back after.

    For a call made from inside a tool, such as the advisor's, which goes to
    another model than the turn's and so needs its own key.
    """
    token = _route.set(route)
    try:
        yield
    finally:
        _route.reset(token)
