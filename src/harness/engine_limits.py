"""The time limits a database engine connects with, as asyncpg's ``connect_args``.

For SQLAlchemy: ``server_settings`` (startup parameters: the server's statement
and idle-in-transaction limits) and ``command_timeout`` (the client's own limit,
which cancels the query on the server when it fires).

A transaction pooler (PgBouncer in transaction mode) refuses any startup
parameter it can't track ("unsupported startup parameter") and so would refuse
every connection. There the server settings are left out and the role's own
settings apply (``ALTER ROLE ... SET``); the client limit holds on every host.

Nothing here knows which hosts are poolers or where the limits are set: the
caller passes the check and the values. ``db_limits.py`` is this service's.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["ServerLimits", "connect_args"]


@dataclass(frozen=True)
class ServerLimits:
    """What the server enforces, in seconds. 0: none is set."""

    statement_timeout_seconds: float = 0
    idle_in_transaction_timeout_seconds: float = 0


def _ms(seconds: float) -> str:
    return str(round(seconds * 1000))


def connect_args(
    url: str,
    *,
    command_timeout_seconds: float,
    server_limits: Callable[[], ServerLimits],
    is_transaction_pooler: Callable[[str], bool],
) -> dict[str, Any]:
    """The limits for one engine's URL. A limit of 0 is left out.

    ``server_limits`` is asked only when they are sent, so not for a pooler's URL.
    """
    args: dict[str, Any] = {}
    if command_timeout_seconds > 0:
        args["command_timeout"] = command_timeout_seconds
    if is_transaction_pooler(url):
        return args
    limits = server_limits()
    settings: dict[str, str] = {}
    if limits.statement_timeout_seconds > 0:
        settings["statement_timeout"] = _ms(limits.statement_timeout_seconds)
    if limits.idle_in_transaction_timeout_seconds > 0:
        settings["idle_in_transaction_session_timeout"] = _ms(
            limits.idle_in_transaction_timeout_seconds
        )
    if settings:
        args["server_settings"] = settings
    return args
