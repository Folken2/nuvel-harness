"""Checks the token before anything reaches the app behind it.

Plain ASGI rather than Starlette's BaseHTTPMiddleware: that one lets WebSockets past
without looking, and ADK has one (/run_live).

Whose token, what it opens and which bodies are bound to it come from the caller
(agent_gate.py is this service's). All of it is required: a gate knows no route of
its own, and no path is open unless the caller names it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from typing import Generic, TypeVar

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .. import wire_names
from .refusal_log import RefusalLog
from .tokens import InvalidToken

logger = logging.getLogger(__name__)

T = TypeVar("T")

# wire_names.INTERNAL_CALL is set on a call this service makes to itself (internal_caller).
# A scope is built in the process, so no request from outside can carry it.

# A refused path is the caller's text and names people and sessions, so a log gets
# only its first segment, and only when that is a plain word.
_ROUTE_WORD = re.compile(r"[A-Za-z0-9_-]{1,32}")


class _TooLarge(Exception):
    pass


class Refused(Exception):
    """What `bind_body` raises for a body that may not go on: the answer's status and code."""

    def __init__(self, status: int, code: str) -> None:
        super().__init__(code)
        self.status = status
        self.code = code


class TokenGate(Generic[T]):
    """`verify` reads a token with the secrets or raises InvalidToken; `allows` says
    whether that token may make this call (method, path); `carries_body` names the
    paths whose body is read, up to `max_body_bytes`, and handed to `bind_body`, which
    returns the body to pass on or raises Refused. `open_paths` need no token."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        secrets: Sequence[str],
        max_body_bytes: int,
        verify: Callable[[str, Sequence[str]], T],
        allows: Callable[[T, str, str], bool],
        carries_body: Callable[[str], bool],
        bind_body: Callable[[T, bytes], bytes],
        open_paths: frozenset[str] = frozenset(),
    ) -> None:
        self.app = app
        self.secrets = list(secrets)
        self.max_body_bytes = max_body_bytes
        self._verify = verify
        self._allows = allows
        self._carries_body = carries_body
        self._bind_body = bind_body
        self._open_paths = open_paths
        # Only a caller without a valid token is capped: anyone can send those. The rest
        # need a token only the web app signs, so each one is logged.
        self.anonymous = RefusalLog(logger)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            # Nothing here uses the live socket, so it isn't open to anyone.
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http" or scope["path"] in self._open_paths or scope.get(wire_names.INTERNAL_CALL):
            await self.app(scope, receive, send)
            return

        header = Headers(scope=scope).get("authorization", "")
        raw = header[7:] if header.startswith("Bearer ") else ""
        try:
            token = self._verify(raw, self.secrets)
        except InvalidToken as error:
            self.anonymous.refused("refused %s %s: %s", _method(scope), _route(scope), error)
            await _refuse(scope, receive, send, 401, "unauthorized")
            return
        if not self._allows(token, scope["method"], scope["path"]):
            logger.warning("refused %s %s for this token", _method(scope), _route(scope))
            await _refuse(scope, receive, send, 403, "forbidden")
            return
        if not self._carries_body(scope["path"]):
            await self.app(scope, receive, send)
            return

        try:
            received = await _read_body(scope, receive, self.max_body_bytes)
        except _TooLarge:
            logger.warning("refused a run: body over %d bytes", self.max_body_bytes)
            await _refuse(scope, receive, send, 413, "body_too_large")
            return
        if received is None:
            return  # The caller left before sending it all.
        try:
            body = self._bind_body(token, received)
        except Refused as refusal:
            logger.warning("refused a run: %s", refusal.code)
            await _refuse(scope, receive, send, refusal.status, refusal.code)
            return
        await self.app(_with_length(scope, len(body)), _replay(body, receive), send)


def internal_caller(app: ASGIApp) -> ASGIApp:
    """The service as the service calls it: a background task (runs/tasks.py) runs for
    the person whose run started it, which was checked then, and long after their token
    has expired."""

    async def call(scope: Scope, receive: Receive, send: Send) -> None:
        await app({**scope, wire_names.INTERNAL_CALL: True}, receive, send)

    return call


def _route(scope: Scope) -> str:
    first = scope["path"].lstrip("/").split("/", 1)[0]
    return "/" + first if _ROUTE_WORD.fullmatch(first) else "/?"


def _method(scope: Scope) -> str:
    method = scope["method"]
    return method if _ROUTE_WORD.fullmatch(method) else "?"


async def _refuse(scope: Scope, receive: Receive, send: Send, status: int, code: str) -> None:
    await JSONResponse({"error": code}, status_code=status)(scope, receive, send)


async def _read_body(scope: Scope, receive: Receive, limit: int) -> bytes | None:
    """The whole body, None if the caller left, _TooLarge past `limit`: a declared
    length over it is refused before a byte is read, and a body without one is
    refused as soon as it passes it."""
    declared = Headers(scope=scope).get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise _TooLarge
    body = bytearray()
    while True:
        message = await receive()
        if message["type"] != "http.request":
            return None
        body += message.get("body", b"")
        if len(body) > limit:
            raise _TooLarge
        if not message.get("more_body", False):
            return bytes(body)


def _replay(body: bytes, receive: Receive) -> Receive:
    """The body once, then the real channel: that's how a Stop arrives."""
    sent = False

    async def replayed() -> Message:
        nonlocal sent
        if sent:
            return await receive()
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return replayed


def _with_length(scope: Scope, length: int) -> Scope:
    headers = [(name, value) for name, value in scope["headers"] if name != b"content-length"]
    return {**scope, "headers": [*headers, (b"content-length", str(length).encode())]}
