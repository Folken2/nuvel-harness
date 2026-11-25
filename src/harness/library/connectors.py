"""A plugin's connectors: each MCP server its ``.mcp.json`` names, and whether it can be tried here.

A connector is tried when it's a server at an address with fixed headers. One that
starts a command, has no address, or needs a value from somewhere this service doesn't
read (``${API_TOKEN}``) is kept with the reason: it's listed, never mounted. Nothing
here connects to anything.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .archive import ArchiveRefused

LOCAL_COMMAND = "it starts a command on a computer, which can't run here"
NO_ADDRESS = "it has no address"
BAD_HEADERS = "its headers aren't text"

# ${NAME} or ${NAME:-default}: a value the plugin expects from its host's environment.
_PLACEHOLDER = re.compile(r"\$\{([^}:]+)[^}]*\}")


@dataclass(frozen=True)
class Connector:
    name: str
    # "remote" for a URL the agent could reach, "local" for a command on a machine.
    kind: str
    url: str | None = None
    # "sse" or "http" (streamable HTTP), as the MCP client names them.
    transport: str = "http"
    headers: tuple[tuple[str, str], ...] = ()
    # Why it can't be tried here, in words that finish "the connector isn't available:".
    problem: str | None = None


def read_connectors(servers: Mapping[str, Any]) -> tuple[Connector, ...]:
    found = []
    for name in sorted(servers):
        server = servers[name]
        if not isinstance(server, dict):
            raise ArchiveRefused("bad_manifest", f"Connector {name} must be a JSON object.")
        found.append(_connector(name, server))
    return tuple(found)


def _connector(name: str, server: Mapping[str, Any]) -> Connector:
    if "command" in server:
        return Connector(name, "local", problem=LOCAL_COMMAND)
    url = server.get("url")
    if not isinstance(url, str) or not url.strip():
        return Connector(name, "remote", problem=NO_ADDRESS)
    transport = "sse" if server.get("type") == "sse" else "http"
    headers = server.get("headers") or {}
    if not isinstance(headers, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
    ):
        return Connector(name, "remote", url=url, transport=transport, problem=BAD_HEADERS)
    needed = sorted({m for text in (url, *headers.values()) for m in _PLACEHOLDER.findall(text)})
    return Connector(
        name,
        "remote",
        url=url,
        transport=transport,
        headers=tuple(sorted(headers.items())),
        problem=_needs(needed) if needed else None,
    )


def _needs(names: list[str]) -> str:
    return f"it needs {', '.join(names)}, which isn't set up here"
