"""A tool call as the actions it takes, each named by the key a person can always allow.

Most calls are one action: the tool itself. A router is the exception: one mount whose
tools are named bare, and which may run other tools through one of its own. Its call is
then one action per tool it runs, so reading an inbox and sending an email through the
same tool are told apart by what's inside. Which routers there are, and how each one's
call is unpacked, is handed in (``Router``); this module knows none by name.

Keys carry where the tool comes from, so no server can pass for another: the agent's
own tools have no prefix, a router's tools and what it runs are bare (its mount name is
reserved by whoever mounts it), and any other server's tool keeps its full
`mcp_<slug>__<tool>` name.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

# How the agent names the tools of an MCP server: mcp_<slug>__<tool>.
MCP_PREFIX = "mcp_"
MCP_SEPARATOR = "__"

# "agent": the agent's own tool. "server": an MCP server's tool. Any other: a router's
# name (``Router.source``), whose tools and what they run get their hints from it.
Source = str
AGENT: Source = "agent"
SERVER: Source = "server"


@dataclass(frozen=True)
class Action:
    # What a person allows for good, and what the policy reads.
    key: str
    args: dict[str, Any] = field(default_factory=dict)
    source: Source = AGENT
    # False when the call's shape couldn't be read, so what it runs is unknown and
    # it asks (policy.py): a check that can't see an action never lets it run.
    readable: bool = True


@dataclass(frozen=True)
class Router:
    """A mount whose tools are keyed bare, under its own source."""

    source: Source
    # Its mount's slug, as in the names of its tools: mcp_<slug>__<tool>.
    slug: str
    # (the tool's bare key, the call's arguments) -> the actions the call takes, when it
    # runs other tools; None when the call is one action, the tool itself.
    unpack: Callable[[str, dict[str, Any]], Optional[list[Action]]] = lambda key, args: None

    @property
    def prefix(self) -> str:
        return f"{MCP_PREFIX}{self.slug}{MCP_SEPARATOR}"


def own_name(action: Action) -> str:
    """The tool's own name, what its words are read from: a server's slug says nothing
    about what the tool does, and mustn't put a read verb in front of a send."""
    if action.source == SERVER:
        return action.key.split(MCP_SEPARATOR, 1)[1]
    return action.key


Unpack = Callable[[str, dict[str, Any]], Optional[list[Action]]]


class ActionReader:
    """Reads a tool call as its actions, for these routers.

    ``own_unpack`` does for the agent's own tools what a router's ``unpack`` does for its
    mount's: given a tool's name and arguments, the actions the call takes, when one call
    does more than its name says (a write that also deletes); None when it is one action.
    """

    def __init__(self, routers: Sequence[Router] = (), own_unpack: Unpack = lambda key, args: None) -> None:
        self._routers = tuple(routers)
        self._own_unpack = own_unpack

    def _router_of(self, tool_name: str) -> Optional[Router]:
        return next((r for r in self._routers if tool_name.startswith(r.prefix)), None)

    def source_of(self, tool_name: str) -> Source:
        router = self._router_of(tool_name)
        if router is not None:
            return router.source
        if tool_name.startswith(MCP_PREFIX) and MCP_SEPARATOR in tool_name:
            return SERVER
        return AGENT

    def key_of(self, tool_name: str) -> str:
        """The tool's key: bare for a router's, the full name for any other server's."""
        router = self._router_of(tool_name)
        return tool_name[len(router.prefix) :] if router is not None else tool_name

    def actions_of(self, tool_name: str, args: dict[str, Any]) -> list[Action]:
        router = self._router_of(tool_name)
        if router is None:
            source = self.source_of(tool_name)
            unpacked = self._own_unpack(tool_name, args) if source == AGENT else None
            return unpacked if unpacked is not None else [Action(key=tool_name, args=args, source=source)]
        key = self.key_of(tool_name)
        unpacked = router.unpack(key, args)
        if unpacked is None:
            return [Action(key=key, args=args, source=router.source)]
        return unpacked
