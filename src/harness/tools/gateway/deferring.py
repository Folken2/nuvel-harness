"""A toolset whose large servers are held back until the model loads their tools.

Its tools are grouped by server, the part of a name before ``__`` (an MCP mount's tools
are ``mcp_<server>__<tool>``; a tool with no ``__`` is a group of its own). A group whose
declarations pass the threshold in tokens is deferred, except its pinned tools (all of
them when the server itself is pinned): they leave this step's list and go in the
catalog, until the conversation loads them. Counted in tokens, not tools, since one tool
can cost thirty times another.

A loaded tool is the real tool, listed by this toolset as before, so every plugin and
ADK's own lookups (an approved call that resumes) see it as they would. Only where its
declaration goes in the request is the gateway's (``toolset.py``).

ADK keeps a toolset's tools for the whole invocation, so a load would wait for the next
turn. Like ADK's own ``SkillToolset``, whose skills load mid-turn, this one asks to be
asked at every step. It is meant to be the outermost wrapper of what it defers, since an
outer toolset would be kept for the invocation again. What it wraps is still listed once
per run, as before (an MCP server's tools are a network call): only what it returns from
that list is chosen again at each step.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable
from typing import Optional

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models import LlmRequest
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.tool_context import ToolContext

from .declarations import tokens_of
from .deferred import DeferredTools, loaded_names

SERVER_MARK = "__"
# Runs whose listing is kept at once; past this the oldest is listed again if asked.
_RUNS_KEPT = 64


def server_of(name: str) -> str:
    return name.split(SERVER_MARK, 1)[0] if SERVER_MARK in name else name


def deferred_names(
    tools: Iterable[BaseTool],
    *,
    threshold_tokens: int,
    pinned: frozenset[str],
    tokens: Callable[[BaseTool], int] = tokens_of,
) -> set[str]:
    """The tools of servers past the threshold, pinned ones left out.

    A pinned entry is a tool's name, or a server's, which pins all its tools: a server
    that is a gateway of its own, with its own search, stays as it is.
    """
    servers: dict[str, list[BaseTool]] = defaultdict(list)
    for tool in tools:
        servers[server_of(tool.name)].append(tool)
    held: set[str] = set()
    for group in servers.values():
        if sum(tokens(tool) for tool in group) > threshold_tokens:
            held.update(
                tool.name for tool in group if tool.name not in pinned and server_of(tool.name) not in pinned
            )
    return held


class DeferringToolset(BaseToolset):
    def __init__(
        self,
        inner: BaseToolset,
        deferred: DeferredTools,
        *,
        threshold_tokens: int,
        pinned: frozenset[str],
    ) -> None:
        super().__init__()
        self._inner = inner
        self._deferred = deferred
        self._threshold_tokens = threshold_tokens
        self._pinned = pinned
        # ADK's switch for keeping the tools per invocation (see SkillToolset): off.
        self._use_invocation_cache = False
        self._listed: OrderedDict[str, list[BaseTool]] = OrderedDict()

    async def _listing(self, readonly_context: ReadonlyContext) -> list[BaseTool]:
        invocation_id = readonly_context.invocation_id
        if invocation_id in self._listed:
            self._listed.move_to_end(invocation_id)
            return self._listed[invocation_id]
        tools = await self._inner.get_tools(readonly_context)
        self._listed[invocation_id] = tools
        while len(self._listed) > _RUNS_KEPT:
            self._listed.popitem(last=False)
        return tools

    def forget(self, invocation_id: str) -> None:
        """The run ended or paused: a run that goes on lists again."""
        self._listed.pop(invocation_id, None)

    async def get_tools(self, readonly_context: Optional[ReadonlyContext] = None) -> list[BaseTool]:
        if readonly_context is None:
            return await self._inner.get_tools(readonly_context)
        tools = await self._listing(readonly_context)
        held = deferred_names(tools, threshold_tokens=self._threshold_tokens, pinned=self._pinned)
        self._deferred.put(
            readonly_context.invocation_id, id(self), {tool.name: tool for tool in tools if tool.name in held}
        )
        loaded = set(loaded_names(readonly_context.state))
        return [tool for tool in tools if tool.name not in held or tool.name in loaded]

    async def process_llm_request(self, *, tool_context: ToolContext, llm_request: LlmRequest) -> None:
        # What the toolset adds besides its tools (a notice in the instruction) stays.
        await self._inner.process_llm_request(tool_context=tool_context, llm_request=llm_request)

    async def close(self) -> None:
        self._listed.clear()
        await self._inner.close()
