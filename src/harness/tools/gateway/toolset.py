"""The gateway's own place in the request: the catalog, ``find_tools`` and loaded tools last.

It goes last in the agent's tools. ADK resolves every toolset's tools, then lets each add
to the request in the agent's order, so by the time this one does, the step's deferred
tools are known and every other declaration is in. On a step with none it adds nothing,
so a turn without large servers sees no change.

Declarations end in a stable order: everything else as before, then ``find_tools``, then
the loaded tools in the order they were loaded. A load so changes the tools section of
the prompt once, at its end, and the prefix a provider caches settles again right after.
"""

from __future__ import annotations

from typing import Any, Optional

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models import LlmRequest
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from ..described import DescribedTool
from .catalog import catalog_text
from .declarations import schema_of
from .deferred import DeferredTools, loaded_names, with_loaded
from .search import is_select, rank, select

# Tools a keyword query loads at most.
MATCHES_LIMIT = 5


def move_to_end(llm_request: LlmRequest, names: list[str]) -> None:
    """These declarations taken from wherever they are and put last, in this order."""
    wanted = set(names)
    taken: dict[str, types.FunctionDeclaration] = {}
    holder: Optional[types.Tool] = None
    for declared in llm_request.config.tools or []:
        if not isinstance(declared, types.Tool) or declared.function_declarations is None:
            continue
        holder = holder or declared
        kept = []
        for declaration in declared.function_declarations:
            if declaration.name in wanted:
                taken[declaration.name] = declaration
            else:
                kept.append(declaration)
        declared.function_declarations = kept
    if holder is None or holder.function_declarations is None:
        return
    holder.function_declarations.extend(taken[name] for name in names if name in taken)


class ToolGatewayToolset(BaseToolset):
    def __init__(self, deferred: DeferredTools) -> None:
        super().__init__()
        self._deferred = deferred
        self._find_tools = DescribedTool(self.find_tools)

    async def get_tools(self, readonly_context: Optional[ReadonlyContext] = None) -> list[BaseTool]:
        # Declared in process_llm_request, only on a step with something to load.
        return []

    async def process_llm_request(self, *, tool_context: ToolContext, llm_request: LlmRequest) -> None:
        deferred = self._deferred.of(tool_context.invocation_id)
        if not deferred:
            return
        llm_request.append_instructions(
            [catalog_text({name: tool.description or "" for name, tool in deferred.items()})]
        )
        await self._find_tools.process_llm_request(tool_context=tool_context, llm_request=llm_request)
        move_to_end(llm_request, [name for name in loaded_names(tool_context.state) if name in deferred])

    async def find_tools(self, query: str, tool_context: ToolContext) -> dict[str, Any]:
        """Loads tools from the "Tools to load" list in your instruction, so you can call them.

        Use it before the first call to any tool on that list: a call to one that isn't
        loaded gets no result. A loaded tool can be called from your next step on and
        stays loaded for the rest of the conversation. Returns each loaded tool's name,
        description and parameters; names that match nothing are listed under not_found,
        and a search that matches nothing says so.

        Args:
            query: "select:" and exact names from the list, comma separated, such as "select:mcp_notes__add_page,mcp_notes__search". Or keywords to search the list by name and description, such as "create calendar event", with "+" before a word every match must have ("+calendar event").
        """
        deferred = self._deferred.of(tool_context.invocation_id)
        catalog = {name: tool.description or "" for name, tool in deferred.items()}
        if is_select(query):
            selected = select(query, catalog)
            names, not_found = list(selected.found), list(selected.not_found)
        else:
            names, not_found = list(rank(query, catalog, MATCHES_LIMIT)), []
        if names:
            with_loaded(tool_context.state, names)
        result: dict[str, Any] = {"loaded": [schema_of(deferred[name]) for name in names]}
        if not_found:
            result["not_found"] = not_found
        if not names and not is_select(query):
            result["matched_nothing"] = f"No tool on the list matches {query!r}. Try other words, or select one by name."
        return result
