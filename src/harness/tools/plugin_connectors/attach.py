"""One plugin connector's tools, or why it has none: the same rules as a person's server.

Its address must be public when this service connects (``user_tools/mcp_address.py``),
and it gets the attach timeout to answer. Whatever goes wrong is one connector's
problem, named, and never raised past here as anything else. Its log lines name the
mount and the kind of error only: a connector's address can carry a key.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.base_tool import BaseTool

from ..user_tools.mcp_address import ServerAddressRefused, check_server_address
from ..user_tools.mcp_limits import McpLimits
from ..user_tools.mcp_toolsets import MAX_TOOL_NAME, TimedMcpToolset, UsableTools
from .mounts import Mount

logger = logging.getLogger(__name__)

ADDRESS_REFUSED = "its address isn't one this service may connect to"
ADDRESS_NOT_FOUND = "its address can't be found"
UNREACHABLE = "it couldn't be reached"
# The server answered 401: each person has to sign in to it, and a plugin's connector
# carries only the plugin's own headers.
SIGN_IN_NEEDED = "it needs a sign-in, which a plugin's connector can't do yet"


def _asks_for_sign_in(error: BaseException) -> bool:
    """Whether a 401 answer is somewhere in what went wrong. The MCP client wraps the
    HTTP error in a task group and then twice in ConnectionError."""
    seen: set[int] = set()
    pending: list[BaseException] = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if getattr(getattr(current, "response", None), "status_code", None) == 401:
            return True
        pending.extend(getattr(current, "exceptions", ()))
        pending.extend(e for e in (current.__cause__, current.__context__) if e is not None)
    return False


def no_answer(limits: McpLimits) -> str:
    """Words for a connector that used the whole attach timeout."""
    return f"it didn't answer within {limits.attach_timeout_seconds:.0f} seconds"


def _why_unusable(left_out: tuple[str, ...]) -> str:
    """Why these tool names can't be used: too long, or characters a provider refuses."""
    long = any(len(name) > MAX_TOOL_NAME for name in left_out)
    odd = any(len(name) <= MAX_TOOL_NAME for name in left_out)
    reasons = [
        *([f"too long (over {MAX_TOOL_NAME} characters with the plugin's name)"] if long else []),
        *(["made of characters a model can't take (only letters, digits, _ and -)"] if odd else []),
    ]
    return " or ".join(reasons)


def all_names_unusable(left_out: tuple[str, ...]) -> str:
    """Words for a connector none of whose tools can be used."""
    return f"its tools' names are {_why_unusable(left_out)}"


def some_names_unusable(left_out: tuple[str, ...]) -> str:
    """Words for a connector that came with some of its tools, not all."""
    return f"{len(left_out)} of its tools have names {_why_unusable(left_out)}"


@dataclass(frozen=True)
class Attached:
    tools: list[BaseTool]
    # The server listed these, but a model provider would refuse their names.
    left_out: tuple[str, ...] = ()


class AttachFailed(Exception):
    def __init__(self, reason: str, *, hung: bool = False, lasting: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        # It used the whole timeout: not worth trying again in the same turn.
        self.hung = hung
        # Trying again won't change it (a sign-in it can't make): worth remembering.
        self.lasting = lasting


async def attach(
    mount: Mount,
    toolset: TimedMcpToolset,
    context: Optional[ReadonlyContext],
    limits: McpLimits,
) -> Attached:
    """The connector's tools through this toolset, or AttachFailed with the reason."""

    async def tools_of() -> UsableTools:
        await check_server_address(
            str(mount.server["url"]), allow_local=limits.allow_local_servers
        )
        return await toolset.usable_tools(context)

    try:
        usable = await asyncio.wait_for(tools_of(), timeout=limits.attach_timeout_seconds)
    except ServerAddressRefused as refused:
        logger.warning("plugin connector refused: %s (%s)", mount.name, refused)
        # A name that doesn't resolve is the connector's fault, not a rule of ours.
        raise AttachFailed(ADDRESS_NOT_FOUND if isinstance(refused.__cause__, OSError) else ADDRESS_REFUSED) from None
    except TimeoutError:
        logger.warning("plugin connector timed out: %s", mount.name)
        raise AttachFailed(no_answer(limits), hung=True) from None
    except Exception as error:  # noqa: BLE001 - one broken connector is only its own problem
        if _asks_for_sign_in(error):
            logger.info("plugin connector needs a sign-in: %s", mount.name)
            raise AttachFailed(SIGN_IN_NEEDED, lasting=True) from None
        # Not the traceback: its frames and messages can hold the address.
        logger.warning("plugin connector unavailable: %s (%s)", mount.name, type(error).__name__)
        raise AttachFailed(UNREACHABLE) from None
    if usable.left_out and not usable.tools:
        raise AttachFailed(all_names_unusable(usable.left_out))
    return Attached(usable.tools, usable.left_out)
