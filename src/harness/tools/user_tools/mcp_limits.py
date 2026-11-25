"""The limits the MCP plumbing runs under, handed to it by whoever builds a toolset.

Nothing here reads a setting: ``limits_from_env.py`` fills one in from this service's.
"""

from __future__ import annotations

from dataclasses import dataclass

# ADK's own waits are 5 seconds to connect and list, and 5 minutes on a request or a
# quiet stream. Each is set this far past the limit it backs, so the limit decides and a
# hang stays distinct from a quick failure.
ADK_WAITS_PAST_SECONDS = 5


@dataclass(frozen=True)
class McpLimits:
    # A server that hangs must not hold the turn: past this, it's skipped.
    attach_timeout_seconds: float
    # Past this, a call gets a failed result and the turn goes on.
    call_timeout_seconds: float
    # Reuse each server's tools/list result instead of re-listing every turn (saves a
    # network round trip per server per turn).
    tool_list_ttl_seconds: float
    # Toolsets kept open. Past it the least recently used idle one is closed.
    cache_limit: int
    # A server may be on this machine or network. Development only.
    allow_local_servers: bool = False
    # Tell servers this host can show MCP Apps pages (tools/mcp_apps/advertise.py).
    advertise_apps: bool = False

    @property
    def connect_timeout_seconds(self) -> float:
        return self.attach_timeout_seconds + ADK_WAITS_PAST_SECONDS

    @property
    def read_timeout_seconds(self) -> float:
        return self.call_timeout_seconds + ADK_WAITS_PAST_SECONDS

    @property
    def close_timeout_seconds(self) -> float:
        """How long a server gets to hang up."""
        return self.attach_timeout_seconds
