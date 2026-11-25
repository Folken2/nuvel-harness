"""MCP Apps (the ``io.modelcontextprotocol/ui`` extension): generic host support.

A server may attach an interactive page to a tool. Design and parts:
docs/mcp-apps.md. Nothing here names a product, a vendor or a setting.
"""

from .limits import PageLimits

__all__ = [
    "PageLimits",
]
