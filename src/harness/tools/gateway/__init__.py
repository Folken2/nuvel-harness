"""The tool gateway: large toolsets listed in a catalog and loaded on demand.

``gateway.py`` puts the pieces together; each other module has one job.
"""

from .gateway import ToolGateway

__all__ = [
    "ToolGateway",
]
