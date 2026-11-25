"""Who a tool call belongs to: its normalised name, source, namespace and sku. Pure.

usage-and-analytics.md section 2, "Tool identity". The person's apps reach the
agent as MCP tools named ``mcp_<server>__<tool>``; a server the product registered
as its connected-app server (``registry.py``) is one of them, so its tools count as
connected apps, not MCP. Everything else the agent defines is the product's own
(``internal``), under the feature the product registered for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..billing.tool_rules import router_web_search_sku
from .registry import DEFAULT_TOOL_FEATURE, REGISTRY, LedgerRegistry

PROVIDER_NATIVE = "provider_native"
CONNECTED_APP = "connected_app"
MCP = "mcp"
INTERNAL = "internal"
UNKNOWN = "unknown"

# A connected-app server's own tools all count under one toolkit (section 2).
TOOL_ROUTER = "tool_router"

_ALIASES = {
    "google_search": "web_search",
    "web_search_preview": "web_search",
    "web": "web_search",
    "code_execution": "code_interpreter",
    "run_javascript": "code_interpreter",
}

# Features of section 2 a tool is by its normalised name, whoever defines it. A product's
# own tools get theirs from the registry; the rest are "tool".
_FEATURES = {
    "web_search": "web_search",
    "code_interpreter": "code_interpreter",
}


@dataclass(frozen=True)
class ToolIdentity:
    name: str
    source: str
    namespace: Optional[str]
    toolkit_slug: Optional[str]
    display_name: str
    billing_sku: str
    feature: str

    @property
    def event_type(self) -> str:
        return f"tool.{self.name}"


def normalise(name: str) -> str:
    """Lower case, no ``server:`` prefix, aliases folded (``google_search`` is ``web_search``)."""
    plain = name.strip().lower()
    if plain.startswith("server:"):
        plain = plain[len("server:"):]
    return _ALIASES.get(plain, plain)


def display_name(name: str) -> str:
    """Words title-cased; ``_`` and ``-`` become spaces."""
    words = name.replace("_", " ").replace("-", " ").split()
    return " ".join(w[:1].upper() + w[1:].lower() for w in words)


def _feature(name: str, registry: LedgerRegistry) -> str:
    return _FEATURES.get(name) or registry.tool_feature(name) or DEFAULT_TOOL_FEATURE


def _mcp(raw: str) -> Optional[tuple[str, str]]:
    """``mcp_<server>__<tool>`` -> (server, tool), as given; None for any other name."""
    if not raw.lower().startswith("mcp_"):
        return None
    server, sep, tool = raw[len("mcp_"):].partition("__")
    if not sep or not server or not tool:
        return None
    return server, tool


def identify(
    raw_name: str, *, namespace: Optional[str] = None, registry: LedgerRegistry = REGISTRY
) -> ToolIdentity:
    """The identity of a tool the agent called. ``namespace`` is where an internal
    tool is defined (its module), for its sku. ``registry`` is what the product registered."""
    mcp = _mcp(raw_name)
    if mcp is not None:
        server, tool = mcp
        server_key, tool_name = server.lower(), normalise(tool)
        if (apps := registry.app_server(server_key)) is not None:
            toolkit = TOOL_ROUTER if tool_name.startswith(apps.meta_prefix) else _toolkit_of(tool_name)
            return ToolIdentity(
                name=tool_name,
                source=CONNECTED_APP,
                namespace=server_key,
                toolkit_slug=toolkit,
                display_name=display_name(tool_name.removeprefix(apps.meta_prefix))
                if toolkit == TOOL_ROUTER
                else display_name(tool_name),
                billing_sku=f"connected_app:{toolkit}:{tool_name}",
                feature=_feature(tool_name, registry),
            )
        return ToolIdentity(
            name=tool_name,
            source=MCP,
            namespace=server_key,
            toolkit_slug=None,
            display_name=f"{display_name(server)}: {display_name(tool)}",
            billing_sku=f"mcp:{server_key}:{tool_name}",
            feature=_feature(tool_name, registry),
        )
    name = normalise(raw_name)
    if not name:
        return ToolIdentity("unknown", UNKNOWN, None, None, "Unknown", "unknown:unknown", "tool")
    space = (namespace or "local").lower()
    return ToolIdentity(
        name=name,
        source=INTERNAL,
        namespace=space,
        toolkit_slug=None,
        display_name=display_name(name),
        billing_sku=f"internal:{space}:{name}",
        feature=_feature(name, registry),
    )


def _toolkit_of(tool: str) -> str:
    """A connected app's own tool is ``<toolkit>_<action>``: its toolkit is the first word."""
    return tool.partition("_")[0] or UNKNOWN


def hosted_web_search(engine: str = "native") -> ToolIdentity:
    """A web search the model provider ran inside a model call (UG-25)."""
    return ToolIdentity(
        name="web_search",
        source=PROVIDER_NATIVE,
        namespace=None,
        toolkit_slug=None,
        display_name="Web Search",
        billing_sku=router_web_search_sku(engine),
        feature="web_search",
    )


def namespace_of(tool: object) -> Optional[str]:
    """Where an internal tool is defined: its function's module, else its class's, last part."""
    func = getattr(tool, "func", None)
    module = getattr(func, "__module__", None) if func is not None else None
    module = module or type(tool).__module__
    if not module:
        return None
    if module.startswith("google.adk"):
        return "adk"
    return module.rsplit(".", 1)[-1]
