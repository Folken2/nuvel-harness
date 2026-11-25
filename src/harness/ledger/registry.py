"""What a product tells the ledger about its operations and its tools. Pure.

The ledger's code writes a row from a kind and a tool's name; what each one is called
in the row is the product's to say. A product fills ``REGISTRY`` once, at import
(``product_kinds.py``), before any row can be written:

- a **kind** of operation: the event type, operation kind, feature and use case its
  row carries (``entries.base_values``);
- a **tool** of the product's own: the feature its calls count under, and whether it
  calls a model itself (``tool_identity.py``, ``tool_meter.py``);
- an **app server**: an MCP server whose tools are the person's connected apps, not
  MCP tools of their own (``tool_identity.py``);
- its **model ids**: how an id says it went through a router, and whose model it is
  (``entries.served_model``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

# The ledger's own operation kinds: the model meter, the stale sweep and the tool meter
# branch on them. What their rows are called is registered like any other kind's.
MODEL = "model"
COMPACTION = "compaction"
# A call priced per picture returned, not per token.
IMAGE = "image"
# A tool call (tool_meter.py): its event type and feature come from its identity.
TOOL = "tool"

# The feature of a tool nothing was registered for.
DEFAULT_TOOL_FEATURE = "tool"


@dataclass(frozen=True)
class Kind:
    """What a row of one kind of operation carries."""

    event_type: str
    operation_kind: str
    feature: str
    use_case: str


@dataclass(frozen=True)
class AppServer:
    """An MCP server whose tools are connected apps.

    ``meta_prefix`` starts the names of the server's own tools (search, execute), in
    lower case: they count under one toolkit, and an app's tools under the app's.
    """

    meta_prefix: str


def _no_router(model: str) -> Optional[str]:
    return None


def _as_it_is(text: str) -> str:
    return text


@dataclass(frozen=True)
class ModelIds:
    """How a product's model ids read: ``<maker prefix>/<name>``, maybe behind a router's mark.

    With none said, no id goes through a router and a prefix is the maker's own name.
    """

    # The router an id goes through, by the name its rows carry, or None.
    router_of: Callable[[str], Optional[str]] = _no_router
    # The id with the router's mark taken off; any other id as it is.
    without_router: Callable[[str], str] = _as_it_is
    # The maker a prefix stands for, as the catalogue names it.
    maker_of: Callable[[str], str] = _as_it_is
    # The id a call through the router goes out under, for the router's own id of a model.
    through_router: Callable[[str], str] = _as_it_is


class LedgerRegistry:
    def __init__(self) -> None:
        self.model_ids = ModelIds()
        self._kinds: dict[str, Kind] = {}
        self._tool_features: dict[str, str] = {}
        self._model_tools: set[str] = set()
        self._app_servers: dict[str, AppServer] = {}

    def add_kind(self, name: str, kind: Kind) -> None:
        """Registers a kind. Saying it again the same way is fine; another way is a clash."""
        known = self._kinds.get(name)
        if known is not None and known != kind:
            raise ValueError(f"ledger kind {name!r} is already registered as {known}")
        self._kinds[name] = kind

    def kind(self, name: str) -> Kind:
        """The kind's row values. An unregistered kind is a ``KeyError``: a row is never guessed."""
        return self._kinds[name]

    def add_tool(self, name: str, *, feature: Optional[str] = None, calls_model: bool = False) -> None:
        """A tool by its normalised name: its feature, and whether it calls a model itself."""
        if feature is not None:
            self._tool_features[name] = feature
        if calls_model:
            self._model_tools.add(name)

    def tool_feature(self, name: str) -> Optional[str]:
        return self._tool_features.get(name)

    def tool_calls_model(self, name: str) -> bool:
        return name in self._model_tools

    def add_app_server(self, server: str, *, meta_prefix: str) -> None:
        self._app_servers[server.lower()] = AppServer(meta_prefix=meta_prefix.lower())

    def app_server(self, server: str) -> Optional[AppServer]:
        return self._app_servers.get(server.lower())

    def set_model_ids(self, ids: ModelIds) -> None:
        self.model_ids = ids


# The one the ledger writes with.
REGISTRY = LedgerRegistry()
