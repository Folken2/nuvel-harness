"""What differs between one MCP mount and another, set by the code that builds its descriptor.

A descriptor carries its rules under ``RULES_KEY`` as a ``MountRules``. One that came as
JSON (a person's server from the web app, a plugin's connector) can't hold one, so it
always gets the defaults: the hardened client, the whole mount refused, its own name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

RULES_KEY = "rules"


@dataclass(frozen=True)
class MountRules:
    # Its builder holds the address to a host list of its own, so the client isn't the
    # one that refuses redirects and anything but a public server.
    own_host_check: bool = False
    # One mount carries many apps: a call that gets no answer refuses that tool for the
    # turn, not everything on the mount.
    many_apps: bool = False
    # What the person reads in place of the mount's name. None: its name.
    shown_as: str | None = None


DEFAULT_RULES = MountRules()


def rules_of(server: dict[str, Any]) -> MountRules:
    rules = server.get(RULES_KEY)
    return rules if isinstance(rules, MountRules) else DEFAULT_RULES
