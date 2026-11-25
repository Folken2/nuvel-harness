"""The limits a page runs under (MCP Apps). Handed in by the product, never read here."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageLimits:
    # Past this, fetching a call's page gives up and the call shows without one.
    fetch_timeout_seconds: float = 8.0
    # The largest page kept, in bytes of HTML.
    max_page_bytes: int = 2_000_000
    # Origins a page may declare per CSP list; more are dropped.
    max_domains: int = 20
