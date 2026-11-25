"""Prompt caching for Claude (and any model that caches a marked prefix).

With a ContextCacheConfig on the App, ADK's LiteLLM model marks two cache
breakpoints per request: the system instruction (which also covers the tool
definitions, since they come first in the prompt) and the last message (the
conversation so far). Later calls in the same turn and the next turn read
that prefix from the provider's cache: faster first token, and cached input
is billed at a fraction of the price. Models that don't cache by marked
prefix have the markers dropped by LiteLLM.

For hits, the prefix must stay byte-identical: the instruction only holds
what is stable for a user over a day, and tools keep a fixed order.

Whether the cache is on and its two numbers are the product's settings: this module
is given them (``PromptCacheSettings``) and reads none. This service's are in
``context_settings.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from google.adk.agents.context_cache_config import ContextCacheConfig

__all__ = ["prompt_cache"]


@dataclass(frozen=True)
class PromptCacheSettings:
    """A prefix under ``min_tokens`` is not marked; a marked one is kept ``ttl_seconds``."""

    min_tokens: int
    ttl_seconds: int


def prompt_cache(settings: PromptCacheSettings) -> ContextCacheConfig:
    return ContextCacheConfig(min_tokens=settings.min_tokens, ttl_seconds=settings.ttl_seconds)
