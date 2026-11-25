"""Whether a provider refused a request because it is too long for the model's context.

LiteLLM names this ``ContextWindowExceededError`` only for some makers' wording; through
a router the same refusal arrives as a plain bad request whose text says it (checked
against the scripted model, ``tests/test_model_call_replay.py``). So the text decides,
by the wordings the makers' APIs and the routers in front of them use. The list was
drawn from two open-source harnesses' lists (pi ``packages/ai/src/utils/overflow.ts``,
opencode ``packages/ai/src/provider-error.ts``), cut to what can reach this service.

Some throttling messages also say "too many tokens" ("too many tokens, please wait"),
so a few wordings are never overflow, whatever else the text says: a rate limit is
retried, not shrunk (``routing/retry_policy.py``).
"""

from __future__ import annotations

import re

_OVERFLOW = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"prompt is too long",  # Anthropic
        r"request_too_large",  # Anthropic, by size (413)
        r"exceeds the context window",  # OpenAI
        r"exceeds (?:the )?(?:model'?s )?maximum context length",  # OpenAI-compatible
        r"maximum context length is \d+ tokens",  # OpenAI, and most endpoints behind a router
        r"exceeds (?:the )?maximum allowed input length",  # some endpoints behind a router
        r"context[_ ]length[_ ]exceeded",  # OpenAI's code
        r"context window exceeded",  # LiteLLM's own check
        r"input token count.*exceeds the maximum",  # Gemini
        r"maximum prompt length is \d+",  # xAI
        r"exceeded model token limit",  # Moonshot
        r"input(?: \(\d+ tokens\))? is longer than the model'?s context length",
        r"reduce the length of the messages",
        r"too large for model with \d+ maximum context length",  # Mistral
        r"input length.*exceeds.*context length",
        r"token limit exceeded",
        r"too many tokens",
    )
)

_NEVER_OVERFLOW = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"rate[ _-]?limit",
        r"too many requests",
        r"too many tokens(?: per (?:minute|day))?, please wait",
        # An output limit above the model's: a shorter prompt doesn't fix it.
        r"max[_ ]tokens must be less than",
    )
)


def is_context_overflow(message: str) -> bool:
    """Whether an error's text says the request didn't fit the model's context."""
    if any(pattern.search(message) for pattern in _NEVER_OVERFLOW):
        return False
    return any(pattern.search(message) for pattern in _OVERFLOW)
