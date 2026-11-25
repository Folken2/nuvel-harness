"""What a model call does after its request failed: one table from the kind of error.

Pure, so the client (``routing/client.py``) only carries the steps out. Drawn from
OpenHands (``llm/utils/retry_mixin.py``, ``llm/fallback_strategy.py``), codex
(``protocol/src/error.rs``) and goose (``goose-providers/src/http_status.rs``):

- a limit that lifts by itself (a rate limit, an overloaded or failing server, a
  dropped connection) is sent again after a wait, the server's own when it names one,
  as long as the wait fits what the call and its run may still spend; once the
  retries are spent it falls back to the router, where there is a fallback;
- a quota or a bill (out of credit, a day's allowance spent) is never sent again on
  the same route, since waiting doesn't lift it; another account may, so it falls back;
- a refused key or an unknown model id falls back, as before;
- a request too long for the model is shrunk once (``context/overflow.py``).

Each request the client sends is one ledger attempt, so a retry is never billed twice
nor lost. The client only retries a request that raised, never a stream it already
returned: once output started, the model's words are the person's to see.
"""

from __future__ import annotations

import enum
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Iterator, Optional, Union

import httpx
import litellm
from litellm.exceptions import MidStreamFallbackError

from ..context.overflow import is_context_overflow

__all__ = ["Settings"]


class Kind(enum.Enum):
    OVERFLOW = "overflow"
    QUOTA = "quota"
    REFUSED = "refused"
    RATE_LIMITED = "rate_limited"
    TRANSIENT = "transient"
    OTHER = "other"


@dataclass(frozen=True)
class Rule:
    retry: bool = False
    fall_back: bool = False
    shrink: bool = False


RULES: dict[Kind, Rule] = {
    Kind.OVERFLOW: Rule(shrink=True),
    Kind.QUOTA: Rule(fall_back=True),
    Kind.REFUSED: Rule(fall_back=True),
    Kind.RATE_LIMITED: Rule(retry=True, fall_back=True),
    Kind.TRANSIENT: Rule(retry=True, fall_back=True),
    Kind.OTHER: Rule(),
}


@dataclass(frozen=True)
class Retry:
    delay: float


@dataclass(frozen=True)
class Shrink:
    pass


@dataclass(frozen=True)
class FallBack:
    pass


@dataclass(frozen=True)
class GiveUp:
    pass


Step = Union[Retry, Shrink, FallBack, GiveUp]


@dataclass(frozen=True)
class Settings:
    retries: int
    first_wait: float
    max_wait: float


# A 429 is a rate limit unless it is a quota that waiting doesn't lift. The window it
# names decides first: Gemini's per-minute limit says "You exceeded your current quota
# ... billing details" and its per-day one says the same, told apart only by the quota
# id (``...PerMinute...``, ``...PerDay...``), and both carry a RetryInfo delay.
_DAY_WINDOW = re.compile(r"per[ _-]?day", re.IGNORECASE)
_SHORT_WINDOW = re.compile(r"per[ _-]?(?:second|minute)|retryDelay", re.IGNORECASE)
# A quota or a bill in the words the providers we route to use: a router ("Insufficient
# credits", "requires more credits"), OpenAI ("insufficient_quota",
# "billing_hard_limit_reached"), Gemini and OpenAI ("exceeded your current quota"),
# Anthropic ("credit balance is too low", "reached your specified API usage limits") and
# xAI ("used all available credits", "spending limit"). Whole phrases, never a bare word:
# a 400 about a thinking ``budget_tokens`` setting is no quota.
_QUOTA = re.compile(
    r"insufficient[_ ]quota|exceeded your current quota|insufficient credits"
    r"|requires more credits|credit balance is too low|billing[_ ]hard[_ ]limit"
    r"|api usage limits|used all (?:available )?credits|spend(?:ing)? limit",
    re.IGNORECASE,
)
# Statuses that say "not now": a server's own timeout (408), a conflict on its side
# (409, which the OpenAI SDK retried before max_retries=0), and the overloaded ones.
_TRANSIENT_STATUSES = frozenset({408, 409, 500, 502, 503, 504, 529})
_SERVER_ERRORS = (
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
    litellm.BadGatewayError,
)
_REFUSED_ERRORS = (
    litellm.AuthenticationError,
    litellm.PermissionDeniedError,
    litellm.NotFoundError,
)


def _status(error: BaseException) -> Optional[int]:
    status = getattr(error, "status_code", None)
    return status if isinstance(status, int) else None


def _causes(error: BaseException) -> Iterator[BaseException]:
    """The error and what it wraps: LiteLLM keeps the transport's error as the cause."""
    seen: set[int] = set()
    current: Optional[BaseException] = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = (
            current.__cause__
            or current.__context__
            or getattr(current, "original_exception", None)
        )


def _the_server_answered(error: BaseException) -> bool:
    return any(isinstance(cause, httpx.HTTPStatusError) for cause in _causes(error))


def _the_connection_failed(error: BaseException) -> bool:
    return any(
        isinstance(cause, (httpx.TransportError, ConnectionError)) for cause in _causes(error)
    )


def _rate_kind(text: str) -> Kind:
    if _DAY_WINDOW.search(text):
        return Kind.QUOTA
    if _SHORT_WINDOW.search(text) or not _QUOTA.search(text):
        return Kind.RATE_LIMITED
    return Kind.QUOTA


def classify(error: BaseException) -> Kind:
    # How LiteLLM raises a lazy stream's refusal on its first chunk (the direct Gemini
    # route): a "server unavailable" around the real error, which is what counts.
    if isinstance(error, MidStreamFallbackError) and error.original_exception is not None:
        return classify(error.original_exception)
    text, status = str(error), _status(error)
    # LiteLLM's label for a timeout of ours and for a server's 408 alike: ours already
    # waited the whole limit, and another would double it.
    if isinstance(error, litellm.Timeout):
        return Kind.TRANSIENT if _the_server_answered(error) else Kind.OTHER
    # Also LiteLLM's label for an error it couldn't map, a bug that a retry repeats.
    if isinstance(error, litellm.APIConnectionError):
        return Kind.TRANSIENT if _the_connection_failed(error) else Kind.OTHER
    # The status first: a 429 that says "too many tokens" is a rate limit, not a long prompt.
    if isinstance(error, litellm.RateLimitError) or status == 429:
        return _rate_kind(text)
    if isinstance(error, litellm.ContextWindowExceededError) or is_context_overflow(text):
        return Kind.OVERFLOW
    # A failing server before the wording: its body may mention billing and still lift.
    if isinstance(error, _SERVER_ERRORS) or status in _TRANSIENT_STATUSES:
        return Kind.TRANSIENT
    if status == 402 or _QUOTA.search(text):
        return Kind.QUOTA
    if isinstance(error, _REFUSED_ERRORS):
        return Kind.REFUSED
    return Kind.OTHER


# The delay in the error's body, more precise than the header: a router's own field,
# and Gemini's RetryInfo ("17s").
_BODY_DELAY = re.compile(
    r'"retry_after_seconds"\s*:\s*([0-9]+(?:\.[0-9]+)?)|"retryDelay"\s*:\s*"([0-9]+(?:\.[0-9]+)?)s"'
)


def _header_delay(name: str, value: Any, now: float) -> Optional[float]:
    if not isinstance(value, str):
        return None
    try:
        seconds = float(value) / (1000 if name == "retry-after-ms" else 1)
    except ValueError:
        if name == "retry-after-ms":
            return None
        try:
            seconds = parsedate_to_datetime(value).timestamp() - now
        except (TypeError, ValueError, OverflowError):
            return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def server_delay(error: BaseException, now: Optional[float] = None) -> Optional[float]:
    """The wait the server asked for, in seconds, or None when it named none."""
    if (match := _BODY_DELAY.search(str(error))) is not None:
        return float(match.group(1) or match.group(2))
    headers = getattr(error, "litellm_response_headers", None)
    if not isinstance(headers, Mapping):
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
    if not isinstance(headers, Mapping):
        return None
    now = time.time() if now is None else now
    delays = [
        delay
        for raw_name, value in headers.items()
        if (name := str(raw_name).lower().removeprefix("llm_provider-"))
        in ("retry-after", "retry-after-ms")
        and (delay := _header_delay(name, value, now)) is not None
    ]
    return max(delays) if delays else None


def backoff(retries_done: int, settings: Settings, jitter: float) -> float:
    """The wait before the next try: doubling from the first, spread by ``jitter`` (0 to 1)
    so calls refused together don't all come back at once."""
    wait = min(settings.first_wait * (2**retries_done), settings.max_wait)
    return wait * (0.8 + 0.4 * jitter)


def next_step(
    kind: Kind,
    *,
    retries_done: int,
    shrunk: bool,
    has_fallback: bool,
    delay: Optional[float],
    settings: Settings,
    jitter: float,
    wait_left: float,
) -> Step:
    """``retries_done`` counts the retries on this route; ``delay`` is the server's;
    ``wait_left`` is what the call and its run may still spend waiting
    (``runs/retry_window.py``)."""
    rule = RULES[kind]
    if rule.shrink:
        return GiveUp() if shrunk else Shrink()
    if rule.retry and retries_done < settings.retries:
        wait = backoff(retries_done, settings, jitter) if delay is None else delay
        if wait <= min(settings.max_wait, wait_left):
            return Retry(wait)
    if rule.fall_back and has_fallback:
        return FallBack()
    return GiveUp()
