"""The table from a failed model request to the client's next step (routing/retry_policy.py)."""

from __future__ import annotations

import httpx
import litellm
import pytest
from litellm.exceptions import MidStreamFallbackError

from harness.runs.retry_window import RetryWindow, wait_left
from harness.routing.retry_policy import (
    FallBack,
    GiveUp,
    Kind,
    Retry,
    Settings,
    Shrink,
    backoff,
    classify,
    next_step,
    server_delay,
)

SETTINGS = Settings(retries=2, first_wait=1.0, max_wait=20.0)


def _error(cls, message="no", status=None, headers=None, cause=None):
    kwargs = {"message": message, "llm_provider": "openai", "model": "m"}
    if cls is litellm.APIError:
        kwargs["status_code"] = status
    error = cls(**kwargs)
    if headers is not None:
        error.litellm_response_headers = headers
    if cause is not None:
        error.__cause__ = cause
    return error


_REQUEST = httpx.Request("POST", "https://x")


def _answered(status):
    """What LiteLLM keeps under an error the server sent: its HTTP status error."""
    return httpx.HTTPStatusError("x", request=_REQUEST, response=httpx.Response(status, request=_REQUEST))


# Gemini's real 429s: the same message, told apart by the quota id; both carry a delay.
_GEMINI_429 = (
    '{"error": {"code": 429, "message": "You exceeded your current quota, please check your plan '
    'and billing details. For more information on this error, head to: '
    'https://ai.google.dev/gemini-api/docs/rate-limits.", "status": "RESOURCE_EXHAUSTED", '
    '"details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": '
    '[{"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests", '
    '"quotaId": "%s"}]}, {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "17s"}]}}'
)
GEMINI_PER_MINUTE = _GEMINI_429 % "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
GEMINI_PER_DAY = _GEMINI_429 % "GenerateRequestsPerDayPerProjectPerModel-FreeTier"


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (_error(litellm.RateLimitError, "Rate limit exceeded"), Kind.RATE_LIMITED),
        (_error(litellm.ServiceUnavailableError, "overloaded"), Kind.TRANSIENT),
        (_error(litellm.InternalServerError, "boom"), Kind.TRANSIENT),
        # A dropped connection is retried; LiteLLM's label for an error it couldn't map is not.
        (_error(litellm.APIConnectionError, "Server disconnected", cause=httpx.ReadError("x")), Kind.TRANSIENT),
        (_error(litellm.APIConnectionError, "refused", cause=ConnectionRefusedError()), Kind.TRANSIENT),
        (_error(litellm.APIConnectionError, "KeyError: 'choices'", cause=KeyError("choices")), Kind.OTHER),
        (_error(litellm.APIConnectionError, "no cause"), Kind.OTHER),
        # A server's 408 is retried; our own timeout already waited the whole limit.
        (_error(litellm.Timeout, "Request timeout", cause=_answered(408)), Kind.TRANSIENT),
        (_error(litellm.APIError, "Conflict", status=409), Kind.TRANSIENT),
        (_error(litellm.APIError, "Overloaded", status=529), Kind.TRANSIENT),
        (_error(litellm.AuthenticationError), Kind.REFUSED),
        (_error(litellm.NotFoundError), Kind.REFUSED),
        # Quota and billing, by status or wording: waiting doesn't lift them.
        (_error(litellm.APIError, "Insufficient credits", status=402), Kind.QUOTA),
        (_error(litellm.RateLimitError, "You exceeded your current quota"), Kind.QUOTA),
        (_error(litellm.RateLimitError, '{"code": "insufficient_quota"}'), Kind.QUOTA),
        (_error(litellm.BadRequestError, "Your credit balance is too low"), Kind.QUOTA),
        (_error(litellm.RateLimitError, "Rate limit exceeded: free-models-per-day"), Kind.QUOTA),
        (_error(litellm.RateLimitError, GEMINI_PER_MINUTE), Kind.RATE_LIMITED),
        (_error(litellm.RateLimitError, GEMINI_PER_DAY), Kind.QUOTA),
        # The makers' and a router's own quota, credit and spend-limit wording.
        (
            _error(litellm.APIError, "This request requires more credits, or fewer max_tokens", status=402),
            Kind.QUOTA,
        ),
        (
            _error(litellm.BadRequestError, "You have reached your specified API usage limits"),
            Kind.QUOTA,
        ),
        (
            _error(
                litellm.APIError,
                "Your team has used all available credits or reached its monthly spending limit",
                status=403,
            ),
            Kind.QUOTA,
        ),
        (_error(litellm.RateLimitError, '{"code": "billing_hard_limit_reached"}'), Kind.QUOTA),
        # A server's failure is retried whatever its body mentions.
        (_error(litellm.ServiceUnavailableError, "billing service unavailable"), Kind.TRANSIENT),
        (_error(litellm.APIError, "upstream budget check failed", status=503), Kind.TRANSIENT),
        (_error(litellm.InternalServerError, "error loading per day stats"), Kind.TRANSIENT),
        # A bad thinking setting is the request's fault, not a quota.
        (
            _error(litellm.BadRequestError, "thinking.budget_tokens: must be less than max_tokens"),
            Kind.OTHER,
        ),
        # The status first: a 429 in overflow's words is a rate limit.
        (_error(litellm.RateLimitError, "token limit exceeded"), Kind.RATE_LIMITED),
        (_error(litellm.APIError, "Too many tokens", status=429), Kind.RATE_LIMITED),
        # Too long, by type or by a router's wording on a plain bad request.
        (_error(litellm.ContextWindowExceededError, "too big"), Kind.OVERFLOW),
        (
            _error(litellm.BadRequestError, "This endpoint's maximum context length is 1000 tokens"),
            Kind.OVERFLOW,
        ),
        (_error(litellm.BadRequestError, "invalid tool schema"), Kind.OTHER),
        (_error(litellm.Timeout, "timed out", cause=httpx.ReadTimeout("x")), Kind.TIMED_OUT),
        # A lazy stream's refusal: what it wraps decides.
        (MidStreamFallbackError("x", "m", "gemini", original_exception=_error(litellm.AuthenticationError)), Kind.REFUSED),
        (MidStreamFallbackError("x", "m", "gemini", original_exception=_error(litellm.RateLimitError, GEMINI_PER_MINUTE)), Kind.RATE_LIMITED),
        (ValueError("not a provider error"), Kind.OTHER),
    ],
)
def test_classify(error, kind):
    assert classify(error) is kind


def test_the_routers_own_field_wins_over_the_header():
    error = _error(
        litellm.RateLimitError,
        '{"error": {"code": 429, "metadata": {"retry_after_seconds": 2.5}}}',
        headers={"retry-after": "3"},
    )
    assert server_delay(error) == 2.5


@pytest.mark.parametrize(
    ("headers", "delay"),
    [
        ({"retry-after": "3"}, 3.0),
        ({"llm_provider-retry-after": "4"}, 4.0),
        ({"retry-after-ms": "1500"}, 1.5),
        ({"Retry-After": "Thu, 01 Jan 1970 00:00:10 GMT"}, 10.0),
        ({"retry-after": "soon"}, None),
        ({"retry-after": "-1"}, None),
        ({"x-other": "3"}, None),
    ],
)
def test_the_header_delay(headers, delay):
    assert server_delay(_error(litellm.RateLimitError, headers=headers), now=0.0) == delay


def test_geminis_retry_info_is_the_delay():
    assert server_delay(_error(litellm.RateLimitError, GEMINI_PER_MINUTE)) == 17.0


def test_the_delay_of_an_httpx_response():
    error = litellm.RateLimitError(
        message="no", llm_provider="x", model="m",
        response=httpx.Response(429, headers={"retry-after": "2"}, request=httpx.Request("POST", "https://x")),
    )
    error.litellm_response_headers = None
    assert server_delay(error) == 2.0


def test_no_delay_named():
    assert server_delay(ValueError("x")) is None


def _step(kind, retries_done=0, shrunk=False, has_fallback=False, delay=None, wait_left=60.0):
    return next_step(
        kind, retries_done=retries_done, shrunk=shrunk, has_fallback=has_fallback,
        delay=delay, settings=SETTINGS, jitter=0.5, wait_left=wait_left,
    )


def test_a_rate_limit_waits_what_the_server_asked():
    assert _step(Kind.RATE_LIMITED, delay=3.0) == Retry(3.0)


def test_without_a_delay_the_wait_doubles():
    assert [_step(Kind.TRANSIENT, retries_done=n) for n in (0, 1)] == [Retry(1.0), Retry(2.0)]


def test_spent_retries_fall_back_or_give_up():
    assert _step(Kind.RATE_LIMITED, retries_done=2, has_fallback=True) == FallBack()
    assert _step(Kind.RATE_LIMITED, retries_done=2) == GiveUp()


def test_a_wait_longer_than_the_most_is_not_waited():
    assert _step(Kind.RATE_LIMITED, delay=60.0, has_fallback=True) == FallBack()
    assert _step(Kind.RATE_LIMITED, delay=60.0) == GiveUp()


def test_a_wait_past_what_is_left_is_not_waited():
    assert _step(Kind.RATE_LIMITED, delay=5.0, wait_left=4.0, has_fallback=True) == FallBack()
    assert _step(Kind.TRANSIENT, wait_left=0.5) == GiveUp()
    assert _step(Kind.TRANSIENT, wait_left=1.0) == Retry(1.0)


def test_what_is_left_is_the_least_of_the_call_the_run_and_the_ceiling():
    assert wait_left(30.0, None) == 30.0
    assert wait_left(30.0, RetryWindow(deadline=100.0, wait_left=10.0), now=0.0) == 10.0
    assert wait_left(30.0, RetryWindow(deadline=105.0, wait_left=60.0), now=100.0) == 5.0
    assert wait_left(30.0, RetryWindow(deadline=90.0, wait_left=60.0), now=100.0) == 0.0


def test_a_quota_is_never_sent_again_on_its_route():
    assert _step(Kind.QUOTA) == GiveUp()
    assert _step(Kind.QUOTA, has_fallback=True) == FallBack()


def test_a_refusal_falls_back_as_before():
    assert _step(Kind.REFUSED, has_fallback=True) == FallBack()
    assert _step(Kind.REFUSED) == GiveUp()


def test_too_long_shrinks_once_and_never_falls_back():
    assert _step(Kind.OVERFLOW, has_fallback=True) == Shrink()
    assert _step(Kind.OVERFLOW, shrunk=True, has_fallback=True) == GiveUp()


def test_other_errors_give_up():
    assert _step(Kind.OTHER, has_fallback=True) == GiveUp()


def test_backoff_is_capped_and_spread():
    assert backoff(10, SETTINGS, jitter=0.5) == 20.0
    assert backoff(0, SETTINGS, jitter=0.0) == pytest.approx(0.8)
    assert backoff(0, SETTINGS, jitter=1.0) == pytest.approx(1.2)
