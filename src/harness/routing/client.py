"""The LiteLLM client: adds the turn's key, retries, and falls back to a router.

A maker's API refuses a call when its key was revoked or it doesn't know the id
``routing/direct.py`` guessed. Either way the same request goes to the fallback router,
so a stored key can only add a route, never break a turn. A rate limit or an
overloaded server is waited out and sent again, then falls back the same way; a
request too long for the model is sent once more with older tool results masked.
``routing/retry_policy.py`` decides each step.

What a request carries beyond that is handed in, since it is a router's or a product's
business, not this client's: transforms of a call's arguments, and preferences, which
are arguments the call goes without when an API doesn't take them
(``routing/turn_client.py`` supplies this service's).
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import litellm

from ..context.masking import mask_old_tool_results
from ..ledger import meter
from ..ledger.client import MeteredLiteLLMClient
from ..runs.model_streams import track
from . import retry_policy
from ..runs.retry_window import current_window, wait_left
from .turn_route import current_route

logger = logging.getLogger(__name__)


# LiteLLM's own flag for "not a call for its MCP gateway". Without it, every call
# with tools imports the proxy's MCP client first, which needs mcp 2.x; this
# service is on mcp 1.x, which ADK's toolsets and our address guard are built on.
# Tools here are ADK's function declarations, never LiteLLM's MCP tools, and
# LiteLLM drops the flag before the request goes out.
_NOT_FOR_THE_MCP_GATEWAY = {"_skip_mcp_handler": True}

# The OpenAI SDK under LiteLLM's OpenAI-compatible calls retries a refused request
# twice by itself, so those requests would be no ledger attempt (seen with the scripted
# model: three requests for one call). Our loop does the retries; LiteLLM consumes the
# flag and sends nothing for it.
_NO_HIDDEN_RETRIES = {"max_retries": 0}


@dataclass(frozen=True)
class ClientSettings:
    retry: retry_policy.Settings
    # The longest one call may spend waiting between its retries.
    retry_budget_seconds: float
    # A request too long for its model is sent again with the tool results of all but
    # this many latest steps masked.
    overflow_kept_steps: int
    # How long a request may go without a byte from the model's API, in seconds, before
    # it fails as timed out: before the answer starts, and between a stream's chunks, so
    # a stalled stream ends the call instead of holding the run. A long answer that keeps
    # streaming is never cut. 0: no limit of ours, LiteLLM's own default applies.
    request_timeout_seconds: float = 0


# (model, the call's arguments) -> the arguments to send.
Transform = Callable[[str, dict[str, Any]], dict[str, Any]]
# model -> arguments to add when the API takes them, and to leave out when it doesn't.
Preference = Callable[[str], dict[str, Any]]


class KeyedLiteLLMClient(MeteredLiteLLMClient):
    """Each request it sends is one ledger attempt (``ledger/client.py``); a
    fallback to the router is the same call's next attempt.

    ``settings`` is asked at each call, so a setting read from the environment is the
    one in force then. ``call_transforms`` see the model the call asked for, once;
    ``request_transforms`` and ``preferences`` see the model each request goes to, which
    after a fallback is another API's.
    """

    def __init__(
        self,
        settings: Callable[[], ClientSettings],
        *,
        call_transforms: Sequence[Transform] = (),
        request_transforms: Sequence[Transform] = (),
        preferences: Sequence[Preference] = (),
        on_whole_answer: Callable[[Any], None] | None = None,
    ) -> None:
        super().__init__(on_whole_answer)
        self._settings = settings
        self._call_transforms = tuple(call_transforms)
        self._request_transforms = tuple(request_transforms)
        self._preferences = tuple(preferences)

    async def acompletion(self, model: Any, messages: Any, tools: Any, **kwargs: Any) -> Any:
        kwargs = {**kwargs, **_NOT_FOR_THE_MCP_GATEWAY, **_NO_HIDDEN_RETRIES}
        for transform in self._call_transforms:
            kwargs = transform(str(model), kwargs)
        meter.new_call()
        try:
            response = await self._routed(model, messages, tools, **kwargs)
        finally:
            meter.end_call()
        # A streamed answer goes in its run's scope, so a Stop can close it.
        return track(response)

    async def _routed(self, model: Any, messages: Any, tools: Any, **kwargs: Any) -> Any:
        """The call on its route, then on the fallback, each step from ``retry_policy``.

        Every request below is one ledger attempt (``MeteredLiteLLMClient``), so a
        retry or a fallback is the same call's next attempt.
        """
        settings = self._settings()
        if settings.request_timeout_seconds > 0 and "timeout" not in kwargs:
            kwargs = {**kwargs, "timeout": settings.request_timeout_seconds}
        route = current_route()
        targets = [(model, kwargs if route is None else {**kwargs, "api_key": route.api_key})]
        if route is not None and route.fallback_model is not None:
            fallback = {k: v for k, v in kwargs.items() if k != "api_key"}
            if route.fallback_key:
                fallback["api_key"] = route.fallback_key
            targets.append((route.fallback_model, fallback))
        call_wait_left = settings.retry_budget_seconds
        window = current_window()
        shrunk = False
        for position, (target, target_kwargs) in enumerate(targets):
            retries_done = 0
            while True:
                try:
                    return await self._requested(target, messages, tools, **target_kwargs)
                except Exception as error:
                    kind = retry_policy.classify(error)
                    step = retry_policy.next_step(
                        kind,
                        retries_done=retries_done,
                        shrunk=shrunk,
                        has_fallback=position + 1 < len(targets),
                        delay=retry_policy.server_delay(error),
                        settings=settings.retry,
                        jitter=random.random(),
                        wait_left=wait_left(call_wait_left, window),
                    )
                    # Never the provider's text in the log: it can carry the URL or a key.
                    if isinstance(step, retry_policy.Retry):
                        logger.warning(
                            "%s: %s (%s); trying again in %.1fs",
                            target, kind.value, type(error).__name__, step.delay,
                        )
                        call_wait_left -= step.delay
                        if window is not None:
                            window.wait_left -= step.delay
                        await asyncio.sleep(step.delay)
                        retries_done += 1
                        continue
                    if isinstance(step, retry_policy.Shrink):
                        smaller = mask_old_tool_results(messages, settings.overflow_kept_steps)
                        shrunk = True
                        if smaller is not None:
                            logger.warning("%s: too long for its context; masking older tool results", target)
                            messages = smaller
                            continue
                    if isinstance(step, retry_policy.FallBack):
                        logger.warning(
                            "%s: %s (%s); using %s",
                            target, kind.value, type(error).__name__, targets[position + 1][0],
                        )
                        break
                    raise
        raise AssertionError("unreachable: the last route gives up rather than falling back")

    async def _requested(self, model: Any, messages: Any, tools: Any, **kwargs: Any) -> Any:
        """One request, with what this model's API should carry.

        LiteLLM refuses an argument it knows a model lacks (a reasoning effort of
        ``minimal`` on a model with no such effort, say) before anything is sent. The
        request then goes without the preferences, at the model's own defaults: a
        preference is never a reason to fail a turn.
        """
        for transform in self._request_transforms:
            kwargs = transform(str(model), kwargs)
        preferred: dict[str, Any] = {}
        for preference in self._preferences:
            preferred.update(preference(str(model)))
        if not preferred:
            return await super().acompletion(model, messages, tools, **kwargs)
        try:
            return await super().acompletion(model, messages, tools, **{**kwargs, **preferred})
        except litellm.UnsupportedParamsError:
            logger.warning("%s doesn't take %s; using its default", model, preferred)
        return await super().acompletion(model, messages, tools, **kwargs)
