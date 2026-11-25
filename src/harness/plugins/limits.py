"""The safety ceiling, as ADK callbacks (limits.md LM-29 to LM-32).

The budget and the rule are ``runs/ceiling.py``. This plugin only counts: each model
call and tool call is a step, each call's cost is added, and before each model call it
asks whether the run may go on. When it may not, the run gets one last call with tool
calls off, to tell the person what was done and what is left (``last_call_tokens`` caps
its reply; 0 turns it off). That reply carries ``wire_names.PAUSED`` in its metadata.
Any call after it, or every call at the ceiling when the last call is off, isn't made:
the run ends with an event with no content that carries the pause. Either is a stop, not
an error, so ADK ends the run as it ends any finished one.

Only the ceiling lives here. Request, month cost and concurrency limits are the web
app's start gate: nothing in this plugin reads them, so a run that has started always
finishes (LM-20, T-29).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from .. import wire_names
from ..runs.retry_window import RetryWindow, use_window
from ..runs.ceiling import (
    Ceiling,
    Pause,
    RunBudget,
    RunBudgets,
    close_last_call,
    make_last_call,
    paused_response,
    reached,
    spent_before,
)

logger = logging.getLogger(__name__)


def _no_call_cost(llm_response: LlmResponse, pricing: object) -> float:
    return 0.0


def _no_tool_cost(tool_name: Optional[str], result: Any) -> float:
    return 0.0


@dataclass(frozen=True)
class RunCosts:
    """How a product counts a run's cost, and where a turn's state says its ceiling.

    With nothing said a run costs nothing and its ceiling is the plugin's own: only
    steps and time can pause it.
    """

    # The state key a turn's own ceiling arrives under, and the one its model's prices do.
    limits_key: Optional[str] = None
    pricing_key: Optional[str] = None
    # What one model call cost, given the prices in the state (or None).
    cost_of_call: Callable[[LlmResponse, object], float] = _no_call_cost
    # What a tool's own provider call cost, from its name and result.
    cost_of_tool: Callable[[Optional[str], Any], float] = _no_tool_cost

    def limits(self, state: Any) -> object:
        return state.get(self.limits_key) if self.limits_key else None

    def pricing(self, state: Any) -> object:
        return state.get(self.pricing_key) if self.pricing_key else None


class RunLimits(BasePlugin):
    """Pauses a run that reaches its safety ceiling: steps, cost or time.

    The ceiling is the one sent with the turn (``costs.limits_key``), else the values
    given here. What a call or a tool cost is the product's to say (``RunCosts``).
    """

    def __init__(
        self,
        max_seconds: float,
        max_cost_usd: float,
        max_steps: int,
        retry_budget_seconds: float = math.inf,
        costs: RunCosts = RunCosts(),
        last_call_tokens: int = 0,
    ) -> None:
        super().__init__(name="run_limits")
        self.costs = costs
        self.last_call_tokens = last_call_tokens
        # Each run whose last call has been made, with the pause it was made for.
        self._last_calls: dict[str, Pause] = {}
        self.fallback = Ceiling(steps=max_steps, cost_usd=max_cost_usd, seconds=max_seconds)
        self.retry_budget_seconds = retry_budget_seconds
        self._runs = RunBudgets(self.fallback)
        self._windows: dict[str, RetryWindow] = {}

    def budget(self, invocation_id: str) -> RunBudget:
        return self._runs.of(invocation_id, time.monotonic())[0]

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        invocation_id = invocation_context.invocation_id
        state = invocation_context.session.state
        pricing = self.costs.pricing(state)
        spent = spent_before(
            [e for e in invocation_context.session.events if e.invocation_id == invocation_id],
            time.monotonic(),
            cost_of_call=lambda call: self.costs.cost_of_call(call, pricing),
            cost_of_tool=lambda result: self.costs.cost_of_tool(result.name, result.response),
        )
        self._runs.start(invocation_id, spent, self.costs.limits(state))

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        budget, ceiling = self._runs.of(
            callback_context.invocation_id,
            time.monotonic(),
            self.costs.limits(callback_context.state),
        )
        pause = reached(budget, ceiling, time.monotonic())
        if pause is None:
            # The call's retries wait no longer than the run may (runs/retry_window.py).
            window = self._windows.setdefault(
                callback_context.invocation_id,
                RetryWindow(
                    deadline=budget.started_at + ceiling.seconds,
                    wait_left=self.retry_budget_seconds,
                ),
            )
            use_window(window)
            return None
        logger.warning(
            "run %s paused at its safety ceiling: %s %s of %s",
            callback_context.invocation_id,
            pause.reason,
            pause.reached,
            pause.ceiling,
        )
        run_id = callback_context.invocation_id
        if self.last_call_tokens <= 0 or run_id in self._last_calls or llm_request is None:
            return paused_response(self._last_calls.get(run_id, pause))
        self._last_calls[run_id] = pause
        make_last_call(llm_request, pause, self.last_call_tokens)
        # Past the ceiling nothing is left to wait for: no retries.
        use_window(RetryWindow(deadline=time.monotonic(), wait_left=0.0))
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        if llm_response.partial or wire_names.PAUSED in (llm_response.custom_metadata or {}):
            return None
        last_call = self._last_calls.get(callback_context.invocation_id)
        if last_call is not None:
            close_last_call(llm_response, last_call)
        budget = self.budget(callback_context.invocation_id)
        budget.steps += 1
        budget.cost_usd += self.costs.cost_of_call(
            llm_response, self.costs.pricing(callback_context.state)
        )
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict,
        tool_context: ToolContext,
        result: dict,
    ) -> Optional[dict]:
        budget = self.budget(tool_context.invocation_id)
        budget.steps += 1
        # A consultation, a picture or a research call is a provider call made inside a
        # tool, past the model callbacks.
        budget.cost_usd += self.costs.cost_of_tool(tool.name, result)
        return None

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        self._runs.end(invocation_context.invocation_id)
        self._last_calls.pop(invocation_context.invocation_id, None)
        self._windows.pop(invocation_context.invocation_id, None)
