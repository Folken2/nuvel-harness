"""model_consult: the agent asks a stronger model about a hard step.

ADK's ``ModelConsultTool`` does the work: it hands the conversation to the
advisor with tools turned off, keeps the per-turn and per-conversation budgets
and never raises into the loop. Ours is a subclass that changes what a service
needs and nothing inside it, and takes each of those as ``ConsultHooks``:

- Which advisor a turn may consult (``advisor_of``). Without one the tool is not
  offered at all.
- How its call goes out (``route``): the id it is called under, and what holds
  around the call. ADK calls the advisor directly, past the plugins, so the route
  is set around the call here.
- How its call is metered (``meter``): one at a time, recorded as the service
  records it, and what an answered consultation says it cost.
- Our own wording on when to ask (``consult_wording.py``).

``tools/advisor.py`` supplies this service's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncContextManager, Callable, ContextManager, Mapping, Optional, Protocol

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.lite_llm import LiteLlm, LiteLLMClient
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.model_consult import ModelConsultContextConfig, ModelConsultTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from .consult_wording import EXECUTOR_INSTRUCTION, TOOL_DESCRIPTION

_UNREACHABLE_MESSAGE = (
    "The advisor can't be reached this turn. Continue with your own best judgment;"
    " don't call this tool again for the same question."
)


@dataclass(frozen=True)
class Advisor:
    """The advisor a turn may consult: its LiteLLM id, catalog prices and catalog entry.

    ``pricing`` is in the service's own shape: only its hooks read it."""

    model: str
    pricing: Optional[dict[str, float]] = None
    id: Optional[str] = None
    name: Optional[str] = None


@dataclass(frozen=True)
class ConsultSettings:
    """The budgets and limits, read once when the agent is built."""

    max_uses: int
    session_max_uses: int
    max_output_tokens: int
    max_context_chars: int
    timeout_seconds: float


class Route(Protocol):
    """How a consultation goes out."""

    def target(self, advisor: Advisor, state: Any) -> str:
        """The id the advisor is called under on this turn."""

    def open(self, advisor: Advisor, state: Any) -> Optional[ContextManager[None]]:
        """What holds around the advisor's call, or None when it can't be reached."""


# Takes the consultation's result once the advisor has answered or failed, records
# it, and gives what an answered one carries besides: its cost, under the service's keys.
Settle = Callable[[dict[str, Any]], Mapping[str, Any]]


class MeteredCall(Protocol):
    """One consultation's metering, begun before its call."""

    def alone(self) -> AsyncContextManager[None]:
        """Held for the whole call, where the service meters one call at a time."""

    def recorded(self) -> ContextManager[Settle]:
        """Around the advisor's call, inside its route: the call is recorded as the
        advisor's, and the function it gives settles it."""


@dataclass(frozen=True)
class ConsultHooks:
    """What a service supplies: the tool's name, the turn's advisor, the model client,
    the route and the metering."""

    name: str
    advisor_of: Callable[[Any], Optional[Advisor]]
    client: Callable[[], LiteLLMClient]
    route: Route
    # Called with the advisor and the tool call's id, before the call.
    meter: Callable[[Advisor, Optional[str]], MeteredCall]


class AdvisorConsultTool(ModelConsultTool):
    """``ModelConsultTool`` on one advisor model, called on the turn's route for it."""

    # Ours, so a call to it can carry the plan (tools/plan_field.py).
    takes_plan_field = True

    def __init__(self, target_model: str, settings: ConsultSettings, hooks: ConsultHooks) -> None:
        super().__init__(
            # The id the call goes out under.
            model=LiteLlm(model=target_model, llm_client=hooks.client()),
            max_uses=settings.max_uses,
            session_max_uses=settings.session_max_uses,
            # LiteLlm doesn't pass ADK's thinking levels on, so none is sent.
            thinking_level=None,
            max_output_tokens=settings.max_output_tokens,
            name=hooks.name,
            description=TOOL_DESCRIPTION,
            executor_instruction=EXECUTOR_INSTRUCTION,
            context_config=ModelConsultContextConfig(max_chars=settings.max_context_chars),
            timeout_seconds=settings.timeout_seconds,
        )
        self._hooks = hooks

    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> dict[str, Any]:
        advisor = self._hooks.advisor_of(tool_context.state)
        if advisor is None:
            return {"status": "error", "message": _UNREACHABLE_MESSAGE}
        going_out = self._hooks.route.open(advisor, tool_context.state)
        if going_out is None:
            return {"status": "error", "message": _UNREACHABLE_MESSAGE}
        # ADK calls the advisor past the plugins, so its call is settled here.
        call = self._hooks.meter(advisor, tool_context.function_call_id)
        async with call.alone():
            with going_out, call.recorded() as settle:
                result = await super().run_async(args=args, tool_context=tool_context)
                extra = settle(result)
        if result.get("status") != "ok":
            return result
        return {**result, **extra}


def usage_metadata(result: dict[str, Any]) -> Optional[types.GenerateContentResponseUsageMetadata]:
    """A consultation's usage in the form a model call's comes in."""
    usage = result.get("usage")
    if not isinstance(usage, dict) or not usage.get("total_tokens"):
        return None
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=int(usage.get("prompt_tokens") or 0),
        cached_content_token_count=int(usage.get("cached_tokens") or 0),
        # Through LiteLLM the output already holds the reasoning.
        candidates_token_count=int(usage.get("output_tokens") or 0),
        thoughts_token_count=int(usage.get("thoughts_tokens") or 0),
        total_token_count=int(usage.get("total_tokens") or 0),
    )


class AdvisorToolset(BaseToolset):
    """Offers model_consult only on a turn that names an advisor."""

    def __init__(self, settings: ConsultSettings, hooks: ConsultHooks) -> None:
        super().__init__()
        self._settings = settings
        self._hooks = hooks
        # One tool per advisor id the calls go out under: a handful at most.
        self._tools: dict[str, AdvisorConsultTool] = {}

    async def get_tools(
        self, readonly_context: Optional[ReadonlyContext] = None
    ) -> list[BaseTool]:
        if readonly_context is None:
            return []
        advisor = self._hooks.advisor_of(readonly_context.state)
        if advisor is None:
            return []
        target = self._hooks.route.target(advisor, readonly_context.state)
        tool = self._tools.get(target)
        if tool is None:
            tool = self._tools[target] = AdvisorConsultTool(target, self._settings, self._hooks)
        return [tool]
