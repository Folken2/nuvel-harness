"""An `LlmAgent` whose model calls leave out ADK's identity step.

ADK adds `You are an agent. Your internal name is "<name>".` and the agent's description to
every model call's instruction. The name is the app's id (sessions, routes and the web app
use it), not something the model should know or say, and the line costs tokens on every
call. The agent's own instruction says who it is.

ADK builds a fresh flow for each run through `_llm_flow`, the hook a subclass overrides,
and names each request processor so a caller can find one without its index. This drops
the one named `identity` and keeps the rest in their order.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow

__all__ = ["LlmAgentWithoutIdentity"]

IDENTITY_STEP = "identity"


class LlmAgentWithoutIdentity(LlmAgent):
    @property
    def _llm_flow(self) -> BaseLlmFlow:
        flow = super()._llm_flow
        flow.request_processors = [
            step for step in flow.request_processors if getattr(step, "name", None) != IDENTITY_STEP
        ]
        return flow
