"""The agent: one ADK LlmAgent, the harness's tools and plugins around it.

The model comes from ASSISTANT_MODEL: a Gemini id ("gemini-2.5-flash", the default,
with GOOGLE_API_KEY set) or any LiteLLM id ("openai/gpt-4.1-mini", "anthropic/...",
with that provider's key set).
"""

from __future__ import annotations

import os
from typing import Union

from google.adk.agents import LlmAgent
from google.adk.apps import App, ResumabilityConfig
from google.adk.models import BaseLlm
from google.adk.models.lite_llm import LiteLlm

from harness.plugins import BadRepliesPlugin, LoopGuardPlugin, PlanFieldPlugin, TurnLogPlugin
from harness.runs import LoopSettings
from harness.tools import ask_tools, plan_tools

INSTRUCTION = """You are a careful assistant.

For any task with more than one step, keep a plan with todo_list and update it as you go.
When a choice is the person's to make, ask with ask_person and wait for the answer.
Use word_count when asked about the length of a text."""


def word_count(text: str) -> dict[str, int]:
    """Counts the words and characters of a text.

    Args:
        text: The text to count.
    """
    return {"words": len(text.split()), "characters": len(text)}


def build_app(model: Union[str, BaseLlm]) -> App:
    agent = LlmAgent(
        name="assistant",
        model=model,
        instruction=INSTRUCTION,
        tools=[word_count, *plan_tools(), *ask_tools()],
    )
    return App(
        name="assistant",
        root_agent=agent,
        # Order matters only where the harness says it does: the plan field comes off a
        # call before replies are repaired, and replies are repaired before the loop
        # guard counts their calls.
        plugins=[
            PlanFieldPlugin(),
            BadRepliesPlugin(),
            # A note to the model at the third identical call, a question to the person at the fifth.
            LoopGuardPlugin(LoopSettings(note_at=3, stop_at=5), ask_tool="ask_person"),
            TurnLogPlugin(),
        ],
        # ask_person pauses the run; resumability lets the answer resume the same one.
        resumability_config=ResumabilityConfig(is_resumable=True),
    )


def _model() -> Union[str, BaseLlm]:
    name = os.environ.get("ASSISTANT_MODEL", "gemini-2.5-flash")
    return LiteLlm(model=name) if "/" in name else name


app = build_app(_model())
