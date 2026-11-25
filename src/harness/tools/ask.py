"""ask_person: one question to the person, and the run waits for the answer.

A long-running tool. A valid question returns nothing, so ADK writes no result and
pauses the run (the app is resumable): the stream ends with the call unanswered, and
the web app shows the question. The answer comes
back as this call's function response, which resumes the same invocation, so the
model reads the question and the answer as one call and carries on.

An invalid question gets an error at once, as any tool would, so the model fixes it.
A second question in the same step gets one too: the person answers one at a time.
"""

from __future__ import annotations

from typing import Annotated, Any

from google.adk.tools import ToolContext

from .described import DescribedLongRunningTool
from pydantic import BaseModel, Field, ValidationError

from .. import run_state

__all__ = ["TOOL_NAME"]

TOOL_NAME = "ask_person"
MIN_OPTIONS = 2
MAX_OPTIONS = 4
MAX_QUESTION_CHARS = 300
MAX_LABEL_CHARS = 80
MAX_CONSEQUENCE_CHARS = 120


class AskOption(BaseModel):
    # The lengths are shown in the schema and checked in _problem, which says which
    # option is over: validated here, the model would get pydantic's message instead.
    label: str = Field(
        description='The choice in a few words, such as "1 July, the letter".',
        json_schema_extra={"maxLength": MAX_LABEL_CHARS},
    )
    consequence: str = Field(
        description='One short line on what follows, such as "38 invoices, 11,940 over".',
        json_schema_extra={"maxLength": MAX_CONSEQUENCE_CHARS},
    )
    # Required, with no default: a schema that offered it as optional led the model to
    # leave it out. At most one is true; none is fine when nothing favours one.
    recommended: bool = Field(
        description=(
            "True on the one option you'd pick, when the facts favour it. False on every "
            "option when it's the person's own choice, such as a day or a name they prefer."
        )
    )


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _problem(question: str, options: list[Any]) -> str | None:
    """What's wrong with the question, in words for the model, or None."""
    if not _one_line(question):
        return "The question is empty. Ask it in one or two sentences."
    if len(_one_line(question)) > MAX_QUESTION_CHARS:
        return f"The question is over {MAX_QUESTION_CHARS} characters. Shorten it."
    if not MIN_OPTIONS <= len(options) <= MAX_OPTIONS:
        return f"Give {MIN_OPTIONS} to {MAX_OPTIONS} options."
    parsed: list[AskOption] = []
    for number, raw in enumerate(options, start=1):
        try:
            option = AskOption.model_validate(raw)
        except ValidationError:
            return f"Option {number} needs a label, a consequence and recommended (true or false)."
        if not _one_line(option.label) or not _one_line(option.consequence):
            return f"Option {number} needs both a label and a consequence."
        if len(_one_line(option.label)) > MAX_LABEL_CHARS:
            return f"Option {number}'s label is over {MAX_LABEL_CHARS} characters."
        if len(_one_line(option.consequence)) > MAX_CONSEQUENCE_CHARS:
            return f"Option {number}'s consequence is over {MAX_CONSEQUENCE_CHARS} characters."
        parsed.append(option)
    if sum(option.recommended for option in parsed) > 1:
        return "Mark at most one option as recommended."
    labels = [_one_line(option.label).lower() for option in parsed]
    if len(set(labels)) != len(labels):
        return "Two options have the same label."
    return None


def ask_person(
    question: str,
    options: Annotated[list[AskOption], Field(min_length=MIN_OPTIONS, max_length=MAX_OPTIONS)],
    tool_context: ToolContext,
) -> dict[str, Any] | None:
    """Asks the person one question and waits for their answer before going on.

    Use it only when you can't go on without the person's decision and a wrong
    guess would waste real work or do something they didn't want, such as two
    sources that disagree on a number, or a choice between approaches with
    different results. Don't use it for something you can find out yourself,
    for permission to take the next step of a task you were asked to do, or to
    ask whether they want more. Do use it when a tool's result tells you to ask
    the person before going on.

    Ask one question at a time. The person sees it with your options and picks
    one, or types something else, and may attach files. The answer is this
    call's result: {"answer": "...", "option": "the label they picked, or null
    if they typed", "files": ["names of files they attached, when any"]}. The
    files are in this conversation's files under those names: open them as you
    would any file the person shared.
    Don't repeat the question in your reply text: the person sees it already.

    Args:
      question: One or two plain sentences, in the user's language, with the
        facts the person needs to decide, such as "The letter says the rate
        starts 1 July and the contract says 15 June. Which date should I use?"
      options: The choices the person picks from.
    """
    problem = _problem(question, options)
    if problem:
        return {"error": f"{problem} The question was not asked."}
    if tool_context.state.get(run_state.ASKED):
        return {"error": "You already asked a question in this step. Ask one at a time."}
    tool_context.state[run_state.ASKED] = True
    # Nothing: ADK writes no result and the run pauses until the person answers.
    return None


def ask_tools() -> list:
    return [DescribedLongRunningTool(ask_person)]
