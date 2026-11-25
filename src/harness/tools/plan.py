"""todo_list: the agent's plan for a longer task, shown to the person as progress.

An update usually rides on the next step's call instead, in a `todos` field the
agent's own tools take (tools/plan_field.py), and becomes a call to this tool
there. Calls to a connected app can't carry it, so this tool stays for those.

The tool keeps nothing. The model sends the whole list on every call and the
result is the snapshot the web app reads, live and after a reload. A call has no effect outside its own
result, so a run resumed after a restart can't repeat anything by writing the
plan again.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from .described import DescribedTool
from pydantic import BaseModel, Field, ValidationError

MAX_ITEMS = 20
MAX_CONTENT_CHARS = 200


class PlanItem(BaseModel):
    # The length is shown in the schema and checked in _checked, which words the refusal.
    content: str = Field(
        description=(
            "What the step does in a few words, in the user's language, such as "
            '"Compare invoices to the contract".'
        ),
        json_schema_extra={"maxLength": MAX_CONTENT_CHARS},
    )
    status: Literal["pending", "in_progress", "completed", "cancelled"] = Field(
        description="cancelled is for a step you dropped."
    )


def _checked(item: Any) -> PlanItem | str:
    """The item, or what to tell the model. ADK hands over a dict it couldn't convert."""
    try:
        parsed = PlanItem.model_validate(item)
    except ValidationError:
        return (
            "Each item needs a content string and a status: "
            "pending, in_progress, completed or cancelled."
        )
    content = " ".join(parsed.content.split())
    if not content:
        return "An item's content is empty. Say what the step is."
    if len(content) > MAX_CONTENT_CHARS:
        return f"An item is over {MAX_CONTENT_CHARS} characters. Keep it to a short phrase."
    return PlanItem(content=content, status=parsed.status)


def todo_list(todos: Annotated[list[PlanItem], Field(max_length=MAX_ITEMS)]) -> dict[str, Any]:
    """Writes your plan for this task, which the person sees as your progress.

    Use it when a task has three or more distinct steps, such as reading
    several sources and then producing a file. Don't use it for a question you
    can just answer or a task of one or two steps.

    Never send todo_list on its own: every response you make costs the person
    time. Put the whole list in the todos field of the call that starts a step
    instead, the first step included, marking that step in_progress and the
    finished ones completed. Your own tools take that field; a connected app's
    tools don't, so for a step that starts with one, send todo_list in the same
    response, before that call.
    Don't send a call only to mark the last step completed. Your final reply
    finishes it.

    Send the whole list every time, in order, and it replaces the last one.
    Keep exactly one step in_progress. Change the list if the work changes.
    An empty list clears the plan.

    Args:
      todos: Every step, in order.
    """
    if len(todos) > MAX_ITEMS:
        return {"error": f"At most {MAX_ITEMS} steps. Group the smaller ones."}
    items = []
    for number, raw in enumerate(todos, start=1):
        item = _checked(raw)
        if isinstance(item, str):
            return {"error": f"Step {number}: {item} The plan was not changed."}
        items.append({"id": str(number), "content": item.content, "status": item.status})
    return {"todos": items}


def plan_tools() -> list:
    return [DescribedTool(todo_list)]
