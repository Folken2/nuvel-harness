"""The plan another call carries : an optional `todos` field on the agent's own tools.

An update sent with todo_list on its own costs a model round trip, about a second the
person waits. So each of the agent's own tools also takes the whole plan in a `todos`
field. Before the calls run, the field comes off the call and becomes a todo_list call of
its own, placed just before the call that carried it, in the same model response. The web
app and a reload then read the plan exactly as they read todo_list's, and the step
that call starts files under the item it marks in progress. The model's history
shows the same thing: a todo_list call sent with the step's call.

Only the agent's own tools get the field: its function tools, and its own tool classes that
say so with `takes_plan_field = True` (the conversation's files, the advisor). Connected apps (MCP) and skill tools
keep the schema their server or ADK gives them: their names and parameters aren't ours,
and one could take a `todos` argument of its own, which stripping would break. A step that
starts with one of those still updates the plan with todo_list.

On a model call where every other tool takes the field, the model isn't shown todo_list at
all: told to use the field, Sonnet still sent todo_list alone, mostly for its first plan,
which is the round trip this saves. The tool stays registered, so the call made from a
field runs as usual.

Pure: no ADK callbacks here (plugins/plan_field.py calls this).
"""

from __future__ import annotations

import copy
import uuid
from collections.abc import Iterable, Mapping
from typing import Any

from google.adk.tools import FunctionTool
from google.adk.tools.base_tool import BaseTool
from google.genai import types

from .plan import todo_list

PLAN_TOOL = todo_list.__name__
FIELD = "todos"

# With the instruction, all the model is told: on most calls it isn't shown todo_list.
# When to plan at all (three or more steps) is the instruction's, so it isn't repeated here.
FIELD_DESCRIPTION = "Optional: your whole plan, when this call starts a step. Replaces the last."
CONTENT_DESCRIPTION = "A few words, in the user's language."
STATUS_DESCRIPTION = "One in_progress: this call's step."

_STATUSES = ["pending", "in_progress", "completed", "cancelled"]


def field_json_schema() -> dict[str, Any]:
    """The field as a JSON schema property, inline so it needs none of the tool's $defs."""
    return {
        "type": "array",
        "description": FIELD_DESCRIPTION,
        "items": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": CONTENT_DESCRIPTION},
                "status": {"type": "string", "enum": list(_STATUSES), "description": STATUS_DESCRIPTION},
            },
            "required": ["content", "status"],
        },
    }


def _field_schema() -> types.Schema:
    """The same field for a declaration written as a `types.Schema`."""
    return types.Schema(
        type=types.Type.ARRAY,
        description=FIELD_DESCRIPTION,
        items=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "content": types.Schema(type=types.Type.STRING, description=CONTENT_DESCRIPTION),
                "status": types.Schema(
                    type=types.Type.STRING, enum=list(_STATUSES), description=STATUS_DESCRIPTION
                ),
            },
            required=["content", "status"],
        ),
    )


def carriers(tools: Mapping[str, BaseTool]) -> set[str]:
    """The tools of a model request that get the field.

    None at all when todo_list isn't offered: then there's no plan for the field to update.
    """
    if PLAN_TOOL not in tools:
        return set()
    return {
        name
        for name, tool in tools.items()
        if name != PLAN_TOOL
        and (isinstance(tool, FunctionTool) or getattr(type(tool), "takes_plan_field", False) is True)
    }


def with_field(declaration: types.FunctionDeclaration) -> types.FunctionDeclaration | None:
    """A copy of the declaration that takes the field, or None when it has one of its own."""
    json_schema = declaration.parameters_json_schema
    if isinstance(json_schema, dict):
        if FIELD in (json_schema.get("properties") or {}):
            return None
        schema = copy.deepcopy(json_schema)
        schema.setdefault("type", "object")
        schema["properties"] = {**(schema.get("properties") or {}), FIELD: field_json_schema()}
        return declaration.model_copy(update={"parameters_json_schema": schema})
    if declaration.parameters is not None:
        if FIELD in (declaration.parameters.properties or {}):
            return None
        parameters = declaration.parameters.model_copy(
            update={"properties": {**(declaration.parameters.properties or {}), FIELD: _field_schema()}}
        )
        return declaration.model_copy(update={"parameters": parameters})
    # A tool with no arguments.
    return declaration.model_copy(
        update={"parameters_json_schema": {"type": "object", "properties": {FIELD: field_json_schema()}}}
    )


def add_field(tools: Iterable[types.Tool], names: set[str]) -> set[str]:
    """Gives the named function declarations the field, in place in the request's tool list.

    Returns the names that now take it: the carriers of this request. A tool whose own
    schema already has a `todos` is left out, so its real argument is never stripped.
    Each changed declaration is a copy, so a declaration a tool keeps between requests
    never gains the field itself.
    """
    taken: set[str] = set()
    if not names:
        return taken
    for tool in tools:
        declarations = getattr(tool, "function_declarations", None)
        if not declarations:
            continue
        for index, declaration in enumerate(declarations):
            if declaration.name not in names:
                continue
            changed = with_field(declaration)
            if changed is not None:
                declarations[index] = changed
                taken.add(declaration.name)
    return taken


def plan_tool_hidden(tools: Mapping[str, BaseTool], names: set[str]) -> bool:
    """Whether the model can do without seeing todo_list: every other tool takes the field."""
    return bool(names) and names == set(tools) - {PLAN_TOOL}


def hide_plan_tool(tools: list[types.Tool]) -> None:
    """Takes todo_list's declaration out of the request's tool list, in place."""
    for tool in list(tools):
        declarations = getattr(tool, "function_declarations", None)
        if not declarations:
            continue
        kept = [d for d in declarations if d.name != PLAN_TOOL]
        if len(kept) == len(declarations):
            continue
        if kept:
            tool.function_declarations = kept
        else:
            tools.remove(tool)


def plan_call_id() -> str:
    """An id for the todo_list call made from a field. Not ADK's `adk-` prefix, which
    ADK strips before a request, since the model's provider must see the call and its
    result paired."""
    return f"call_plan_{uuid.uuid4().hex[:24]}"


def lift(content: types.Content | None, names: set[str]) -> bool:
    """Takes the field off the named calls and makes it a todo_list call, in place.

    The todo_list call goes just before the first call that carried a plan, with the last
    plan carried, so that call files under the step it starts. A response that already
    calls todo_list keeps its own call: the fields only come off. Returns whether anything
    changed.
    """
    if not content or not content.parts or not names:
        return False
    plan: Any = None
    first: int | None = None
    explicit = False
    changed = False
    for index, part in enumerate(content.parts):
        call = part.function_call
        if call is None:
            continue
        if call.name == PLAN_TOOL:
            explicit = True
            continue
        if call.name not in names or not call.args or FIELD not in call.args:
            continue
        args = dict(call.args)
        carried = args.pop(FIELD)
        call.args = args
        changed = True
        if isinstance(carried, list):
            plan = carried
            if first is None:
                first = index
    if explicit or first is None:
        return changed
    carrier = content.parts[first]
    plan_part = types.Part(
        function_call=types.FunctionCall(id=plan_call_id(), name=PLAN_TOOL, args={FIELD: plan})
    )
    # A thinking model's signature belongs on the response's first call.
    if carrier.thought_signature:
        plan_part.thought_signature = carrier.thought_signature
        carrier.thought_signature = None
    content.parts.insert(first, plan_part)
    return True
