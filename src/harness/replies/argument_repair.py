"""Repairs a tool call's arguments, only where the tool's schema leaves no doubt.

Models send a number as text ("5"), a flag as "true", an object or a list written out as
JSON text, or ``{}`` for an optional field that can't be an object. The tool then fails, or
a connected app refuses the call, and a model step goes on a retry. Each of these has one
reading the schema allows, so it is changed to that and every change is logged. Anything
else is left as it was, for the tool to judge: an argument that fits two types, a
string where a string is allowed, a field the schema doesn't name, a null.

After opencode's ``packages/core/src/plugin/tool-input-repair.ts``, without its removal
of fields the schema doesn't name: that drops what the model meant to send.

The schema is JSON Schema, as ADK declares a tool (``parameters_json_schema``) or as the
genai ``Schema`` turns into one (``declared_schemas``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Optional

from google.genai import types

_MAX_DEPTH = 8
_SAFE_INTEGER = 2**53 - 1
_PLAIN_TYPES = {"string", "number", "integer", "boolean", "array", "object"}
_UNCERTAIN = ("anyOf", "oneOf", "allOf", "enum", "const", "$ref", "nullable")


@dataclass(frozen=True)
class Repair:
    """One change: where (``a.b[2]``) and what, never the value (it can be the person's data)."""

    path: str
    change: str

    def __str__(self) -> str:
        return f"{self.path or '(arguments)'}: {self.change}"


def declared_schemas(tools: Iterable[types.Tool]) -> dict[str, dict[str, Any]]:
    """Each function declaration's parameters as JSON Schema, by tool name."""
    schemas: dict[str, dict[str, Any]] = {}
    for tool in tools:
        for declaration in tool.function_declarations or []:
            if not declaration.name:
                continue
            if isinstance(declaration.parameters_json_schema, dict):
                schemas[declaration.name] = declaration.parameters_json_schema
            elif declaration.parameters is not None:
                schemas[declaration.name] = declaration.parameters.json_schema.model_dump(
                    mode="json", exclude_none=True, by_alias=True
                )
    return schemas


def repair_arguments(args: dict[str, Any], schema: dict[str, Any]) -> tuple[dict[str, Any], list[Repair]]:
    """The arguments with what the schema settles changed, and the changes; the same
    dict, untouched, when there are none."""
    repairs: list[Repair] = []
    repaired = _repair(args, schema, schema, "", 0, repairs)
    return (repaired if isinstance(repaired, dict) else args), repairs


def _matches(value: Any, type_name: Any) -> bool:
    if type_name == "null":
        return value is None
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, dict)
    return False


def _resolve(ref: str, root: dict[str, Any]) -> Optional[dict[str, Any]]:
    for prefix, key in (("#/$defs/", "$defs"), ("#/definitions/", "definitions")):
        if ref.startswith(prefix) and "/" not in ref[len(prefix) :]:
            found = (root.get(key) or {}).get(ref[len(prefix) :].replace("~1", "/").replace("~0", "~"))
            return found if isinstance(found, dict) else None
    return None


def _repair(
    value: Any, schema: Any, root: dict[str, Any], path: str, depth: int, repairs: list[Repair]
) -> Any:
    if depth > _MAX_DEPTH or not isinstance(schema, dict):
        return value
    ref = schema.get("$ref")
    if isinstance(ref, str):
        target = _resolve(ref, root)
        return _repair(value, target, root, path, depth + 1, repairs) if target else value

    kind = schema.get("type")
    if isinstance(kind, list):
        if any(_matches(value, each) for each in kind):
            return value
        others = [each for each in kind if each != "null"]
        if len(others) != 1:
            return value
        return _repair(value, {**schema, "type": others[0]}, root, path, depth + 1, repairs)

    if kind is None:
        return _repair_branches(value, schema, root, path, depth, repairs)

    if kind in ("number", "integer"):
        number = _number(value, integer=kind == "integer")
        if number is None:
            return value
        repairs.append(Repair(path, f"text to {kind}"))
        return number
    if kind == "boolean":
        if value in ("true", "false"):
            repairs.append(Repair(path, "text to boolean"))
            return value == "true"
        return value
    if kind == "object":
        return _repair_object(value, schema, root, path, depth, repairs)
    if kind == "array":
        return _repair_array(value, schema, root, path, depth, repairs)
    return value


def _repair_branches(
    value: Any, schema: dict[str, Any], root: dict[str, Any], path: str, depth: int, repairs: list[Repair]
) -> Any:
    """``anyOf`` or ``oneOf`` (not both): repaired only to the one branch that isn't null,
    as an optional argument is declared (``Optional[int]``)."""
    if ("anyOf" in schema) == ("oneOf" in schema) or value is None:
        return value
    branches = schema.get("anyOf", schema.get("oneOf"))
    if not isinstance(branches, list) or not all(isinstance(b, dict) for b in branches):
        return value
    others = [b for b in branches if b.get("type") != "null"]
    if len(others) != 1:
        return value
    branch = others[0]
    # A branch with no plain type may take the value as it is; one that matches does.
    if branch.get("type") not in _PLAIN_TYPES or _matches(value, branch["type"]):
        return value
    return _repair(value, branch, root, path, depth + 1, repairs)


def _number(value: Any, *, integer: bool) -> Optional[float | int]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    if integer:
        if not parsed.is_integer() or abs(parsed) > _SAFE_INTEGER:
            return None
        return int(parsed)
    return int(parsed) if parsed.is_integer() and "." not in value and "e" not in value.lower() else parsed


def _from_json_text(value: Any, wanted: type, path: str, change: str, repairs: list[Repair]) -> Any:
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value)
    except ValueError:
        return value
    if not isinstance(parsed, wanted):
        return value
    repairs.append(Repair(path, change))
    return parsed


def _plain(property_schema: dict[str, Any]) -> bool:
    """One plain type and nothing that could accept null or ``{}`` too."""
    kind = property_schema.get("type")
    return isinstance(kind, str) and kind != "null" and not any(key in property_schema for key in _UNCERTAIN)


def _repair_object(
    value: Any, schema: dict[str, Any], root: dict[str, Any], path: str, depth: int, repairs: list[Repair]
) -> Any:
    parsed = _from_json_text(value, dict, path, "JSON text to object", repairs)
    if not isinstance(parsed, dict):
        return value
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    composed = any(key in schema for key in ("allOf", "anyOf", "oneOf"))
    result = parsed
    for key, current in parsed.items():
        prop = properties.get(key)
        if not isinstance(prop, dict):
            continue
        where = f"{path}.{key}" if path else key
        # Only ``{}`` where an object can't be meant. A null stays: for some apps it
        # means "clear this field", which leaving it out would change.
        empty_placeholder = current == {} and prop.get("type") != "object"
        if key not in required and not composed and _plain(prop) and empty_placeholder:
            result = {k: v for k, v in result.items() if k != key}
            repairs.append(Repair(where, "empty optional value left out"))
            continue
        repaired = _repair(current, prop, root, where, depth + 1, repairs)
        if repaired is not current:
            result = {**result, key: repaired}
    return result


def _repair_array(
    value: Any, schema: dict[str, Any], root: dict[str, Any], path: str, depth: int, repairs: list[Repair]
) -> Any:
    parsed = _from_json_text(value, list, path, "JSON text to array", repairs)
    if not isinstance(parsed, list):
        return value
    items = schema.get("items")
    if not isinstance(items, dict):
        return parsed
    result = parsed
    for index, item in enumerate(parsed):
        repaired = _repair(item, items, root, f"{path}[{index}]", depth + 1, repairs)
        if repaired is not item:
            if result is parsed:
                result = list(parsed)
            result[index] = repaired
    return result
