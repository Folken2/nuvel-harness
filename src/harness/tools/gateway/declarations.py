"""A tool's declaration as the model gets it, and what it costs in tokens.

The declaration is read the way ADK builds a request (``LlmRequest.append_tools``), so it is
the one a provider would be sent. Tokens are counted on its JSON with LiteLLM's default
tokenizer: a maker's own counts somewhat differently, so the figure is for comparing
toolsets with a threshold, not for billing.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import litellm
from google.adk.models import LlmRequest
from google.adk.tools.base_tool import BaseTool
from google.genai import types


def declaration_of(tool: BaseTool) -> types.FunctionDeclaration | None:
    request = LlmRequest()
    request.append_tools([tool])
    for declared in request.config.tools or []:
        if isinstance(declared, types.Tool):
            for declaration in declared.function_declarations or []:
                return declaration
    return None


def schema_of(tool: BaseTool) -> dict[str, Any]:
    """Name, description and parameters, as JSON the model can read back."""
    declaration = declaration_of(tool)
    if declaration is None:
        return {"name": tool.name, "description": tool.description or ""}
    parameters: Any = declaration.parameters_json_schema
    if parameters is None and declaration.parameters is not None:
        parameters = declaration.parameters.model_dump(mode="json", exclude_none=True)
    schema: dict[str, Any] = {"name": tool.name, "description": declaration.description or ""}
    if parameters is not None:
        schema["parameters"] = parameters
    return schema


@lru_cache(maxsize=4096)
def _tokens(text: str) -> int:
    return int(litellm.token_counter(text=text))


def tokens_of(tool: BaseTool) -> int:
    return _tokens(json.dumps(schema_of(tool), sort_keys=True))
