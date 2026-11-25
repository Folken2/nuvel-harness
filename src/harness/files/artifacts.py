"""Optional ADK artifact storage for user-uploaded files (off by default).

When enabled, SaveFilesAsArtifactsPlugin moves files out of user messages into
the artifact service (leaving a reference in the session history), and the
agent reads them on demand with ADK's load_artifacts tool. This keeps file
bytes out of the session store and out of every later model call.

Whether it is on is the product's setting; so is the storage, any ADK artifact
service (gs://, file://, or one of the product's own).
"""

from __future__ import annotations

import contextvars

from google.adk.models import LlmRequest
from google.adk.tools import ToolContext
from google.adk.tools.load_artifacts_tool import LoadArtifactsTool
from google.genai import types

from ..runs.own_messages import own_message
from .file_list import needs_telling, note
from .made_files import made_in_run
from .layout import layout
from .tool_result_copy import is_copy

# The files this run's abilities made, for the request being built. A context variable,
# because the tool's name filter is given the names and nothing of the run.
_made: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "made_files", default=frozenset()
)


# The names ADK listed for the request being built, as ``_listed`` returned them: the
# filter is the one place that is handed them.
_names: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar("listed_files", default=())


def _listed(names: list[str]) -> list[str]:
    """Without the files this run made, nor the kept copies of cut tool results: loaded
    whole, a copy would put back in the request the size it was cut to keep out. The
    model reads one with read_document, by the name the cut result's note gives. Nor a
    page a server attached to a call (MCP Apps): the person sees it, the model never."""
    made = _made.get()
    listed = [name for name in names if name not in made and not is_copy(name) and not layout().is_page(name)]
    _names.set(tuple(listed))
    return listed


# ADK's own ("Loads artifacts into the session for this request.") names its internals.
DESCRIPTION = """Opens files of this conversation so you can read them: the person's uploads and files made earlier.

The files are listed in a message of their own; pass their names as listed. Their
contents are shown to you in the next step only, so open a file again to read it again."""
ARTIFACT_NAMES_DESCRIPTION = "The files to open, by their names exactly as listed."


class ConversationFilesTool(LoadArtifactsTool):
    """ADK's load_artifacts, which lists the conversation's files to the model and tells
    it to load any the question is about, without the files this run just made.

    A picture or a report an ability saved a moment ago is about the question by
    definition, so listed, it was loaded again and again (ten times in one test run)
    though the tool's result already said everything. It is listed from the next turn
    on, where "change that picture" needs it.

    The kept whole of a cut tool result (files/tool_result_copy.py) is never listed.

    The list isn't put in the system instruction, where ADK puts it: there each new file
    changed the start of every later prompt, and with it the provider's cache. It is told
    in a message of its own, written into the history when it changes (file_list.py).
    """

    # Ours, so a call to it can carry the plan (tools/plan_field.py).
    takes_plan_field = True

    def __init__(self) -> None:
        super().__init__(process_artifact_names=_listed)
        self.description = DESCRIPTION

    def _get_declaration(self) -> types.FunctionDeclaration | None:
        declaration = super()._get_declaration()
        schema = declaration.parameters_json_schema if declaration is not None else None
        names = (schema.get("properties") or {}).get("artifact_names") if isinstance(schema, dict) else None
        if isinstance(names, dict):
            names["description"] = ARTIFACT_NAMES_DESCRIPTION
        return declaration

    async def process_llm_request(
        self, *, tool_context: ToolContext, llm_request: LlmRequest
    ) -> None:
        made = _made.set(frozenset(made_in_run(tool_context.state)))
        names = _names.set(())
        try:
            # The tool's own declaration: what ADK's base step does.
            llm_request.append_tools([self])
            # ADK's step does two things. A file the model has just asked for is attached
            # to the request, which is wanted. The list is queued for the instruction,
            # which isn't: that part lands on a request of its own and goes no further.
            aside = LlmRequest(contents=list(llm_request.contents))
            await self._append_artifacts_to_llm_request(tool_context=tool_context, llm_request=aside)
            llm_request.contents = aside.contents
            await self._tell_the_list(tool_context, llm_request, list(_names.get()))
        finally:
            _names.reset(names)
            _made.reset(made)

    async def _tell_the_list(
        self, tool_context: ToolContext, llm_request: LlmRequest, listed: list[str]
    ) -> None:
        """Writes the note when the request's latest one doesn't say this list: into the
        session, for every later request, and into this one, built before it was there."""
        if not needs_telling(llm_request.contents, listed):
            return
        session = tool_context.session
        event = own_message(
            tool_context.invocation_id,
            note(listed, self.name),
            session.events[-1] if session.events else None,
        )
        await tool_context.get_invocation_context().session_service.append_event(session, event)
        if event.content:
            llm_request.contents.append(event.content)
