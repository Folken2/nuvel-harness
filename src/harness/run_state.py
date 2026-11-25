"""State the agent keeps for itself during one invocation.

Written and read only by the agent: the web app sends none of these and reads none.
All use ADK's ``temp:`` prefix, so each lasts the invocation and is never saved. The
web app's keys are in state_keys.py.
"""

__all__ = ["ALL"]

# The files the built-in abilities made in this run (files/made_files.py).
MADE_FILES = "temp:agent_made_files"
# Which tools of this model call took the plan field (tools/plan_field.py), so the
# response callback knows which calls to take it off.
PLAN_FIELD_TOOLS = "temp:agent_plan_field_tools"
# For the model's bad replies (plugins/bad_replies.py): the declared tools' parameter
# schemas on this model call, by name (the response callback has no request); the ids of
# the calls of a reply cut at the output limit, which never run; and the kinds of note
# this run has sent the model (replies/agent_note.py), each once.
TOOL_SCHEMAS = "temp:agent_tool_schemas"
CUT_CALLS = "temp:agent_cut_calls"
NOTES_SENT = "temp:agent_notes_sent"
# Whether the stream of the model call going on has written a tool call, so a stream that
# ends cut at the output limit with no call in it is read as a dropped call (replies/cut_reply.py).
CALL_STREAMED = "temp:agent_call_streamed"
# Set on the first question of a step (tools/ask.py). temp: state is dropped when the run
# pauses, so the next step, after the answer, can ask again.
ASKED = "temp:agent_asked"

ALL = frozenset(
    {MADE_FILES, PLAN_FIELD_TOOLS, TOOL_SCHEMAS, CUT_CALLS, NOTES_SENT, CALL_STREAMED, ASKED}
)
