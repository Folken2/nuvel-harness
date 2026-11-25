"""An example product on nuvel-harness. ADK loads this folder as the app.

A product says two things before anything else runs: its namespace (the prefix on the
keys the harness puts on events) and where a conversation's files live.
"""

from harness import wire_names
from harness.files import FileLayout, set_layout

wire_names.use_namespace("assistant")
set_layout(
    FileLayout(
        original="original/",
        output="output/",
        reading="markdown/",
        tool_results="tool-results/",
        pages="app-pages/",
    )
)

from . import agent as agent  # noqa: E402  after the namespace and layout
