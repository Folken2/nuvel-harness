"""The harness alone. A product names its wire namespace and its file layout first; these
tests stand in for one with neutral values."""

from harness import wire_names
from harness.files import FileLayout, set_layout

wire_names.use_namespace("test")
set_layout(
    FileLayout(
        original="original/",
        output="output/",
        reading="markdown/",
        tool_results="tool-results/",
        pages="app-pages/",
    )
)
