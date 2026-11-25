"""Whether a run has read anything from outside the conversation since the person last
wrote: an app's records, an email, a web page, a document, code output.

Read off the session's events, not kept in the run's state, so it holds across a pause
for an approval or a question, when a run's temporary state is dropped. A tool's answer
counts as outside unless its name is in ``inside``, the product's list of tools whose
answers carry nothing a third party wrote. Answers the person gives (to a question, to an
approval) are theirs, and so are their messages; the agent's own notes in their turn are
not their words, and don't end the search. A file the person attached is not their words
either: an uploaded PDF or a pasted document was written by someone else, so a message that
carries one counts as outside, whether the file is still inline or ADK's artifact plugin has
moved it out and left a reference.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from google.adk.events import Event

from ..runs.own_messages import is_own_message


# What SaveFilesAsArtifactsPlugin leaves in a message in place of the file it moved out.
_MOVED_FILE = "[Uploaded Artifact:"


def _carries_file(event: Event) -> bool:
    if event.author != "user" or is_own_message(event) or event.content is None:
        return False
    return any(
        part.inline_data is not None or part.file_data is not None or (part.text or "").lstrip().startswith(_MOVED_FILE)
        for part in event.content.parts or []
    )


def _person_wrote(event: Event) -> bool:
    if event.author != "user" or is_own_message(event) or event.content is None:
        return False
    return any(part.text and part.text.strip() for part in event.content.parts or [])


def read_outside_since_person(events: Sequence[Event], inside: Iterable[str]) -> bool:
    """Whether a tool outside ``inside`` has answered since the person's last message."""
    allowed = frozenset(inside)
    for event in reversed(events):
        if _carries_file(event):
            return True
        if _person_wrote(event):
            return False
        if event.author == "user":
            continue
        if any(response.name not in allowed for response in event.get_function_responses()):
            return True
    return False
