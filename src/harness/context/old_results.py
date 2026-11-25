"""Older tool results a request goes without, before anything is summarized.

Tool results are most of a long run's prompt, and the old ones are rarely read again. So
once a prompt nears the size at which history would be rolled up, the older results are
sent as a short preview with a note, with no model call. Often that is enough and no
summary is written at all.

Three rules keep it cheap and safe:

- the newest results stay whole: every result of the step the model is on, then older
  ones while they fit in the protected share;
- results are masked in batches, only when one batch saves enough. A masked result
  changes the prompt from there on, so the provider's cached prefix is lost once per
  batch, never at every call;
- the same results are masked the same way at every later call, from a mark kept in the
  session: the key of the last masked result. A request masks everything up to it.

Pure, on the request's contents: the session keeps every result whole, and so does the
file the plugin saves (``plugins/old_results.py``), which the note names.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any, Optional

from google.genai import types

# The session state key the mark is kept under. Saved with the session, unlike the
# agent's other state (run_state.py), so the next turn masks the same results.
MARK_KEY = "agent_masked_until"
# What a masked result keeps of its start, so the model still sees what it was.
PREVIEW_CHARS = 200
# A result under this isn't worth a note of about the same size.
MIN_MASKED_CHARS = 1_000


@dataclass(frozen=True)
class MaskSettings:
    """Shares of the size at which the run's history is rolled up: results are masked
    once the prompt reaches ``trigger`` of it, the newest ``protect`` of it in results
    stays whole, and a batch must save ``min_saving`` of it."""

    trigger: float
    protect: float
    min_saving: float


@dataclass(frozen=True)
class Result:
    """One tool result in a request: where it is, and what masking it would save."""

    content: int
    part: int
    key: str
    name: str
    text: str
    # In the step the model is on: never masked.
    newest_step: bool
    maskable: bool

    @property
    def tokens(self) -> int:
        return tokens_of(len(self.text))

    @property
    def saving(self) -> int:
        """Roughly what masking it saves, in tokens."""
        return max(self.tokens - tokens_of(PREVIEW_CHARS + 400), 0) if self.maskable else 0


def tokens_of(chars: int) -> int:
    """A size in tokens, by the usual rule of four characters each."""
    return chars // 4


def _text(response: Any) -> str:
    return json.dumps(response, default=str, ensure_ascii=False, sort_keys=True)


def _key(call_id: Optional[str], name: str, text: str) -> str:
    digest = hashlib.sha256(f"{call_id or ''}\x00{name}\x00{text}".encode()).hexdigest()
    return digest[:24]


def results_of(contents: list[types.Content], never: Collection[str] = ()) -> list[Result]:
    """Every tool result of a request, oldest first. ``never`` names tools whose results
    always stay whole (instructions the model works from, say)."""
    found: list[tuple[int, int, types.FunctionResponse]] = []
    last_model = -1
    for index, content in enumerate(contents):
        if content.role == "model":
            last_model = index
        for position, part in enumerate(content.parts or []):
            if part.function_response is not None:
                found.append((index, position, part.function_response))
    results = []
    for index, position, response in found:
        name = response.name or ""
        text = _text(response.response)
        results.append(
            Result(
                content=index,
                part=position,
                key=_key(response.id, name, text),
                name=name,
                text=text,
                newest_step=index > last_model,
                maskable=name not in never and len(text) >= MIN_MASKED_CHARS,
            )
        )
    return results


def marked(results: list[Result], mark: Optional[str]) -> int:
    """How many of the results the mark covers, from the oldest: up to its first match.
    None when it matches nothing, as after the results it named were rolled up."""
    if mark is None:
        return 0
    return next((index + 1 for index, result in enumerate(results) if result.key == mark), 0)


def next_mark(
    results: list[Result],
    mark: Optional[str],
    prompt_tokens: Optional[int],
    threshold: int,
    settings: MaskSettings,
) -> Optional[str]:
    """The new mark when a further batch should be masked now; None to leave it.

    ``prompt_tokens`` is what the run's last model call was sent, as the provider
    counted it; with none yet, nothing is masked.
    """
    if prompt_tokens is None or prompt_tokens < settings.trigger * threshold:
        return None
    start = marked(results, mark)
    # The newest results stay whole: the step the model is on, then what fits the share.
    protected = len(results)
    kept = 0
    for index in range(len(results) - 1, start - 1, -1):
        result = results[index]
        if not result.newest_step and kept + result.tokens > settings.protect * threshold:
            break
        kept += result.tokens
        protected = index
    batch = [result for result in results[start:protected] if result.maskable]
    if not batch or sum(result.saving for result in batch) < settings.min_saving * threshold:
        return None
    return batch[-1].key


def to_mask(results: list[Result], mark: Optional[str], after: Optional[str] = None) -> list[Result]:
    """The results a request with this mark goes without; with ``after``, only those past
    that earlier mark, which is the batch the mark's move added."""
    return [result for result in results[marked(results, after) : marked(results, mark)] if result.maskable]


def masked(
    contents: list[types.Content], results: list[Result], note: Callable[[Result], str]
) -> list[types.Content]:
    """The contents with each of ``results`` as its preview and its note. Contents and
    parts that change are new objects: a request shares its parts with the session."""
    by_content: dict[int, dict[int, Result]] = {}
    for result in results:
        by_content.setdefault(result.content, {})[result.part] = result
    sent = list(contents)
    for index, changes in by_content.items():
        parts = list(contents[index].parts or [])
        for position, result in changes.items():
            response = parts[position].function_response
            assert response is not None
            parts[position] = parts[position].model_copy(
                update={
                    "function_response": response.model_copy(
                        update={"response": {"preview": result.text[:PREVIEW_CHARS], "note": note(result)}}
                    )
                }
            )
        sent[index] = contents[index].model_copy(update={"parts": parts})
    return sent
