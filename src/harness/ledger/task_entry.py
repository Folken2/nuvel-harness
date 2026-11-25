"""A background task's own interaction in the usage ledger (usage-and-analytics.md UG-5).

A task is a run of its own that the agent starts from a turn's tool call, so no
request opens its interaction: the agent does, from the parent turn's ledger state,
and the task's run writes its model calls under it. Whoever watches the task to its
end closes it (the web app).

The parent's state and the task's are in the product's shape, so the parser and the
function that writes the task's state are arguments (``ledger/web_app_state.py``).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Callable, Optional

from .context import StateParser, context_from_state
from .entries import task_interaction_values
from .writer import current_writer

logger = logging.getLogger(__name__)

# The task's ledger state from its parent's: (parent, the task's interaction id, task id).
TaskState = Callable[[Mapping[str, Any], str, str], dict[str, object]]


async def open_task_state(
    parent: object, task_id: str, *, parser: StateParser, as_task: TaskState
) -> Optional[dict[str, object]]:
    """The task's ledger state, its interaction opened; None when the parent's turn
    isn't recorded, or the entry couldn't be written (the task runs on unmetered)."""
    context = context_from_state(parent, parser)
    writer = current_writer()
    if context is None or writer is None or not isinstance(parent, Mapping):
        return None
    try:
        interaction_id = await writer.store.open_background(task_interaction_values(context, task_id))
    except Exception:  # noqa: BLE001 - recording must never stop a task
        logger.error("ledger: couldn't open task %s's entry", task_id, exc_info=True)
        return None
    if interaction_id is None:
        return None
    return as_task(parent, interaction_id, task_id)
