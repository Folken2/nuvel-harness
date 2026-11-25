"""start_task, check_task, wait_for_tasks: background work that reports back.

The tools only reach the board (runs/tasks.py) through the run they're part of
(runs/run_scope.py), so a task belongs to the conversation that started it and no
other can read it. A task's own run doesn't get them: it can't start more tasks.
What else a task doesn't get is left out by ./not_in_task.py.

The tools are given their limits (``TaskToolLimits``): whether tasks are on, how long a
wait may last, and the two caps their refusals name. Each is read when a tool runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Callable, Optional

from pydantic import Field

from google.adk.agents.readonly_context import ReadonlyContext
from .described import DescribedTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.tool_context import ToolContext

from ..runs.run_scope import current_run_scope
from ..runs.tasks import TaskLimitReached, TaskNotStarted, TaskRunLimitReached
from .not_in_task import in_task

START_TASK = "start_task"
CHECK_TASK = "check_task"
WAIT_FOR_TASKS = "wait_for_tasks"
DESCRIPTION_MAX_CHARS = 80
PROMPT_MAX_CHARS = 20_000
NO_RUN = {"error": "Tasks can't be started here."}


@dataclass(frozen=True)
class TaskToolLimits:
    """What a service says about its tasks, each asked when a tool runs."""

    enabled: Callable[[], bool]
    # The caps the board holds (runs/tasks.py), as the refusals tell the model.
    max_running: Callable[[], int]
    max_per_turn: Callable[[], int]
    wait_max_seconds: Callable[[], float]


TaskTool = Callable[..., Any]

# The two tools that look at a task that is still going: the same answer twice is the
# task taking its time, not a loop (runs/loop_guard.py).
WAITING_TOOLS = frozenset({"check_task", "wait_for_tasks"})


def task_tools(limits: TaskToolLimits) -> tuple[TaskTool, TaskTool, TaskTool]:
    """start_task, check_task and wait_for_tasks under these limits. Their names,
    docstrings and arguments are what the model is shown."""

    async def start_task(description: str, prompt: str, tool_context: ToolContext) -> dict[str, Any]:
        """Starts a background task: a separate agent does a self-contained job and reports back.

        Use it for work that takes many steps and doesn't need you in between, such as
        researching a question across several sources or drafting a document, while you
        go on with this conversation. Start several at once for independent pieces. Don't
        use it for something one or two tool calls answer, or for work on the person's
        apps or this conversation's files: a task has neither, and can't ask the person
        anything.

        The task starts with nothing of this conversation: put everything it needs in the
        prompt, including what to look at, what to leave out and the form the report should
        take.

        It returns at once with a task_id. The report arrives later as a message starting
        "[Task report]". With nothing else to do, call wait_for_tasks rather than ending
        your turn.

        Args:
          description: What the task is, in 3 to 8 words, shown to the person, such as
            "Research competitor pricing".
          prompt: The full instructions for the task.
        """
        scope = current_run_scope()
        if scope is None:
            return NO_RUN
        description = " ".join(description.split())
        if not description or len(description) > DESCRIPTION_MAX_CHARS:
            return {"error": f"Give a description of 1 to {DESCRIPTION_MAX_CHARS} characters."}
        if not prompt.strip() or len(prompt) > PROMPT_MAX_CHARS:
            return {"error": f"Give a prompt of 1 to {PROMPT_MAX_CHARS} characters."}
        try:
            task = await scope.tasks.start(
                scope.key, description, prompt.strip(), tool_context.state.to_dict()
            )
        except TaskLimitReached:
            return {
                "error": (
                    f"The person already has {limits.max_running()} tasks running, the "
                    "most at once. Wait for one to finish, or do this yourself."
                )
            }
        except TaskRunLimitReached:
            return {
                "error": (
                    f"This turn already started {limits.max_per_turn()} tasks, the most one "
                    "turn may. Use their reports, or do the rest yourself."
                )
            }
        except TaskNotStarted as failed:
            return {"error": f"The task didn't start: {failed.detail}"}
        return {"task_id": task.id, "status": task.status, "started_at": task.run.started_at}


    def check_task(task_id: str) -> dict[str, Any]:
        """Checks a background task without waiting: its status ("running", "finished",
        "failed" or "stopped"), the steps it has done, and its report once it has finished.
        To wait for the report instead, use wait_for_tasks.

        Args:
          task_id: The id start_task returned.
        """
        scope = current_run_scope()
        if scope is None:
            return NO_RUN
        task = scope.tasks.get(scope.key, task_id)
        if task is None:
            return {
                "error": (
                    "No task with that id here. If the service restarted, it may be coming "
                    "back: check again in a minute."
                )
            }
        return task.view()


    async def wait_for_tasks(
        task_ids: Annotated[list[str], Field(min_length=1)],
        timeout_seconds: Annotated[int, Field(ge=0)] = 60,
    ) -> dict[str, Any]:
        """Waits until all the given background tasks have ended, or the timeout passes,
        and returns each one's status and report. Use it when you need their reports and
        have nothing else to do meanwhile. You can wait again.

        Args:
          task_ids: Ids start_task returned.
          timeout_seconds: How long to wait at most, in seconds. A longer wait is cut to
            the service's limit; wait again for more.
        """
        scope = current_run_scope()
        if scope is None:
            return NO_RUN
        tasks = [scope.tasks.get(scope.key, task_id) for task_id in task_ids]
        known = [task for task in tasks if task is not None]
        timeout = max(0.0, min(float(timeout_seconds), limits.wait_max_seconds()))
        if known:
            await scope.tasks.wait(known, timeout)
        return {
            "tasks": [
                task.view() if task else {"task_id": task_id, "error": "No task with that id here."}
                for task_id, task in zip(task_ids, tasks)
            ]
        }

    return start_task, check_task, wait_for_tasks


class TaskToolset(BaseToolset):
    """The task tools, on a conversation's own runs only."""

    def __init__(self, limits: TaskToolLimits) -> None:
        super().__init__()
        self._limits = limits
        start_task, check_task, wait_for_tasks = task_tools(limits)
        self._tools: list[BaseTool] = [
            DescribedTool(start_task),
            DescribedTool(check_task),
            DescribedTool(wait_for_tasks),
        ]

    async def get_tools(self, readonly_context: Optional[ReadonlyContext] = None) -> list[BaseTool]:
        if readonly_context is None or not self._limits.enabled():
            return []
        if in_task(readonly_context.state):
            return []
        return list(self._tools)
