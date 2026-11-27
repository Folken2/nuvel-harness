"""The runs in progress, and the events each has produced so far.

ASSUMPTION: exactly one instance of this service per deployment (a design
decision). This dict is the only record of which runs are live, so a second instance
would neither see another's runs nor stop them. To run more than one, this module is
what changes: route by session, or move the frames to a shared bus.

Memory only: a restart loses the frames, though not the completed steps, which ADK
keeps in the session store and ./resume.py rebuilds a run from. A finished run stays for FINISHED_KEPT_SECONDS so a person
who reopens the page just after it ended, and the web app's save, can still read it.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field

from .steering import RunInbox, Steer, steer_frame

# How a background task's session is named after its parent's (./tasks.py builds them).
TASK_SESSION_MARK = ".task-"

logger = logging.getLogger(__name__)

# Long enough for the web app's steps before a run: a web app that died in them can't hold
# the session for longer than this.
RESERVED_SECONDS = 180.0
FINISHED_KEPT_SECONDS = 600.0
# So a quiet stretch (a long tool, a slow model) doesn't look like a dead connection.
KEEPALIVE_SECONDS = 15.0
KEEPALIVE_FRAME = ": keepalive\n\n"

RunKey = tuple[str, str, str]  # app, user, session


class RunStatus(str, enum.Enum):
    RESERVED = "reserved"
    RUNNING = "running"
    FINISHED = "finished"
    STOPPED = "stopped"
    FAILED = "failed"

    @property
    def active(self) -> bool:
        return self in (RunStatus.RESERVED, RunStatus.RUNNING)


class RunActive(Exception):
    """The session already has a run: two at once would interleave its history."""


class RunsFull(Exception):
    """As many runs are live as the service admits: a new one waits for one to end."""


class RunNotReserved(Exception):
    """Nothing is waiting to start under that id: stopped, or gone."""


@dataclass
class Run:
    id: str
    status: RunStatus = RunStatus.RESERVED
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    frames: list[str] = field(default_factory=list)
    stop_requested: asyncio.Event = field(default_factory=asyncio.Event)
    _changed: asyncio.Event = field(default_factory=asyncio.Event)
    _when_ended: list[Callable[[], None]] = field(default_factory=list)
    # What the person sent to this run while it works (./steering.py).
    inbox: RunInbox = field(init=False)

    def __post_init__(self) -> None:
        self.inbox = RunInbox(on_taken=self._took)

    def _took(self, steer: Steer) -> None:
        self.add_frame(steer_frame(steer))

    def add_frame(self, frame: str) -> None:
        self.frames.append(frame)
        self._wake()

    def end(self, status: RunStatus) -> None:
        if not self.status.active:
            return
        self.status = RunStatus.STOPPED if self.stop_requested.is_set() else status
        self.ended_at = time.time()
        self.inbox.close()
        self._wake()
        when_ended, self._when_ended = self._when_ended, []
        for callback in when_ended:
            callback()

    def when_ended(self, callback: Callable[[], None]) -> None:
        """Calls back once the run has ended: now, if it already has."""
        if self.status.active:
            self._when_ended.append(callback)
        else:
            callback()

    def _wake(self) -> None:
        changed, self._changed = self._changed, asyncio.Event()
        changed.set()

    async def frames_from(self, after: int) -> AsyncIterator[str]:
        """Every frame past `after`, live until the run ends. Reading changes nothing."""
        index = max(after, 0)
        while True:
            changed = self._changed
            while index < len(self.frames):
                yield self.frames[index]
                index += 1
            if not self.status.active:
                return
            try:
                await asyncio.wait_for(changed.wait(), KEEPALIVE_SECONDS)
            except TimeoutError:
                yield KEEPALIVE_FRAME


def _of_session(key_session: str, session: str | None) -> bool:
    """The session itself, its background tasks' sessions (./tasks.py), or any with None."""
    return session is None or key_session == session or key_session.startswith(f"{session}{TASK_SESSION_MARK}")


def _is_task(key: RunKey) -> bool:
    return TASK_SESSION_MARK in key[2]


class RunRegistry:
    """One run per session. See the module docstring for why this can be a dict.

    ``max_live_runs`` caps the runs a person started that are live at once (reserved or
    running), so a burst can't take the service past what it was sized for: a reserve
    past it raises RunsFull. Background tasks don't count and aren't refused, since
    their parent was admitted and the task board limits them (./tasks.py). A run back
    after a restart counts but is never refused: it was admitted before. None: no cap.
    """

    def __init__(self, max_live_runs: int | None = None) -> None:
        if max_live_runs is not None and max_live_runs < 1:
            raise ValueError(f"max_live_runs must be at least 1, got {max_live_runs}")
        self._runs: dict[RunKey, Run] = {}
        self._max_live_runs = max_live_runs

    def live_runs(self) -> int:
        """The runs counted against ``max_live_runs``: live, and not a background task."""
        return sum(1 for key, run in self._runs.items() if run.status.active and not _is_task(key))

    def reserve(self, key: RunKey) -> Run:
        """Claims the session before the web app's own steps, so a Stop has a target."""
        self._forget_lapsed()
        existing = self._runs.get(key)
        if existing and existing.status.active:
            raise RunActive()
        if (
            self._max_live_runs is not None
            and not _is_task(key)
            and self.live_runs() >= self._max_live_runs
        ):
            raise RunsFull()
        run = Run(id=str(uuid.uuid4()))
        self._runs[key] = run
        return run

    def restore(self, key: RunKey, started_at: float) -> Run:
        """A run back after a restart: as it began, already under way."""
        self._forget_lapsed()
        existing = self._runs.get(key)
        if existing and existing.status.active:
            raise RunActive()
        run = Run(id=str(uuid.uuid4()), status=RunStatus.RUNNING, started_at=started_at)
        self._runs[key] = run
        return run

    def forget(self, key: RunKey, run: Run) -> None:
        if self._runs.get(key) is run:
            del self._runs[key]

    @contextlib.contextmanager
    def until_driven(self, key: RunKey, run: Run, *, restored: bool = False) -> Iterator[None]:
        """The steps between a run being RUNNING and its driver having it. Nothing else
        ends a RUNNING run, so one left by an error here (a cancelled request too) would
        hold the session until a restart: it ends failed instead. A restored one is
        forgotten as well, so the session is as the restart left it and can be resumed."""
        try:
            yield
        except BaseException:
            run.end(RunStatus.FAILED)
            if restored:
                self.forget(key, run)
            raise

    def get(self, key: RunKey) -> Run | None:
        self._forget_lapsed()
        return self._runs.get(key)

    def begin(self, key: RunKey, run_id: str) -> Run:
        run = self.get(key)
        if not run or run.id != run_id or run.status is not RunStatus.RESERVED:
            raise RunNotReserved()
        run.status = RunStatus.RUNNING
        return run

    def steer(self, key: RunKey, steer: Steer) -> bool:
        """Whether the session has a run that may still take it."""
        run = self.get(key)
        return bool(run and run.status.active and run.inbox.put(steer))

    def stop(self, key: RunKey) -> bool:
        """Whether there was a run to stop. A reserved one ends here; a running one
        ends when its driver has unwound, which the driver reports."""
        run = self.get(key)
        if not run or not run.status.active:
            return False
        run.stop_requested.set()
        if run.status is RunStatus.RESERVED:
            run.end(RunStatus.STOPPED)
        return True

    def stop_all_of(self, app: str, user: str, session: str | None = None) -> None:
        """A delete: what's about to be erased must not keep running, nor stay here to be
        read. Its frames go as soon as it has ended, not FINISHED_KEPT_SECONDS later."""
        for key, run in list(self._runs.items()):
            if key[0] == app and key[1] == user and _of_session(key[2], session):
                self.stop(key)
                run.when_ended(lambda key=key, run=run: self._drop(key, run))

    def _drop(self, key: RunKey, run: Run) -> None:
        # A viewer still reading holds the run, so its frames go too, not only the entry.
        run.frames.clear()
        self.forget(key, run)

    def _forget_lapsed(self) -> None:
        now = time.time()
        for key, run in list(self._runs.items()):
            if run.status is RunStatus.RESERVED and now - run.started_at > RESERVED_SECONDS:
                logger.warning("run %s was reserved and never started", run.id)
                run.end(RunStatus.FAILED)
            if run.ended_at is not None and now - run.ended_at > FINISHED_KEPT_SECONDS:
                del self._runs[key]
