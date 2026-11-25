"""The web app's calls for a session's run."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Collection

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from google.adk.sessions import BaseSessionService

from .. import wire_names
from .driver import RunStart, drive_run
from .regenerate import RewindFailed, message_to_retry, rewind_for_regenerate
from .registry import Run, RunActive, RunKey, RunNotReserved, RunRegistry, RunStatus
from .resume import AlreadyEnded, NotResumed, past_frames, read_history, restore_to_resume
from .run_scope import RunScope, reset_run_scope, set_run_scope
from .steering import Steer
from .tasks import TaskBoard

BASE = "/runs/apps/{app_name}/users/{user_id}/sessions/{session_id}"
# What one "Send now" may carry. The web app keeps a message shorter than this.
STEER_MAX_CHARS = 32_000
# Only the caller's identity goes through: drive_run sets the rest of the request.
FORWARDED_HEADERS = frozenset({b"authorization"})

# The tasks need a strong reference, or they can be collected mid-run.
_driving: set[asyncio.Task[None]] = set()


def _describe(run: Run) -> dict:
    return {
        "runId": run.id,
        "status": run.status.value,
        "startedAt": run.started_at,
        "endedSecondsAgo": None if run.ended_at is None else time.time() - run.ended_at,
    }


async def _sse(run: Run, after: int) -> AsyncIterator[str]:
    async for frame in run.frames_from(after):
        yield frame
    # Last, so a reader can tell a run that ended from a connection that dropped.
    end = {wire_names.RUN_FRAME: {"status": run.status.value}}
    yield f"data: {json.dumps(end)}\n\n"


def _forwarded(request: Request) -> list[tuple[bytes, bytes]]:
    return [h for h in request.scope["headers"] if h[0] in FORWARDED_HEADERS]


async def _drive(
    app: FastAPI, run: Run, body: bytes, request: Request, scope: RunScope
) -> Response:
    start = RunStart()
    # The run's tools find their run here (./run_scope.py): the task copies the context.
    token = set_run_scope(scope)
    try:
        task = asyncio.create_task(drive_run(app, run, body, _forwarded(request), start))
    finally:
        reset_run_scope(token)
    _driving.add(task)
    task.add_done_callback(_driving.discard)

    await start.answered.wait()
    if start.status != 200:
        return Response(
            content=start.detail, status_code=start.status or 500, media_type="text/plain"
        )
    return JSONResponse(_describe(run), status_code=202)


def add_run_routes(
    app: FastAPI,
    registry: RunRegistry,
    sessions: BaseSessionService,
    tasks: TaskBoard | None = None,
    safe_to_repeat: Collection[str] = (),
) -> None:
    """Without a board the service has no background tasks: none can start.
    ``safe_to_repeat`` names the tools a resume asks the model to call again (resume.py)."""
    tasks = tasks or TaskBoard(registry, max_running=0)
    tasks.bind(app)

    @app.post(BASE)
    async def reserve_run(app_name: str, user_id: str, session_id: str) -> Response:
        try:
            run = registry.reserve((app_name, user_id, session_id))
        except RunActive:
            return JSONResponse({"error": "run_active"}, status_code=409)
        return JSONResponse(_describe(run), status_code=201)

    @app.post(f"{BASE}/start")
    async def start_run(
        app_name: str, user_id: str, session_id: str, runId: str, request: Request
    ) -> Response:
        key: RunKey = (app_name, user_id, session_id)
        # The auth middleware has already bound this body to the token.
        body = await request.body()
        try:
            run = registry.begin(key, runId)
        except RunNotReserved:
            return JSONResponse({"error": "run_stopped"}, status_code=409)
        with registry.until_driven(key, run):
            # After begin, so a run that was stopped first never loses the turn it repeats.
            message_id = message_to_retry(body)
            if message_id:
                try:
                    await rewind_for_regenerate(
                        app, app_name, user_id, session_id, message_id, _forwarded(request)
                    )
                except RewindFailed as failed:
                    run.end(RunStatus.FAILED)
                    return Response(
                        content=f"rewind: {failed.status} {failed.detail}",
                        status_code=502,
                        media_type="text/plain",
                    )
        return await _drive(app, run, body, request, RunScope(key, tasks))

    @app.post(f"{BASE}/resume")
    async def resume_run(
        app_name: str,
        user_id: str,
        session_id: str,
        since: float,
        afterSteps: int,
        request: Request,
    ) -> Response:
        """Picks up the run started at `since` after a restart. `afterSteps` is how far it
        had got the last time it was resumed: without new steps it's a crash loop."""
        key: RunKey = (app_name, user_id, session_id)
        # The auth middleware has already bound this body to the token.
        body = json.loads(await request.body())
        try:
            run, restored = await restore_to_resume(
                registry, sessions, key, since, afterSteps, body, safe_to_repeat
            )
        except RunActive:
            return JSONResponse({"error": "run_active"}, status_code=409)
        if isinstance(restored, NotResumed):
            return JSONResponse({"error": restored.error}, status_code=409)
        if isinstance(restored, AlreadyEnded):
            return JSONResponse({**_describe(run), "steps": restored.steps})
        answer = await _drive(app, run, restored.body, request, RunScope(key, tasks))
        if answer.status_code != 202:
            return answer
        # A background task's run: followed again, so its report reaches its parent.
        tasks.adopt_from(key, run, restored.state)
        return JSONResponse({**_describe(run), "steps": restored.steps}, status_code=202)

    @app.post(f"{BASE}/replay")
    async def replay_run(
        app_name: str, user_id: str, session_id: str, since: float, status: str
    ) -> Response:
        """What a run did before a restart, without running any more of it: it was stopped
        during the outage, or it isn't getting anywhere."""
        if status not in (RunStatus.STOPPED.value, RunStatus.FAILED.value):
            return JSONResponse({"error": "invalid_status"}, status_code=400)
        key: RunKey = (app_name, user_id, session_id)
        try:
            run = registry.restore(key, since)
        except RunActive:
            return JSONResponse({"error": "run_active"}, status_code=409)
        with registry.until_driven(key, run, restored=True):
            session = await sessions.get_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
            history = read_history(session.events, since) if session else None
            for frame in past_frames(history) if history else ():
                run.add_frame(frame)
            run.end(RunStatus(status))
        return JSONResponse(_describe(run))

    @app.get(BASE)
    async def run_status(app_name: str, user_id: str, session_id: str) -> Response:
        run = registry.get((app_name, user_id, session_id))
        if not run:
            return JSONResponse({"error": "no_run"}, status_code=404)
        return JSONResponse(_describe(run))

    @app.get(f"{BASE}/events")
    async def run_events(
        app_name: str, user_id: str, session_id: str, after: int = 0
    ) -> Response:
        run = registry.get((app_name, user_id, session_id))
        if not run:
            return JSONResponse({"error": "no_run"}, status_code=404)
        return StreamingResponse(_sse(run, after), media_type="text/event-stream")

    @app.post(f"{BASE}/steer")
    async def steer_run(
        app_name: str, user_id: str, session_id: str, request: Request
    ) -> Response:
        """A message for the run going on, taken at its next model call. Accepted is
        not taken: the run may end first, and the web app's queue still has it."""
        try:
            body = json.loads(await request.body())
            steer = Steer(id=body["id"], text=body["text"])
        except (ValueError, KeyError, TypeError):
            return JSONResponse({"error": "invalid_body"}, status_code=400)
        if (
            not isinstance(steer.id, str)
            or not isinstance(steer.text, str)
            or not steer.id
            or not steer.text.strip()
            or len(steer.text) > STEER_MAX_CHARS
        ):
            return JSONResponse({"error": "invalid_body"}, status_code=400)
        if not registry.steer((app_name, user_id, session_id), steer):
            return JSONResponse({"error": "no_run"}, status_code=409)
        return JSONResponse({"accepted": True}, status_code=202)

    @app.post(f"{BASE}/stop")
    async def stop_run(app_name: str, user_id: str, session_id: str) -> Response:
        """The conversation's run and its background tasks (./tasks.py)."""
        key: RunKey = (app_name, user_id, session_id)
        stopped = registry.stop(key)
        stopped_tasks = tasks.stop_children(key)
        return JSONResponse({"stopped": stopped or stopped_tasks > 0})
