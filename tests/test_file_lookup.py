"""An upload's edits reach the agent: the web app saves a person's
edit of an upload as the next version of ``original/<name>``, and the lookup the
agent's tools use reads the newest version of that name first."""

from __future__ import annotations

import asyncio

from google.adk.artifacts import InMemoryArtifactService
from google.genai import types

from harness.files.lookup import load_original_or_named


class _Context:
    """The one call of ToolContext the lookup makes, over a real artifact service."""

    def __init__(self, service: InMemoryArtifactService) -> None:
        self._service = service

    async def load_artifact(self, filename: str, version: int | None = None):
        return await self._service.load_artifact(
            app_name="app", user_id="u", session_id="s", filename=filename, version=version
        )


def _docx(data: bytes) -> types.Part:
    return types.Part.from_bytes(
        data=data,
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


def test_an_edited_upload_reads_as_its_newest_original() -> None:
    async def run() -> types.Part | None:
        service = InMemoryArtifactService()
        for name, data in (
            ("original/report.docx", b"as uploaded"),
            ("report.docx", b"an agent copy"),
            ("original/report.docx", b"as the person edited it"),
        ):
            await service.save_artifact(
                app_name="app", user_id="u", session_id="s", filename=name, artifact=_docx(data)
            )
        return await load_original_or_named(_Context(service), "report.docx")  # type: ignore[arg-type]

    part = asyncio.run(run())
    assert part is not None and part.inline_data is not None
    assert part.inline_data.data == b"as the person edited it"
