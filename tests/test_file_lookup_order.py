"""Finding a conversation file by the name a model gives it."""

import asyncio
from types import SimpleNamespace

from google.genai import types

from harness.files.lookup import load_original_or_named


class _Files:
    """A store of named files; with times, it also answers which version came last."""

    def __init__(self, names, times=None):
        self.parts = {name: types.Part.from_text(text=name) for name in names}
        self.times = times
        self.asked = []

    async def load_artifact(self, name):
        self.asked.append(name)
        return self.parts.get(name)


class _VersionedFiles(_Files):
    async def get_artifact_version(self, name):
        if name not in self.parts:
            return None
        return SimpleNamespace(create_time=self.times[name])


def _found(files, name):
    part = asyncio.run(load_original_or_named(files, name))
    return part.text if part is not None else None


def test_the_kept_original_comes_first_with_no_times_to_compare():
    assert _found(_Files(["original/a.xlsx", "a.xlsx"]), "a.xlsx") == "original/a.xlsx"
    assert _found(_Files(["a.xlsx"]), "original/a.xlsx") == "a.xlsx"


def test_the_agents_later_edit_of_an_upload_wins_over_the_upload():
    # Uploaded report.docx; run 1 edits it and shares report.docx; run 2 asks for it,
    # by either name. Before, the unedited upload came back.
    files = _VersionedFiles(
        ["original/report.docx", "report.docx"],
        {"original/report.docx": 100.0, "report.docx": 200.0},
    )

    assert _found(files, "output/report.docx") == "report.docx"
    assert _found(files, "report.docx") == "report.docx"
    assert _found(files, "original/report.docx") == "report.docx"


def test_the_persons_later_edit_of_an_upload_wins_over_the_agents_file():
    # The person's edit of an upload is saved under original/<name>, so it is the
    # newest of the two then.
    files = _VersionedFiles(
        ["original/report.docx", "report.docx"],
        {"original/report.docx": 300.0, "report.docx": 200.0},
    )

    assert _found(files, "report.docx") == "original/report.docx"


def test_a_file_named_under_output_is_found_by_its_own_name():
    # A run saved output/chart.png, a later call names it so.
    assert _found(_Files(["chart.png"]), "output/chart.png") == "chart.png"
    assert _found(_Files(["original/r.docx"]), "output/r.docx") == "original/r.docx"


def test_an_exact_match_wins_and_output_alone_names_nothing():
    files = _Files(["output/x.csv", "x.csv"])
    assert _found(files, "output/x.csv") == "output/x.csv"
    assert _found(_Files(["x.csv"]), "output/") is None


def test_the_prefix_is_tried_only_after_the_name_as_given():
    files = _Files([])
    _found(files, "output/c.png")
    assert files.asked == ["original/output/c.png", "output/c.png", "original/c.png", "c.png"]


def test_a_store_that_cant_say_keeps_the_original_first():
    class Broken(_Files):
        async def get_artifact_version(self, name):
            raise RuntimeError("down")

    assert _found(Broken(["original/a.docx", "a.docx"]), "a.docx") == "original/a.docx"


class _Session:
    """ToolContext's two artifact calls the lookup makes, over ADK's own in-memory store,
    which records each version's save time."""

    def __init__(self):
        from google.adk.artifacts import InMemoryArtifactService

        self.service = InMemoryArtifactService()
        self._scope = {"app_name": "app", "user_id": "u", "session_id": "s"}

    async def save(self, name, data):
        await self.service.save_artifact(**self._scope, filename=name, artifact=types.Part.from_text(text=data))
        await asyncio.sleep(0.01)  # distinct save times

    async def load_artifact(self, filename, version=None):
        return await self.service.load_artifact(**self._scope, filename=filename, version=version)

    async def get_artifact_version(self, filename, version=None):
        return await self.service.get_artifact_version(**self._scope, filename=filename, version=version)


def _latest(saves, name):
    async def run():
        session = _Session()
        for each, data in saves:
            await session.save(each, data)
        part = await load_original_or_named(session, name)
        return part.text if part is not None else None

    return asyncio.run(run())


def test_the_latest_save_wins_for_an_agent_reshare_and_for_a_persons_edit():
    # Agent re-share after the upload: the agent's copy.
    assert _latest([("original/report.docx", "upload"), ("report.docx", "agent edit")], "report.docx") == "agent edit"
    assert _latest([("original/report.docx", "upload"), ("report.docx", "agent edit")], "output/report.docx") == "agent edit"
    # The person's edit after the agent's copy, saved under original/.
    saves = [("original/report.docx", "upload"), ("report.docx", "agent copy"), ("original/report.docx", "person edit")]
    assert _latest(saves, "report.docx") == "person edit"


_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _latest_part(saves, name):
    async def run():
        session = _Session()
        for each, part in saves:
            await session.service.save_artifact(**session._scope, filename=each, artifact=part)
            await asyncio.sleep(0.01)
        return await load_original_or_named(session, name)

    return asyncio.run(run())


def test_an_uploads_text_copy_never_hides_its_original():
    # The web app keeps original/orders.xlsx, then ADK's SaveFilesAsArtifactsPlugin saves
    # the text the model was shown under orders.xlsx, a moment later. That copy is newer,
    # but code asking for the workbook needs the workbook, not a five-row preview.
    original = types.Part.from_bytes(data=b"PK workbook", mime_type=_XLSX)
    preview = types.Part.from_bytes(data=b'Spreadsheet "orders.xlsx": ...', mime_type="text/plain")
    saves = [("original/orders.xlsx", original), ("orders.xlsx", preview)]
    for name in ("orders.xlsx", "original/orders.xlsx", "output/orders.xlsx"):
        part = _latest_part(saves, name)
        assert part is not None and part.inline_data.data == b"PK workbook", name


def test_the_agents_later_workbook_still_wins_over_the_upload_and_its_copy():
    saves = [
        ("original/orders.xlsx", types.Part.from_bytes(data=b"PK upload", mime_type=_XLSX)),
        ("orders.xlsx", types.Part.from_bytes(data=b"preview", mime_type="text/plain")),
        ("orders.xlsx", types.Part.from_bytes(data=b"PK agent edit", mime_type=_XLSX)),
    ]
    part = _latest_part(saves, "orders.xlsx")
    assert part is not None and part.inline_data.data == b"PK agent edit"
