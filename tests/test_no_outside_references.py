"""Text in this repo is for outside readers. A rule id or ticket from a product's own design
documents means nothing here, so none may appear in the source, tests, examples or docs."""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKED = ("src", "tests", "examples", "docs")
SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".txt"}

# Rule and ticket ids (UG-14, LM-29, T-2), section citations and decision records.
OUTSIDE = re.compile(r"\b(?:UG|LM|MC|AT|T|D)-\d+\b|\bowner decision\b|\bSection \d+,")


def _found(text: str) -> list[str]:
    return [match.group(0) for match in OUTSIDE.finditer(text)]


def _files() -> list[pathlib.Path]:
    files = [path for path in ROOT.glob("*.md")]
    for folder in CHECKED:
        files += [path for path in (ROOT / folder).rglob("*") if path.suffix in SUFFIXES]
    return [path for path in files if path.resolve() != pathlib.Path(__file__).resolve()]


def test_no_text_cites_an_outside_document():
    found = [
        f"{path.relative_to(ROOT)}: {hit}"
        for path in sorted(_files())
        for hit in _found(path.read_text(encoding="utf-8"))
    ]
    assert found == []


def test_the_check_sees_a_citation():
    assert _found('"""Settled once (UG-14, owner decision 2). See Section 2, "settlement"."""') == [
        "UG-14",
        "owner decision",
        "Section 2,",
    ]
    assert _found("a 429 from the server, HTTP/1.1, utf-8, sha-256") == []
