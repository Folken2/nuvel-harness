"""Text in this repo is for outside readers. A rule id or ticket from a product's own design
documents or tracker means nothing here, so none may appear in the source, tests, examples,
docs, packaging or CI."""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKED = ("src", "tests", "examples", "docs", ".github")
SUFFIXES = {".py", ".pyi", ".md", ".toml", ".yml", ".yaml", ".txt", ".json", ".sh", ".cfg"}
NAMED = {"NOTICE", "LICENSE"}

# Ticket and rule ids (AB-12, KEY-3400, T-3), section citations and decision records.
# Standards that look like ids (UTF-8, SHA-256, RFC-6749) are text, not references.
OUTSIDE = re.compile(
    r"\b(?!(?:UTF|SHA|ISO|RFC|CVE|HTTP|TLS|PEP)-)[A-Z]{2,5}-\d+\b"
    r"|\b[TD]-\d+\b"
    r"|\bowner decision\b"
    r"|\bSection \d+(?:\.\d+)*,"
)


def _found(text: str) -> list[str]:
    return [match.group(0) for match in OUTSIDE.finditer(text)]


def _text(path: pathlib.Path) -> bool:
    return path.is_file() and (path.suffix in SUFFIXES or path.name in NAMED)


def _files() -> list[pathlib.Path]:
    files = [path for path in ROOT.iterdir() if _text(path)]
    for folder in CHECKED:
        files += [path for path in (ROOT / folder).rglob("*") if _text(path)]
    return [path for path in files if path.resolve() != pathlib.Path(__file__).resolve()]


def test_no_text_cites_an_outside_document():
    found = [
        f"{path.relative_to(ROOT)}: {hit}"
        for path in sorted(_files())
        for hit in _found(path.read_text(encoding="utf-8"))
    ]
    assert found == []


def test_the_check_sees_a_citation():
    text = '"""Settled once (AB-12, KEY-3400, T-3, owner decision 2). See Section 4.1, "x"."""'
    assert _found(text) == ["AB-12", "KEY-3400", "T-3", "owner decision", "Section 4.1,"]


def test_the_check_leaves_standards_and_plain_text_alone():
    text = "UTF-8 text, SHA-256, RFC-6749, HTTP/1.1, a 429, contract C-1182, Apache-2.0"
    assert _found(text) == []


def test_the_check_reads_packaging_and_ci():
    names = {path.relative_to(ROOT).as_posix() for path in _files()}
    assert {"NOTICE", "pyproject.toml", ".github/workflows/tests.yml"} <= names
