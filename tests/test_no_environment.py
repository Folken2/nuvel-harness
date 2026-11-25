"""The harness reads no environment. A product reads its settings and passes them in, so one
process can hold two configurations and a test never depends on the shell it runs in."""

import ast
import pathlib

import harness

HARNESS = pathlib.Path(harness.__file__).parent


def _reads(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ("environ", "environb", "getenv", "putenv"):
            lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module == "os":
            if any(alias.name in ("environ", "environb", "getenv", "putenv") for alias in node.names):
                lines.append(node.lineno)
    return lines


def test_no_module_reads_the_environment():
    found = [
        f"{path.relative_to(HARNESS.parent)}:{line}"
        for path in sorted(HARNESS.rglob("*.py"))
        for line in _reads(ast.parse(path.read_text()))
    ]
    assert found == []


def test_the_check_sees_a_read():
    assert _reads(ast.parse("import os\nos.environ['A']\nos.getenv('B')\nfrom os import environ"))
