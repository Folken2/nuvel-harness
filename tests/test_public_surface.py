"""The harness's public surface: the names its packages list in ``__all__``.

A product imports only those (it holds that line in its own tests). Here: every listed name
resolves, no re-export hides a module, every public function and method is fully typed, and
every module imports first, in a fresh process.
"""

import ast
import importlib
import inspect
import pathlib
import subprocess
import sys
import types

import harness

HARNESS = pathlib.Path(harness.__file__).parent


def _public(module: types.ModuleType) -> set[str]:
    return set(getattr(module, "__all__", ()))


def _attribute(module: types.ModuleType, name: str) -> object:
    """``module.name``, importing it when it is a submodule not loaded yet."""
    if name in vars(module):
        return vars(module)[name]
    try:
        return importlib.import_module(f"{module.__name__}.{name}")
    except ModuleNotFoundError:
        return getattr(module, name)


def _public_objects() -> tuple[dict[str, object], list[str]]:
    """Every public name, all the way down from ``harness``, and the listed names that don't resolve."""
    found: dict[str, object] = {}
    missing = []
    pending = ["harness"]
    while pending:
        module = importlib.import_module(pending.pop())
        for name in sorted(_public(module)):
            try:
                value = _attribute(module, name)
            except (ImportError, AttributeError):
                missing.append(f"{module.__name__}.{name}")
                continue
            found[f"{module.__name__}.{name}"] = value
            if isinstance(value, types.ModuleType) and value.__name__ == f"{module.__name__}.{name}":
                pending.append(value.__name__)
    return found, missing


def test_every_public_name_is_there():
    """A name listed in ``__all__`` resolves, all the way down from ``harness``."""
    found, missing = _public_objects()
    assert len(found) >= 150
    assert missing == []


def _untyped(function: types.FunctionType) -> list[str]:
    signature = inspect.signature(function)
    gaps = [
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.name not in ("self", "cls") and parameter.annotation is parameter.empty
    ]
    if signature.return_annotation is signature.empty and function.__name__ != "__init__":
        gaps.append("return")
    return gaps


def test_the_public_surface_is_typed():
    """Each public function, and each public method of a public class, says its parameters'
    types and what it returns. A dataclass field's default (a lambda) is typed by its field."""
    untyped = {}
    for dotted, value in _public_objects()[0].items():
        if inspect.isfunction(value) and value.__module__.startswith("harness."):
            functions = {dotted: value}
        elif inspect.isclass(value) and value.__module__.startswith("harness."):
            functions = {
                f"{dotted}.{name}": member.__func__ if isinstance(member, (staticmethod, classmethod)) else member
                for name, member in vars(value).items()
                if not name.startswith("_") or name == "__init__"
            }
        else:
            continue
        for name, function in functions.items():
            ours = inspect.isfunction(function) and (function.__module__ or "").startswith("harness.")
            if ours and function.__name__ != "<lambda>" and (gaps := _untyped(function)):
                untyped[name] = gaps
    assert untyped == {}


def test_no_public_name_hides_a_module():
    """``from .overage import overage`` in a package file would make ``harness.ledger.overage``
    the function for some importers and the module for others. A function named like its
    module is reached through the module: the package lists the module."""
    hidden = []
    for init in sorted(HARNESS.rglob("__init__.py")):
        for node in ast.parse(init.read_text()).body:
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                names = {path.stem for path in init.parent.glob("*.py")} | {
                    path.parent.name for path in init.parent.glob("*/__init__.py")
                }
                hidden += [f"{init.relative_to(HARNESS.parent)}: {a.name}" for a in node.names if a.name in names]
    assert hidden == []


# Run in a child process: third-party libraries load once, then each harness module is
# imported first in its own fork, with no harness module loaded. A package file that
# re-exports eagerly can close a cycle only one import order sees.
_EACH_ALONE = """
import importlib, os, sys
for library in ("google.adk", "google.genai", "litellm", "fastapi", "sqlalchemy", "httpx", "mcp", "pydantic"):
    importlib.import_module(library)
failed, running, pending = [], 0, list(sys.argv[1:])
while pending or running:
    while pending and running < (os.cpu_count() or 2):
        name = pending.pop()
        if os.fork() == 0:
            try:
                importlib.import_module(name)
            except BaseException as error:
                print(f"{name}: {type(error).__name__}: {error}", flush=True)
                os._exit(1)
            os._exit(0)
        running += 1
    _, status = os.wait()
    running -= 1
    failed += [status] if status else []
sys.exit(1 if failed else 0)
"""


def test_every_harness_module_imports_first():
    names = sorted(
        ".".join(("harness", *path.relative_to(HARNESS).with_suffix("").parts)).removesuffix(".__init__")
        for path in HARNESS.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    assert len(names) >= 130
    done = subprocess.run(
        [sys.executable, "-c", _EACH_ALONE, *names], capture_output=True, text=True, timeout=600
    )
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-2000:]
