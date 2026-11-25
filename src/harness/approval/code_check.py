"""Whether code the agent wrote only computes, read from its syntax tree.

The classifier's verdict on run_code is advice (policy.second_check): code it may
let run unasked must also pass this. It fails closed, in two layers:

- What code names from a module must be on the list it is checked against
  (``Allowlist``; this service's is code_allowlist.py): every
  import, and every attribute chain that starts at an imported name, is resolved to
  a dotted path (`np.linalg.inv` is `numpy.linalg.inv`) and checked. A module is
  never used whole (passed around, it could reach anything on it).
- On any other object, names that reach Python's machinery or run something are
  refused (dunders, private names, `exec`, `getattr`, `.query`, `.load`...), since
  what an object is can't be read from the text.

The sandbox, with no network, stays the boundary; this is the check before it.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["Allowlist", "only_computes"]


@dataclass(frozen=True)
class Allowlist:
    """What code may name from a module, as two questions."""

    # Whether this dotted path, as its parts, may be named at all.
    path_allowed: Callable[[list[str]], bool]
    # Whether this dotted path is a module, which may never be used whole.
    is_module: Callable[[str], bool]

# Builtins that reach Python's machinery, so what runs can't be read from the text.
_BUILTINS = frozenset(
    {
        "__import__", "exec", "eval", "compile", "getattr", "setattr", "delattr", "globals",
        "locals", "vars", "breakpoint", "memoryview", "__builtins__",
        # A class from its instance (`type(archive)(...)`), and a parent's methods.
        "type", "super",
    }
)
# File methods that delete or relink, whatever object they're called on.
_DELETES = frozenset({"unlink", "rmdir", "rmtree", "symlink_to", "hardlink_to", "chmod", "chown"})


# Attribute names that reach a module's machinery or run something, whatever object
# holds them: modules import `os` and friends and keep them as attributes
# (`pathlib.os`, `tempfile._os`), and some library calls evaluate a string as
# Python (`pd.eval`, `DataFrame.query`, `sympify`) or unpickle a file. Matched with
# leading underscores dropped.
_REACHES = frozenset(
    {
        "os", "sys", "subprocess", "shutil", "builtins", "importlib", "ctypes", "socket",
        "pty", "posix", "nt", "posixsubprocess", "multiprocessing", "pickle", "marshal",
        "runpy", "system", "popen", "startfile", "fork", "forkpty", "kill", "killpg",
        "check_output", "check_call", "getoutput", "getstatusoutput", "attrgetter",
        "methodcaller", "eval", "query", "sympify", "parse_expr", "read_pickle", "run",
        "call", "Popen", "import_module", "import_optional_dependency", "ctypeslib",
        "load_library", "formula", "from_formula", "read_array", "open_memmap",
        "read_hdf", "HDFStore",
        # Python's machinery under an attribute name.
        "getattr", "setattr", "delattr", "exec", "globals", "locals", "vars", "inspect",
        "bltns", "get_type_hints", "get_annotations", "ForwardRef", "evaluate_forward_ref",
        "singledispatch", "singledispatchmethod", "make_dataclass", "get_field", "vformat",
        # numpy's archive class unpickles when told to positionally, and its file
        # opener fetches; the module holding both. A formatter reads attributes.
        "npyio", "NpzFile", "DataSource", "Formatter",
        # Frames, code and tracebacks, which hold a running function's globals.
        "f_back", "f_builtins", "f_globals", "f_locals", "f_code", "gi_frame", "gi_code",
        "cr_frame", "cr_code", "ag_frame", "ag_code", "tb_frame", "tb_next",
        # A plotting backend is a module imported by name.
        "backend", "switch_backend", "rc_context", "rc_file",
    }
)
# Prefixes of `os` functions that replace or start a process (execv, spawnl,
# posix_spawn), for an `os` reached under another name. Not `execute`: DuckDB's.
_PROCESS_PREFIXES = ("exec", "spawn", "posix_spawn")
_NOT_PROCESS = frozenset({"execute", "executemany"})
# Private attributes are where modules keep what they import (`tempfile._os`,
# `pandas.compat._optional`), so none may be named but namedtuple's public API.
_PRIVATE_OK = frozenset({"_asdict", "_replace", "_fields", "_make", "_field_defaults"})
# Keyword arguments that turn loading data into running code. `np.load` takes it
# third, so `load` may only be called directly on one file (see only_computes).
_KEYWORDS = frozenset({"allow_pickle", "backend"})
# A format string reads attributes of its arguments (`"{0.__class__}"`), so it may
# only be one written in the code, which the dunder check reads. One built at run
# time could spell a dunder that check never sees (`"_" * 2`).
_FORMATS = frozenset({"format", "format_map"})
# Calls that copy attributes named by strings (functools.wraps, update_wrapper).
_COPIES_NAMED = frozenset({"wraps", "update_wrapper"})
# Loaders that unpickle when called on an object (a results class's `.load(path)`).
# Module paths that are listed (`np.load`, `json.loads`) are checked by the list.
# `compile` too. Allowed only at these exact paths: a class reached through a listed
# module can still unpickle (`ARIMAResults.load(path)`).
_OBJECT_LOADS = frozenset({"load", "loads", "compile"})
_SAFE_LOADS = frozenset({"numpy.load", "json.load", "json.loads", "re.compile"})
# The backends `matplotlib.use` may name, each literally.
_BACKENDS = frozenset({"Agg", "agg", "pdf", "svg", "ps", "cairo"})


# The one script idiom that names a dunder: `if __name__ == "__main__":`.
_IDIOM = frozenset({"__name__", "__main__"})


def _dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__") and name not in _IDIOM


def _name_reaches(name: str) -> bool:
    private = name.startswith("_") and name not in _PRIVATE_OK
    process = name.startswith(_PROCESS_PREFIXES) and name not in _NOT_PROCESS
    pickles = "pickle" in name.lower()
    return private or process or pickles or _dunder(name) or name in _DELETES or name.lstrip("_") in _REACHES


def _reaches_out(node: ast.AST) -> bool:
    """A node that leaves pure computation, whatever module it's in."""
    if isinstance(node, ast.Name):
        return node.id in _BUILTINS or _dunder(node.id)
    if isinstance(node, ast.Attribute):
        literal = isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        return _name_reaches(node.attr) or (node.attr in _FORMATS and not literal)
    if isinstance(node, ast.keyword):
        return node.arg in _KEYWORDS
    if isinstance(node, ast.MatchClass):
        # `case object(__globals__=g)` reads an attribute named by a plain string,
        # which no Attribute or Name node carries.
        return any(_name_reaches(attr) or attr in _FORMATS for attr in node.kwd_attrs)
    if isinstance(node, ast.Call):
        # `**` unpacking hides which keywords are passed (a `backend=`). `{**x}`, a
        # dict literal, still merges dicts.
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
        # `wraps` copies the attributes its `assigned` and `updated` name, as strings
        # that may be built at run time: only the function it wraps may be passed.
        copies = name in _COPIES_NAMED and (len(node.args) > 1 or bool(node.keywords))
        return copies or any(keyword.arg is None for keyword in node.keywords)
    if isinstance(node, (ast.arg, ast.AnnAssign, ast.FunctionDef, ast.AsyncFunctionDef)):
        # A string annotation is evaluated by whatever reads annotations.
        annotation = node.returns if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else node.annotation
        return isinstance(annotation, ast.Constant) and isinstance(annotation.value, str)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        # A dunder anywhere in a string, for anything that formats or evaluates it
        # (`"{0.__class__}".format`). A URL: pandas and others read from one without
        # importing a network module. "backend": a key that names a module to import.
        text = node.value
        # The pickle switch by name, for anything that unpacks a dict into a call.
        lowered = text.lower()
        return ("__" in text and text not in _IDIOM) or "://" in text or "backend" in lowered or "pickle" in lowered
    return False


def _bind_imports(node: ast.AST, bound: dict[str, str], allowlist: Allowlist) -> bool:
    """Imports must be on the list; the names they bind are collected with the
    dotted path each stands for."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            parts = alias.name.split(".")
            if not _parts_allowed(parts, allowlist):
                return False
            # `import a.b` binds `a`; `import a.b as c` binds `c` to `a.b`.
            if alias.asname:
                bound[alias.asname] = alias.name
            else:
                bound[parts[0]] = parts[0]
    elif isinstance(node, ast.ImportFrom):
        # `*` would bind names this check never sees; `load` only as `x.load(file)`.
        if node.level or not node.module:
            return False
        for alias in node.names:
            # `rcParams` only as an attribute, where its keys are checked.
            if alias.name in ("*", "load", "rcParams"):
                return False
            path = f"{node.module}.{alias.name}"
            if not _parts_allowed(path.split("."), allowlist):
                return False
            bound[alias.asname or alias.name] = path
    return True


def _parts_allowed(parts: list[str], allowlist: Allowlist) -> bool:
    # Past the package, no part may reach out (`os.path.os`, `x.compat._optional`),
    # even where the list allows a whole module.
    return allowlist.path_allowed(parts) and not any(_name_reaches(part) for part in parts[1:])


def _chain(node: ast.Name, parents: dict[int, ast.AST]) -> list[ast.Attribute]:
    """The attribute accesses stacked on a name: `np.linalg.inv` from `np`."""
    chain: list[ast.Attribute] = []
    current: ast.AST = node
    parent = parents.get(id(current))
    while isinstance(parent, ast.Attribute) and parent.value is current:
        chain.append(parent)
        current = parent
        parent = parents.get(id(current))
    return chain


def _literal_backend(node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    """`matplotlib.use("Agg")`: called directly, on a backend named literally."""
    call = parents.get(id(node))
    if not isinstance(call, ast.Call) or call.func is not node or len(call.args) != 1:
        return False
    arg = call.args[0]
    literal_keywords = all(isinstance(keyword.value, ast.Constant) for keyword in call.keywords)
    return isinstance(arg, ast.Constant) and arg.value in _BACKENDS and literal_keywords


def _literal_key(node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    """`rcParams["font.size"]`: subscripted with a literal key (never the backend,
    which the string check refuses)."""
    subscript = parents.get(id(node))
    if not isinstance(subscript, ast.Subscript) or subscript.value is not node:
        return False
    return isinstance(subscript.slice, ast.Constant) and isinstance(subscript.slice.value, str)


def only_computes(code: str, allowlist: Allowlist) -> bool:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    nodes = list(ast.walk(tree))
    bound: dict[str, str] = {}
    if not all(_bind_imports(node, bound, allowlist) for node in nodes):
        return False
    if any(_reaches_out(node) for node in nodes):
        return False
    parents = {id(child): node for node in nodes for child in ast.iter_child_nodes(node)}
    # The dotted path each attribute on a module chain stands for.
    on_the_list: dict[int, str] = {}
    for node in nodes:
        if not isinstance(node, ast.Name) or node.id not in bound:
            continue
        chain = _chain(node, parents)
        path = bound[node.id]
        full = path.split(".") + [attribute.attr for attribute in chain]
        if allowlist.is_module(".".join(full)):
            # Used whole (bound bare, or a chain ending at a submodule): passed
            # around, it could reach anything on it, past the submodule rule.
            return False
        if not _parts_allowed(full, allowlist):
            return False
        if ".".join(full[:2]) == "matplotlib.use" and not _literal_backend(chain[-1] if chain else node, parents):
            return False
        for at, attribute in enumerate(chain, start=1):
            on_the_list[id(attribute)] = ".".join(path.split(".") + [a.attr for a in chain[:at]])
    # `load` (np.load, json.load) may only be called directly, on one file: passed
    # around (`partial(np.load, ...)`) or given more, it could allow pickles.
    loads = {
        id(node.func)
        for node in nodes
        if isinstance(node, ast.Call)
        and len(node.args) == 1
        and not isinstance(node.args[0], ast.Starred)
        and all(keyword.arg is not None for keyword in node.keywords)
    }
    for node in nodes:
        if isinstance(node, ast.Attribute) and node.attr == "rcParams" and not _literal_key(node, parents):
            return False
    for node in nodes:
        if not isinstance(node, ast.Attribute) or node.attr not in _OBJECT_LOADS:
            continue
        if on_the_list.get(id(node)) not in _SAFE_LOADS:
            return False
        if node.attr == "load" and id(node) not in loads:
            return False
    return True
