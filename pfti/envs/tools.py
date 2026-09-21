"""Mock tools with seeded, deterministic failures (plan §2.1).

Each tool takes a hidden World you control per scenario and fails
deterministically when preconditions are violated, with a
machine-readable error_class. Because failures depend only on state,
`would_fail(tool, args, world)` is a free ground-truth ORACLE --
computable without an LLM. That oracle is what makes matcher
precision/recall measurable (shadow mode).
"""
from __future__ import annotations

from ..core.signature import path_prefix

ERROR_CLASSES = ["PermissionDenied", "NotFound", "RateLimited",
                 "StaleState", "InvalidArgument", "QuotaExceeded"]


class ToolError(Exception):
    def __init__(self, error_class: str, msg: str):
        super().__init__(f"{error_class}: {msg}")
        self.error_class = error_class
        self.msg = msg


class World:
    """Hidden world state, one instance per scenario."""

    def __init__(self, dirs=("/",), files=None, perm=None, disk_quota=10**9,
                 api_budget=None, tables=(), stale=()):
        self.dirs = set(dirs)
        self.files = dict(files or {})          # path -> content
        self.perm = dict(perm or {})            # "/prefix" -> "ro"|"rw"
        self.disk_quota = disk_quota
        self.disk_used = sum(len(str(v)) for v in self.files.values())
        self.api_budget = dict(api_budget or {})  # endpoint -> remaining calls
        self.tables = set(tables)
        self.stale = set(stale)                 # paths with stale content

    def perm_of(self, prefix: str):
        return self.perm.get(prefix)

    @staticmethod
    def parent(path: str) -> str:
        p = path.rsplit("/", 1)[0]
        return p or "/"


# ---------- pure precondition checks (shared by tools AND the oracle) ------

def _check_write_file(world, path, content=""):
    if World.parent(path) not in world.dirs:
        return ("NotFound", f"parent dir {World.parent(path)} does not exist")
    if world.perm.get(path_prefix(path)) == "ro":
        return ("PermissionDenied", f"{path_prefix(path)} is read-only")
    if world.disk_used + len(str(content)) > world.disk_quota:
        return ("QuotaExceeded", "disk quota exceeded")
    return None


def _check_read_file(world, path):
    if path not in world.files:
        return ("NotFound", f"{path} does not exist")
    if path in world.stale:
        return ("StaleState", f"{path} content is stale")
    return None


def _check_list_dir(world, path):
    if path not in world.dirs:
        return ("NotFound", f"dir {path} does not exist")
    return None


def _check_run_code(world, code, path=None, content=""):
    if "BUG" in str(code):
        return ("InvalidArgument", "code fails to compile")
    if path is not None:  # code that writes a file hits write preconditions
        return _check_write_file(world, path, content)
    return None


def _check_call_api(world, endpoint, payload=None):
    if endpoint not in world.api_budget:
        return ("NotFound", f"unknown endpoint {endpoint}")
    if world.api_budget[endpoint] <= 0:
        return ("RateLimited", f"rate limit exhausted for {endpoint}")
    return None


def _check_query_db(world, table, query=""):
    if table not in world.tables:
        return ("NotFound", f"table {table} does not exist")
    if "drop" in str(query).lower():
        return ("InvalidArgument", "destructive query rejected")
    return None


_CHECKS = {
    "write_file": _check_write_file,
    "read_file": _check_read_file,
    "list_dir": _check_list_dir,
    "run_code": _check_run_code,
    "call_api": _check_call_api,
    "query_db": _check_query_db,
}


def would_fail(tool: str, args: dict, world) -> tuple | None:
    """Ground-truth oracle: (error_class, msg) if the call would fail now."""
    return _CHECKS[tool](world, **args)


# ---------- executable tools (check, then mutate) ---------------------------

def _guard(tool, world, **args):
    err = _CHECKS[tool](world, **args)
    if err:
        raise ToolError(*err)


def write_file(world, path, content=""):
    _guard("write_file", world, path=path, content=content)
    world.files[path] = content
    world.disk_used += len(str(content))
    return f"wrote {path}"


def read_file(world, path):
    _guard("read_file", world, path=path)
    return world.files[path]


def list_dir(world, path):
    _guard("list_dir", world, path=path)
    return sorted(p for p in world.files if World.parent(p) == path)


def run_code(world, code, path=None, content=""):
    _guard("run_code", world, code=code, path=path, content=content)
    if path is not None:
        world.files[path] = content
        world.disk_used += len(str(content))
    return "code ran ok"


def call_api(world, endpoint, payload=None):
    _guard("call_api", world, endpoint=endpoint, payload=payload)
    world.api_budget[endpoint] -= 1
    return {"endpoint": endpoint, "status": 200}


def query_db(world, table, query=""):
    _guard("query_db", world, table=table, query=query)
    return [{"table": table, "row": 1}]


TOOLS = {
    "write_file": write_file, "read_file": read_file, "list_dir": list_dir,
    "run_code": run_code, "call_api": call_api, "query_db": query_db,
}
