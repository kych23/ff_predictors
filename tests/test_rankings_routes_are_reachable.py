"""The static mount must never shadow an API route.

`api.py` ends with `app.mount("/", StaticFiles(...))`, guarded by
`DIST.exists()`. `Mount("/")` matches EVERY path and Starlette takes the first
match in registration order, so a route added below that block silently 404s.

The obvious test — "GET /api/rankings does not 404" — is worthless here.
`web/dist/` is gitignored and `DIST.exists()` is evaluated once at import, so
on CI and on any fresh clone the mount is never registered, the route resolves
trivially, and the test passes green while proving nothing. It would only be
live on a laptop that had run `npm run build`, which is exactly the machine
least likely to notice.

So both tests below are structural and hold with or without a build.
"""
from __future__ import annotations

import ast
from pathlib import Path

from starlette.routing import Mount

from src.app.web import api as api_mod

API_PATH = Path(api_mod.__file__)


def test_no_api_route_is_registered_after_a_catch_all_mount():
    """Ordering, asserted on the live app object."""
    routes = api_mod.app.routes
    mounts = [i for i, route in enumerate(routes) if isinstance(route, Mount)]
    api_routes = [i for i, route in enumerate(routes)
                  if getattr(route, "path", "").startswith("/api/")]
    assert api_routes, "no /api routes registered at all"
    if mounts:
        assert max(api_routes) < min(mounts), (
            "an /api route is registered after the static mount and will "
            "silently 404 wherever web/dist exists")


def test_no_route_decorator_appears_below_the_dist_guard():
    """The same guarantee, read off the SOURCE rather than the running app.

    This one is unconditional: it does not care whether `web/dist` exists, so
    it fails on a bare clone too — which is where the mistake would otherwise
    sit undetected until draft night.
    """
    tree = ast.parse(API_PATH.read_text())

    guard_line = None
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            if (isinstance(test, ast.Call)
                    and isinstance(test.func, ast.Attribute)
                    and test.func.attr == "exists"
                    and isinstance(test.func.value, ast.Name)
                    and test.func.value.id == "DIST"):
                guard_line = node.lineno
    assert guard_line is not None, "the DIST.exists() guard has moved or gone"

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            if (isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "app"
                    and func.attr in {"get", "post", "put", "delete", "patch"}
                    and decorator.args
                    and isinstance(decorator.args[0], ast.Constant)
                    and str(decorator.args[0].value).startswith("/api/")
                    and decorator.lineno > guard_line):
                offenders.append((node.name, decorator.lineno))

    assert not offenders, (
        f"these routes are declared below the DIST.exists() guard at line "
        f"{guard_line} and will be shadowed by the static mount: {offenders}")


def test_the_catalogue_literal_is_not_shadowed_by_the_board_id_route():
    """`/api/rankings/catalogue` and `/api/rankings/{board_id}` both match the
    same URL. FastAPI takes the first, so the literal must be registered
    first."""
    paths = [getattr(route, "path", "") for route in api_mod.app.routes]
    assert paths.index("/api/rankings/catalogue") \
        < paths.index("/api/rankings/{board_id}")
