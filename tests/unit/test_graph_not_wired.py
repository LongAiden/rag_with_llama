"""
Guards that the knowledge graph feature stays out of the live workflow.

The graph code lives in the active tree (`src/app/graph/` and friends) but is
deliberately unreachable: nothing mounts its router and no live code path
imports it. That is easy to undo by accident — one `include_router` call, or one
stray import in a shared `__init__` — so it is asserted here rather than left to
review.

If you are intentionally enabling the feature, delete this file in the same
commit that mounts the router.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"

GRAPH_MODULE_PREFIXES = (
    "app.graph",
    "app.config.graph_config",
    "app.models.graph_models",
    "app.ingestion.extraction",
    "app.api.routes.graph_routes",
)


def test_graph_router_is_not_mounted():
    """No /graph endpoint is served by the app."""
    from app.api.app import app

    graph_routes = [
        route.path for route in app.routes
        if getattr(route, "path", "").startswith("/graph")
    ]

    assert graph_routes == []


def test_graph_modules_are_not_imported_at_startup():
    """Importing the API and worker entrypoints must not pull in graph code.

    Run in a subprocess: this test session has already imported the graph
    modules for the graph unit tests, so `sys.modules` here proves nothing.
    """
    probe = f"""
import sys
sys.path.insert(0, {str(SRC)!r})

import app.api.app
import app.worker.celery_app
import app.worker.ingestion_tasks

prefixes = {GRAPH_MODULE_PREFIXES!r}
leaked = sorted(m for m in sys.modules if m.startswith(prefixes))
print("LEAKED:" + ",".join(leaked))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, cwd=REPO_ROOT,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(Path.home()),
             "LOGFIRE_SEND_TO_LOGFIRE": "false", "TOKENIZERS_PARALLELISM": "false"},
    )

    assert result.returncode == 0, f"probe failed:\n{result.stderr[-2000:]}"

    leaked_line = next(
        (line for line in result.stdout.splitlines() if line.startswith("LEAKED:")),
        None,
    )
    assert leaked_line is not None, f"probe produced no result:\n{result.stdout[-2000:]}"

    leaked = [m for m in leaked_line[len("LEAKED:"):].split(",") if m]
    assert leaked == [], f"graph modules reachable from the live entrypoints: {leaked}"


def test_no_live_module_imports_the_graph_feature():
    """Static check, so a lazy import inside a function is caught too."""
    live_files = [
        path for path in (REPO_ROOT / "src" / "app").rglob("*.py")
        if not str(path.relative_to(SRC)).replace("\\", "/").startswith(
            ("app/graph/", "app/ingestion/extraction/")
        )
        and path.name not in ("graph_routes.py", "graph_config.py", "graph_models.py")
    ]

    offenders = []
    for path in live_files:
        text = path.read_text(encoding="utf-8")
        for prefix in GRAPH_MODULE_PREFIXES:
            if f"import {prefix}" in text or f"from {prefix}" in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} -> {prefix}")

    assert offenders == [], f"live code imports graph modules: {offenders}"


@pytest.mark.parametrize("filename", ["002_create_llm_interactions.sql"])
def test_graph_migration_is_not_applied_on_init(filename):
    """Postgres runs only the .sql files directly in the mounted migrations dir.

    The graph schema lives in `optional/`, a subdirectory initdb ignores, so a
    fresh volume comes up without empty entities/relationships tables.
    """
    migrations = REPO_ROOT / "deploy" / "migrations"

    applied = sorted(p.name for p in migrations.glob("*.sql"))

    assert filename in applied, "expected the live migrations to still be here"
    assert "001_create_graph_tables.sql" not in applied
    assert (migrations / "optional" / "001_create_graph_tables.sql").is_file()
