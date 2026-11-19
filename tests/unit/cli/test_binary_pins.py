"""Upstream source pins: every in-tree build recipe builds an exact commit."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from symfluence.cli.binary_pins import PINS_FILE, SHA_RE, apply_pins, load_pins

pytestmark = [pytest.mark.unit, pytest.mark.quick]

SRC = Path(__file__).resolve().parents[3] / "src" / "symfluence"
SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "update_binary_pins.py"
_ADD = re.compile(r"build_instructions\.add\(\s*['\"]([\w-]+)['\"]")


def _in_tree_tool_names() -> set[str]:
    """Tool names registered by build-instruction modules shipped in this repo."""
    files = [*SRC.glob("models/*/build_instructions.py"), SRC / "cli" / "external_tools_config.py"]
    return {m for f in files for m in _ADD.findall(f.read_text(encoding="utf-8"))}


@pytest.fixture(scope="module")
def tools():
    from symfluence.cli.external_tools_config import get_external_tools_definitions
    return get_external_tools_definitions()


def test_every_in_tree_recipe_with_a_repository_is_pinned(tools):
    names = _in_tree_tool_names()
    assert {"summa", "fuse", "parflow", "vic"} <= names  # the scan sees the recipes
    unpinned = {
        name: tools[name].get("git_hash")
        for name in names
        if name in tools and tools[name].get("repository")
        and not SHA_RE.match(str(tools[name].get("git_hash") or ""))
    }
    assert not unpinned, f"recipes without a 40-hex source pin: {unpinned}"


def test_pin_file_entries_are_well_formed_and_match_their_recipe(tools):
    pins = load_pins()
    assert pins, f"no pins read from {PINS_FILE}"
    for name, pin in pins.items():
        assert SHA_RE.match(pin.get("commit", "")), name
        assert pin.get("repository"), name
        assert "ref" in pin, name  # branch/tag the pin came from (None = default branch)
        if name in tools:
            recipe = tools[name]
            assert recipe["source_pin"]["pinned_by"] == "binary_pins.json", name
            assert recipe["git_hash"] == pin["commit"], name


def test_recipe_held_commit_wins_over_pin_file():
    recipe = {"repository": "https://h/x.git", "branch": "b", "git_hash": "a" * 40}
    out = apply_pins({"x": recipe}, {"x": {"repository": "https://h/x.git", "commit": "b" * 40}})
    assert out["x"]["git_hash"] == "a" * 40
    assert out["x"]["source_pin"]["pinned_by"] == "recipe"


def test_pin_for_another_repository_is_not_applied():
    recipe = {"repository": "https://h/new.git", "branch": "main"}
    apply_pins({"x": recipe}, {"x": {"repository": "https://h/old.git", "commit": "b" * 40}})
    assert "git_hash" not in recipe


def test_apply_pins_is_idempotent_and_follows_pin_changes():
    recipe = {"repository": "https://h/x", "branch": None}
    apply_pins({"x": recipe}, {"x": {"repository": "https://h/x.git", "ref": None, "commit": "b" * 40}})
    assert recipe["git_hash"] == "b" * 40  # .git suffix difference tolerated
    apply_pins({"x": recipe}, {"x": {"repository": "https://h/x", "commit": "c" * 40}})
    assert recipe["git_hash"] == "c" * 40
    assert recipe["source_pin"]["pinned_by"] == "binary_pins.json"


def _load_script():
    spec = importlib.util.spec_from_file_location("update_binary_pins", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_update_script_check_reports_drift_without_failing(tmp_path, monkeypatch, capsys):
    script = _load_script()
    pins = tmp_path / "pins.json"
    pins.write_text(json.dumps({"tools": {
        "a": {"repository": "https://h/a.git", "ref": "main", "commit": "1" * 40},
    }}))
    recipes = {"a": {"repository": "https://h/a.git", "branch": "main"}}
    monkeypatch.setattr(script, "_recipes", lambda: recipes)
    monkeypatch.setattr(script, "resolve_remote_ref", lambda repo, ref: "2" * 40)

    assert script.main(["--check", "--pins-file", str(pins)]) == 0
    assert "behind upstream" in capsys.readouterr().out

    recipes["b"] = {"repository": "https://h/b.git", "branch": None}
    assert script.main(["--check", "--pins-file", str(pins)]) == 1  # missing pin

    assert script.main(["--write", "--pins-file", str(pins)]) == 0
    written = json.loads(pins.read_text())["tools"]
    assert written["a"]["commit"] == "2" * 40 and written["a"]["ref"] == "main"
    assert written["b"]["commit"] == "2" * 40 and written["b"]["ref"] is None


# ---------------------------------------------------------------------------
# Installer: a pin is what gets checked out, and is recorded
# ---------------------------------------------------------------------------

def _git_env():
    return {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


@pytest.fixture
def two_commit_origin(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    origin = tmp_path / "origin"
    origin.mkdir()
    env = _git_env()
    run = lambda *a: subprocess.run(["git", *a], cwd=origin, check=True, env=env,  # noqa: E731
                                    capture_output=True, text=True)
    run("init", "-q", "-b", "main")
    (origin / "f.txt").write_text("first\n")
    run("add", "-A")
    run("commit", "-qm", "1")
    first = run("rev-parse", "HEAD").stdout.strip()
    (origin / "f.txt").write_text("second\n")
    run("commit", "-qam", "2")
    return origin, first


def test_pinned_clone_checks_out_the_pin_not_the_branch_tip(two_commit_origin, tmp_path):
    from symfluence.cli.services.tool_installer import ToolInstaller

    origin, first = two_commit_origin
    target = tmp_path / "tool"
    installer = ToolInstaller(external_tools={})
    assert installer._clone_repository(str(origin), "main", target, git_hash=first)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=target, capture_output=True,
                          text=True, check=True).stdout.strip()
    assert head == first
    assert (target / "f.txt").read_text() == "first\n"
    remote = subprocess.run(["git", "config", "--get", "remote.origin.url"], cwd=target,
                            capture_output=True, text=True).stdout.strip()
    assert remote == str(origin)  # build steps can still `git fetch origin ...`


def test_source_record_reaches_the_install_manifest(two_commit_origin, tmp_path):
    from symfluence.cli.services.tool_installer import ToolInstaller

    origin, first = two_commit_origin
    installs = tmp_path / "installs"
    tool_dir = installs / "tool"
    tools = {"tool": {"repository": str(origin), "branch": "main", "install_dir": "tool",
                      "git_hash": first,
                      "source_pin": {"ref": "main", "commit": first,
                                     "pinned_by": "binary_pins.json"}}}
    installer = ToolInstaller(external_tools=tools)
    installer._clone_repository(str(origin), "main", tool_dir, git_hash=first)
    installer._write_source_record("tool", tool_dir, str(origin), "main", first,
                                   pinned_by="binary_pins.json")

    record = json.loads((tool_dir / ".symfluence_source.json").read_text())
    assert record["repository"] == str(origin)
    assert record["pinned_commit"] == record["commit"] == first
    assert record["tracked_changes"] is False

    installer._generate_toolchain_metadata(installs)
    entry = json.loads((installs / "toolchain.json").read_text())["tools"]["tool"]
    assert entry["repository"] == str(origin)
    assert entry["pinned_commit"] == first
    assert entry["commit"] == first
    assert entry["source"]["pinned_by"] == "binary_pins.json"
