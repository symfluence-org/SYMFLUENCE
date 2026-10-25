"""Unit tests for ToolInstaller behavior and result reporting."""
from __future__ import annotations

import os
import shutil
import stat
import sys
from unittest.mock import MagicMock

import pytest


def _make_exe(path):
    """Create a tiny executable shell stub at *path*."""
    with open(path, "w") as fh:
        fh.write("#!/bin/sh\n")
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink shim")
def test_bare_compiler_shim_exposes_versioned_gfortran(mock_external_tools, tmp_path):
    """When only `gfortran-NN` exists, a bare `gfortran` is shimmed onto PATH.

    Mirrors the macOS-runner case where Homebrew's gcc ships only versioned
    Fortran binaries, breaking Makefiles/preflights that call bare `gfortran`.
    """
    from symfluence.cli.services.tool_installer import ToolInstaller

    installer = ToolInstaller(external_tools=mock_external_tools)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _make_exe(bindir / "gfortran-9")
    _make_exe(bindir / "gfortran-14")  # newest — should win

    env = {"PATH": str(bindir)}
    assert shutil.which("gfortran", path=env["PATH"]) is None

    installer._ensure_bare_compiler_shim(env, "gfortran")

    resolved = shutil.which("gfortran", path=env["PATH"])
    assert resolved is not None
    assert os.path.realpath(resolved) == os.path.realpath(bindir / "gfortran-14")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink shim")
def test_bare_compiler_shim_is_noop_when_bare_present(mock_external_tools, tmp_path):
    """If a bare `gfortran` is already reachable, PATH is left untouched."""
    from symfluence.cli.services.tool_installer import ToolInstaller

    installer = ToolInstaller(external_tools=mock_external_tools)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _make_exe(bindir / "gfortran")

    env = {"PATH": str(bindir)}
    installer._ensure_bare_compiler_shim(env, "gfortran")

    assert env["PATH"] == str(bindir)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink shim")
def test_bare_compiler_shim_noop_without_any_compiler(mock_external_tools, tmp_path):
    """With neither bare nor versioned compiler, PATH is unchanged (no crash)."""
    from symfluence.cli.services.tool_installer import ToolInstaller

    installer = ToolInstaller(external_tools=mock_external_tools)
    bindir = tmp_path / "empty"
    bindir.mkdir()

    env = {"PATH": str(bindir)}
    installer._ensure_bare_compiler_shim(env, "gfortran")

    assert env["PATH"] == str(bindir)


def test_clone_repository_sparse_excludes_paths(mock_external_tools, tmp_path):
    """`sparse_exclude` keeps host-incompatible fixture paths out of the checkout.

    Reproduces the t-route case where a test-fixture directory must be skipped
    (on Windows its filenames are illegal); here we just assert the directory is
    absent from the working tree while normal sources remain.
    """
    import subprocess

    from symfluence.cli.services.tool_installer import ToolInstaller

    if shutil.which("git") is None:
        pytest.skip("git not available")

    origin = tmp_path / "origin"
    origin.mkdir()
    # Two test/ fixture dirs — one top-level, one deeply nested — mirroring how
    # t-route scatters them. A bare `test/` pattern must catch both.
    top = origin / "test" / "HurricaneLaura"
    nested = origin / "src" / "pkg" / "network" / "test" / "fixtures"
    top.mkdir(parents=True)
    nested.mkdir(parents=True)
    (top / "restart.nc").write_text("fixture")
    (nested / "data.ncdf").write_text("fixture")
    (origin / "src" / "pkg" / "normal.py").write_text("print('hi')\n")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "init", "-q"], cwd=origin, check=True, env=env)
    subprocess.run(["git", "add", "-A"], cwd=origin, check=True, env=env)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=origin, check=True, env=env)

    installer = ToolInstaller(external_tools=mock_external_tools)
    target = tmp_path / "clone"
    ok = installer._clone_repository(
        str(origin), None, target,
        sparse_exclude=["test/"],  # any-depth, matches both dirs above
    )

    assert ok is True
    assert (target / "src" / "pkg" / "normal.py").exists()
    assert not (target / "test" / "HurricaneLaura" / "restart.nc").exists()
    assert not (target / "src" / "pkg" / "network" / "test" / "fixtures" / "data.ncdf").exists()


def test_clone_with_retry_recovers_from_transient_failure(mock_external_tools, tmp_path, monkeypatch):
    """A clone that fails once (e.g. flaky SourceForge) is retried and succeeds.

    The partial target is cleaned between attempts and the backoff sleep is
    stubbed so the test stays fast.
    """
    import subprocess

    from symfluence.cli.services import tool_installer as ti

    target = tmp_path / "clone"
    calls = {"n": 0}

    def fake_run(cmd, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # Simulate a partial clone left behind by the failed attempt.
            target.mkdir(parents=True, exist_ok=True)
            raise subprocess.CalledProcessError(128, cmd, stderr="access denied")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(ti.subprocess, "run", fake_run)
    monkeypatch.setattr(ti.time, "sleep", lambda *_: None)

    installer = ti.ToolInstaller(external_tools=mock_external_tools)
    installer._clone_with_retry(["git", "clone", "x", str(target)], {}, target)

    assert calls["n"] == 2  # failed once, succeeded on retry


def test_clone_with_retry_reraises_after_exhausting_attempts(mock_external_tools, tmp_path, monkeypatch):
    """When every attempt fails the last error is re-raised so install fails hard."""
    import subprocess

    from symfluence.cli.services import tool_installer as ti

    def always_fail(cmd, *args, **kwargs):
        raise subprocess.CalledProcessError(128, cmd, stderr="access denied")

    monkeypatch.setattr(ti.subprocess, "run", always_fail)
    monkeypatch.setattr(ti.time, "sleep", lambda *_: None)

    installer = ti.ToolInstaller(external_tools=mock_external_tools)
    with pytest.raises(subprocess.CalledProcessError):
        installer._clone_with_retry(["git", "clone", "x", "y"], {}, tmp_path / "c", attempts=3)


def test_install_fails_fast_when_required_tool_missing(mock_external_tools, tmp_path):
    """A tool with unmet `requires` should not proceed to build."""
    from symfluence.cli.services.tool_installer import ToolInstaller

    installer = ToolInstaller(external_tools=mock_external_tools)

    installer._load_config = MagicMock(return_value={"SYMFLUENCE_DATA_DIR": str(tmp_path)})
    installer._clone_repository = MagicMock(return_value=True)
    installer._check_system_dependencies = MagicMock(return_value=[])
    installer._run_build_commands = MagicMock(return_value=True)
    installer._verify_installation = MagicMock(return_value=True)

    result = installer.install(specific_tools=["summa"], force=True)

    assert all(call.args[0] != "summa" for call in installer._run_build_commands.call_args_list)
    assert "summa" in result["failed"]
    assert "summa" not in result["successful"]


def test_install_rebuilds_when_existing_dir_fails_verification(mock_external_tools, tmp_path):
    """An install dir left by a failed build must be rebuilt, not skipped.

    The clone step creates the directory before the build runs, so a build that
    fails for missing system libraries leaves source behind with no binary.
    Re-running must not report that tool as "skipped - already exists".
    """
    from symfluence.cli.services.tool_installer import ToolInstaller

    installer = ToolInstaller(external_tools=mock_external_tools)

    # Directory exists but holds no verifiable artifact (bin/pitremove etc.).
    (tmp_path / "installs" / "TauDEM").mkdir(parents=True)

    installer._load_config = MagicMock(return_value={"SYMFLUENCE_DATA_DIR": str(tmp_path)})
    installer._clone_repository = MagicMock(return_value=True)
    installer._check_system_dependencies = MagicMock(return_value=[])
    installer._run_build_commands = MagicMock(return_value=True)

    def _verify(_name, _info, install_dir):
        return (install_dir / "bin" / "pitremove").exists()

    installer._verify_installation = MagicMock(side_effect=_verify)

    result = installer.install(specific_tools=["taudem"], force=False)

    assert "taudem" not in result["skipped"]
    installer._run_build_commands.assert_called_once()
    assert "taudem" in result["failed"]  # rebuild still lacks the artifact


def test_install_skips_when_existing_dir_verifies(mock_external_tools, tmp_path):
    """A genuinely installed tool must still skip without rebuilding."""
    from symfluence.cli.services.tool_installer import ToolInstaller

    installer = ToolInstaller(external_tools=mock_external_tools)

    taudem_dir = tmp_path / "installs" / "TauDEM" / "bin"
    taudem_dir.mkdir(parents=True)
    _make_exe(taudem_dir / "pitremove")

    installer._load_config = MagicMock(return_value={"SYMFLUENCE_DATA_DIR": str(tmp_path)})
    installer._clone_repository = MagicMock(return_value=True)
    installer._check_system_dependencies = MagicMock(return_value=[])
    installer._run_build_commands = MagicMock(return_value=True)

    result = installer.install(specific_tools=["taudem"], force=False)

    assert "taudem" in result["skipped"]
    installer._run_build_commands.assert_not_called()
    installer._clone_repository.assert_not_called()


def test_install_marks_verification_failure_as_failed(mock_external_tools, tmp_path):
    """Verification failure should remove a tool from successful installs."""
    from symfluence.cli.services.tool_installer import ToolInstaller

    installer = ToolInstaller(external_tools=mock_external_tools)

    installer._load_config = MagicMock(return_value={"SYMFLUENCE_DATA_DIR": str(tmp_path)})
    installer._clone_repository = MagicMock(return_value=True)
    installer._check_system_dependencies = MagicMock(return_value=[])
    installer._run_build_commands = MagicMock(return_value=True)
    installer._verify_installation = MagicMock(return_value=False)

    result = installer.install(specific_tools=["taudem"], force=True)

    assert "taudem" in result["failed"]
    assert "taudem" not in result["successful"]
    assert any("taudem: installation verification failed" in e for e in result["errors"])


def test_move_to_front_promotes_an_existing_path_entry():
    """An entry already on PATH is moved ahead, not skipped.

    In the Windows release job the MSYS2 mingw64 bin sat behind conda's
    Library\\mingw-w64\\bin; skipping it because it was "already on PATH" made
    cc1.exe load conda's older runtime DLLs, and gcc failed CMake's compiler
    test with no diagnostic.
    """
    from symfluence.cli.services.tool_installer import _move_to_front

    path = ";".join(["usr", "conda-mingw", "windows", "mingw64", "mingw64"])
    assert _move_to_front(path, "mingw64", ";").split(";") == [
        "mingw64", "usr", "conda-mingw", "windows"]
    assert _move_to_front("", "mingw64", ";") == "mingw64"
