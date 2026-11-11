# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Top-level doctor command handler for SYMFLUENCE CLI.

Provides comprehensive system diagnostics including environment info,
path resolution reporting, and binary/library checks.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from argparse import Namespace
from importlib.metadata import PackageNotFoundError
from pathlib import Path

from ..exit_codes import ExitCode
from .base import BaseCommand, cli_exception_handler, supports_dry_run


class DoctorCommands(BaseCommand):
    """Handlers for the top-level doctor command."""

    @staticmethod
    @supports_dry_run
    @cli_exception_handler
    def doctor(args: Namespace) -> int:
        """
        Execute: symfluence doctor

        Run comprehensive system diagnostics covering:
        1. Python environment (version, venv, package versions)
        2. Path resolution (CODE_DIR, DATA_DIR with source reasoning)
        3. Binary/toolchain/library checks (delegates to BinaryManager.doctor)

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for issues found)
        """
        console = BaseCommand._console
        issues_found = False

        console.info("Running SYMFLUENCE system diagnostics...")
        console.rule("Environment")

        # --- Python environment ---
        console.info(f"Python:  {sys.version.split()[0]}  ({sys.executable})")

        in_venv = sys.prefix != sys.base_prefix
        if in_venv:
            console.info(f"Venv:    {sys.prefix}")
        else:
            console.warning("No virtual environment detected")

        # SYMFLUENCE version + install location
        try:
            import symfluence
            from symfluence.symfluence_version import __version__
            pkg_path = Path(symfluence.__file__).parent
            console.info(f"SYMFLUENCE: {__version__}  ({pkg_path})")
        except ImportError:
            console.warning("SYMFLUENCE package version not available")
            issues_found = True

        # Key package versions
        _report_package(console, "numpy")
        _report_package(console, "xarray")

        # GDAL: check both Python and system
        try:
            from osgeo import gdal
            py_ver = gdal.VersionInfo("VERSION_NUM")
            # Format VERSION_NUM (e.g., 3080300 -> 3.8.3)
            major = int(py_ver) // 1000000
            minor = (int(py_ver) // 10000) % 100
            patch = (int(py_ver) // 100) % 100
            py_gdal_str = f"{major}.{minor}.{patch}"
            console.info(f"GDAL (Python): {py_gdal_str}")
        except ImportError:
            py_gdal_str = None
            console.warning("GDAL Python bindings not installed")
            issues_found = True

        sys_gdal = shutil.which("gdal-config")
        if sys_gdal:
            try:
                result = subprocess.run(
                    ["gdal-config", "--version"],
                    capture_output=True, text=True, timeout=5
                )
                sys_gdal_ver = result.stdout.strip()
                console.info(f"GDAL (system): {sys_gdal_ver}  ({sys_gdal})")
                if py_gdal_str and sys_gdal_ver and not sys_gdal_ver.startswith(
                    py_gdal_str.rsplit(".", 1)[0]
                ):
                    console.warning(
                        f"GDAL version mismatch: Python={py_gdal_str}, system={sys_gdal_ver}"
                    )
                    issues_found = True
            except (subprocess.SubprocessError, OSError, ValueError):
                console.info(f"GDAL (system): gdal-config found but failed  ({sys_gdal})")
        else:
            console.warning("gdal-config not found on PATH")

        # --- Path resolution ---
        console.newline()
        console.rule("Path Resolution")

        _report_code_dir(console)
        data_ok = _report_data_dir(console)
        if not data_ok:
            issues_found = True

        # --- Binary / toolchain / library checks ---
        console.newline()
        console.rule("Binaries & Libraries")

        try:
            from symfluence.cli.binary_service import BinaryManager
            binary_manager = BinaryManager()
            binary_ok = binary_manager.doctor()
            if not binary_ok:
                issues_found = True
        except ImportError as e:
            console.warning(f"Binary diagnostics unavailable: {e}")
            issues_found = True

        # --- Summary ---
        console.newline()
        console.rule()
        if issues_found:
            console.warning("Diagnostics completed with warnings")
            return ExitCode.SUCCESS  # warnings are not fatal
        else:
            console.success("All diagnostics passed")
            return ExitCode.SUCCESS


def _report_package(console, name: str) -> None:
    """Report a single Python package version."""
    try:
        from importlib.metadata import version
        ver = version(name)
        console.info(f"{name}: {ver}")
    except PackageNotFoundError:
        console.info(f"{name}: not installed")


def _report_code_dir(console) -> None:
    """Report SYMFLUENCE_CODE_DIR resolution with source reasoning."""
    env_val = os.environ.get("SYMFLUENCE_CODE_DIR")
    if env_val:
        console.info(f"CODE_DIR: {env_val}")
        console.indent("source: SYMFLUENCE_CODE_DIR environment variable")
        return

    # Check if we're inside a repo checkout
    try:
        import symfluence
        pkg_dir = Path(symfluence.__file__).parent
        # Walk up looking for .git or pyproject.toml
        candidate = pkg_dir
        for _ in range(6):
            if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
                console.info(f"CODE_DIR: {candidate}")
                console.indent("source: detected repository root from package location")
                return
            candidate = candidate.parent
    except (ImportError, AttributeError, OSError, RuntimeError):
        pass

    console.warning("CODE_DIR: not set")
    console.indent("Set SYMFLUENCE_CODE_DIR env var or run from a repository checkout")


def _report_data_dir(console) -> bool:
    """Report SYMFLUENCE_DATA_DIR resolution with source reasoning. Returns True if OK."""
    env_val = os.environ.get("SYMFLUENCE_DATA_DIR")
    if env_val:
        p = Path(env_val)
        console.info(f"DATA_DIR: {p}")
        console.indent("source: SYMFLUENCE_DATA_DIR environment variable")
        return _check_dir(console, p)

    # Use the canonical resolver (sibling of code dir)
    try:
        from symfluence.core.config.factories import _resolve_default_data_dir
        resolved = Path(_resolve_default_data_dir())
        console.info(f"DATA_DIR: {resolved}")
        console.indent("source: sibling of detected code directory")
        if resolved.exists():
            return _check_dir(console, resolved)
        # Directory doesn't exist yet — that's fine, it will be created on first install
        console.indent("directory will be created on first install")
        return True
    except (ImportError, OSError, RuntimeError):
        pass

    # Last resort fallback
    home_default = Path.home() / "SYMFLUENCE_data"
    if home_default.exists():
        console.info(f"DATA_DIR: {home_default}")
        console.indent("source: default ~/SYMFLUENCE_data directory")
        return _check_dir(console, home_default)

    console.warning("DATA_DIR: not set")
    console.indent(
        "Set SYMFLUENCE_DATA_DIR env var or run from a repository checkout"
    )
    return True  # not fatal — will be set per-project via config


def _check_dir(console, path: Path) -> bool:
    """Check directory exists, is writable, and report disk space."""
    ok = True
    if not path.exists():
        console.warning(f"  directory does not exist: {path}")
        ok = False
    elif not os.access(path, os.W_OK):
        console.warning(f"  directory is not writable: {path}")
        ok = False
    else:
        try:
            usage = shutil.disk_usage(path)
            free_gb = usage.free / (1024 ** 3)
            total_gb = usage.total / (1024 ** 3)
            console.indent(f"disk: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
        except OSError:
            pass
    return ok
