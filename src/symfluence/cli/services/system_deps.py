# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
System dependency registry and checker for SYMFLUENCE.

Loads the canonical dependency definitions from system_deps.yml and provides
platform-aware detection, version checking, and install-command generation.
"""

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class Platform(Enum):
    """Detected package-manager platform."""
    APT = "apt"
    DNF = "dnf"
    BREW = "brew"
    CONDA = "conda"
    HPC_MODULE = "hpc_module"
    MSYS2 = "msys2"
    WSL = "wsl"
    UNKNOWN = "unknown"


@dataclass
class DepCheckResult:
    """Result of checking a single system dependency."""
    dep_id: str
    display_name: str
    found: bool
    path: Optional[str] = None
    version: Optional[str] = None
    version_ok: bool = True
    install_command: Optional[str] = None
    needed_for: List[str] = field(default_factory=list)
    required: bool = True
    category: str = "core_library"


class SystemDepsRegistry:
    """
    Singleton registry that loads system_deps.yml and checks dependencies.

    Usage::

        registry = SystemDepsRegistry()
        results = registry.check_deps_for_tool("summa")
        for r in results:
            print(r.display_name, r.found, r.version)
    """

    _instance: Optional["SystemDepsRegistry"] = None
    _registry: Optional[Dict[str, Any]] = None

    def __new__(cls) -> "SystemDepsRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._registry is None:
            self._load_registry()
            self._platform = self._detect_platform()

    # ── Registry loading ─────────────────────────────────────────────

    def _load_registry(self) -> None:
        """Load the YAML registry using importlib.resources (same pattern as resources/manager.py)."""
        if sys.version_info >= (3, 9):
            from importlib.resources import files
        else:
            from importlib_resources import files

        registry_traversable = files("symfluence.resources") / "system_deps.yml"

        if hasattr(registry_traversable, "__fspath__"):
            registry_path = Path(registry_traversable)
        else:
            registry_path = Path(str(registry_traversable))

        with open(registry_path, "r", encoding="utf-8") as f:
            self._registry = yaml.safe_load(f)

    # ── Platform detection ───────────────────────────────────────────

    @staticmethod
    def _detect_platform() -> Platform:
        """Detect the current package-manager platform.

        Priority: HPC → CONDA → (win32: MSYS2 → WSL) → apt → dnf → brew → UNKNOWN
        """
        # HPC module system (check common env vars first)
        for var in ("LMOD_CMD", "MODULESHOME", "LMOD_DIR"):
            if os.environ.get(var):
                return Platform.HPC_MODULE

        # Conda environment
        if os.environ.get("CONDA_PREFIX"):
            return Platform.CONDA

        # Windows-specific: MSYS2 then WSL
        if sys.platform == "win32":
            # MSYS2 — same paths as ToolInstaller._find_bash()
            for candidate in [
                r"C:\msys64\usr\bin\bash.exe",
                r"C:\msys2\usr\bin\bash.exe",
            ]:
                if os.path.isfile(candidate):
                    return Platform.MSYS2

            # WSL — probe via `wsl -e echo ok`
            wsl = shutil.which("wsl")
            if wsl:
                try:
                    probe = subprocess.run(
                        [wsl, "-e", "echo", "ok"],
                        capture_output=True, text=True, timeout=10,
                    )
                    if probe.returncode == 0 and "ok" in probe.stdout:
                        return Platform.WSL
                except (subprocess.TimeoutExpired, OSError):
                    pass

        # System package managers
        if shutil.which("apt-get"):
            return Platform.APT
        if shutil.which("dnf"):
            return Platform.DNF
        if shutil.which("brew"):
            return Platform.BREW

        return Platform.UNKNOWN

    @property
    def platform(self) -> Platform:
        """Currently detected platform."""
        return self._platform

    # ── Dependency checking ──────────────────────────────────────────

    def check_dep(self, dep_id: str) -> DepCheckResult:
        """
        Check whether a single dependency is available.

        Args:
            dep_id: Key from the ``dependencies`` section of system_deps.yml.

        Returns:
            DepCheckResult with detection details.
        """
        assert self._registry is not None  # guaranteed by __init__
        dep_info = self._registry["dependencies"].get(dep_id)
        if dep_info is None:
            return DepCheckResult(
                dep_id=dep_id,
                display_name=dep_id,
                found=False,
                install_command=None,
            )

        display_name = dep_info.get("display_name", dep_id)
        check = dep_info.get("check", {})
        command = check.get("command", "")
        needed_for = dep_info.get("needed_for", [])
        required = dep_info.get("required", True)
        category = dep_info.get("category", "core_library")

        found = False
        path: Optional[str] = None
        version: Optional[str] = None
        version_ok = True

        # Try primary command, then alternatives
        commands_to_try = [command] if command else []
        commands_to_try.extend(check.get("alternatives", []))

        # Also try pkg-config as a fallback (useful for BLAS which has no CLI command)
        pkg_config_name = check.get("pkg_config", "")

        for cmd in commands_to_try:
            if not cmd:
                continue
            location = shutil.which(cmd)
            if location:
                found = True
                path = location
                version = self._extract_version(cmd, check)
                break

        # WSL fallback: probe inside WSL when running on the Windows side
        if not found and self._platform == Platform.WSL:
            for cmd in commands_to_try:
                if not cmd:
                    continue
                wsl_found, wsl_version = self._check_wsl_command(cmd, check)
                if wsl_found:
                    found = True
                    path = f"(wsl: {cmd})"
                    version = wsl_version
                    break

        # Fallback: pkg-config probe
        if not found and pkg_config_name:
            found, version = self._check_pkg_config(pkg_config_name)
            if found:
                path = f"(pkg-config: {pkg_config_name})"

        # Conda prefix fallback: on Conda (especially Windows), CLI tools
        # like nc-config are often absent.  Probe $CONDA_PREFIX for known
        # header files to confirm the library is actually installed.
        if not found and self._platform == Platform.CONDA:
            conda_header = check.get("conda_header", "")
            if conda_header:
                found, path = self._check_conda_prefix(conda_header)

        # Homebrew keg-only fallback: on macOS, packages like openblas and
        # lapack are "keg-only" (not symlinked into the Homebrew prefix), so
        # their .pc files are invisible to pkg-config by default.  Probe the
        # keg's own pkgconfig directory.
        if not found and pkg_config_name and self._platform == Platform.BREW:
            brew_formulas = dep_info.get("packages", {}).get("brew", "")
            if brew_formulas:
                found, version = self._check_brew_keg_pkg_config(
                    pkg_config_name, brew_formulas.split()
                )
                if found:
                    path = f"(brew keg: {pkg_config_name})"

        # Version comparison
        min_version = dep_info.get("min_version")
        if found and version and min_version:
            version_ok = self._version_ge(version, min_version)

        install_command = self._get_install_command(dep_info, self._platform)

        return DepCheckResult(
            dep_id=dep_id,
            display_name=display_name,
            found=found,
            path=path,
            version=version,
            version_ok=version_ok,
            install_command=install_command,
            needed_for=needed_for,
            required=required,
            category=category,
        )

    def check_deps_for_tool(self, tool_name: str) -> List[DepCheckResult]:
        """
        Check all dependencies (required + optional) for a given tool.

        Args:
            tool_name: Tool key from ``tool_requirements`` (e.g. ``"summa"``).

        Returns:
            List of DepCheckResult, one per dependency.
        """
        assert self._registry is not None
        tool_reqs = self._registry.get("tool_requirements", {}).get(tool_name)
        if tool_reqs is None:
            return []

        results: List[DepCheckResult] = []
        for dep_id in tool_reqs.get("required", []):
            result = self.check_dep(dep_id)
            result.required = True
            results.append(result)
        for dep_id in tool_reqs.get("optional", []):
            result = self.check_dep(dep_id)
            result.required = False
            results.append(result)
        return results

    def get_required_deps_for_tool(self, tool_name: str) -> List[DepCheckResult]:
        """Return only the *required* deps for a tool."""
        return [r for r in self.check_deps_for_tool(tool_name) if r.required]

    def check_all_deps(self) -> List[DepCheckResult]:
        """Check every dependency in the registry."""
        assert self._registry is not None
        return [self.check_dep(dep_id) for dep_id in self._registry.get("dependencies", {})]

    def get_tool_names(self) -> List[str]:
        """Return all tool names that have registered requirements."""
        assert self._registry is not None
        return list(self._registry.get("tool_requirements", {}).keys())

    # ── Version helpers ──────────────────────────────────────────────

    @staticmethod
    def _extract_version(command: str, check: Dict[str, Any]) -> Optional[str]:
        """Run a command and extract its version string."""
        version_flag = check.get("version_flag", "--version")
        version_regex = check.get("version_regex", "")
        if not version_regex:
            return None

        cmd = [command]
        if version_flag:
            cmd.append(version_flag)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = result.stdout + result.stderr
            match = re.search(version_regex, output)
            if match:
                return match.group(1)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        return None

    @staticmethod
    def _check_wsl_command(cmd: str, check: Dict[str, Any]) -> tuple:
        """Probe for a command inside WSL and optionally extract its version.

        Returns:
            (found, version) tuple.
        """
        try:
            result = subprocess.run(
                ["wsl", "-e", "bash", "-c", f"command -v {cmd}"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return False, None
        except (subprocess.TimeoutExpired, OSError):
            return False, None

        # Extract version inside WSL
        version_flag = check.get("version_flag", "--version")
        version_regex = check.get("version_regex", "")
        if not version_regex:
            return True, None

        version_cmd = cmd
        if version_flag:
            version_cmd = f"{cmd} {version_flag}"
        try:
            result = subprocess.run(
                ["wsl", "-e", "bash", "-c", version_cmd],
                capture_output=True, text=True, timeout=10,
            )
            output = result.stdout + result.stderr
            match = re.search(version_regex, output)
            if match:
                return True, match.group(1)
        except (subprocess.TimeoutExpired, OSError):
            pass

        return True, None

    @staticmethod
    def _check_pkg_config(name: str) -> tuple:
        """Check via pkg-config and return (found, version)."""
        pkg_config = shutil.which("pkg-config")
        if not pkg_config:
            return False, None
        try:
            result = subprocess.run(
                [pkg_config, "--modversion", name],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return True, result.stdout.strip()
        except (subprocess.TimeoutExpired, OSError):
            pass
        return False, None

    @staticmethod
    def _check_conda_prefix(header_relpath: str) -> tuple:
        """Probe ``$CONDA_PREFIX`` for a known header file.

        On Windows Conda, CLI tools like ``nc-config`` are often absent, but
        the libraries are installed under ``$CONDA_PREFIX/Library/`` (Windows)
        or ``$CONDA_PREFIX/`` (Unix).

        Args:
            header_relpath: Relative path to check, e.g. ``"include/netcdf.h"``.

        Returns:
            ``(found, path_str)`` — *found* is True if the header exists.
        """
        prefix = os.environ.get("CONDA_PREFIX", "")
        if not prefix:
            return False, None

        # Windows Conda puts headers in Library/include/, Unix in include/
        candidates = [
            os.path.join(prefix, "Library", header_relpath),
            os.path.join(prefix, header_relpath),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return True, f"(conda: {candidate})"
        return False, None

    @staticmethod
    def _check_brew_keg_pkg_config(pkg_config_name: str, brew_formulas: List[str]) -> tuple:
        """Probe Homebrew keg-only packages for pkg-config metadata.

        Keg-only formulae (openblas, lapack, zlib, etc.) are not symlinked
        into the Homebrew prefix, so their ``.pc`` files are invisible to
        ``pkg-config`` by default.  This method runs ``brew --prefix <formula>``
        for each formula, collects ``lib/pkgconfig`` directories that exist,
        and re-runs ``pkg-config`` with those paths prepended.

        Args:
            pkg_config_name: The pkg-config module name (e.g. ``"openblas"``).
            brew_formulas: One or more Homebrew formula names to probe.

        Returns:
            ``(found, version)`` tuple, same contract as ``_check_pkg_config``.
        """
        pkg_config = shutil.which("pkg-config")
        brew = shutil.which("brew")
        if not pkg_config or not brew:
            return False, None

        keg_pc_dirs: List[str] = []
        for formula in brew_formulas:
            try:
                result = subprocess.run(
                    [brew, "--prefix", formula],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    prefix = result.stdout.strip()
                    pc_dir = os.path.join(prefix, "lib", "pkgconfig")
                    if os.path.isdir(pc_dir):
                        keg_pc_dirs.append(pc_dir)
            except (subprocess.TimeoutExpired, OSError):
                continue

        if not keg_pc_dirs:
            return False, None

        # Prepend keg dirs to existing PKG_CONFIG_PATH and retry
        extra = ":".join(keg_pc_dirs)
        env = os.environ.copy()
        env["PKG_CONFIG_PATH"] = extra + ":" + env.get("PKG_CONFIG_PATH", "")

        try:
            result = subprocess.run(
                [pkg_config, "--modversion", pkg_config_name],
                capture_output=True, text=True, timeout=5,
                env=env,
            )
            if result.returncode == 0:
                return True, result.stdout.strip()
        except (subprocess.TimeoutExpired, OSError):
            pass

        return False, None

    @staticmethod
    def _version_ge(actual: str, minimum: str) -> bool:
        """Return True if *actual* >= *minimum* using numeric tuple comparison."""
        def _parse(v: str):
            return tuple(int(x) for x in re.findall(r"\d+", v))
        try:
            return _parse(actual) >= _parse(minimum)
        except (ValueError, TypeError):
            return True  # can't parse → assume OK

    # ── Install command generation ───────────────────────────────────

    def generate_install_script(self, tool_name: Optional[str] = None) -> Optional[str]:
        """
        Generate a shell script that installs missing dependencies.

        Args:
            tool_name: If given, only deps for this tool. Otherwise all deps.

        Returns:
            Shell script string, or None if nothing to install.
        """
        if tool_name:
            results = self.check_deps_for_tool(tool_name)
        else:
            results = self.check_all_deps()

        missing = [r for r in results if not r.found and r.install_command]
        if not missing:
            return None

        assert self._registry is not None
        lines: List[str] = []

        if self._platform in (Platform.APT, Platform.DNF, Platform.BREW,
                               Platform.HPC_MODULE):
            lines.extend(["#!/usr/bin/env bash", "set -e", ""])

        if self._platform == Platform.APT:
            pkgs = []
            for r in missing:
                dep_info = self._registry["dependencies"].get(r.dep_id, {})
                pkg = dep_info.get("packages", {}).get("apt", "")
                if pkg:
                    pkgs.extend(pkg.split())
            if pkgs:
                lines.append("sudo apt-get update")
                lines.append(f"sudo apt-get install -y {' '.join(sorted(set(pkgs)))}")
        elif self._platform == Platform.DNF:
            pkgs = []
            for r in missing:
                dep_info = self._registry["dependencies"].get(r.dep_id, {})
                pkg = dep_info.get("packages", {}).get("dnf", "")
                if pkg:
                    pkgs.extend(pkg.split())
            if pkgs:
                lines.append(f"sudo dnf install -y {' '.join(sorted(set(pkgs)))}")
        elif self._platform == Platform.BREW:
            pkgs = []
            for r in missing:
                dep_info = self._registry["dependencies"].get(r.dep_id, {})
                pkg = dep_info.get("packages", {}).get("brew", "")
                if pkg:
                    pkgs.extend(pkg.split())
            if pkgs:
                lines.append(f"brew install {' '.join(sorted(set(pkgs)))}")
        elif self._platform == Platform.CONDA:
            # On Windows, prefer conda_win package names (m2- prefixed)
            pkg_key = "conda_win" if sys.platform == "win32" else "conda"
            pkgs = []
            for r in missing:
                dep_info = self._registry["dependencies"].get(r.dep_id, {})
                pkg = dep_info.get("packages", {}).get(pkg_key, "") or \
                      dep_info.get("packages", {}).get("conda", "")
                if pkg:
                    pkgs.extend(pkg.split())
            if pkgs:
                lines.append(f"conda install -c conda-forge {' '.join(sorted(set(pkgs)))}")
        elif self._platform == Platform.HPC_MODULE:
            for r in missing:
                dep_info = self._registry["dependencies"].get(r.dep_id, {})
                pkg = dep_info.get("packages", {}).get("hpc_module", "")
                if pkg:
                    lines.append(f"module load {pkg}")
        elif self._platform == Platform.MSYS2:
            pkgs = []
            for r in missing:
                dep_info = self._registry["dependencies"].get(r.dep_id, {})
                pkg = dep_info.get("packages", {}).get("msys2", "")
                if pkg:
                    pkgs.extend(pkg.split())
            if pkgs:
                lines.append(f"pacman -S --noconfirm {' '.join(sorted(set(pkgs)))}")
        elif self._platform == Platform.WSL:
            # Single wsl -e invocation running apt inside WSL
            pkgs = []
            for r in missing:
                dep_info = self._registry["dependencies"].get(r.dep_id, {})
                pkg = dep_info.get("packages", {}).get("apt", "")
                if pkg:
                    pkgs.extend(pkg.split())
            if pkgs:
                pkg_str = " ".join(sorted(set(pkgs)))
                lines.append(
                    f'wsl -e bash -c "sudo apt-get update && '
                    f'sudo apt-get install -y {pkg_str}"'
                )
        else:
            # Unknown platform — list individual commands
            for r in missing:
                if r.install_command:
                    lines.append(r.install_command)

        # Filter out preamble-only scripts
        content_lines = [l for l in lines if l and not l.startswith("#") and l != "set -e"]
        if not content_lines:
            return None

        return "\n".join(lines) + "\n"

    @staticmethod
    def _get_install_command(dep_info: Dict[str, Any], platform: Platform) -> Optional[str]:
        """Generate a platform-specific install hint."""
        packages = dep_info.get("packages", {})

        # Resolve the package name key for the current platform
        if platform == Platform.CONDA and sys.platform == "win32":
            pkg = packages.get("conda_win") or packages.get("conda")
        elif platform == Platform.WSL:
            # WSL runs Ubuntu by default — reuse apt package names
            pkg = packages.get("apt")
        else:
            pkg = packages.get(platform.value)

        if not pkg:
            return None

        if platform == Platform.APT:
            return f"sudo apt-get install -y {pkg}"
        elif platform == Platform.DNF:
            return f"sudo dnf install -y {pkg}"
        elif platform == Platform.BREW:
            return f"brew install {pkg}"
        elif platform == Platform.CONDA:
            return f"conda install -c conda-forge {pkg}"
        elif platform == Platform.HPC_MODULE:
            return f"module load {pkg}"
        elif platform == Platform.MSYS2:
            return f"pacman -S --noconfirm {pkg}"
        elif platform == Platform.WSL:
            return f"wsl -e sudo apt-get install -y {pkg}"
        return None
