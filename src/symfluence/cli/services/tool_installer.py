# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Tool installation service for SYMFLUENCE.

Handles cloning repositories, running build commands, and verifying installations.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..console import Console
from .base import BaseService


class ToolInstaller(BaseService):
    """
    Service for installing external tools from source.

    Handles:
    - Repository cloning
    - Build command execution
    - Dependency resolution
    - Installation verification
    """

    def __init__(
        self,
        external_tools: Optional[Dict[str, Any]] = None,
        console: Optional[Console] = None,
    ):
        """
        Initialize the ToolInstaller.

        Args:
            external_tools: Dictionary of tool definitions. If None, loads on demand.
            console: Console instance for output.
        """
        super().__init__(console=console)
        self._external_tools = external_tools

    @property
    def external_tools(self) -> Dict[str, Any]:
        """Lazy load external tools definitions."""
        if self._external_tools is None:
            from ..external_tools_config import get_external_tools_definitions
            self._external_tools = get_external_tools_definitions()
        return self._external_tools

    @staticmethod
    def _find_bash() -> Optional[str]:
        """
        Locate a usable bash executable for building from source.

        On Unix this is simply /bin/bash.  On Windows we prefer a native
        bash that can use the conda-forge build toolchain (gfortran, cmake,
        make) installed directly in the conda environment.  Search order:

        1. Conda m2-bash  (CONDA_PREFIX/Library/usr/bin/bash.exe)
        2. MSYS2           (C:/msys64/usr/bin/bash.exe)
        3. Git Bash         (C:/Program Files/Git/usr/bin/bash.exe)
        4. WSL              (last resort — uses a separate Linux filesystem)
        5. Anything else on PATH

        Returns:
            Absolute path to bash, or None if not found.
        """
        if sys.platform != "win32":
            if os.path.exists("/bin/bash"):
                return "/bin/bash"
            return shutil.which("bash")

        # 1. Conda m2-bash — lives inside the active conda environment and
        #    shares PATH with conda-forge gfortran/cmake/make.
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if conda_prefix:
            conda_bash = os.path.join(
                conda_prefix, "Library", "usr", "bin", "bash.exe"
            )
            if os.path.isfile(conda_bash):
                return conda_bash

        # 2. MSYS2 (full Unix toolchain on Windows)
        for candidate in [
            r"C:\msys64\usr\bin\bash.exe",
            r"C:\msys2\usr\bin\bash.exe",
        ]:
            if os.path.isfile(candidate):
                return candidate

        # 3. Git Bash
        for candidate in [
            r"C:\Program Files\Git\usr\bin\bash.exe",
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files (x86)\Git\usr\bin\bash.exe",
        ]:
            if os.path.isfile(candidate):
                return candidate

        # 4. WSL — last resort.  Provides a Linux environment but requires
        #    separate toolchain installation inside the WSL distro.
        wsl = shutil.which("wsl")
        if wsl:
            try:
                probe = subprocess.run(
                    [wsl, "-e", "bash", "-c", "echo ok"],
                    capture_output=True, text=True, timeout=10,
                )
                if probe.returncode == 0 and "ok" in probe.stdout:
                    return wsl  # handled specially in _run_build_commands
            except (subprocess.TimeoutExpired, OSError):
                pass

        # 5. Anything else on PATH
        return shutil.which("bash")

    def _preflight_check(self, tool_name: str) -> bool:
        """
        Verify that essential build tools are available before attempting a build.

        Uses the canonical SystemDepsRegistry when the tool has registered
        requirements; falls back to a legacy check for unregistered tools.

        Args:
            tool_name: Name of the tool about to be built (for messaging).

        Returns:
            True if the environment looks viable, False otherwise.
        """
        # Bash availability (shared by both paths)
        bash = self._find_bash()
        if bash is None:
            self._console.error(
                "No bash shell found.  Building from source requires bash "
                "(Git for Windows, MSYS2, or WSL)."
            )
            if sys.platform == "win32":
                self._console.indent(
                    "Install Git for Windows (https://git-scm.com) which includes Git Bash,"
                )
                self._console.indent(
                    "or use WSL: wsl --install"
                )
            return False

        # Try registry-driven check
        try:
            from .system_deps import SystemDepsRegistry

            registry = SystemDepsRegistry()
            results = registry.check_deps_for_tool(tool_name)

            if not results:
                # Tool not in registry — fall back to legacy
                return self._legacy_preflight_check(tool_name)

            # Display dependency table
            rows = []
            missing_required = []
            for r in results:
                if r.found:
                    status = "[green]OK[/green]"
                    if not r.version_ok:
                        status = "[yellow]OLD[/yellow]"
                    ver = r.version or "-"
                    loc = r.path or "-"
                else:
                    status = "[red]MISSING[/red]"
                    ver = "-"
                    loc = "-"
                    if r.required:
                        missing_required.append(r)

                req_flag = "required" if r.required else "optional"
                install = r.install_command or "-"
                rows.append([r.display_name, status, ver, loc, install, req_flag])

            self._console.table(
                columns=["Dependency", "Status", "Version", "Location", "Install With", ""],
                rows=rows,
                title=f"System Dependencies for {tool_name}",
            )

            if missing_required:
                names = ", ".join(r.display_name for r in missing_required)
                self._console.error(
                    f"Missing required dependencies for {tool_name}: {names}"
                )
                self._console.newline()
                self._console.indent("Install the missing dependencies:")
                for r in missing_required:
                    if r.install_command:
                        self._console.indent(f"  {r.install_command}")
                self._console.newline()
                self._console.indent(
                    "See docs/SYSTEM_REQUIREMENTS.md for full platform recipes."
                )

                if getattr(self, "_force", False):
                    self._console.warning(
                        "--force passed: proceeding despite missing dependencies"
                    )
                    return True

                return False

            return True

        except Exception:  # noqa: BLE001 — top-level fallback
            # Registry unavailable — fall through to legacy
            return self._legacy_preflight_check(tool_name)

    def _legacy_preflight_check(self, tool_name: str) -> bool:
        """
        Legacy preflight: check cmake, gfortran, make via shutil.which.

        Used as a fallback when the tool is not in the registry.

        Args:
            tool_name: Name of the tool about to be built.

        Returns:
            True if all essential build tools are found, False otherwise.
        """
        bash = self._find_bash()
        is_wsl = (
            sys.platform == "win32"
            and bash
            and os.path.basename(bash).lower() == "wsl.exe"
        )

        # Check against the build environment PATH (which may include
        # additional directories from _get_clean_build_env), not just
        # the current process PATH.
        build_env = self._get_clean_build_env()
        build_path = build_env.get("PATH", "")

        missing = []
        for tool in ("cmake", "gfortran", "make"):
            if is_wsl:
                assert bash is not None  # guaranteed by is_wsl check
                probe = subprocess.run(
                    [bash, "-e", "bash", "-c", f"command -v {tool}"],
                    capture_output=True, text=True,
                )
                if probe.returncode != 0:
                    missing.append(tool)
            elif not shutil.which(tool, path=build_path):
                missing.append(tool)

        if missing:
            self._console.error(
                f"Missing required build tools for {tool_name}: {', '.join(missing)}"
            )
            if sys.platform == "win32":
                self._console.indent(
                    "On Windows, install via conda-forge (recommended):"
                )
                self._console.indent(
                    "  conda install -c conda-forge gfortran cmake m2-make m2-bash"
                )
                self._console.indent(
                    "Or via MSYS2 (https://www.msys2.org):"
                )
                self._console.indent(
                    "  pacman -S mingw-w64-x86_64-cmake mingw-w64-x86_64-gcc-fortran make"
                )
            else:
                self._console.indent(
                    "Install them with your system package manager, e.g.:"
                )
                self._console.indent(
                    "  sudo apt install cmake gfortran make   # Debian/Ubuntu"
                )
                self._console.indent(
                    "  brew install cmake gcc make             # macOS"
                )
                # HPC hint: detect module system (lmod/environment-modules).
                # 'module' is a shell function, not a binary, so check env vars.
                has_modules = any(
                    os.environ.get(v) for v in
                    ("LMOD_DIR", "MODULESHOME", "LOADEDMODULES")
                )
                if has_modules:
                    self._console.indent(
                        "  module load cmake gcc              # HPC (check 'module avail')"
                    )

            if getattr(self, "_force", False):
                self._console.warning(
                    "--force passed: proceeding despite missing build tools"
                )
                return True

            return False

        return True

    def _get_clean_build_env(self) -> Dict[str, str]:
        """
        Get a clean environment for build processes.

        Removes MAKE-related variables that can cause spurious make calls
        during git submodule operations (common issue in 2i2c/JupyterHub).

        Returns:
            Clean environment dictionary for subprocess calls.
        """
        env = os.environ.copy()
        sep = os.pathsep  # ";" on Windows, ":" on Unix

        # Remove MAKE-related variables that can trigger unwanted make calls
        make_vars = ["MAKEFLAGS", "MAKELEVEL", "MAKE", "MFLAGS", "MAKEOVERRIDES"]
        for var in make_vars:
            env.pop(var, None)

        # On Windows, ensure conda's build-tool directories AND a source
        # of standard Unix utilities (awk, mkdir, rm, uname, ...) are on PATH.
        if sys.platform == "win32":
            conda_prefix = env.get("CONDA_PREFIX", "")
            if conda_prefix:
                extra_dirs = [
                    os.path.join(conda_prefix, "Library", "bin"),
                    os.path.join(conda_prefix, "Library", "usr", "bin"),
                    # MinGW binutils (ar, as, ld, ranlib, strip, nm, etc.)
                    os.path.join(conda_prefix, "Library", "x86_64-w64-mingw32", "bin"),
                ]
                current_path = env.get("PATH", "")
                for d in reversed(extra_dirs):
                    if os.path.isdir(d) and d not in current_path:
                        env["PATH"] = d + sep + env["PATH"]

            # Conda's m2-bash only ships bash + make.  Standard Unix utils
            # (awk, mkdir, rm, ls, uname, tar, ...) must come from Git Bash,
            # MSYS2, or additional m2- conda packages.  Append the first
            # available source of these utilities to PATH.
            unix_util_dirs = [
                # MSYS2 standalone install
                r"C:\msys64\usr\bin",
                r"C:\msys2\usr\bin",
                # Git for Windows (most common on dev machines)
                r"C:\Program Files\Git\usr\bin",
                r"C:\Program Files (x86)\Git\usr\bin",
            ]
            current_path = env.get("PATH", "")
            for d in unix_util_dirs:
                if os.path.isdir(d) and d not in current_path:
                    # Append (not prepend) so conda tools take priority
                    env["PATH"] = env["PATH"] + sep + d
                    break

            # Tell cmake to use MSYS Makefiles generator and the MSYS2 make
            # that ships with conda's m2-make package.
            env.setdefault("CMAKE_GENERATOR", "MSYS Makefiles")
            make_exe = shutil.which("make")
            if make_exe:
                env.setdefault("CMAKE_MAKE_PROGRAM", make_exe)

            # Set compiler names (not MSYS2 paths) so cmake resolves them
            # on PATH with the .exe extension.
            if shutil.which("gcc"):
                env.setdefault("CC", "gcc")
                env.setdefault("CXX", "g++")
            if shutil.which("gfortran"):
                env.setdefault("FC", "gfortran")

            # Ensure temp directory is set for Fortran compiler backends
            # (cc1.exe/f951.exe). Without TMP/TEMP, gfortran falls back to
            # C:\Windows which is not writable by normal users.
            import tempfile
            tmp_dir = tempfile.gettempdir()
            for var in ("TMPDIR", "TMP", "TEMP"):
                env.setdefault(var, tmp_dir)

        # Compiler / PATH setup for Unix (conda compilers, 2i2c, etc.)
        if sys.platform != "win32":
            # Check if conda compilers are available (required for ABI compatibility
            # with conda-forge libraries like GDAL built with GCC 13+)
            conda_prefix = env.get("CONDA_PREFIX", "")
            conda_gcc = os.path.join(conda_prefix, "bin", "x86_64-conda-linux-gnu-gcc")
            conda_gxx = os.path.join(conda_prefix, "bin", "x86_64-conda-linux-gnu-g++")

            if conda_prefix and os.path.exists(conda_gcc) and os.path.exists(conda_gxx):
                # Use conda compilers for ABI compatibility with conda libraries
                env["CC"] = conda_gcc
                env["CXX"] = conda_gxx
                # Ensure conda bin is first in PATH
                conda_bin = os.path.join(conda_prefix, "bin")
                if conda_bin not in env.get("PATH", "").split(sep)[0]:
                    env["PATH"] = conda_bin + sep + env.get("PATH", "")
            elif os.path.exists("/srv/conda/envs/notebook"):
                # 2i2c environment without conda compilers - try system compilers
                # but warn that this may fail if conda GDAL needs newer ABI
                if "/usr/bin" not in env.get("PATH", "").split(sep)[0]:
                    env["PATH"] = "/usr/bin" + sep + env.get("PATH", "")
                if os.path.exists("/usr/bin/gcc"):
                    env["CC"] = "/usr/bin/gcc"
                if os.path.exists("/usr/bin/g++"):
                    env["CXX"] = "/usr/bin/g++"
                if os.path.exists("/usr/bin/ld"):
                    env["LD"] = "/usr/bin/ld"
            else:
                # Non-2i2c, non-conda environment (HPC, desktop Linux).
                # Append /usr/bin as a fallback so basic tools are reachable,
                # but do NOT prepend — that would override HPC module-loaded
                # compilers (e.g. gfortran from `module load gcc/13.3`).
                current_path = env.get("PATH", "")
                if "/usr/bin" not in current_path.split(sep):
                    env["PATH"] = current_path + sep + "/usr/bin"

        # Ensure build scripts install Python packages into the same
        # interpreter that is running symfluence.  Without this, Docker/2i2c
        # images where multiple conda envs exist can end up with troute (or
        # other pip-installed tools) landing in the wrong environment.
        env.setdefault("SYMFLUENCE_PYTHON", sys.executable)

        # Set SYMFLUENCE_PATCHED if patched mode is enabled
        if getattr(self, '_patched', False):
            env["SYMFLUENCE_PATCHED"] = "1"

        return env

    def install(
        self,
        specific_tools: Optional[List[str]] = None,
        symfluence_instance=None,
        force: bool = False,
        dry_run: bool = False,
        patched: bool = False,
        branch_override: Optional[str] = None,
        git_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Clone and install external tool repositories with dependency resolution.

        Args:
            specific_tools: List of specific tools to install. If None, installs all.
            symfluence_instance: Optional SYMFLUENCE instance with config.
            force: If True, reinstall even if already exists.
            dry_run: If True, only show what would be done.
            patched: If True, apply SYMFLUENCE patches (e.g., RHESSys GW recharge).
            branch_override: If set, override the default branch for the tool(s).
            git_hash: If set, checkout this specific commit hash after cloning.

        Returns:
            Dictionary with installation results.
        """
        # Store flags for use in _get_clean_build_env and _preflight_check
        self._patched = patched
        self._force = force
        action = "Planning" if dry_run else "Installing"
        self._console.panel(f"{action} External Tools", style="blue")

        if dry_run:
            self._console.info("[DRY RUN] No actual installation will occur")
            self._console.rule()

        installation_results = {
            "successful": [],
            "failed": [],
            "skipped": [],
            "errors": [],
            "dry_run": dry_run,
        }

        config = self._load_config(symfluence_instance)
        install_base_dir = self._get_data_dir(config) / "installs"

        self._console.info(f"Installation directory: {install_base_dir}")

        if not dry_run:
            install_base_dir.mkdir(parents=True, exist_ok=True)

        # Determine which tools to install
        if specific_tools is None:
            # Install all non-optional tools by default
            tools_to_install = [
                name for name, info in self.external_tools.items()
                if not info.get('optional', False)
            ]
        else:
            tools_to_install = []
            for tool in specific_tools:
                if tool in self.external_tools:
                    tools_to_install.append(tool)
                else:
                    self._console.warning(f"Unknown tool: {tool}")
                    installation_results["errors"].append(f"Unknown tool: {tool}")

        # Resolve dependencies and sort by install order
        tools_to_install = self._resolve_dependencies(tools_to_install)

        self._console.info(f"Installing tools in order: {', '.join(tools_to_install)}")

        # Install each tool
        for tool_name in tools_to_install:
            tool_info = self.external_tools[tool_name]
            self._console.newline()
            self._console.info(f"[bold]{action} {tool_name.upper()}:[/bold]")
            self._console.indent(tool_info.get("description", ""))

            tool_install_dir = install_base_dir / tool_info.get("install_dir", tool_name)
            repository_url = tool_info.get("repository")
            branch = branch_override if branch_override else tool_info.get("branch")

            try:
                # Check if already exists
                if tool_install_dir.exists() and not force:
                    self._console.indent(f"Skipping - already exists at: {tool_install_dir}")
                    self._console.indent("Use --force to reinstall")
                    installation_results["skipped"].append(tool_name)
                    continue

                if dry_run:
                    self._console.indent(f"Would clone: {repository_url}")
                    if branch:
                        self._console.indent(f"Would checkout branch: {branch}")
                    if git_hash:
                        self._console.indent(f"Would checkout commit: {git_hash}")
                    self._console.indent(f"Target directory: {tool_install_dir}")
                    self._console.indent("Would run build commands:")
                    for cmd in tool_info.get("build_commands", []):
                        self._console.indent(f"  {cmd[:100]}...", level=2)
                    installation_results["successful"].append(f"{tool_name} (dry run)")
                    continue

                # Remove existing if force reinstall
                if tool_install_dir.exists() and force:
                    self._console.indent(f"Removing existing installation: {tool_install_dir}")
                    # On Windows, git repos contain read-only pack files and
                    # may have broken symlinks/junctions (e.g. SUNDIALS docs).
                    # This handler retries with permissions cleared, and
                    # silently ignores paths that no longer exist.
                    import stat
                    def _force_remove_readonly(func, path, exc_info):
                        try:
                            os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                            func(path)
                        except FileNotFoundError:
                            pass  # already gone — nothing to do
                        except OSError:
                            # Broken symlink / junction — try each removal method
                            for remover in (os.unlink, os.rmdir):
                                try:
                                    remover(path)
                                    return
                                except OSError:
                                    continue
                    shutil.rmtree(tool_install_dir, onerror=_force_remove_readonly)

                # Clone repository or create directory
                if not self._clone_repository(
                    repository_url, branch, tool_install_dir, git_hash=git_hash
                ):
                    installation_results["failed"].append(tool_name)
                    continue

                # Check dependencies
                missing_deps = self._check_system_dependencies(
                    tool_info.get("dependencies", [])
                )
                if missing_deps:
                    self._console.warning(
                        f"Missing system dependencies: {', '.join(missing_deps)}"
                    )
                    self._console.indent(
                        "These may be available as modules - check with 'module avail'"
                    )
                    installation_results["errors"].append(
                        f"{tool_name}: missing system dependencies {missing_deps}"
                    )

                # Check required tools
                if tool_info.get("requires"):
                    required_tools = tool_info.get("requires", [])
                    missing_required_tool = False
                    for req_tool in required_tools:
                        req_tool_info = self.external_tools.get(req_tool, {})
                        req_tool_dir = install_base_dir / req_tool_info.get(
                            "install_dir", req_tool
                        )
                        if not req_tool_dir.exists():
                            error_msg = (
                                f"{tool_name} requires {req_tool} but it's not installed"
                            )
                            self._console.error(error_msg)
                            installation_results["errors"].append(error_msg)
                            installation_results["failed"].append(tool_name)
                            missing_required_tool = True
                            break
                    if missing_required_tool:
                        continue

                # Run build commands
                if tool_info.get("build_commands"):
                    success = self._run_build_commands(
                        tool_name, tool_info, tool_install_dir
                    )
                    if success:
                        installation_results["successful"].append(tool_name)
                    else:
                        installation_results["failed"].append(tool_name)
                        installation_results["errors"].append(f"{tool_name} build failed")
                else:
                    self._console.success("No build required")
                    installation_results["successful"].append(tool_name)

                # Verify installation
                verified = self._verify_installation(tool_name, tool_info, tool_install_dir)
                if not verified:
                    installation_results["errors"].append(
                        f"{tool_name}: installation verification failed"
                    )
                    if tool_name in installation_results["successful"]:
                        installation_results["successful"] = [
                            t for t in installation_results["successful"] if t != tool_name
                        ]
                    if tool_name not in installation_results["failed"]:
                        installation_results["failed"].append(tool_name)

            except subprocess.CalledProcessError as e:
                if repository_url:
                    error_msg = f"Failed to clone {repository_url}: {e.stderr if e.stderr else str(e)}"
                else:
                    error_msg = f"Failed during installation: {e.stderr if e.stderr else str(e)}"
                self._console.error(error_msg)
                installation_results["failed"].append(tool_name)
                installation_results["errors"].append(f"{tool_name}: {error_msg}")

            except Exception as e:  # noqa: BLE001 — top-level fallback
                error_msg = f"Installation error: {str(e)}"
                self._console.error(error_msg)
                installation_results["failed"].append(tool_name)
                installation_results["errors"].append(f"{tool_name}: {error_msg}")

        # Print summary
        self._print_installation_summary(installation_results, dry_run)

        # Generate toolchain metadata after successful installs
        if not dry_run and installation_results["successful"]:
            self._generate_toolchain_metadata(install_base_dir)

        return installation_results

    def _clone_repository(
        self,
        repository_url: Optional[str],
        branch: Optional[str],
        target_dir: Path,
        git_hash: Optional[str] = None,
    ) -> bool:
        """
        Clone a git repository, optionally checking out a specific commit.

        Args:
            repository_url: URL of the repository to clone.
            branch: Branch to checkout. If None, uses default branch.
            target_dir: Target directory for the clone.
            git_hash: If set, checkout this specific commit after cloning.

        Returns:
            True if successful, False otherwise.
        """
        if repository_url:
            self._console.indent(f"Cloning from: {repository_url}")
            # Use shallow clone unless a specific commit hash is needed
            shallow = git_hash is None
            if branch:
                self._console.indent(f"Checking out branch: {branch}")
                clone_cmd = [
                    "git",
                    "clone",
                    *(["--depth", "1"] if shallow else []),
                    "-b",
                    branch,
                    repository_url,
                    str(target_dir),
                ]
            else:
                clone_cmd = [
                    "git",
                    "clone",
                    *(["--depth", "1"] if shallow else []),
                    repository_url,
                    str(target_dir),
                ]

            subprocess.run(
                clone_cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=600,  # 10-minute timeout to prevent hanging clones
                env=self._get_clean_build_env(),
            )
            self._console.success("Clone successful")

            # Checkout specific commit hash if requested
            if git_hash:
                self._console.indent(f"Checking out commit: {git_hash}")
                subprocess.run(
                    ["git", "checkout", git_hash],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=str(target_dir),
                    env=self._get_clean_build_env(),
                )
                self._console.success(f"Checked out {git_hash}")
        else:
            self._console.indent("Creating installation directory")
            target_dir.mkdir(parents=True, exist_ok=True)
            self._console.success(f"Directory created: {target_dir}")

        return True

    def _run_build_commands(
        self,
        tool_name: str,
        tool_info: Dict[str, Any],
        install_dir: Path,
    ) -> bool:
        """
        Run build commands for a tool.

        Args:
            tool_name: Name of the tool being built.
            tool_info: Tool configuration dictionary.
            install_dir: Installation directory.

        Returns:
            True if build successful, False otherwise.
        """
        # Pre-flight: ensure bash and build tools are available
        if not self._preflight_check(tool_name):
            return False

        bash = self._find_bash()
        # _preflight_check already validated bash exists, but guard anyway
        if bash is None:
            self._console.error("Cannot locate bash — aborting build.")
            return False

        self._console.indent("Running build commands...")

        is_wsl = (
            sys.platform == "win32"
            and os.path.basename(bash).lower() == "wsl.exe"
        )
        if sys.platform == "win32":
            if is_wsl:
                self._console.indent("Using WSL build environment")
            else:
                self._console.indent(f"Using native bash: {bash}")

        original_dir = os.getcwd()
        os.chdir(install_dir)

        try:
            combined_script = "\n".join(tool_info.get("build_commands", []))

            # Write the script to a temp file rather than passing via -c.
            # This avoids Windows command-line length limits (~32K) and
            # quoting issues with long multi-line bash scripts.
            script_file = Path(install_dir) / "_build.sh"
            script_file.write_text(combined_script + "\n", encoding="utf-8", newline="\n")

            # Security note: build_commands are multi-line bash scripts from
            # internal tool definitions, not user input.
            if is_wsl:
                # Convert the Windows install_dir to a WSL path so the
                # script runs in the correct directory.
                wsl_dir = subprocess.run(
                    [bash, "-e", "wslpath", "-a", str(install_dir)],
                    capture_output=True, text=True,
                ).stdout.strip()
                wsl_script = f"{wsl_dir}/_build.sh"
                cmd = [bash, "-e", "bash", wsl_script]
            else:
                # Direct bash invocation (Unix, Git Bash, MSYS2).
                cmd = [bash, str(script_file)]

            build_result = subprocess.run(  # nosec B603
                cmd,
                check=True,
                capture_output=True,
                text=True,
                env=self._get_clean_build_env(),
            )

            # Show output for critical tools
            if tool_name in ["summa", "sundials", "mizuroute", "fuse", "ngen", "cfuse", "enzyme"]:
                if build_result.stdout:
                    self._console.indent("=== Build Output ===", level=2)
                    for line in build_result.stdout.strip().split("\n"):
                        self._console.indent(line, level=3)
            else:
                if build_result.stdout:
                    lines = build_result.stdout.strip().split("\n")
                    for line in lines[-10:]:
                        self._console.indent(line, level=3)

            self._console.success("Build successful")
            return True

        except subprocess.CalledProcessError as build_error:
            self._console.error(f"Build failed: {build_error}")
            if build_error.stdout:
                self._console.indent("=== Build Output ===", level=2)
                for line in build_error.stdout.strip().split("\n"):
                    self._console.indent(line, level=3)
            if build_error.stderr:
                self._console.indent("=== Error Output ===", level=2)
                for line in build_error.stderr.strip().split("\n"):
                    self._console.indent(line, level=3)
            return False

        finally:
            os.chdir(original_dir)

    def _verify_installation(
        self, tool_name: str, tool_info: Dict[str, Any], install_dir: Path
    ) -> bool:
        """
        Verify that a tool was installed correctly.

        Args:
            tool_name: Name of the tool.
            tool_info: Tool configuration dictionary.
            install_dir: Installation directory.

        Returns:
            True if verification passed, False otherwise.
        """
        try:
            verify = tool_info.get("verify_install")
            if verify and isinstance(verify, dict):
                check_type = verify.get("check_type", "exists_all")
                candidates = [install_dir / p for p in verify.get("file_paths", [])]

                if check_type in ("python_module", "python_import"):
                    module_name = verify.get("python_import", tool_name)
                    try:
                        # Use a subprocess for the import check — the build
                        # ran in a child process so packages it installed via
                        # pip are not visible to our in-process importlib.
                        import subprocess as _sp
                        import sys as _sys
                        _r = _sp.run(
                            [_sys.executable, "-c", f"import {module_name}"],
                            capture_output=True, timeout=15,
                        )
                        ok = _r.returncode == 0
                    except Exception:  # noqa: BLE001 — top-level fallback
                        ok = False
                    status = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
                    self._console.indent(f"Install verification ({check_type}): {status}")
                    return ok
                elif check_type == "exists_any":
                    ok = any(p.exists() for p in candidates)
                elif check_type in ("exists_all", "exists"):
                    ok = all(p.exists() for p in candidates)
                else:
                    ok = False

                status = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
                self._console.indent(f"Install verification ({check_type}): {status}")
                for p in candidates:
                    check = "[green]Y[/green]" if p.exists() else "[red]N[/red]"
                    self._console.indent(f"  {check} {p}", level=2)
                return ok

            exe_name = tool_info.get("default_exe")
            if not exe_name:
                return False

            possible_paths = [
                install_dir / exe_name,
                install_dir / "bin" / exe_name,
                install_dir / "build" / exe_name,
                install_dir / "route" / "bin" / exe_name,
                install_dir / exe_name.replace(".exe", ""),
                install_dir / "install" / "sundials" / exe_name,
            ]

            for exe_path in possible_paths:
                if exe_path.exists():
                    self._console.success(f"Executable/library found: {exe_path}")
                    return True

            return False

        except Exception as e:  # noqa: BLE001 — top-level fallback
            self._console.warning(f"Verification error: {str(e)}")
            return False

    def _resolve_dependencies(self, tools: List[str]) -> List[str]:
        """
        Resolve dependencies between tools and return sorted list.

        Args:
            tools: List of tool names to install.

        Returns:
            Sorted list with dependencies included.
        """
        tools_with_deps = set(tools)
        for tool in tools:
            if tool in self.external_tools and self.external_tools.get(tool, {}).get(
                "requires"
            ):
                required = self.external_tools.get(tool, {}).get("requires", [])
                tools_with_deps.update(required)

        return sorted(
            tools_with_deps,
            key=lambda t: (self.external_tools.get(t, {}).get("order", 999), t),
        )

    def _check_system_dependencies(self, dependencies: List[str]) -> List[str]:
        """
        Check which system dependencies are missing.

        Args:
            dependencies: List of required system binaries.

        Returns:
            List of missing dependencies.
        """
        missing_deps = []
        for dep in dependencies:
            if not shutil.which(dep):
                missing_deps.append(dep)
        return missing_deps

    def _print_installation_summary(
        self, results: Dict[str, Any], dry_run: bool
    ) -> None:
        """
        Print installation summary.

        Args:
            results: Installation results dictionary.
            dry_run: Whether this was a dry run.
        """
        successful_count = len(results["successful"])
        failed_count = len(results["failed"])
        skipped_count = len(results["skipped"])

        self._console.newline()
        self._console.info("Installation Summary:")
        if dry_run:
            self._console.indent(f"Would install: {successful_count} tools")
            self._console.indent(f"Would skip: {skipped_count} tools")
        else:
            self._console.indent(f"Successful: {successful_count} tools")
            self._console.indent(f"Failed: {failed_count} tools")
            self._console.indent(f"Skipped: {skipped_count} tools")

        if results["errors"]:
            self._console.newline()
            self._console.error("Errors encountered:")
            for error in results["errors"]:
                self._console.indent(f"- {error}")

    def _generate_toolchain_metadata(self, install_base_dir: Path) -> None:
        """
        Generate toolchain.json with compiler, library, and tool metadata.

        Called automatically after successful installs so that
        ``symfluence doctor`` can report toolchain information.

        Args:
            install_base_dir: The ``installs/`` directory.
        """
        import json
        import platform
        from datetime import datetime, timezone

        toolchain: Dict[str, Any] = {}

        # Platform
        arch = platform.machine()
        os_name = {"Darwin": "macos", "Linux": "linux"}.get(
            platform.system(), platform.system().lower()
        )
        toolchain["platform"] = f"{os_name}-{arch}"
        toolchain["build_date"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        # SYMFLUENCE version
        try:
            from symfluence.symfluence_version import __version__
            toolchain["symfluence_version"] = __version__
        except ImportError:
            toolchain["symfluence_version"] = "unknown"

        # Compilers
        compilers: Dict[str, str] = {}
        for name, cmds in [
            ("fortran", ["gfortran --version", "ifort --version"]),
            ("c", ["gcc --version", "clang --version"]),
            ("cxx", ["g++ --version", "clang++ --version"]),
        ]:
            for cmd in cmds:
                exe = cmd.split()[0]
                if shutil.which(exe):
                    try:
                        r = subprocess.run(
                            cmd.split(), capture_output=True, text=True, timeout=5,
                        )
                        compilers[name] = r.stdout.strip().split("\n")[0]
                    except Exception:  # noqa: BLE001 — top-level fallback
                        compilers[name] = exe
                    break
            else:
                compilers[name] = "not found"
        toolchain["compilers"] = compilers

        # Libraries
        libraries: Dict[str, str] = {}
        for lib_name, cmd in [
            ("netcdf", "nc-config --version"),
            ("netcdf_fortran", "nf-config --version"),
            ("hdf5", "h5cc -showconfig"),
        ]:
            exe = cmd.split()[0]
            if shutil.which(exe):
                try:
                    r = subprocess.run(
                        cmd.split(), capture_output=True, text=True, timeout=5,
                    )
                    out = r.stdout.strip()
                    if lib_name == "hdf5":
                        for line in out.split("\n"):
                            if "HDF5 Version" in line:
                                out = line.split(":")[-1].strip()
                                break
                    libraries[lib_name] = out.split("\n")[0]
                except Exception:  # noqa: BLE001 — top-level fallback
                    libraries[lib_name] = "unknown"
            else:
                libraries[lib_name] = "not found"
        toolchain["libraries"] = libraries

        # Installed tools — scan git repos for commit/branch info
        tools_meta: Dict[str, Any] = {}
        for name, info in self.external_tools.items():
            tool_dir = install_base_dir / info.get("install_dir", name)
            if not tool_dir.exists():
                continue
            entry: Dict[str, Any] = {"installed": True}
            git_dir = tool_dir / ".git"
            if git_dir.exists():
                try:
                    commit = subprocess.run(
                        ["git", "rev-parse", "HEAD"],
                        capture_output=True, text=True, cwd=tool_dir, timeout=5,
                    ).stdout.strip()
                    branch = subprocess.run(
                        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                        capture_output=True, text=True, cwd=tool_dir, timeout=5,
                    ).stdout.strip()
                    entry["commit"] = commit
                    entry["branch"] = branch
                except Exception:  # noqa: BLE001 — top-level fallback
                    pass
            tools_meta[name] = entry
        toolchain["tools"] = tools_meta

        # Write
        out_path = install_base_dir / "toolchain.json"
        try:
            out_path.write_text(
                json.dumps(toolchain, indent=2) + "\n", encoding="utf-8",
            )
            self._console.success(f"Toolchain metadata written to {out_path}")
        except OSError as e:
            self._console.warning(f"Could not write toolchain metadata: {e}")
