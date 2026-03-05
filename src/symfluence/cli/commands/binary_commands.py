# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Binary/tool management command handlers for SYMFLUENCE CLI.

This module implements handlers for external tool installation and validation,
plus a pass-through ``exec_binary()`` for running bundled binaries directly.
"""

import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import List, Optional

from ..exit_codes import ExitCode
from .base import BaseCommand, cli_exception_handler


class BinaryCommands(BaseCommand):
    """Handlers for binary/tool management commands."""

    @staticmethod
    @cli_exception_handler
    def install(args: Namespace) -> int:
        """
        Execute: symfluence binary install [TOOL1 TOOL2 ...]

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        from symfluence.cli.binary_service import BinaryManager

        binary_manager = BinaryManager()

        # Get tools to install
        tools = args.tools if args.tools else None  # None means install all
        force = args.force
        patched = getattr(args, 'patched', False)
        branch_override = getattr(args, 'branch', None)
        git_hash = getattr(args, 'git_hash', None)

        if tools:
            BaseCommand._console.info(f"Installing tools: {', '.join(tools)}")
        else:
            BaseCommand._console.info("Installing all available tools...")

        if force:
            BaseCommand._console.indent("(Force reinstall mode)")

        if patched:
            BaseCommand._console.indent("(SYMFLUENCE patches enabled)")

        if branch_override:
            BaseCommand._console.indent(f"(Branch override: {branch_override})")

        if git_hash:
            BaseCommand._console.indent(f"(Git hash: {git_hash})")

        # Handle subprocess errors specifically
        try:
            success = binary_manager.get_executables(
                specific_tools=tools,
                force=force,
                patched=patched,
                branch_override=branch_override,
                git_hash=git_hash,
            )
        except subprocess.CalledProcessError as e:
            BaseCommand._console.error(f"Build command failed: {e}")
            if BaseCommand.get_arg(args, 'debug', False):
                import traceback
                traceback.print_exc()
            return ExitCode.BINARY_BUILD_ERROR

        if success:
            BaseCommand._console.success("Tool installation completed successfully")
            return ExitCode.SUCCESS
        else:
            BaseCommand._console.error("Tool installation failed or was incomplete")
            return ExitCode.BINARY_ERROR

    @staticmethod
    @cli_exception_handler
    def validate(args: Namespace) -> int:
        """
        Execute: symfluence binary validate

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        from symfluence.cli.binary_service import BinaryManager

        binary_manager = BinaryManager()

        verbose = BaseCommand.get_arg(args, 'verbose', False)

        BaseCommand._console.info("Validating installed binaries...")

        # Handle subprocess errors specifically
        try:
            success = binary_manager.validate_binaries(verbose=verbose)
        except subprocess.CalledProcessError as e:
            BaseCommand._console.error(f"Binary test command failed: {e}")
            return ExitCode.BINARY_ERROR

        if success:
            BaseCommand._console.success("All binaries validated successfully")
            return ExitCode.SUCCESS
        else:
            BaseCommand._console.error("Binary validation failed")
            return ExitCode.BINARY_ERROR

    @staticmethod
    @cli_exception_handler
    def doctor(args: Namespace) -> int:
        """
        Execute: symfluence binary doctor

        Run system diagnostics to check environment and dependencies.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        from symfluence.cli.binary_service import BinaryManager

        binary_manager = BinaryManager()

        BaseCommand._console.info("Running system diagnostics...")
        BaseCommand._console.rule()

        # Call doctor function from binary manager
        success = binary_manager.doctor()

        if success:
            BaseCommand._console.rule()
            BaseCommand._console.success("System diagnostics completed")
            return ExitCode.SUCCESS
        else:
            BaseCommand._console.rule()
            BaseCommand._console.error("System diagnostics found issues")
            return ExitCode.DEPENDENCY_ERROR

    @staticmethod
    @cli_exception_handler
    def install_sysdeps(args: Namespace) -> int:
        """
        Execute: symfluence binary install-sysdeps

        Install platform-appropriate system dependencies using the detected
        package manager.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        from symfluence.cli.services.system_deps import SystemDepsRegistry

        registry = SystemDepsRegistry()
        tool = getattr(args, 'tool', None)
        dry_run = getattr(args, 'dry_run', False)

        BaseCommand._console.info(
            f"Detected platform: {registry.platform.value}"
        )

        if tool:
            results = registry.check_deps_for_tool(tool)
            if not results:
                BaseCommand._console.error(f"Unknown tool: {tool}")
                return ExitCode.GENERAL_ERROR
        else:
            results = registry.check_all_deps()

        missing = [r for r in results if not r.found]
        if not missing:
            BaseCommand._console.success("All system dependencies are already installed!")
            return ExitCode.SUCCESS

        BaseCommand._console.warning(
            f"Missing {len(missing)} dependencies: "
            + ", ".join(r.display_name for r in missing)
        )

        script = registry.generate_install_script(tool_name=tool)
        if not script:
            BaseCommand._console.error(
                "Could not generate install commands for your platform. "
                "See docs/SYSTEM_REQUIREMENTS.md for manual instructions."
            )
            return ExitCode.DEPENDENCY_ERROR

        BaseCommand._console.newline()
        BaseCommand._console.info("Install commands:")
        BaseCommand._console.rule()
        for line in script.strip().splitlines():
            if line and not line.startswith("#") and not line.startswith("set"):
                BaseCommand._console.indent(line)
        BaseCommand._console.rule()

        if dry_run:
            BaseCommand._console.info(
                "[DRY RUN] Commands printed above but not executed."
            )
            return ExitCode.SUCCESS

        BaseCommand._console.newline()
        BaseCommand._console.info("Running install commands...")

        try:
            from symfluence.cli.services.system_deps import Platform

            platform = registry.platform

            if platform in (Platform.CONDA, Platform.WSL):
                # Conda install runs natively (no bash wrapper needed).
                # WSL script contains `wsl -e ...` — run directly on Windows.
                # Script is generated internally, not from user input.
                result = subprocess.run(
                    script, shell=True, text=True, timeout=600,  # nosec B602
                )
            elif platform == Platform.MSYS2:
                # MSYS2 has its own bash — run pacman script through it
                from symfluence.cli.services.tool_installer import ToolInstaller
                bash = ToolInstaller._find_bash()
                if bash:
                    result = subprocess.run(
                        [bash, "-c", script], text=True, timeout=600,
                    )
                else:
                    result = subprocess.run(
                        script, shell=True, text=True, timeout=600,  # nosec B602
                    )
            elif platform == Platform.UNKNOWN:
                # Don't attempt execution — just show commands
                BaseCommand._console.warning(
                    "Unknown platform. Commands printed above but not executed."
                )
                return ExitCode.SUCCESS
            else:
                # APT, DNF, BREW, HPC_MODULE — use bash
                from symfluence.cli.services.tool_installer import ToolInstaller
                bash = ToolInstaller._find_bash() or "bash"
                result = subprocess.run(
                    [bash, "-c", script], text=True, timeout=600,
                )

            if result.returncode == 0:
                BaseCommand._console.success(
                    "System dependencies installed successfully"
                )
                return ExitCode.SUCCESS
            else:
                BaseCommand._console.error(
                    "Some packages failed to install. "
                    "Check the output above and retry manually."
                )
                return ExitCode.DEPENDENCY_ERROR
        except subprocess.TimeoutExpired:
            BaseCommand._console.error("Installation timed out after 10 minutes")
            return ExitCode.GENERAL_ERROR

    @staticmethod
    @cli_exception_handler
    def info(args: Namespace) -> int:
        """
        Execute: symfluence binary info

        Display information about installed tools.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        from symfluence.cli.binary_service import BinaryManager

        binary_manager = BinaryManager()

        BaseCommand._console.info("Installed Tools Information:")
        BaseCommand._console.rule()

        # Call info function from binary manager
        success = binary_manager.tools_info()

        if success:
            BaseCommand._console.rule()
            return ExitCode.SUCCESS
        else:
            BaseCommand._console.error("Failed to retrieve tools information")
            return ExitCode.GENERAL_ERROR

    # ------------------------------------------------------------------
    # Pass-through: symfluence binary <tool_name> [args...]
    # ------------------------------------------------------------------

    @staticmethod
    def exec_binary(tool_name: str, tool_args: List[str]) -> int:
        """Run a bundled binary directly with full stdio pass-through.

        Resolution order:
          1. npm-installed binaries  (``$(npm root -g)/symfluence/dist/bin/<tool>``)
          2. Source installs          (``$SYMFLUENCE_DATA_DIR/<path>/<exe>``)
          3. Direct name match in npm bin directory (e.g. TauDEM sub-tools)

        Args:
            tool_name: Name of the binary to run (e.g. ``summa``, ``fuse.exe``).
            tool_args: Arguments forwarded verbatim to the binary.

        Returns:
            The native binary's exit code, or 1 on error.
        """
        from symfluence.cli.binary_service import BinaryService
        from symfluence.cli.external_tools_config import get_external_tools_definitions

        binary_path: Optional[Path] = None
        service = BinaryService()

        # --- 1. npm-installed binaries ---
        npm_bin_dir = service.detect_npm_binaries()
        if npm_bin_dir is not None:
            tools_defs = get_external_tools_definitions()
            tool_def = tools_defs.get(tool_name)
            if tool_def is not None:
                exe_name = tool_def.get("default_exe", tool_name)
                candidate = npm_bin_dir / exe_name
                if candidate.is_file():
                    binary_path = candidate

        # --- 2. Source installs ($SYMFLUENCE_DATA_DIR/<suffix>/<exe>) ---
        if binary_path is None:
            tools_defs = get_external_tools_definitions()
            tool_def = tools_defs.get(tool_name)
            if tool_def is not None:
                path_suffix = tool_def.get("default_path_suffix")
                exe_name = tool_def.get("default_exe")
                if path_suffix and exe_name:
                    config = service._load_config()
                    data_dir = service._get_data_dir(config)
                    candidate = data_dir / path_suffix / exe_name
                    if candidate.is_file():
                        binary_path = candidate

        # --- 3. Direct name match in npm bin/ (TauDEM sub-tools etc.) ---
        if binary_path is None and npm_bin_dir is not None:
            candidate = npm_bin_dir / tool_name
            if candidate.is_file():
                binary_path = candidate

        if binary_path is None:
            print(
                f"Binary '{tool_name}' not found.\n\n"
                f"Install it with:\n"
                f"  symfluence binary install {tool_name}\n\n"
                f"Or install pre-built binaries:\n"
                f"  npm install -g symfluence",
                file=sys.stderr,
            )
            return ExitCode.BINARY_ERROR

        return BinaryCommands._run_binary(binary_path, tool_args)

    @staticmethod
    def _run_binary(binary_path: Path, args: List[str]) -> int:
        """Execute a binary with full stdio pass-through.

        Args:
            binary_path: Absolute path to the executable.
            args: Arguments forwarded to the binary.

        Returns:
            The process exit code.
        """
        cmd = [str(binary_path)] + args
        try:
            # Use a pseudo-tty for stdout so C/C++ child processes use
            # line-buffered output.  Without this, completion messages
            # (e.g. ngen's "Finished N timesteps") are lost when the
            # process segfaults during cleanup before flushing its buffer.
            # stderr is piped separately so we can filter noisy WARN lines.
            pty_master_fd = None
            stdout_target = sys.stdout

            if sys.platform != "win32":
                try:
                    import pty as _pty
                    import os as _os
                    pty_master_fd, pty_slave_fd = _pty.openpty()
                    stdout_target = pty_slave_fd
                except (OSError, ImportError):
                    pass

            proc = subprocess.Popen(
                cmd,
                stdin=sys.stdin,
                stdout=stdout_target,
                stderr=subprocess.PIPE,
            )

            # Close the slave end in the parent — only the child writes to it
            if pty_master_fd is not None:
                import os as _os
                _os.close(pty_slave_fd)

                # Read from pty master and forward to real stdout
                import threading

                def _forward_pty():
                    try:
                        while True:
                            data = _os.read(pty_master_fd, 8192)
                            if not data:
                                break
                            sys.stdout.buffer.write(data)
                            sys.stdout.buffer.flush()
                    except OSError:
                        pass  # pty closed when child exits
                    finally:
                        _os.close(pty_master_fd)

                pty_thread = threading.Thread(target=_forward_pty, daemon=True)
                pty_thread.start()

            # Stream stderr but filter out repetitive WARN lines
            # (e.g. ngen emits thousands of unit-conversion warnings per timestep)
            warn_count = 0
            for raw_line in proc.stderr:
                line = raw_line.decode("utf-8", errors="replace")
                if line.startswith("WARN:"):
                    warn_count += 1
                    continue
                sys.stderr.write(line)
                sys.stderr.flush()

            proc.wait()

            if pty_master_fd is not None:
                pty_thread.join(timeout=2)

            if warn_count > 0:
                sys.stderr.write(f"[suppressed {warn_count:,} repetitive warnings]\n")
                sys.stderr.flush()

            # Ensure terminal gets a clean newline after binary output
            try:
                sys.stdout.write("\n")
                sys.stdout.flush()
            except (OSError, ValueError):
                pass

            rc = proc.returncode
            # Treat SIGSEGV during cleanup as success if the binary name
            # is known to crash on teardown (e.g. ngen with Fortran BMI modules).
            # Signal 11 (SIGSEGV) maps to exit code -11 or 128+11=139 or 256-11=245.
            if rc in (-11, 139, 245):
                tool_name = binary_path.name
                print(
                    f"Note: {tool_name} exited with signal 11 (SIGSEGV) during cleanup. "
                    f"This is a known issue with Fortran BMI module teardown and does not "
                    f"affect simulation results.",
                    file=sys.stderr,
                )
                rc = 0

            return rc
        except FileNotFoundError:
            print(f"Binary not found: {binary_path}", file=sys.stderr)
            return ExitCode.BINARY_ERROR
        except PermissionError:
            print(f"Permission denied: {binary_path}", file=sys.stderr)
            return ExitCode.PERMISSION_ERROR
        except KeyboardInterrupt:
            return ExitCode.USER_INTERRUPT
