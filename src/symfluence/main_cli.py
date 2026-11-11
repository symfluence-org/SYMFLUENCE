# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SYMFLUENCE Command-Line Interface entry point.

Provides the main() function that serves as the entry point for the
`symfluence` command. Handles argument parsing, command dispatch, and
error handling for all CLI operations.

The main() function is called by the console_scripts entry point defined
in pyproject.toml and is responsible for:
- Creating the CLI argument parser
- Parsing command-line arguments
- Dispatching to appropriate command handlers
- Handling interrupts and exceptions gracefully
"""
from __future__ import annotations


def enable_crash_diagnostics() -> bool:
    """
    Install Python's fault handler so a native crash is not silent.

    Workflows load large compiled stacks (PyTorch, GDAL, HDF5/netCDF, MPI-linked
    model binaries). When one of those faults, the interpreter dies without
    writing anything: the run leaves a log that stops mid-step and a shell exit
    status of 139 (Git Bash maps a Windows ``STATUS_ACCESS_VIOLATION`` onto
    SIGSEGV), with no indication of which call crashed.

    ``faulthandler`` handles this on every supported platform — on Windows it
    installs an unhandled-exception filter that prints ``Windows fatal
    exception: access violation`` plus the Python stack of every thread before
    the process dies. That output goes to stderr, which the workflow logs
    capture, turning an opaque exit code into a located crash.

    Set ``SYMFLUENCE_NO_FAULTHANDLER=1`` to opt out.

    Returns:
        True if the fault handler was enabled.
    """
    import os

    if os.environ.get("SYMFLUENCE_NO_FAULTHANDLER"):
        return False

    try:
        import faulthandler

        if not faulthandler.is_enabled():
            faulthandler.enable()
        return True
    except (ImportError, OSError, ValueError):
        # A closed/redirected stderr is the only realistic failure here, and it
        # must never stop the CLI from starting.
        return False


def main():
    """
    Main entry point for SYMFLUENCE CLI.

    Uses subcommand architecture for clean command organization.
    """
    import sys

    from symfluence.core.exceptions import SYMFLUENCEError

    enable_crash_diagnostics()

    # Windows consoles default to cp1252 which cannot encode Rich's Unicode
    # box-drawing and emoji characters.  Reconfigure to UTF-8 so output
    # renders correctly in modern terminals (Windows Terminal, VS Code, etc.).
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")

    from symfluence.cli.argument_parser import CLIParser

    # --- binary pass-through: symfluence binary <tool> [args...] ---
    # Intercept before argparse so tool flags (e.g. -m, --help) are not
    # consumed by symfluence's own parser.
    _BINARY_ACTIONS = {'install', 'validate', 'doctor', 'install-sysdeps', 'info'}
    argv = sys.argv[1:]
    # Global options may precede ``binary``.  Strip only that prefix for
    # dispatch detection; everything after the tool name remains opaque and is
    # forwarded verbatim to the external executable. The option-name sets are
    # the single source shared with the parser (argument_parser._GLOBAL_OPTION_SPECS).
    from symfluence.cli.argument_parser import GLOBAL_FLAG_OPTIONS, GLOBAL_VALUE_OPTIONS
    binary_argv = list(argv)
    stripped_globals = []
    while binary_argv:
        option = binary_argv[0].partition('=')[0]
        if binary_argv[0] in GLOBAL_FLAG_OPTIONS or (
            option in GLOBAL_VALUE_OPTIONS and '=' in binary_argv[0]
        ):
            stripped_globals.append(binary_argv.pop(0))
        elif binary_argv[0] in GLOBAL_VALUE_OPTIONS and len(binary_argv) >= 2:
            stripped_globals.extend(binary_argv[:2])
            del binary_argv[:2]
        else:
            break
    if (
        len(binary_argv) >= 2
        and binary_argv[0] == 'binary'
        and binary_argv[1] not in _BINARY_ACTIONS
        and not binary_argv[1].startswith('-')
    ):
        # Stripped globals must be honored, not silently dropped: --dry-run
        # previews the exec instead of running the real binary.
        if '--dry-run' in stripped_globals:
            print(f"[dry-run] would execute: {binary_argv[1]} "
                  + ' '.join(binary_argv[2:]))
            return 0
        from symfluence.cli.commands.binary_commands import BinaryCommands
        return BinaryCommands.exec_binary(binary_argv[1], binary_argv[2:])

    try:
        # Create parser and parse arguments
        parser = CLIParser()

        # Bare `symfluence` should orient the user, not print an argparse error
        if not argv:
            parser.parser.print_help()
            return 0

        args = parser.parse_args()

        # Global flags that shape console output (--quiet) must be applied
        # before any command prints.
        from symfluence.cli.console import apply_global_flags
        apply_global_flags(args)

        # Execute the command handler
        if hasattr(args, 'func'):
            # dispatch_command refuses handlers that cannot honour --dry-run.
            from symfluence.cli.commands.base import dispatch_command
            return dispatch_command(args)
        else:
            # No command specified - should not happen due to required=True on subparsers
            parser.parser.print_help()
            return 1

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user", file=sys.stderr)
        return 130
    except (SYMFLUENCEError, FileNotFoundError, ValueError) as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 — top-level fallback
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        # Honor --debug / SYMFLUENCE_DEBUG so the traceback is recoverable on this
        # fallback path too (the CLI's own exception decorator handles tracebacks
        # for dispatched commands; this catch-all otherwise swallows them).
        import os
        if '--debug' in sys.argv or os.environ.get('SYMFLUENCE_DEBUG'):
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
