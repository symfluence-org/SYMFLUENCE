# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
FEWS CLI command handlers.

Provides ``symfluence fews pre|post|run|launch`` subcommands for
operating the Delft-FEWS General Adapter integration.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from ..exit_codes import ExitCode
from .base import BaseCommand, cli_exception_handler

logger = logging.getLogger(__name__)


DELTARES_FEWS_URL = "https://oss.deltares.nl/web/delft-fews"


def _diag_path(run_info_path: Path) -> Path:
    """Where FEWS expects the diagnostics file for this run.

    The run file names it (``<outputDiagnosticFile>``); hand-written run files
    without one get ``toFews/diag.xml`` next to the run file. A run file that
    cannot be parsed also falls back there, so the parse error itself can
    still be reported to FEWS.
    """
    fallback = run_info_path.parent / "toFews" / "diag.xml"
    from symfluence.fews.exceptions import FEWSAdapterError
    from symfluence.fews.run_info import parse_run_info
    try:
        run_info = parse_run_info(run_info_path)
    except (FEWSAdapterError, OSError, ValueError):
        return fallback
    return run_info.output_diagnostic_file or fallback


def _run_model(config_path: Path, overrides: dict, debug: bool = False) -> list:
    """Run the SYMFLUENCE workflow for a FEWS run; return failed step names.

    The FEWS-derived overrides (run period, DOMAIN_NAME, ...) are applied on
    top of the base configuration -- running the base config unchanged would
    simulate whatever period it happens to contain.
    """
    from symfluence import SYMFLUENCE

    symfluence = SYMFLUENCE(config_path, config_overrides=dict(overrides), debug_mode=debug)
    symfluence.run_workflow()
    return list(symfluence.get_failed_steps())


def _find_fews_installation():
    """Locate a real Delft-FEWS installation, or return ``(None, searched)``.

    ``FEWS_HOME`` wins; otherwise the directory ``symfluence binary install
    openfews`` prepares. It only counts as an installation when Deltares'
    FEWS jars are actually there -- SYMFLUENCE creates the directory and the
    adapter template but cannot ship FEWS itself.
    """
    import os

    candidates = []
    if os.environ.get("FEWS_HOME"):
        candidates.append(Path(os.environ["FEWS_HOME"]))
    try:
        from symfluence.cli.services.tool_installer import ToolInstaller
        installer = ToolInstaller()
        data_dir, _ = installer._resolve_data_dir(installer._load_config(None))
        candidates.append(Path(data_dir) / "installs" / "openfews")
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        pass

    for home in candidates:
        lib = home / "lib"
        if lib.is_dir() and any(lib.rglob("*fews*.jar")) or (
            (home / "bin").is_dir() and any((home / "bin").rglob("Delft_FEWS*.jar"))
        ):
            return home, candidates
    return None, candidates


class FEWSCommands(BaseCommand):
    """Command handlers for the ``symfluence fews`` category."""

    @staticmethod
    @cli_exception_handler
    def pre(args: Namespace) -> int:
        """Run the FEWS pre-adapter."""
        from symfluence.fews.pi_diagnostics import DiagnosticsCollector
        from symfluence.fews.pre_adapter import FEWSPreAdapter

        run_info_path = Path(args.run_info)
        base_config = Path(args.config) if getattr(args, 'config', None) else None
        data_format = getattr(args, 'format', 'netcdf-cf')
        id_map_path = getattr(args, 'id_map', None)

        diag = DiagnosticsCollector(_diag_path(run_info_path))

        try:
            adapter = FEWSPreAdapter(
                run_info_path=run_info_path,
                base_config_path=base_config,
                data_format=data_format,
                id_map_path=id_map_path,
            )
            adapter.run(diag=diag)
            diag.info("Pre-adapter completed successfully")
            FEWSCommands._console.success("FEWS pre-adapter completed")
            return ExitCode.SUCCESS
        except Exception as exc:  # noqa: BLE001 — wrap-and-raise to domain error
            diag.fatal(str(exc))
            raise
        finally:
            diag.write()

    @staticmethod
    @cli_exception_handler
    def post(args: Namespace) -> int:
        """Run the FEWS post-adapter."""
        from symfluence.fews.pi_diagnostics import DiagnosticsCollector
        from symfluence.fews.post_adapter import FEWSPostAdapter

        run_info_path = Path(args.run_info)
        config_path = Path(args.config) if getattr(args, 'config', None) else None
        data_format = getattr(args, 'format', 'netcdf-cf')
        id_map_path = getattr(args, 'id_map', None)

        diag = DiagnosticsCollector(_diag_path(run_info_path))

        try:
            adapter = FEWSPostAdapter(
                run_info_path=run_info_path,
                config_path=config_path,
                data_format=data_format,
                id_map_path=id_map_path,
            )
            output_dir = adapter.run(diag=diag)
            diag.info("Post-adapter completed successfully")
            FEWSCommands._console.success(f"FEWS post-adapter completed -> {output_dir}")
            return ExitCode.SUCCESS
        except Exception as exc:  # noqa: BLE001 — wrap-and-raise to domain error
            diag.fatal(str(exc))
            raise
        finally:
            diag.write()

    @staticmethod
    @cli_exception_handler
    def run_full(args: Namespace) -> int:
        """Run the full FEWS adapter cycle: pre -> model -> post."""
        from symfluence.fews.pi_diagnostics import DiagnosticsCollector
        from symfluence.fews.post_adapter import FEWSPostAdapter
        from symfluence.fews.pre_adapter import FEWSPreAdapter

        run_info_path = Path(args.run_info)
        base_config = Path(args.config) if getattr(args, 'config', None) else None
        data_format = getattr(args, 'format', 'netcdf-cf')
        id_map_path = getattr(args, 'id_map', None)

        diag = DiagnosticsCollector(_diag_path(run_info_path))

        try:
            # Without a configuration there is no model to run and no output
            # to locate; "skip the model and export whatever is lying around"
            # is not a successful FEWS run.
            if base_config is None or not base_config.is_file():
                message = (
                    "'fews run' requires --config <SYMFLUENCE config file>"
                    + (f" (not found: {base_config})" if base_config else "")
                    + "; the model cannot be run without it."
                )
                diag.fatal(message)
                FEWSCommands._console.error(message)
                return ExitCode.CONFIG_ERROR if base_config else ExitCode.USAGE_ERROR

            # Pre-adapter
            diag.info("Starting pre-adapter")
            pre = FEWSPreAdapter(
                run_info_path=run_info_path,
                base_config_path=base_config,
                data_format=data_format,
                id_map_path=id_map_path,
            )
            config, run_info = pre.run(diag=diag)
            FEWSCommands._console.info("Pre-adapter complete")

            notice = (
                f"FEWS forcing was translated to {pre.forcing_path}, but the SYMFLUENCE "
                "workflow reads the forcing configured in the SYMFLUENCE config; the "
                "FEWS-supplied forcing is not injected into the model run."
            )
            diag.warning(notice)
            FEWSCommands._console.warning(notice)

            # Model execution, with the FEWS overrides (period, domain, ...)
            diag.info(f"Starting model execution with overrides {pre.config_overrides}")
            failed = _run_model(
                base_config, pre.config_overrides, debug=getattr(args, 'debug', False)
            )
            if failed:
                message = f"Model execution failed in step(s): {', '.join(failed)}"
                diag.fatal(message)
                FEWSCommands._console.error(message)
                return ExitCode.WORKFLOW_ERROR
            diag.info("Model execution completed")

            # Post-adapter
            diag.info("Starting post-adapter")
            post = FEWSPostAdapter(
                run_info_path=run_info_path,
                config_path=base_config,
                data_format=data_format,
                id_map_path=id_map_path,
                config=config,
            )
            output_dir = post.run(diag=diag)
            diag.info("Full FEWS adapter cycle completed")
            FEWSCommands._console.success(f"FEWS run completed -> {output_dir}")
            return ExitCode.SUCCESS
        except Exception as exc:  # noqa: BLE001 — wrap-and-raise to domain error
            diag.fatal(str(exc))
            raise
        finally:
            diag.write()

    @staticmethod
    @cli_exception_handler
    def launch(args: Namespace) -> int:
        """Launch an existing Delft-FEWS installation with SYMFLUENCE on its PATH.

        SYMFLUENCE does not distribute or download Delft-FEWS (it is obtained
        from Deltares). Without a real installation this fails; it never
        pretends to have started anything.
        """
        import os

        config_path = getattr(args, 'config', None)

        fews_home, searched = _find_fews_installation()
        if fews_home is None:
            FEWSCommands._console.error(
                "No Delft-FEWS installation found. Delft-FEWS must be obtained from "
                f"Deltares ({DELTARES_FEWS_URL}); SYMFLUENCE does not download it."
            )
            FEWSCommands._console.indent(
                "Set FEWS_HOME to the installation directory. "
                "'symfluence binary install openfews' only generates the General "
                "Adapter configuration template."
            )
            for path in searched:
                FEWSCommands._console.indent(f"searched: {path}")
            return ExitCode.BINARY_ERROR

        launcher = fews_home / "bin" / "fews.sh"
        if not launcher.is_file():
            FEWSCommands._console.error(
                f"Delft-FEWS found at {fews_home} but it has no bin/fews.sh launcher; "
                "start FEWS with its own launcher."
            )
            return ExitCode.BINARY_ERROR

        cmd = [str(launcher)]
        FEWSCommands._console.info(f"Launching Delft-FEWS: {' '.join(cmd)}")

        # Set environment so the General Adapter can find SYMFLUENCE
        env = os.environ.copy()
        env["SYMFLUENCE_CLI"] = sys.executable + " -m symfluence"
        if config_path:
            env["SYMFLUENCE_CONFIG"] = str(config_path)

        try:
            proc = subprocess.run(cmd, env=env)
            return proc.returncode
        except (FileNotFoundError, PermissionError):
            FEWSCommands._console.error(f"Cannot execute: {launcher}")
            return ExitCode.BINARY_ERROR
        except KeyboardInterrupt:
            FEWSCommands._console.info("Delft-FEWS terminated by user")
            return ExitCode.SUCCESS
