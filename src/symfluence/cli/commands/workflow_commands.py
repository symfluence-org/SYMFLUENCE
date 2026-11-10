# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Workflow command handlers for SYMFLUENCE CLI.

This module implements handlers for the workflow command category.
"""
from __future__ import annotations

from argparse import Namespace

from symfluence.workflow_steps import (
    WORKFLOW_STEP_ALIAS_REVERSE,
    WORKFLOW_STEP_DESCRIPTION_MAP,
)

from ..exit_codes import ExitCode
from .base import (
    BaseCommand,
    cli_exception_handler,
    cli_exception_handler_with_profiling,
)


def _report_failed_steps(symfluence, what: str) -> int:
    """Return WORKFLOW_ERROR (and name the steps) when any step failed, else SUCCESS.

    With ``--continue-on-error`` / ``STOP_ON_ERROR: false`` a failing step is
    logged and the run carries on; the command must still exit non-zero.
    """
    get_failed = getattr(symfluence, 'get_failed_steps', None)
    failed = list(get_failed()) if callable(get_failed) else []
    if failed:
        BaseCommand._console.error(
            f"{what} finished with {len(failed)} failed step(s): {', '.join(failed)}"
        )
        return ExitCode.WORKFLOW_ERROR
    return ExitCode.SUCCESS


def _force_rerun_overrides(args: Namespace) -> dict:
    """``--force-rerun`` -> FORCE_RUN_ALL_STEPS override, so step internals
    (e.g. the forcing remap's per-file skip cache) honour it too."""
    if BaseCommand.get_arg(args, 'force_rerun', False):
        return {'FORCE_RUN_ALL_STEPS': True}
    return {}


class WorkflowCommands(BaseCommand):
    """Handlers for workflow category commands."""

    # Workflow step definitions (from original CLIArgumentManager)
    WORKFLOW_STEPS = dict(WORKFLOW_STEP_DESCRIPTION_MAP)

    @staticmethod
    @cli_exception_handler_with_profiling
    def run(args: Namespace) -> int:
        """
        Execute: symfluence workflow run

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        from symfluence import SYMFLUENCE

        config_path = BaseCommand.get_config_path(args)

        # Validate config exists
        if not BaseCommand.validate_config(config_path, required=True):
            return ExitCode.CONFIG_ERROR

        # Initialize SYMFLUENCE instance
        BaseCommand._console.info("Starting full workflow execution...")
        config_overrides = _force_rerun_overrides(args)
        if BaseCommand.get_arg(args, 'continue_on_error', False):
            config_overrides['STOP_ON_ERROR'] = False

        symfluence = SYMFLUENCE(
            config_path,
            config_overrides=config_overrides,
            debug_mode=BaseCommand.get_arg(args, 'debug', False),
            visualize=BaseCommand.get_arg(args, 'visualise', False),
            diagnostic=BaseCommand.get_arg(args, 'diagnostic', False),
            quiet_mode=BaseCommand.get_arg(args, 'quiet', False)
        )

        # Execute full workflow. Without --force-rerun pass None so the
        # config's FORCE_RUN_ALL_STEPS decides (False would override it).
        symfluence.run_workflow(
            force_run=True if BaseCommand.get_arg(args, 'force_rerun', False) else None
        )

        exit_code = _report_failed_steps(symfluence, "Workflow")
        if exit_code == ExitCode.SUCCESS:
            BaseCommand._console.success("Workflow execution completed successfully")
        return exit_code

    @staticmethod
    @cli_exception_handler_with_profiling
    def run_step(args: Namespace) -> int:
        """
        Execute: symfluence workflow step STEP_NAME

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        from symfluence import SYMFLUENCE

        config_path = BaseCommand.get_config_path(args)

        if not BaseCommand.validate_config(config_path, required=True):
            return ExitCode.CONFIG_ERROR

        BaseCommand._console.info(f"Executing step: {args.step_name}")
        BaseCommand._console.indent(WorkflowCommands.WORKFLOW_STEPS.get(args.step_name, ''))

        symfluence = SYMFLUENCE(
            config_path,
            config_overrides=_force_rerun_overrides(args) or None,
            debug_mode=BaseCommand.get_arg(args, 'debug', False),
            visualize=BaseCommand.get_arg(args, 'visualise', False),
            diagnostic=BaseCommand.get_arg(args, 'diagnostic', False),
            quiet_mode=BaseCommand.get_arg(args, 'quiet', False)
        )

        # Run single step
        symfluence.run_individual_steps([args.step_name])

        exit_code = _report_failed_steps(symfluence, f"Step '{args.step_name}'")
        if exit_code == ExitCode.SUCCESS:
            BaseCommand._console.success(f"Step '{args.step_name}' completed successfully")
        return exit_code

    @staticmethod
    @cli_exception_handler_with_profiling
    def run_steps(args: Namespace) -> int:
        """
        Execute: symfluence workflow steps STEP1 STEP2 ...

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        from symfluence import SYMFLUENCE

        config_path = BaseCommand.get_config_path(args)

        if not BaseCommand.validate_config(config_path, required=True):
            return ExitCode.CONFIG_ERROR

        BaseCommand._console.info(f"Executing {len(args.step_names)} steps:")
        for step_name in args.step_names:
            BaseCommand._console.indent(f"{step_name}: {WorkflowCommands.WORKFLOW_STEPS.get(step_name, '')}")

        symfluence = SYMFLUENCE(
            config_path,
            config_overrides=_force_rerun_overrides(args) or None,
            debug_mode=BaseCommand.get_arg(args, 'debug', False),
            visualize=BaseCommand.get_arg(args, 'visualise', False),
            diagnostic=BaseCommand.get_arg(args, 'diagnostic', False),
            quiet_mode=BaseCommand.get_arg(args, 'quiet', False)
        )

        # Run multiple steps in order
        symfluence.run_individual_steps(args.step_names)

        exit_code = _report_failed_steps(symfluence, "Steps")
        if exit_code == ExitCode.SUCCESS:
            BaseCommand._console.success(f"All {len(args.step_names)} steps completed successfully")
        return exit_code

    @staticmethod
    @cli_exception_handler
    def status(args: Namespace) -> int:
        """
        Execute: symfluence workflow status

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        from symfluence import SYMFLUENCE

        config_path = BaseCommand.get_config_path(args)

        if not BaseCommand.validate_config(config_path, required=True):
            return ExitCode.CONFIG_ERROR

        symfluence = SYMFLUENCE(
            config_path,
            debug_mode=BaseCommand.get_arg(args, 'debug', False),
            visualize=BaseCommand.get_arg(args, 'visualise', False),
            diagnostic=BaseCommand.get_arg(args, 'diagnostic', False),
            quiet_mode=BaseCommand.get_arg(args, 'quiet', False)
        )

        # Show workflow status
        BaseCommand._console.info("Workflow Status:")
        BaseCommand._console.rule()

        status_info = symfluence.get_workflow_status()
        BaseCommand._console.info(status_info)

        return ExitCode.SUCCESS

    @staticmethod
    @cli_exception_handler
    def validate(args: Namespace) -> int:
        """
        Execute: symfluence workflow validate

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        config_path = BaseCommand.get_config_path(args)

        # Load and validate config using typed system
        config = BaseCommand.load_typed_config(config_path, required=True)
        if config is None:
            return ExitCode.CONFIG_ERROR

        BaseCommand._console.success("Configuration validated successfully")
        return ExitCode.SUCCESS

    @staticmethod
    def list_steps(args: Namespace) -> int:
        """
        Execute: symfluence workflow list-steps

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        BaseCommand._console.info("Available workflow steps:")
        BaseCommand._console.rule()
        for i, (step_name, description) in enumerate(WorkflowCommands.WORKFLOW_STEPS.items(), 1):
            aliases = WORKFLOW_STEP_ALIAS_REVERSE.get(step_name, [])
            alias_str = f'  (aliases: {", ".join(aliases)})' if aliases else ''
            BaseCommand._console.info(f"{i:2}. {step_name:35s} - {description}{alias_str}")
        BaseCommand._console.rule()
        BaseCommand._console.info(f"Total: {len(WorkflowCommands.WORKFLOW_STEPS)} steps")
        BaseCommand._console.info("Tip: Use aliases for quicker typing, e.g. 'symfluence workflow step map'")
        return ExitCode.SUCCESS

    @staticmethod
    @cli_exception_handler
    def resume(args: Namespace) -> int:
        """
        Execute: symfluence workflow resume STEP_NAME

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        from symfluence import SYMFLUENCE

        config_path = BaseCommand.get_config_path(args)

        if not BaseCommand.validate_config(config_path, required=True):
            return ExitCode.CONFIG_ERROR

        # Get all steps from the resume point onwards
        step_list = list(WorkflowCommands.WORKFLOW_STEPS.keys())
        if args.step_name not in step_list:
            BaseCommand._console.error(f"Unknown step: {args.step_name}")
            return ExitCode.USAGE_ERROR

        resume_index = step_list.index(args.step_name)
        steps_to_run = step_list[resume_index:]

        BaseCommand._console.info(f"Resuming workflow from: {args.step_name}")
        BaseCommand._console.indent(f"Will execute {len(steps_to_run)} steps:")
        for step in steps_to_run:
            BaseCommand._console.indent(f"- {step}")

        # resume always re-executes the listed steps; --force-rerun additionally
        # reaches the step internals (per-file skip caches) via FORCE_RUN_ALL_STEPS.
        symfluence = SYMFLUENCE(
            config_path,
            config_overrides=_force_rerun_overrides(args) or None,
            debug_mode=BaseCommand.get_arg(args, 'debug', False),
            visualize=BaseCommand.get_arg(args, 'visualise', False),
            diagnostic=BaseCommand.get_arg(args, 'diagnostic', False),
            quiet_mode=BaseCommand.get_arg(args, 'quiet', False)
        )

        # Run steps from resume point
        symfluence.run_individual_steps(steps_to_run)

        exit_code = _report_failed_steps(symfluence, "Resumed workflow")
        if exit_code == ExitCode.SUCCESS:
            BaseCommand._console.success(f"Workflow resumed and completed from '{args.step_name}'")
        return exit_code

    @staticmethod
    @cli_exception_handler
    def clean(args: Namespace) -> int:
        """
        Execute: symfluence workflow clean

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        from symfluence import SYMFLUENCE

        config_path = BaseCommand.get_config_path(args)

        if not BaseCommand.validate_config(config_path, required=True):
            return ExitCode.CONFIG_ERROR

        level = args.level
        dry_run = BaseCommand.get_arg(args, 'dry_run', False)

        # Require confirmation for non-dry-run destructive operations
        if not dry_run and level in ('outputs', 'all'):
            if not BaseCommand.confirm_action(
                f"This will delete {level} files. Are you sure?"
            ):
                BaseCommand._console.info("Operation cancelled")
                return ExitCode.SUCCESS

        BaseCommand._console.info(f"Cleaning {level} files...")
        if dry_run:
            BaseCommand._console.indent("(DRY RUN - no files will be deleted)")

        symfluence = SYMFLUENCE(config_path,
                                debug_mode=BaseCommand.get_arg(args, 'debug', False),
                                quiet_mode=BaseCommand.get_arg(args, 'quiet', False))

        if hasattr(symfluence, 'clean_workflow_files'):
            symfluence.clean_workflow_files(level=level, dry_run=dry_run)
        else:
            # Feature in development - provide helpful guidance
            BaseCommand._console.warning("[BETA] Automated cleaning is under development")
            BaseCommand._console.info(f"Manual cleanup guidance for '{level}' level:")
            if level == 'intermediate':
                BaseCommand._console.indent("Remove temporary files: rm -rf <domain>/temp/*")
            elif level == 'outputs':
                BaseCommand._console.indent("Remove model outputs: rm -rf <domain>/simulations/*/output/*")
            elif level == 'all':
                BaseCommand._console.indent("Remove temp files: rm -rf <domain>/temp/*")
                BaseCommand._console.indent("Remove outputs: rm -rf <domain>/simulations/*/output/*")
                BaseCommand._console.indent("Remove logs: rm -rf <domain>/logs/*")
            else:
                BaseCommand._console.indent(f"Clean target: <domain>/{level}/*")
            BaseCommand._console.info("Use --dry-run to preview what would be cleaned")

        if not dry_run:
            BaseCommand._console.success(f"Cleaned {level} files")
        return ExitCode.SUCCESS

    @staticmethod
    @cli_exception_handler
    def diagnose(args: Namespace) -> int:
        """
        Execute: symfluence workflow diagnose [--step STEP]

        Run diagnostic plots on existing workflow outputs without re-running the workflow.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        from symfluence import SYMFLUENCE

        config_path = BaseCommand.get_config_path(args)

        if not BaseCommand.validate_config(config_path, required=True):
            return ExitCode.CONFIG_ERROR

        step = BaseCommand.get_arg(args, 'step', None)

        BaseCommand._console.info("Running diagnostics on existing workflow outputs...")
        if step:
            BaseCommand._console.indent(f"Step: {step}")

        # Initialize SYMFLUENCE with diagnostic mode enabled
        symfluence = SYMFLUENCE(
            config_path,
            debug_mode=BaseCommand.get_arg(args, 'debug', False),
            diagnostic=True,  # Always enable diagnostic mode for this command
            quiet_mode=BaseCommand.get_arg(args, 'quiet', False)
        )

        # Run diagnostics
        if step:
            results = symfluence.run_diagnostics_for_step(step)
        else:
            results = symfluence.run_all_diagnostics()

        # Report results
        if results:
            BaseCommand._console.success(f"Generated {len(results)} diagnostic plot(s):")
            for path in results:
                BaseCommand._console.indent(f"- {path}")
        else:
            BaseCommand._console.warning("No diagnostic plots were generated. Check that workflow outputs exist.")

        return ExitCode.SUCCESS
