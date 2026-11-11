# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Project command handlers for SYMFLUENCE CLI.

This module implements handlers for the project command category,
including initialization and pour point setup.
"""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from ..exit_codes import ExitCode
from ..validators import validate_bounding_box, validate_coordinates, validate_date_range
from .base import BaseCommand, cli_exception_handler, supports_dry_run


class ProjectCommands(BaseCommand):
    """Handlers for project category commands."""

    @staticmethod
    @cli_exception_handler
    def init(args: Namespace) -> int:
        """
        Execute: symfluence project init [PRESET]

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        # Check for interactive mode
        if BaseCommand.get_arg(args, 'interactive', False):
            from symfluence.cli.wizard import ProjectWizard
            wizard = ProjectWizard()
            return wizard.run(
                output_dir=BaseCommand.get_arg(args, 'output_dir', './'),
                scaffold=BaseCommand.get_arg(args, 'scaffold', False)
            )

        # Import initialization manager
        from symfluence.cli.services import InitializationManager

        init_manager = InitializationManager()

        # Build initialization operations dict
        preset_name = args.preset if args.preset else None

        # Use get_args_dict for cleaner extraction
        cli_overrides = BaseCommand.get_args_dict(args, [
            'domain', 'model', 'start_date', 'end_date',
            'forcing', 'discretization', 'definition_method'
        ])

        # Validate date range if both dates provided
        if cli_overrides.get('start_date') and cli_overrides.get('end_date'):
            result = validate_date_range(cli_overrides['start_date'], cli_overrides['end_date'])
            if result.is_err:
                error = result.first_error()
                BaseCommand._console.error(f"Invalid date range: {error.message if error else 'validation failed'}")
                return ExitCode.VALIDATION_ERROR

        output_dir = BaseCommand.get_arg(args, 'output_dir', './')
        scaffold = BaseCommand.get_arg(args, 'scaffold', False)
        minimal = BaseCommand.get_arg(args, 'minimal', False)
        comprehensive = BaseCommand.get_arg(args, 'comprehensive', True)

        BaseCommand._console.info("Initializing SYMFLUENCE project...")

        # Call initialization manager
        config = init_manager.generate_config(
            preset_name=preset_name,
            cli_overrides=cli_overrides,
            minimal=minimal,
            comprehensive=comprehensive
        )

        # 2. Determine output path
        domain_name = config.get("DOMAIN_NAME", "unnamed_project")
        output_dir_path = Path(output_dir)
        output_file = output_dir_path / f"config_{domain_name}.yaml"

        # 3. Write config file
        written_path = init_manager.write_config(config, output_file)
        BaseCommand._console.success(f"Created config file: {written_path}")

        # 4. Create scaffold if requested
        if scaffold:
            BaseCommand._console.info("Creating project scaffold...")
            domain_dir = init_manager.create_scaffold(config)
            BaseCommand._console.success(f"Created project structure at: {domain_dir}")
        else:
            BaseCommand._console.info(f"To create project structure, run: symfluence workflow step setup_project --config {written_path}")

        return ExitCode.SUCCESS

    @staticmethod
    @cli_exception_handler
    def pour_point(args: Namespace) -> int:
        """
        Execute: symfluence project pour-point LAT/LON

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        # Validate coordinates using Result pattern
        coord_result = validate_coordinates(args.coordinates)
        if coord_result.is_err:
            error = coord_result.first_error()
            BaseCommand._console.error(f"Invalid coordinates: {error.message if error else 'validation failed'}")
            return ExitCode.VALIDATION_ERROR

        # Validate bounding box if provided
        bounding_box_coords = BaseCommand.get_arg(args, 'bounding_box_coords', None)
        if bounding_box_coords:
            bbox_result = validate_bounding_box(bounding_box_coords)
            if bbox_result.is_err:
                error = bbox_result.first_error()
                BaseCommand._console.error(f"Invalid bounding box: {error.message if error else 'validation failed'}")
                return ExitCode.VALIDATION_ERROR

        BaseCommand._console.info("Setting up pour point workflow...")
        BaseCommand._console.indent(f"Coordinates: {args.coordinates}")
        BaseCommand._console.indent(f"Domain name: {args.domain_name}")
        BaseCommand._console.indent(f"Definition method: {args.domain_def}")

        from symfluence.project.pour_point_workflow import setup_pour_point_workflow

        # Get output directory from args or use default
        output_dir = Path(BaseCommand.get_arg(args, 'output_dir', '.'))

        result = setup_pour_point_workflow(
            coordinates=args.coordinates,
            domain_def_method=args.domain_def,
            domain_name=args.domain_name,
            bounding_box_coords=bounding_box_coords,
            output_dir=output_dir,
        )

        BaseCommand._console.success("Pour point workflow setup completed")
        BaseCommand._console.indent(f"Config file: {result.config_file}")
        if result.used_auto_bounding_box:
            BaseCommand._console.indent(f"Auto-generated bounding box: {result.bounding_box_coords}")
        return ExitCode.SUCCESS

    @staticmethod
    @supports_dry_run
    @cli_exception_handler
    def list_presets(args: Namespace) -> int:
        """
        Execute: symfluence project list-presets

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        from symfluence.cli.services import InitializationManager

        init_manager = InitializationManager()

        # InitializationManager handles all output formatting
        init_manager.list_presets()

        return ExitCode.SUCCESS

    @staticmethod
    @supports_dry_run
    @cli_exception_handler
    def show_preset(args: Namespace) -> int:
        """
        Execute: symfluence project show-preset PRESET_NAME

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        from symfluence.cli.services import InitializationManager

        init_manager = InitializationManager()

        preset_name = args.preset_name

        # InitializationManager handles all output formatting
        preset_info = init_manager.show_preset(preset_name)

        if preset_info:
            return ExitCode.SUCCESS
        else:
            # Error already printed by manager if invalid
            return ExitCode.FILE_NOT_FOUND
