# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Configuration management command handlers for SYMFLUENCE CLI.

This module implements handlers for configuration file management and validation.
"""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from ..exit_codes import ExitCode
from .base import BaseCommand, cli_exception_handler, supports_dry_run


class ConfigCommands(BaseCommand):
    """Handlers for configuration management commands."""

    @staticmethod
    @supports_dry_run
    @cli_exception_handler
    def list_templates(args: Namespace) -> int:
        """
        Execute: symfluence config list-templates

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        # Get template files from SYMFLUENCE resources
        from symfluence.resources import list_config_templates

        BaseCommand._console.info("Available configuration templates:")
        BaseCommand._console.rule()

        templates = list_config_templates()
        if templates:
            for i, template_path in enumerate(templates, 1):
                template_name = Path(template_path).name
                # Skip backup files
                if template_name.endswith('_backup.yaml'):
                    continue
                description = ""
                if 'quickstart_minimal' in template_name:
                    description = " (Minimal template - 10 required fields only)"
                elif 'comprehensive' in template_name:
                    description = " (Complete reference - 406+ options)"
                elif 'camels' in template_name:
                    description = " (CAMELS dataset template)"
                elif 'fluxnet' in template_name:
                    description = " (FLUXNET sites template)"
                elif 'norswe' in template_name:
                    description = " (Norwegian SWE template)"
                elif template_name.startswith('config_template.yaml'):
                    description = " (Standard template with common options)"
                BaseCommand._console.info(f"{i:2}. {template_name}{description}")
            BaseCommand._console.rule()
            BaseCommand._console.info(f"Total: {len([t for t in templates if not Path(t).name.endswith('_backup.yaml')])} templates")
        else:
            BaseCommand._console.info("No template files found")

        return ExitCode.SUCCESS

    @staticmethod
    def update(args: Namespace) -> int:
        """
        Execute: symfluence config update CONFIG_FILE

        This feature is not yet implemented.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        config_file = args.config_file

        # Validate file exists
        if not Path(config_file).exists():
            BaseCommand._console.error(f"Config file not found: {config_file}")
            return ExitCode.FILE_NOT_FOUND

        # Feature not implemented - provide helpful guidance
        BaseCommand._console.warning("Config update command is not yet implemented")
        BaseCommand._console.info("Workarounds:")
        BaseCommand._console.indent("- Use 'symfluence project init --interactive' to create a new config")
        BaseCommand._console.indent(f"- Edit the file directly: {config_file}")
        return ExitCode.USAGE_ERROR

    @staticmethod
    @supports_dry_run
    @cli_exception_handler
    def validate(args: Namespace) -> int:
        """
        Execute: symfluence config validate

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        from symfluence import SYMFLUENCE
        from symfluence.core.exceptions import ConfigurationError, SYMFLUENCEError

        config_path = BaseCommand.get_config_path(args)

        BaseCommand._console.info(f"Validating configuration: {config_path}")

        # Load config using typed system to validate
        config = BaseCommand.load_typed_config(config_path, required=True)
        if config is None:
            return ExitCode.CONFIG_ERROR

        BaseCommand._console.success("Configuration file is valid YAML")

        # Try to initialize SYMFLUENCE to validate structure. Point the data
        # dir at a throwaway location: validation must not create directories
        # or logs under the user's real SYMFLUENCE_DATA_DIR as a side effect.
        import tempfile
        try:
            # ignore_cleanup_errors: the run log inside stays open on Windows
            with tempfile.TemporaryDirectory(
                prefix="symfluence_validate_", ignore_cleanup_errors=True
            ) as tmp_data_dir:
                SYMFLUENCE(
                    config_path,
                    config_overrides={'SYMFLUENCE_DATA_DIR': tmp_data_dir},
                    debug_mode=BaseCommand.get_arg(args, 'debug', False),
                )
            BaseCommand._console.success("Configuration validated successfully")
            return ExitCode.SUCCESS
        except ConfigurationError as e:
            BaseCommand._console.error(f"Configuration structure validation failed: {e}")
            return ExitCode.CONFIG_ERROR
        except (ValueError, TypeError) as e:
            BaseCommand._console.error(f"Configuration contains invalid values: {e}")
            return ExitCode.CONFIG_ERROR
        except SYMFLUENCEError as e:
            BaseCommand._console.error(f"Configuration structure validation failed: {e}")
            return ExitCode.CONFIG_ERROR

    @staticmethod
    @supports_dry_run
    @cli_exception_handler
    def validate_env(args: Namespace) -> int:
        """
        Execute: symfluence config validate-env

        Validate system environment for SYMFLUENCE.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        import platform
        import sys

        BaseCommand._console.info("Validating system environment...")
        BaseCommand._console.rule()

        BaseCommand._console.info(f"Python version: {sys.version}")
        BaseCommand._console.info(f"Platform: {platform.platform()}")
        BaseCommand._console.info(f"Python executable: {sys.executable}")

        # Check for required packages
        required_packages = ['yaml', 'numpy', 'pandas', 'xarray', 'geopandas']
        missing_packages = []

        for package in required_packages:
            try:
                __import__(package)
                BaseCommand._console.success(f"{package:15s} - installed")
            except ImportError:
                BaseCommand._console.error(f"{package:15s} - MISSING")
                missing_packages.append(package)

        BaseCommand._console.rule()

        if missing_packages:
            BaseCommand._console.error(f"Missing packages: {', '.join(missing_packages)}")
            return ExitCode.DEPENDENCY_ERROR
        else:
            BaseCommand._console.success("Environment validation passed")
            return ExitCode.SUCCESS

    @staticmethod
    @supports_dry_run
    @cli_exception_handler
    def resolve(args: Namespace) -> int:
        """
        Execute: symfluence config resolve

        Show the fully resolved configuration after merging all sources
        (defaults, file, env vars, overrides).

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        import json

        import yaml

        flat_mode = BaseCommand.get_arg(args, 'flat', False)
        json_mode = BaseCommand.get_arg(args, 'as_json', False)
        diff_mode = BaseCommand.get_arg(args, 'diff', False)
        section = BaseCommand.get_arg(args, 'section', None)

        # --section and --flat are incompatible
        if section and flat_mode:
            BaseCommand._console.error(
                "--section and --flat cannot be combined. "
                "Sections apply to the nested config structure only."
            )
            return ExitCode.USAGE_ERROR

        config_path = BaseCommand.get_config_path(args)
        config = BaseCommand.load_typed_config(config_path, required=True)
        if config is None:
            return ExitCode.CONFIG_ERROR

        # Build the output dict
        if flat_mode:
            output = config.to_dict(flatten=True)
        else:
            output = config.to_dict(flatten=False)

        # Apply --diff: keep only values that differ from Pydantic defaults
        if diff_mode:
            from symfluence.core.config.models import SymfluenceConfig

            # Build a reference config with the same required fields but all
            # optional fields at their Pydantic defaults.
            try:
                ref_config = SymfluenceConfig.from_minimal(
                    domain_name=config.domain.name,
                    model=str(config.model.hydrological_model),
                    forcing_dataset=str(config.forcing.dataset),
                    EXPERIMENT_TIME_START=config.domain.time_start,
                    EXPERIMENT_TIME_END=config.domain.time_end,
                    SYMFLUENCE_DATA_DIR=str(config.system.data_dir),
                    SYMFLUENCE_CODE_DIR=str(config.system.code_dir),
                )
                defaults = ref_config.to_dict(flatten=True)
            except Exception:  # noqa: BLE001 — top-level fallback
                # Fallback: empty defaults means everything is shown
                defaults = {}

            if flat_mode:
                # Filter flat dict: keep keys where value differs from default
                output = {
                    k: v for k, v in output.items()
                    if k not in defaults or defaults[k] != v
                }
            else:
                # For nested mode: compare via flat dicts, then rebuild nested
                flat_resolved = config.to_dict(flatten=True)
                changed_keys = {
                    k for k, v in flat_resolved.items()
                    if k not in defaults or defaults[k] != v
                }
                from symfluence.core.config.transformers import transform_flat_to_nested
                changed_flat = {k: flat_resolved[k] for k in changed_keys}
                output = transform_flat_to_nested(changed_flat)

        # Apply --section filter (nested mode only)
        if section:
            if section in output:
                output = {section: output[section]}
            else:
                output = {section: {}}

        # Format and print
        if json_mode:
            text = json.dumps(output, indent=2, default=str)
        else:
            text = yaml.dump(output, default_flow_style=False, sort_keys=True)

        # Print raw output (pipeable, no prefixes)
        print(text, end='' if not json_mode else '\n')

        return ExitCode.SUCCESS
