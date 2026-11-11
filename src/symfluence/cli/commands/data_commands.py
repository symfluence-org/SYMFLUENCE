# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Data acquisition command handlers for SYMFLUENCE CLI.

This module implements handlers for standalone data download, listing,
and inspection — a thin CLI layer over the AcquisitionRegistry.
"""
from __future__ import annotations

import logging
from argparse import Namespace
from pathlib import Path

from ..exit_codes import ExitCode
from .base import BaseCommand, cli_exception_handler, supports_dry_run

logger = logging.getLogger(__name__)


def _bbox_from_shapefile(shapefile_path: str) -> str:
    """
    Extract a bounding box string from a shapefile.

    Reads the shapefile with geopandas, reprojects to EPSG:4326,
    and returns the bounding box in SYMFLUENCE convention:
    ``lat_max/lon_min/lat_min/lon_max``.

    Args:
        shapefile_path: Path to a .shp file.

    Returns:
        Bounding box string in ``lat_max/lon_min/lat_min/lon_max`` format.

    Raises:
        FileNotFoundError: If the shapefile does not exist.
        RuntimeError: If geopandas is not installed or the file cannot be read.
    """
    import geopandas as gpd

    path = Path(shapefile_path)
    if not path.exists():
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")

    gdf = gpd.read_file(path)
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    lon_min, lat_min, lon_max, lat_max = gdf.total_bounds
    return f"{lat_max}/{lon_min}/{lat_min}/{lon_max}"


class DataCommands(BaseCommand):
    """Handlers for standalone data acquisition commands."""

    @staticmethod
    @supports_dry_run
    @cli_exception_handler
    def download(args: Namespace) -> int:
        """
        Execute: symfluence data download DATASET

        Downloads a dataset using the registered acquisition handler.
        Builds a minimal config dict from CLI args, or loads a full
        config file if --config is provided.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        import symfluence.data.acquisition  # noqa: F401 — package import triggers handler registration
        from symfluence.data.acquisition.registry import AcquisitionRegistry

        dataset_name = args.dataset

        # Resolve --shapefile to --bbox if provided
        shapefile_path = getattr(args, 'shapefile', None)
        if shapefile_path and not args.bbox:
            try:
                args.bbox = _bbox_from_shapefile(shapefile_path)
                BaseCommand._console.info(f"Extracted bbox from shapefile: {args.bbox}")
            except FileNotFoundError as exc:
                BaseCommand._console.error(str(exc))
                return ExitCode.VALIDATION_ERROR
            except Exception as exc:  # noqa: BLE001 — top-level fallback
                BaseCommand._console.error(f"Failed to read shapefile: {exc}")
                return ExitCode.RUNTIME_ERROR
            # Derive domain name from shapefile filename if still default
            if getattr(args, 'domain_name', 'standalone') == 'standalone':
                args.domain_name = Path(shapefile_path).stem

        # Determine output directory
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = Path('./data') / dataset_name

        # Build config: either from --config file or from CLI args
        if getattr(args, 'config', None):
            # Load full config, then overlay CLI args
            config = BaseCommand.load_typed_config(args.config)
            if config is None:
                return ExitCode.CONFIG_ERROR
            # Overlay CLI overrides onto the typed config's backing dict
            config_dict = config.to_dict() if hasattr(config, 'to_dict') else {}
            if args.bbox:
                config_dict['BOUNDING_BOX_COORDS'] = args.bbox
            if args.start:
                config_dict['EXPERIMENT_TIME_START'] = args.start
            if args.end:
                config_dict['EXPERIMENT_TIME_END'] = args.end
            if args.domain_name != 'standalone':
                config_dict['DOMAIN_NAME'] = args.domain_name
            if args.force:
                config_dict['FORCE_DOWNLOAD'] = True
        else:
            # Validate required args when no config file
            if not args.bbox or not args.start or not args.end:
                BaseCommand._console.error(
                    "When --config is not provided, --bbox, --start, and --end are required."
                )
                return ExitCode.VALIDATION_ERROR

            config_dict = {
                'DOMAIN_NAME': args.domain_name,
                'BOUNDING_BOX_COORDS': args.bbox,
                'EXPERIMENT_TIME_START': args.start,
                'EXPERIMENT_TIME_END': args.end,
                'DATA_DIR': str(output_dir.parent),
                'FORCE_DOWNLOAD': args.force,
            }

        # Apply --vars override
        if getattr(args, 'vars', None):
            config_dict['DAYMET_VARIABLES'] = [
                v.strip() for v in args.vars.split(',')
            ]

        # Apply --extra KEY=VALUE overrides
        for extra in (args.extra or []):
            if '=' not in extra:
                BaseCommand._console.error(
                    f"Invalid --extra format: '{extra}'. Expected KEY=VALUE."
                )
                return ExitCode.VALIDATION_ERROR
            key, value = extra.split('=', 1)
            # Coerce common string values to Python types
            if value.lower() in ('true', 'yes'):
                value = True
            elif value.lower() in ('false', 'no'):
                value = False
            config_dict[key] = value

        BaseCommand._console.info(f"Downloading dataset: {dataset_name}")
        BaseCommand._console.indent(f"Bounding box: {config_dict.get('BOUNDING_BOX_COORDS', 'from config')}")
        BaseCommand._console.indent(f"Period: {config_dict.get('EXPERIMENT_TIME_START', '?')} to {config_dict.get('EXPERIMENT_TIME_END', '?')}")
        BaseCommand._console.indent(f"Output: {output_dir}")

        if BaseCommand.get_arg(args, 'dry_run', False):
            if not AcquisitionRegistry.is_registered(dataset_name):
                available = ', '.join(AcquisitionRegistry.list_datasets())
                BaseCommand._console.error(
                    f"Unknown dataset: '{dataset_name}'. Available: {available}"
                )
                return ExitCode.VALIDATION_ERROR
            BaseCommand._console.info(
                "[DRY RUN] Nothing was downloaded and no files were written"
            )
            return ExitCode.SUCCESS

        handler = AcquisitionRegistry.get_handler(dataset_name, config_dict, logger)
        result_path = handler.download(output_dir)

        BaseCommand._console.success(f"Download complete: {result_path}")
        return ExitCode.SUCCESS

    @staticmethod
    @supports_dry_run
    @cli_exception_handler
    def list_datasets(args: Namespace) -> int:
        """
        Execute: symfluence data list

        Lists all registered acquisition datasets.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        import symfluence.data.acquisition  # noqa: F401 — package import triggers handler registration
        from symfluence.data.acquisition.registry import AcquisitionRegistry

        datasets = AcquisitionRegistry.list_datasets()

        if not datasets:
            BaseCommand._console.warning("No acquisition handlers registered.")
            return ExitCode.SUCCESS

        BaseCommand._console.info(f"Available datasets ({len(datasets)}):")
        BaseCommand._console.rule()

        rows = [[name] for name in datasets]
        BaseCommand._console.table(
            columns=["Dataset"],
            rows=rows,
            title="Registered Acquisition Handlers",
        )

        return ExitCode.SUCCESS

    @staticmethod
    @supports_dry_run
    @cli_exception_handler
    def info(args: Namespace) -> int:
        """
        Execute: symfluence data info DATASET

        Shows information about a specific dataset handler.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        import symfluence.data.acquisition  # noqa: F401 — package import triggers handler registration
        from symfluence.data.acquisition.registry import AcquisitionRegistry

        dataset_name = args.dataset

        if not AcquisitionRegistry.is_registered(dataset_name):
            available = ', '.join(AcquisitionRegistry.list_datasets())
            BaseCommand._console.error(
                f"Unknown dataset: '{dataset_name}'. Available: {available}"
            )
            return ExitCode.VALIDATION_ERROR

        handler_class = AcquisitionRegistry._get_handler_class(dataset_name)

        BaseCommand._console.info(f"Dataset: {dataset_name}")
        BaseCommand._console.rule()

        # Show class name
        BaseCommand._console.indent(f"Handler class: {handler_class.__name__}")

        # Show docstring
        doc = handler_class.__doc__
        if doc:
            BaseCommand._console.newline()
            BaseCommand._console.indent("Description:")
            for line in doc.strip().splitlines():
                stripped = line.strip()
                if stripped:
                    BaseCommand._console.indent(stripped, level=2)
        else:
            BaseCommand._console.indent("No description available.")

        # Show config keys used by the handler (scan for config_dict.get calls)
        BaseCommand._console.newline()
        BaseCommand._console.indent("Common config keys (auto-detected):")
        config_keys = _extract_config_keys(handler_class)
        if config_keys:
            for key in sorted(config_keys):
                BaseCommand._console.indent(f"- {key}", level=2)
        else:
            BaseCommand._console.indent("(none detected)", level=2)

        return ExitCode.SUCCESS


def _extract_config_keys(handler_class) -> set:
    """
    Best-effort extraction of config keys from a handler class.

    Scans the source code for config_dict.get('KEY') and
    config_dict['KEY'] patterns.

    Args:
        handler_class: The handler class to inspect

    Returns:
        Set of config key strings found
    """
    import inspect
    import re

    keys = set()
    try:
        source = inspect.getsource(handler_class)
        # Match config_dict.get('KEY' or config_dict['KEY']
        patterns = [
            r"config_dict\.get\(['\"](\w+)['\"]",
            r"config_dict\[['\"](\w+)['\"]\]",
        ]
        for pattern in patterns:
            keys.update(re.findall(pattern, source))
    except (TypeError, OSError):
        pass

    # Exclude generic keys that come from base class / CLI
    generic_keys = {
        'DOMAIN_NAME', 'BOUNDING_BOX_COORDS',
        'EXPERIMENT_TIME_START', 'EXPERIMENT_TIME_END',
        'DATA_DIR', 'FORCE_DOWNLOAD',
    }
    return keys - generic_keys
