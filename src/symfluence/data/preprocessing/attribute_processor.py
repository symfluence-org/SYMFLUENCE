# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Refactored attribute processor - backward-compatible wrapper.

This module provides a backward-compatible wrapper around the new modular
attribute processing system. The original monolithic attributeProcessor class
has been split into specialized processors, but this wrapper maintains the
original interface for existing code.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from symfluence.core.config.coercion import coerce_config
from symfluence.core.mixins import ConfigMixin

from .attribute_processors import (
    ClimateProcessor,
    ElevationProcessor,
    GeologyProcessor,
    HydrologyProcessor,
    LandCoverProcessor,
    SoilProcessor,
)


class attributeProcessor(ConfigMixin):
    """
    Backward-compatible attribute processor wrapper.

    Delegates to specialized processors while maintaining the original interface.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """Initialize attribute processor with specialized sub-processors."""
        self._config = coerce_config(config, warn=False)
        self.logger = logger

        # Initialize specialized processors
        self.elevation = ElevationProcessor(config, logger)
        self.geology = GeologyProcessor(config, logger)
        self.soil = SoilProcessor(config, logger)
        self.landcover = LandCoverProcessor(config, logger)
        self.climate = ClimateProcessor(config, logger)
        self.hydrology = HydrologyProcessor(config, logger)

        # Expose commonly used properties from base processor
        self.data_dir = self.elevation.data_dir
        self.domain_name = self.elevation.domain_name
        self.project_dir = self.elevation.project_dir
        self.catchment_path = self.elevation.catchment_path

        # For backward compatibility with DEM-related methods
        self.dem_dir = self.elevation.dem_dir
        self.slope_dir = self.elevation.slope_dir
        self.aspect_dir = self.elevation.aspect_dir

    # Delegate elevation methods
    def find_dem_file(self) -> Path:
        """Find DEM file - delegates to ElevationProcessor."""
        return self.elevation.find_dem_file()

    def generate_slope_and_aspect(self, dem_file: Path) -> Dict[str, Path]:
        """Generate slope and aspect - delegates to ElevationProcessor."""
        return self.elevation.generate_slope_and_aspect(dem_file)

    def calculate_statistics(self, raster_file: Path, attribute_name: str) -> Dict[str, float]:
        """Calculate statistics - delegates to ElevationProcessor."""
        return self.elevation.calculate_statistics(raster_file, attribute_name)

    def _process_elevation_attributes(self) -> Dict[str, float]:
        """Process elevation attributes - delegates to ElevationProcessor."""
        return self.elevation.process()

    # Delegate hydrology methods
    def calculate_water_balance(self) -> Dict[str, Any]:
        """Calculate water balance - delegates to HydrologyProcessor."""
        return self.hydrology.calculate_water_balance()

    def calculate_streamflow_signatures(self) -> Dict[str, Any]:
        """Calculate streamflow signatures - delegates to HydrologyProcessor."""
        return self.hydrology.calculate_streamflow_signatures()

    def calculate_baseflow_attributes(self) -> Dict[str, Any]:
        """Calculate baseflow attributes - delegates to HydrologyProcessor."""
        return self.hydrology.calculate_baseflow_attributes()

    def enhance_river_network_analysis(self, current_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhance river network analysis - delegates to HydrologyProcessor."""
        if current_results is None:
            current_results = {}
        return self.hydrology.enhance_river_network_analysis(current_results)

    def _process_plugin_attributes(self) -> Dict[str, Any]:
        """Run attribute processors contributed by external packages.

        Plugins advertise themselves under the ``symfluence.attribute_processors``
        entry-point group and run automatically during ``acquire_attributes``.
        Control via config:

        * ``ATTRIBUTE_PLUGINS_ENABLED`` (bool, default ``True``) - master switch.
        * ``ATTRIBUTE_PLUGINS_EXCLUDE`` (list[str], default ``[]``) - plugin
          names to skip.

        A plugin that errors is logged and skipped; it never aborts the run.
        """
        results: Dict[str, Any] = {}

        enabled = self._get_config_value(
            lambda: None,
            default=True,
            dict_key='ATTRIBUTE_PLUGINS_ENABLED',
        )
        if not enabled:
            return results

        from .attribute_processors.plugins import discover_attribute_plugins

        exclude = self._get_config_value(
            lambda: None,
            default=[],
            dict_key='ATTRIBUTE_PLUGINS_EXCLUDE',
        ) or []
        exclude = set(exclude)
        # The pipeline can inject additional exclusions (e.g. providers already
        # served by an AttributeBackend, to avoid double-extraction).
        exclude |= getattr(self, '_plugin_exclude_override', set())

        for name, cls in discover_attribute_plugins(self.logger):
            if name in exclude:
                self.logger.info(f"Skipping excluded attribute plugin '{name}'")
                continue
            try:
                self.logger.info(f"Processing attributes from plugin '{name}'")
                plugin_results = cls(self._config, self.logger).process()
                if plugin_results:
                    results.update(plugin_results)
                    self.logger.info(f"Plugin '{name}' contributed {len(plugin_results)} attributes")
            except Exception as e:  # noqa: BLE001 — a plugin must not break attribute processing
                self.logger.warning(f"Attribute plugin '{name}' failed and was skipped: {e}")

        return results

    # Main processing method
    def process_attributes(self) -> pd.DataFrame:
        """
        Process catchment attributes from available data sources.

        This is the main entry point that orchestrates all attribute processing.

        Returns:
            pd.DataFrame: Complete dataframe of catchment attributes
        """
        self.logger.info("Starting attribute processing")

        try:
            # Initialize results dictionary
            all_results = {}

            # Process elevation attributes (DEM, slope, aspect)
            self.logger.info("Processing elevation attributes")
            elevation_results = self.elevation.process()
            all_results.update(elevation_results)

            # Process geological attributes
            self.logger.info("Processing geological attributes")
            geology_results = self.geology.process()
            all_results.update(geology_results)

            # Process soil attributes
            self.logger.info("Processing soil attributes")
            soil_results = self.soil.process()
            all_results.update(soil_results)

            # Process land cover attributes
            self.logger.info("Processing land cover attributes")
            landcover_results = self.landcover.process()
            all_results.update(landcover_results)

            # Process climate attributes
            self.logger.info("Processing climate attributes")
            climate_results = self.climate.process()
            all_results.update(climate_results)

            # Process hydrological attributes
            self.logger.info("Processing hydrological attributes")
            hydrology_results = self.hydrology.process()
            all_results.update(hydrology_results)

            # Process externally-registered attribute plugins (e.g. climaclass
            # climate classification). Discovered via entry points; opt-out only.
            plugin_results = self._process_plugin_attributes()
            all_results.update(plugin_results)

            # Convert to DataFrame
            if all_results:
                # Check if we have distributed HRUs
                is_lumped = self._get_config_value(lambda: self.config.domain.definition_method, dict_key='DOMAIN_DEFINITION_METHOD') == 'lumped'

                if is_lumped:
                    # Single row for lumped catchment
                    df = pd.DataFrame([all_results])
                    df.index = [1]  # Basin ID
                    df.index.name = 'basin_id'
                else:
                    # Multiple rows for distributed HRUs
                    # Extract HRU IDs from keys
                    hru_ids = set()
                    for key in all_results.keys():
                        if key.startswith("HRU_"):
                            hru_id = int(key.split("_")[1])
                            hru_ids.add(hru_id)

                    # Create multi-index DataFrame
                    if hru_ids:
                        rows = []
                        for hru_id in sorted(hru_ids):
                            row = {}
                            prefix = f"HRU_{hru_id}_"
                            for key, value in all_results.items():
                                if key.startswith(prefix):
                                    clean_key = key.replace(prefix, "")
                                    row[clean_key] = value
                            rows.append(row)

                        df = pd.DataFrame(rows)
                        df['hru_id'] = sorted(hru_ids)
                        df['basin_id'] = 1
                        df.set_index(['basin_id', 'hru_id'], inplace=True)
                    else:
                        # Fallback to lumped
                        df = pd.DataFrame([all_results])
                        df.index = [1]
                        df.index.name = 'basin_id'

                self.logger.info(f"Attribute processing complete. Generated {len(df)} rows with {len(df.columns)} attributes")
                return df
            else:
                self.logger.warning("No attributes were processed")
                return pd.DataFrame()

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error in attribute processing: {str(e)}")
            raise

    def process_profile_attributes(self, profile_name: str = 'core'):
        """Run attribute processors enabled by the given profile.

        Writes results as CSVs to ``data/attributes/{category}/`` for
        each processor category.  The ``AttributesNetCDFBuilder`` picks
        up these CSVs during ``build_model_ready_store``.

        Parameters
        ----------
        profile_name : str
            One of 'core', 'camels_spat', 'full'.
        """
        from symfluence.core.mixins.project import resolve_data_subdir

        _PROFILE_CATEGORIES = {
            'core': [],
            'camels_spat': ['climate', 'soil', 'geology', 'landcover'],
            'full': ['climate', 'soil', 'geology', 'landcover', 'hydrology'],
        }

        categories = _PROFILE_CATEGORIES.get(profile_name, [])
        if not categories:
            return

        self.logger.info(
            f"Processing extended attributes (profile: {profile_name}, "
            f"categories: {categories})"
        )

        attrs_dir = resolve_data_subdir(self.project_dir, 'attributes')

        _PROCESSOR_MAP = {
            'climate': (self.climate, attrs_dir / 'climate', 'worldclim_attributes.csv'),
            'soil': (self.soil, attrs_dir / 'soilclass', 'soil_extended_attributes.csv'),
            'geology': (self.geology, attrs_dir / 'geology', 'glhymps_attributes.csv'),
            'landcover': (self.landcover, attrs_dir / 'landclass', 'landcover_extended_attributes.csv'),
            'hydrology': (self.hydrology, attrs_dir / 'hydrology', 'hydrology_attributes.csv'),
        }

        for cat in categories:
            processor, out_dir, filename = _PROCESSOR_MAP[cat]
            try:
                self.logger.info(f"Processing {cat} attributes")
                results = processor.process()
                if results:
                    processor._write_results_csv(results, out_dir, filename)
                else:
                    self.logger.info(f"No {cat} attributes produced (data may not be available)")
            except Exception as e:  # noqa: BLE001
                self.logger.warning(f"{cat} attribute processing failed (non-fatal): {e}")
