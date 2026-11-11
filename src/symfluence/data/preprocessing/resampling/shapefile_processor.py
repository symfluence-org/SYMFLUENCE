# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Shapefile Processor

Handles shapefile CRS conversion and HRU ID uniqueness.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import geopandas as gpd

from symfluence.core.mixins import ConfigMixin

from .geometry_validator import GeometryValidator


class ShapefileProcessor(ConfigMixin):
    """
    Processes shapefiles for EASYMORE remapping compatibility.

    Ensures shapefiles are in WGS84 (EPSG:4326) and have unique HRU IDs.
    """

    def __init__(self, config: dict, logger: logging.Logger = None):
        """
        Initialize shapefile processor.

        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger or logging.getLogger(__name__)
        self.geometry_validator = GeometryValidator(self.logger)

    def ensure_unique_hru_ids(
        self,
        shapefile_path: Path,
        hru_id_field: str
    ) -> Tuple[Path, str]:
        """
        Ensure HRU IDs are unique in the shapefile.

        For lumped catchments with duplicate HRU_IDs, dissolves features by HRU_ID.
        For distributed models, creates new unique sequential IDs.

        Args:
            shapefile_path: Path to the shapefile
            hru_id_field: Name of the HRU ID field

        Returns:
            tuple: (updated_shapefile_path, actual_hru_id_field_used)
        """
        try:
            shapefile_path = Path(shapefile_path)

            # Check for existing processed versions
            dissolved_path = shapefile_path.parent / f"{shapefile_path.stem}_dissolved.shp"
            unique_ids_path = shapefile_path.parent / f"{shapefile_path.stem}_unique_ids.shp"

            for existing_path in [dissolved_path, unique_ids_path]:
                if existing_path.exists():
                    try:
                        existing_gdf = gpd.read_file(existing_path)
                        if hru_id_field in existing_gdf.columns:
                            if existing_gdf[hru_id_field].nunique() == len(existing_gdf):
                                self.logger.debug(
                                    f"Using existing processed shapefile: {existing_path.name}"
                                )
                                return existing_path, hru_id_field
                    except Exception as e:  # noqa: BLE001 — preprocessing resilience
                        self.logger.debug(f"Could not use existing {existing_path.name}: {e}", exc_info=True)

            # Read the shapefile
            gdf = gpd.read_file(shapefile_path)
            self.logger.debug(f"Checking HRU ID uniqueness in {shapefile_path.name}")
            self.logger.debug(f"Available fields: {list(gdf.columns)}")

            if hru_id_field not in gdf.columns:
                raise ValueError(f"HRU ID field '{hru_id_field}' not found in shapefile")

            # Check for uniqueness
            original_count = len(gdf)
            unique_count = gdf[hru_id_field].nunique()

            self.logger.debug(
                f"Shapefile has {original_count} rows, {unique_count} unique {hru_id_field} values"
            )

            if unique_count == original_count:
                self.logger.debug(f"All {hru_id_field} values are unique")
                return shapefile_path, hru_id_field

            # Handle duplicates
            self.logger.info(f"Found {original_count - unique_count} duplicate {hru_id_field} values")

            if unique_count <= 10:
                # Lumped/semi-distributed: dissolve features
                return self._dissolve_features(gdf, hru_id_field, shapefile_path)
            else:
                # Distributed: create unique sequential IDs
                return self._create_unique_ids(gdf, hru_id_field, shapefile_path)

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error ensuring unique HRU IDs: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def _dissolve_features(
        self,
        gdf,
        hru_id_field: str,
        shapefile_path: Path
    ) -> Tuple[Path, str]:
        """Dissolve features by HRU_ID for lumped catchments."""
        unique_count = gdf[hru_id_field].nunique()
        original_count = len(gdf)

        self.logger.info(f"Detected lumped/semi-distributed catchment ({unique_count} unique HRUs)")
        self.logger.info(f"Dissolving {original_count} features into {unique_count} HRUs")

        # Build aggregation dictionary
        agg_dict = {}
        for col in gdf.columns:
            if col == 'geometry' or col == hru_id_field:
                continue
            elif col in ['GRU_ID', 'gru_to_seg']:
                agg_dict[col] = 'first'
            elif gdf[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                if col.lower().endswith('_area') or col in ['HRU_area', 'GRU_area', 'HRUarea', 'GRUarea']:
                    agg_dict[col] = 'sum'
                else:
                    agg_dict[col] = 'mean'
            else:
                agg_dict[col] = 'first'

        gdf_dissolved = gdf.dissolve(by=hru_id_field, aggfunc=agg_dict)
        gdf_dissolved = gdf_dissolved.reset_index()

        self.logger.info(f"Dissolved into {len(gdf_dissolved)} features")

        # Validate and repair geometries
        gdf_dissolved = self.geometry_validator.validate_and_repair(gdf_dissolved)

        output_path = shapefile_path.parent / f"{shapefile_path.stem}_dissolved.shp"
        gdf_dissolved.to_file(output_path)
        self.logger.info(f"Dissolved shapefile saved to: {output_path}")

        # Verify
        verify_gdf = gpd.read_file(output_path)
        if verify_gdf[hru_id_field].nunique() == len(verify_gdf):
            self.logger.info(f"Verification successful: {len(verify_gdf)} unique HRUs")
            return output_path, hru_id_field
        else:
            raise ValueError("Could not dissolve features by HRU_ID")

    def _create_unique_ids(
        self,
        gdf,
        hru_id_field: str,
        shapefile_path: Path
    ) -> Tuple[Path, str]:
        """Create unique sequential IDs for distributed models."""
        self.logger.info(
            f"Detected distributed model ({gdf[hru_id_field].nunique()} unique HRUs, "
            f"{len(gdf)} features)"
        )
        self.logger.info("Creating new unique sequential IDs for each feature")

        new_hru_field = "hru_id_new"

        if new_hru_field in gdf.columns and gdf[new_hru_field].nunique() == len(gdf):
            self.logger.info(f"Using existing unique field: {new_hru_field}")
            gdf_updated = gdf.copy()
        else:
            self.logger.info(f"Creating new unique IDs in field: {new_hru_field}")
            gdf_updated = gdf.copy()
            gdf_updated[new_hru_field] = range(1, len(gdf_updated) + 1)

        output_path = shapefile_path.parent / f"{shapefile_path.stem}_unique_ids.shp"
        gdf_updated.to_file(output_path)
        self.logger.info(f"Updated shapefile with unique IDs saved to: {output_path}")

        # Verify and find actual field name (may be truncated by shapefile)
        verify_gdf = gpd.read_file(output_path)
        self.logger.debug(f"Fields in saved shapefile: {list(verify_gdf.columns)}")

        possible_fields = [col for col in verify_gdf.columns if col.startswith('hru_id')]
        if not possible_fields:
            raise ValueError("Could not find unique HRU ID field in saved shapefile")

        actual_saved_field = possible_fields[0]
        self.logger.info(f"Using field '{actual_saved_field}' from saved shapefile")

        if verify_gdf[actual_saved_field].nunique() == len(verify_gdf):
            self.logger.info(f"Verification successful: All {actual_saved_field} values are unique")
            return output_path, actual_saved_field
        else:
            raise ValueError("Could not create unique HRU IDs")

    def ensure_wgs84(
        self,
        shapefile_path: Path,
        output_suffix: str = "_wgs84",
        is_target: Optional[bool] = None,
    ) -> Union[Path, Tuple[Path, str]]:
        """
        Ensure shapefile is in WGS84 (EPSG:4326).

        Args:
            shapefile_path: Path to the shapefile
            output_suffix: Suffix for WGS84 version
            is_target: True for the target (catchment/HRU) shapefile, False
                for the source (forcing grid) shapefile. Pass it explicitly;
                when omitted the role is guessed from the FILE name only
                (never from parent directories, which may contain "catchment"
                in a data root).

        Returns:
            For target shapefiles: (wgs84_path, hru_id_field)
            For source shapefiles: wgs84_path
        """
        shapefile_path = Path(shapefile_path)
        if is_target is None:
            is_target_shapefile = 'catchment' in shapefile_path.name.lower()
        else:
            is_target_shapefile = bool(is_target)

        try:
            gdf = gpd.read_file(shapefile_path)
            current_crs = gdf.crs

            self.logger.debug(f"Checking CRS for {shapefile_path.name}: {current_crs}")

            # For target shapefiles, ensure unique HRU IDs first
            if is_target_shapefile:
                hru_id_field = self._get_config_value(lambda: self.config.paths.catchment_hruid, dict_key='CATCHMENT_SHP_HRUID')
                shapefile_path, actual_hru_field = self.ensure_unique_hru_ids(
                    shapefile_path, hru_id_field
                )
                gdf = gpd.read_file(shapefile_path)
                current_crs = gdf.crs

            # Check if already in WGS84
            if current_crs is not None and current_crs.to_epsg() == 4326:
                self.logger.debug(f"Shapefile {shapefile_path.name} already in WGS84")
                if is_target_shapefile:
                    return shapefile_path, actual_hru_field
                else:
                    return shapefile_path

            # Create WGS84 version
            wgs84_shapefile = shapefile_path.parent / f"{shapefile_path.stem}{output_suffix}.shp"

            # Check if WGS84 version already exists
            if wgs84_shapefile.exists():
                try:
                    wgs84_gdf = gpd.read_file(wgs84_shapefile)
                    if wgs84_gdf.crs is not None and wgs84_gdf.crs.to_epsg() == 4326:
                        if is_target_shapefile:
                            possible_fields = [
                                col for col in wgs84_gdf.columns if col.startswith('hru_id')
                            ]
                            if possible_fields and wgs84_gdf[possible_fields[0]].nunique() == len(wgs84_gdf):
                                self.logger.info(
                                    f"WGS84 version with unique IDs exists: {wgs84_shapefile.name}"
                                )
                                return wgs84_shapefile, possible_fields[0]
                        else:
                            self.logger.info(f"WGS84 version already exists: {wgs84_shapefile.name}")
                            return wgs84_shapefile
                except Exception as e:  # noqa: BLE001 — preprocessing resilience
                    self.logger.warning(f"Error reading existing WGS84 file: {e}. Recreating.", exc_info=True)

            # Convert to WGS84
            self.logger.info(f"Converting {shapefile_path.name} from {current_crs} to WGS84")
            gdf_wgs84 = gdf.to_crs('EPSG:4326')
            gdf_wgs84.to_file(wgs84_shapefile)
            self.logger.info(f"WGS84 shapefile created: {wgs84_shapefile}")

            if is_target_shapefile:
                saved_gdf = gpd.read_file(wgs84_shapefile)
                possible_fields = [col for col in saved_gdf.columns if col.startswith('hru_id')]
                if possible_fields:
                    return wgs84_shapefile, possible_fields[0]
                else:
                    return wgs84_shapefile, actual_hru_field
            else:
                return wgs84_shapefile

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error ensuring WGS84 for {shapefile_path}: {str(e)}")
            raise
