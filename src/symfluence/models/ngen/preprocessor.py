# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
NGen Model Preprocessor.

Handles spatial preprocessing and configuration generation for the NOAA NextGen Framework.
Uses shared utilities for time window management and forcing data processing.
"""

import json
import logging
import sys
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import yaml

from symfluence.models.base import BaseModelPreProcessor
from symfluence.models.ngen.config_generator import NgenConfigGenerator
from symfluence.models.registry import ModelRegistry
from symfluence.models.utilities import ForcingDataProcessor, TimeWindowManager


@ModelRegistry.register_preprocessor('NGEN')
class NgenPreProcessor(BaseModelPreProcessor):  # type: ignore[misc]
    """
    Preprocessor for NextGen Framework.

    Handles conversion of SYMFLUENCE data to ngen-compatible formats including:
    - Catchment geometry (geopackage)
    - Nexus points (GeoJSON)
    - Forcing data (NetCDF)
    - Model configurations (CFE, PET, NOAH-OWP)
    - Realization configuration (JSON)

    Inherits observation loading from ObservationLoaderMixin.
    """


    MODEL_NAME = "NGEN"
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the NextGen preprocessor.

        Sets up NGEN-specific configuration including module availability
        checking (SLOTH, PET, NOAH-OWP, CFE) and library path resolution.

        Args:
            config: Configuration dictionary or SymfluenceConfig object containing
                NGEN settings, module selection, and installation paths.
            logger: Logger instance for status messages and debugging.

        Note:
            Modules are selected via ``NGEN_MODULES_SELECTED`` (comma-separated
            string, e.g. ``"SLOTH,PET,CFE"``).  A module is only enabled when
            it appears in the selection AND the corresponding shared library
            exists on disk.
        """
        # Initialize base class (handles standard paths and directories)
        super().__init__(config, logger)

        # NGen-specific configuration (typed config)
        self.hru_id_col = self._get_config_value(
            lambda: self.config.paths.catchment_hruid,
            default='HRU_ID'
        )

        self._ngen_lib_paths = self._resolve_ngen_lib_paths()
        # Ensure we check for existence using the correctly resolved paths
        self._available_modules = {}
        for name, path in self._ngen_lib_paths.items():
            exists = path.exists()
            self._available_modules[name] = exists
            self.logger.info(f"Module {name} path: {path} (exists={exists})")
            if not exists:
                self.logger.warning(f"NGEN module library missing for {name}: {path}")

        # Determine which modules to include based on NGEN_MODULES_SELECTED
        # AND library availability.  A module is only enabled when it appears
        # in the selection AND the .so/.dylib exists on disk.
        modules_selected_str = self._get_config_value(
            lambda: self.config.model.ngen.modules_selected,
            default='SLOTH,PET,CFE',
        )
        _selected = {m.strip().upper() for m in modules_selected_str.split(',') if m.strip()}

        _all_modules = ['SLOTH', 'PET', 'NOAH', 'CFE', 'TOPMODEL', 'SACSMA', 'SNOW17']
        _resolved = {}
        for mod_name in _all_modules:
            config_enabled = mod_name in _selected
            lib_exists = self._available_modules.get(mod_name, False)
            if config_enabled and not lib_exists:
                self.logger.warning(
                    f"{mod_name} is selected in NGEN_MODULES_SELECTED but library not found — "
                    f"disabling to prevent ngen runtime failure"
                )
            _resolved[mod_name] = config_enabled and lib_exists

        self._include_sloth = _resolved['SLOTH']
        self._include_pet = _resolved['PET']
        self._include_noah = _resolved['NOAH']
        self._include_cfe = _resolved['CFE']
        self._include_topmodel = _resolved['TOPMODEL']
        self._include_sacsma = _resolved['SACSMA']
        self._include_snow17 = _resolved['SNOW17']

        # Validate module exclusivity
        self._validate_module_exclusivity()

        # Determine the active runoff module name for logging
        active_runoff = [m for m in ['CFE', 'TOPMODEL', 'SACSMA']
                         if getattr(self, f'_include_{m.lower()}', False)]
        runoff_name = active_runoff[0] if active_runoff else 'none'

        # Coupling logic: NOAH/Snow-17 and PET serve complementary roles.
        # The land-surface/snow module outputs net water input as the runoff module's
        # precipitation input. PET provides potential ET for soil moisture accounting.
        if self._include_noah and self._include_pet:
            self.logger.info(
                f"NOAH+PET both enabled: using QINSUR-based coupling. "
                f"NOAH provides QINSUR (post-snow/interception water) to {runoff_name} as precipitation; "
                f"PET provides potential ET for {runoff_name}'s soil moisture depletion."
            )
        elif self._include_noah and not self._include_pet:
            # Get the ET fallback configuration
            self._noah_et_fallback = self._get_config_value(
                lambda: self.config.model.ngen.noah_et_fallback,
                default='EVAPOTRANS'
            )
            valid_fallbacks = ['EVAPOTRANS', 'ETRAN', 'ECAN', 'QSEVA']
            if self._noah_et_fallback not in valid_fallbacks:
                self.logger.warning(
                    f"Invalid NGEN_NOAH_ET_FALLBACK '{self._noah_et_fallback}', using 'EVAPOTRANS'"
                )
                self._noah_et_fallback = 'EVAPOTRANS'

            self.logger.info(
                f"NOAH enabled but PET disabled: {runoff_name} will receive NOAH's {self._noah_et_fallback} "
                f"(actual ET) instead of potential ET. For physically correct potential ET, "
                f"add PET to NGEN_MODULES_SELECTED."
            )
        else:
            self._noah_et_fallback = None

        # Log module configuration
        self.logger.info("NGEN module configuration:")
        self.logger.info(f"  SLOTH: {'ENABLED' if self._include_sloth else 'DISABLED'}")
        self.logger.info(f"  PET: {'ENABLED' if self._include_pet else 'DISABLED'}")
        self.logger.info(f"  NOAH-OWP: {'ENABLED' if self._include_noah else 'DISABLED'}")
        self.logger.info(f"  CFE: {'ENABLED' if self._include_cfe else 'DISABLED'}")
        self.logger.info(f"  TOPMODEL: {'ENABLED' if self._include_topmodel else 'DISABLED'}")
        self.logger.info(f"  SAC-SMA: {'ENABLED' if self._include_sacsma else 'DISABLED'}")
        self.logger.info(f"  Snow-17: {'ENABLED' if self._include_snow17 else 'DISABLED'}")

    def _validate_module_exclusivity(self):
        """Validate that mutually exclusive modules are not both enabled.

        Enforces:
        - Rainfall-runoff slot: exactly one of CFE, TOPMODEL, SAC-SMA
        - Snow/land-surface slot: at most one of NOAH, Snow-17
        """
        # Rainfall-runoff: at most one
        runoff_modules = [m for m in ['CFE', 'TOPMODEL', 'SACSMA']
                          if getattr(self, f'_include_{m.lower()}', False)]
        if len(runoff_modules) > 1:
            keep = runoff_modules[0]
            for drop in runoff_modules[1:]:
                setattr(self, f'_include_{drop.lower()}', False)
            self.logger.warning(
                f"Multiple runoff modules enabled ({', '.join(runoff_modules)}). "
                f"Keeping {keep}, disabling {', '.join(runoff_modules[1:])}."
            )

        # Snow/land-surface: at most one
        if self._include_noah and self._include_snow17:
            self.logger.warning(
                "Both NOAH and Snow-17 enabled. They occupy the same slot. "
                "Keeping NOAH, disabling Snow-17."
            )
            self._include_snow17 = False

    def _detect_npm_lib_dir(self) -> Optional[Path]:
        """Detect the npm-installed symfluence dist/lib/ directory."""
        import subprocess as _sp
        try:
            result = _sp.run(
                ["npm", "root", "-g"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                lib_dir = Path(result.stdout.strip()) / "symfluence" / "dist" / "lib"
                if lib_dir.is_dir():
                    return lib_dir
        except (FileNotFoundError, OSError, _sp.TimeoutExpired, _sp.SubprocessError):
            pass
        return None

    def _resolve_ngen_lib_paths(self) -> Dict[str, Path]:
        lib_ext = ".dylib" if sys.platform == "darwin" else ".so"
        install_path = self._get_config_value(
            lambda: self.config.model.ngen.install_path,
            default='default'
        )

        # Module library names (shared across all resolution strategies)
        module_libs = {
            "SLOTH": f"libslothmodel{lib_ext}",
            "PET": f"libpetbmi{lib_ext}",
            "NOAH": f"libsurfacebmi{lib_ext}",
            "CFE": f"libcfebmi{lib_ext}",
            "TOPMODEL": f"libtopmodelbmi{lib_ext}",
            "SACSMA": f"libsacbmi{lib_ext}",
            "SNOW17": f"libsnow17_bmi{lib_ext}",
        }

        # --- 1. Try npm-bundled libraries first (when install_path is default) ---
        if install_path == 'default':
            npm_lib_dir = self._detect_npm_lib_dir()
            if npm_lib_dir is not None:
                npm_paths = {}
                all_found = True
                for name, libname in module_libs.items():
                    candidate = npm_lib_dir / libname
                    if candidate.exists():
                        npm_paths[name] = candidate
                    else:
                        all_found = False
                if all_found:
                    self.logger.info(f"Resolved NGEN libraries from npm bundle: {npm_lib_dir}")
                    return npm_paths
                else:
                    self.logger.info("npm bundle found but missing some NGEN libraries, falling back")

        # --- 2. Resolve from install path (explicit or default fallback) ---
        if install_path == 'default':
            ngen_base = self.data_dir / 'installs' / 'ngen'
        else:
            p = Path(install_path)
            if p.name == 'cmake_build':
                ngen_base = p.parent
            else:
                ngen_base = p

        self.logger.info(f"Resolved NGEN_BASE to: {ngen_base}")

        # Check both ngen_base/extern and ngen_base/cmake_build/extern
        paths = {}
        module_subpaths = {
            "SLOTH": ("extern/sloth/cmake_build", f"libslothmodel{lib_ext}"),
            "PET": ("extern/evapotranspiration/evapotranspiration/cmake_build", f"libpetbmi{lib_ext}"),
            "NOAH": ("extern/noah-owp-modular/cmake_build", f"libsurfacebmi{lib_ext}"),
            "CFE": ("extern/cfe/cmake_build", f"libcfebmi{lib_ext}"),
            "TOPMODEL": ("extern/topmodel/cmake_build", f"libtopmodelbmi{lib_ext}"),
            "SACSMA": ("extern/sac-sma/cmake_build", f"libsacbmi{lib_ext}"),
            "SNOW17": ("extern/snow17/cmake_build", f"libsnow17_bmi{lib_ext}"),
        }

        for name, (subpath, libname) in module_subpaths.items():
            # Try direct extern
            p1 = ngen_base / subpath / libname
            # Try extern under cmake_build
            p2 = ngen_base / "cmake_build" / subpath / libname

            if p1.exists():
                paths[name] = p1
            elif p2.exists():
                paths[name] = p2
            else:
                # Fallback to p1 for consistent missing path reporting
                paths[name] = p1

        return paths

    def _copy_noah_parameter_tables(self):
        """
        Copy Noah-OWP parameter tables from package data to domain settings.
        """
        if not self._include_noah:
            return

        self.logger.info("Copying Noah-OWP parameter tables")
        from symfluence.resources import get_base_settings_dir

        try:
            noah_base_dir = get_base_settings_dir('NOAH')
            source_param_dir = noah_base_dir / 'parameters'
        except FileNotFoundError:
            self.logger.warning("NOAH base settings not found in package; skipping parameter table copy")
            return

        dest_param_dir = self.setup_dir / 'NOAH' / 'parameters'
        param_files = ['GENPARM.TBL', 'MPTABLE.TBL', 'SOILPARM.TBL']

        for param_file in param_files:
            source_file = source_param_dir / param_file
            dest_file = dest_param_dir / param_file
            if source_file.exists():
                copyfile(source_file, dest_file)

    def run_preprocessing(self):
        """
        Execute complete NGEN preprocessing workflow.

        Runs the full preprocessing pipeline using the template method pattern:
        1. Create directories for NGEN modules (CFE, PET, NOAH)
        2. Copy base settings (Noah-OWP parameter tables)
        3. Prepare forcing data (NetCDF and CSV formats)
        4. Create catchment geopackage and nexus GeoJSON
        5. Generate model configurations for all enabled modules
        6. Generate realization config tying everything together

        Returns:
            Path: Path to setup directory containing all NGEN configurations.
        """
        self.logger.info("Starting NextGen preprocessing")
        return self.run_preprocessing_template()

    def create_directories(self, additional_dirs=None):
        """Override to add NGen-specific directories."""
        ngen_dirs = [
            self.setup_dir / "CFE",
            self.setup_dir / "PET",
            self.setup_dir / "NOAH",
            self.setup_dir / "NOAH" / "parameters",
            self.setup_dir / "TOPMODEL",
            self.setup_dir / "SACSMA",
            self.setup_dir / "SNOW17",
            self.forcing_dir / "csv"
        ]
        if additional_dirs:
            ngen_dirs.extend(additional_dirs)
        super().create_directories(additional_dirs=ngen_dirs)

    def copy_base_settings(self, source_dir: Optional[Path] = None, file_patterns: Optional[List[str]] = None):
        """Override to copy Noah-OWP parameter tables."""
        if source_dir:
            return super().copy_base_settings(source_dir, file_patterns)
        self._copy_noah_parameter_tables()

    def _prepare_forcing(self) -> None:
        """NGEN-specific forcing data preparation."""
        self._forcing_file = self.prepare_forcing_data()

    def _create_model_configs(self) -> None:
        """NGEN-specific configuration file creation."""
        self._nexus_file = self.create_nexus_geojson()
        self._catchment_file = self.create_catchment_geopackage()

        self.generate_model_configs()
        self.generate_realization_config(
            self._catchment_file,
            self._nexus_file,
            self._forcing_file
        )

    def create_nexus_geojson(self) -> Path:
        """
        Create nexus GeoJSON from river network topology.

        Generates a GeoJSON file defining nexus points (flow exchange locations)
        at river segment endpoints. For distributed domains, creates a nexus
        for each segment; for lumped domains, creates a single outlet nexus.

        Returns:
            Path: Path to created nexus.geojson file.

        Note:
            - Nexus points connect catchments (waterbodies) in the NGEN framework
            - Terminal nexuses (outlets) have empty 'toid' and type='poi'
            - Internal nexuses have toid pointing to downstream waterbody
        """
        self.logger.info("Creating nexus GeoJSON")
        river_network_file = self.get_river_network_path()
        if not river_network_file.exists():
            return self._create_simple_nexus()

        river_gdf = gpd.read_file(river_network_file)
        seg_id_col = self._get_config_value(
            lambda: self.config.paths.river_network_segid,
            default='LINKNO'
        )
        downstream_col = self._get_config_value(
            lambda: self.config.paths.river_network_downsegid,
            default='DSLINKNO'
        )

        nexus_features = []
        for idx, row in river_gdf.iterrows():
            seg_id = row[seg_id_col]
            downstream_id = row[downstream_col]
            geom = row.geometry
            endpoint = geom.coords[-1] if geom.geom_type == 'LineString' else (geom.x, geom.y)
            nexus_id = f"nex-{int(seg_id)}"
            nexus_type = "poi" if (downstream_id == 0 or pd.isna(downstream_id)) else "nexus"
            toid = "" if nexus_type == "poi" else f"wb-{int(downstream_id)}"

            nexus_features.append({
                "type": "Feature", "id": nexus_id,
                "properties": {"toid": toid, "hl_id": None, "hl_uri": "NA", "type": nexus_type},
                "geometry": {"type": "Point", "coordinates": list(endpoint)}
            })

        nexus_file = self.setup_dir / "nexus.geojson"
        with open(nexus_file, 'w', encoding='utf-8') as f:
            json.dump({"type": "FeatureCollection", "name": "nexus", "xy_coordinate_resolution": 1e-06, "features": nexus_features}, f, indent=2)
        return nexus_file

    def _create_simple_nexus(self) -> Path:
        """Create a simple single-nexus for lumped catchments."""
        catchment_file = self.get_catchment_path()
        catchment_gdf = gpd.read_file(catchment_file)
        catchment_utm = catchment_gdf.to_crs(catchment_gdf.estimate_utm_crs())
        centroid = catchment_utm.geometry.centroid.to_crs("EPSG:4326").iloc[0]
        catchment_id = str(catchment_gdf[self.hru_id_col].iloc[0])

        nexus_file = self.setup_dir / "nexus.geojson"
        with open(nexus_file, 'w', encoding='utf-8') as f:
            json.dump({"type": "FeatureCollection", "name": "nexus", "xy_coordinate_resolution": 1e-06, "features": [{
                "type": "Feature", "id": f"nex-{catchment_id}",
                "properties": {"toid": "", "hl_id": None, "hl_uri": "NA", "type": "poi"},
                "geometry": {"type": "Point", "coordinates": [centroid.x, centroid.y]}
            }]}, f, indent=2)
        return nexus_file

    def create_catchment_geopackage(self) -> Path:
        """
        Create NGEN-compatible geopackage and GeoJSON catchment files.

        Transforms catchment shapefile to NGEN format with required columns:
        - divide_id: Catchment identifier prefixed with 'cat-'
        - toid: Target nexus identifier prefixed with 'nex-'
        - areasqkm: Catchment area in square kilometers
        - type: Always 'network' for active catchments

        Creates both GeoPackage (EPSG:5070) and GeoJSON (EPSG:4326) outputs
        since NGEN requires specific CRS for different operations.

        Returns:
            Path: Path to created geopackage file.
        """
        from shapely.geometry import mapping

        catchment_file = self.get_catchment_path()
        catchment_gdf = gpd.read_file(catchment_file)
        divides_gdf = catchment_gdf.copy()
        divides_gdf['divide_id'] = divides_gdf[self.hru_id_col].apply(lambda x: f'cat-{x}')
        divides_gdf['toid'] = divides_gdf[self.hru_id_col].apply(lambda x: f'nex-{x}')
        divides_gdf['type'] = 'network'
        utm_crs = divides_gdf.estimate_utm_crs()
        divides_gdf['areasqkm'] = divides_gdf.to_crs(utm_crs).geometry.area / 1e6

        for col in ['ds_id', 'lengthkm', 'tot_drainage_areasqkm', 'has_flowline']:
            if col not in divides_gdf.columns:
                divides_gdf[col] = 0.0 if col != 'has_flowline' else False

        # Add 'id' column for NGEN compatibility
        divides_gdf['id'] = divides_gdf['divide_id']

        divides_gdf = divides_gdf[['id', 'divide_id', 'toid', 'type', 'areasqkm', 'geometry', 'ds_id', 'lengthkm', 'tot_drainage_areasqkm', 'has_flowline']]

        # For GPKG, use EPSG:5070 (Albers Equal Area)
        gpkg_gdf = divides_gdf.copy()
        if gpkg_gdf.crs != "EPSG:5070":
            gpkg_gdf = gpkg_gdf.to_crs("EPSG:5070")
        # Reset index to avoid duplicate 'id' column error when writing
        # The 'id' column is already present in the data for NGEN compatibility
        gpkg_gdf = gpkg_gdf.reset_index(drop=True)

        gpkg_file = self.setup_dir / f"{self.domain_name}_catchments.gpkg"
        gpkg_gdf.to_file(gpkg_file, layer='divides', driver='GPKG')

        # For GeoJSON, use EPSG:4326 (WGS84) and manually set feature-level id
        # NGEN requires feature-level id field which geopandas doesn't set automatically
        geojson_gdf = divides_gdf.to_crs("EPSG:4326")

        features = []
        for _, row in geojson_gdf.iterrows():
            # Build properties dict, handling NaN values
            props: Dict[str, Any] = {}
            for k, v in row.drop('geometry').to_dict().items():
                if isinstance(v, float) and np.isnan(v):
                    props[k] = None
                else:
                    props[k] = v

            feat = {
                'type': 'Feature',
                'id': row['divide_id'],  # Feature-level id required by NGEN
                'properties': props,
                'geometry': mapping(row.geometry)
            }
            features.append(feat)

        geojson_data = {
            'type': 'FeatureCollection',
            'name': f'{self.domain_name}_catchments',
            'features': features
        }

        geojson_file = self.setup_dir / f"{self.domain_name}_catchments.geojson"
        with open(geojson_file, 'w', encoding='utf-8') as f:
            json.dump(geojson_data, f, indent=2)

        self.logger.info(f"Created NGEN catchment files: {gpkg_file.name}, {geojson_file.name}")
        return gpkg_file

    def _convert_forcing_units_to_ngen(self, forcing_data: xr.Dataset) -> xr.Dataset:
        """
        Convert forcing data units to NGEN/AORC standards.

        Expected NGEN/AORC units:
        - pptrate: kg m⁻² s⁻¹ (equivalent to mm/s)
        - airtemp: K (Kelvin)
        - airpres: Pa (Pascals)
        - spechum: kg/kg (dimensionless)
        - SWRadAtm, LWRadAtm: W/m²

        Args:
            forcing_data: Input forcing dataset

        Returns:
            Dataset with units converted to NGEN standards
        """
        from symfluence.core.constants import UnitDetectionThresholds

        # Temperature: Convert °C to K if needed
        if 'air_temperature' in forcing_data:
            temp_mean = float(forcing_data['air_temperature'].mean())
            if temp_mean < UnitDetectionThresholds.TEMP_KELVIN_VS_CELSIUS:
                self.logger.info("Converting temperature from °C to K (+273.15)")
                forcing_data['air_temperature'] = forcing_data['air_temperature'] + 273.15
            else:
                self.logger.debug(f"Temperature appears to be in K (mean={temp_mean:.1f})")

        # Precipitation rate: Convert mm/day or mm/hour to kg m⁻² s⁻¹ (mm/s)
        if 'precipitation_flux' in forcing_data:
            precip_units = forcing_data['precipitation_flux'].attrs.get('units', '').lower()
            precip_mean = float(forcing_data['precipitation_flux'].mean())
            precip_max = float(forcing_data['precipitation_flux'].max())

            if 'mm/day' in precip_units or 'mm day' in precip_units or 'mm d-1' in precip_units:
                self.logger.info("Converting precipitation from mm/day to kg m⁻² s⁻¹ (÷86400)")
                forcing_data['precipitation_flux'] = forcing_data['precipitation_flux'] / 86400.0
            elif 'mm/h' in precip_units or 'mm h-1' in precip_units or ('mm' in precip_units and 'hour' in precip_units):
                self.logger.info("Converting precipitation from mm/hour to kg m⁻² s⁻¹ (÷3600)")
                forcing_data['precipitation_flux'] = forcing_data['precipitation_flux'] / 3600.0
            elif precip_mean > 0.1 or precip_max > 1.0:
                # Heuristic: if values are large, likely mm/day not mm/s
                self.logger.warning(
                    f"Precipitation values appear too large for mm/s (mean={precip_mean:.3f}, max={precip_max:.3f}). "
                    f"Assuming mm/day and converting to kg m⁻² s⁻¹ (÷86400)"
                )
                forcing_data['precipitation_flux'] = forcing_data['precipitation_flux'] / 86400.0
            elif 'kg' in precip_units and 's' in precip_units:
                self.logger.debug("Precipitation already in kg m⁻² s⁻¹")
            elif 'mm' in precip_units and 's' in precip_units:
                self.logger.debug("Precipitation already in mm/s (equivalent to kg m⁻² s⁻¹)")
            else:
                self.logger.debug(f"Precipitation units unclear ({precip_units}), assuming already in kg m⁻² s⁻¹")

        # Air pressure: Convert hPa/kPa to Pa if needed
        if 'surface_air_pressure' in forcing_data:
            pres_units = forcing_data['surface_air_pressure'].attrs.get('units', '').lower()
            pres_mean = float(forcing_data['surface_air_pressure'].mean())

            if 'hpa' in pres_units or (pres_mean > 100 and pres_mean < 2000):
                # Likely hPa (typical range 950-1050 hPa)
                self.logger.info("Converting pressure from hPa to Pa (×100)")
                forcing_data['surface_air_pressure'] = forcing_data['surface_air_pressure'] * 100.0
            elif 'kpa' in pres_units or (pres_mean > 10 and pres_mean < 200):
                # Likely kPa (typical range 95-105 kPa)
                self.logger.info("Converting pressure from kPa to Pa (×1000)")
                forcing_data['surface_air_pressure'] = forcing_data['surface_air_pressure'] * 1000.0
            elif pres_mean > 50000 and pres_mean < 110000:
                self.logger.debug(f"Pressure appears to be in Pa (mean={pres_mean:.0f})")
            else:
                self.logger.warning(
                    f"Pressure units unclear (units={pres_units}, mean={pres_mean:.0f}). "
                    f"Assuming Pa if > 10000, otherwise converting from hPa"
                )
                if pres_mean < 10000:
                    forcing_data['surface_air_pressure'] = forcing_data['surface_air_pressure'] * 100.0

        # Specific humidity: should be kg/kg (0-0.1 range), sometimes given as g/kg
        if 'specific_humidity' in forcing_data:
            hum_units = forcing_data['specific_humidity'].attrs.get('units', '').lower().strip()
            hum_max = float(forcing_data['specific_humidity'].max())

            # Check if units explicitly indicate g/kg (but NOT kg/kg or kg kg-1)
            is_g_per_kg = (
                hum_units in ('g/kg', 'g kg-1', 'g/kg-1')
                or (hum_units.startswith('g') and 'kg' in hum_units and not hum_units.startswith('kg'))
            )

            if is_g_per_kg or hum_max > 1.0:
                self.logger.info("Converting specific humidity from g/kg to kg/kg (÷1000)")
                forcing_data['specific_humidity'] = forcing_data['specific_humidity'] / 1000.0
            else:
                self.logger.debug(f"Specific humidity appears to be in kg/kg (units='{hum_units}', max={hum_max:.6f})")

        # Radiation (SWRadAtm, LWRadAtm) should be in W/m² - typically already correct
        for rad_var in ['surface_downwelling_shortwave_flux', 'surface_downwelling_longwave_flux']:
            if rad_var in forcing_data:
                self.logger.debug(f"{rad_var} assuming W/m² (standard unit)")

        return forcing_data

    def prepare_forcing_data(self) -> Path:
        """
        Convert forcing data to NGEN format (NetCDF and CSV).

        Loads basin-averaged forcing data and transforms it to NGEN's expected
        format with AORC/GRIB standard variable names. Creates both NetCDF
        (for ngen-cal) and CSV (for individual catchment forcing) outputs.

        Processing steps:
        1. Load forcing from basin-averaged NetCDF files
        2. Resample to hourly if needed (NGEN requires hourly)
        3. Extend time window to provide lookahead buffer for NGEN
        4. Map variable names to AORC standards (TMP_2maboveground, etc.)
        5. Write NetCDF with catchment-id dimension
        6. Write per-catchment CSV files for CFE/PET modules

        Returns:
            Path: Path to created forcing.nc file.

        Note:
            NGEN requires forcing data beyond the configured end_time for
            internal interpolation. This method adds a 4-timestep buffer.
        """
        catchment_gdf = gpd.read_file(self.get_catchment_path())
        catchment_ids = [f"cat-{x}" for x in catchment_gdf[self.hru_id_col].astype(str).tolist()]
        fdp = ForcingDataProcessor(self.config, self.logger)
        forcing_data = fdp.load_forcing_data(self.forcing_basin_path)
        forcing_data = forcing_data.sortby('time')

        # Normalize legacy SUMMA-style names to CFIF standard at the boundary
        from symfluence.data.preprocessing.cfif.variables import normalize_to_cfif
        forcing_data = normalize_to_cfif(forcing_data)

        # Convert units to NGEN/AORC standards (K, Pa, kg/m²/s, W/m²)
        forcing_data = self._convert_forcing_units_to_ngen(forcing_data)

        twm = TimeWindowManager(self.config, self.logger)
        try:
            start_time, end_time = twm.get_simulation_times(forcing_path=self.forcing_basin_path)
        except (ValueError, KeyError, TypeError) as e:
            self.logger.debug(f"Could not get simulation times from TimeWindowManager, using config: {e}")
            start_time = pd.to_datetime(self.time_start)
            end_time = pd.to_datetime(self.time_end)

        time_values = pd.to_datetime(forcing_data.time.values)
        inferred_step_seconds = None
        if len(time_values) > 1:
            time_deltas = np.diff(time_values).astype('timedelta64[s]').astype(int)
            inferred_step_seconds = int(np.median(time_deltas))

        if inferred_step_seconds and inferred_step_seconds != 3600:
            self.logger.info(
                f"Resampling forcing data from {inferred_step_seconds} seconds to hourly for NGEN"
            )
            # Use mass-conserving resampling: sum for fluxes, mean for state variables
            # Convert precipitation rate to depth for resampling, then back to rate
            if 'precipitation_flux' in forcing_data:
                # Convert mm/s to mm for the source timestep
                forcing_data['pptrate_depth'] = forcing_data['precipitation_flux'] * inferred_step_seconds

            # Define resampling strategy for each variable
            resample_dict = {}
            for var in forcing_data.data_vars:
                if var in ['precipitation_flux', 'pptrate_depth']:
                    continue  # Handle precipitation separately
                elif var in ['surface_downwelling_shortwave_flux', 'surface_downwelling_longwave_flux']:
                    # Radiation: use mean (could also use interpolation, but mean is safer)
                    resample_dict[var] = 'mean'
                else:
                    # State variables (temp, pressure, humidity): use mean
                    resample_dict[var] = 'mean'

            # Resample
            forcing_data_resampled = forcing_data.resample(time='1h').mean()

            # Handle precipitation separately with mass conservation
            if 'pptrate_depth' in forcing_data:
                # Sum the precipitation depth over the resampled period
                precip_depth_hourly = forcing_data['pptrate_depth'].resample(time='1h').sum()
                # Convert back to rate (mm/h → mm/s)
                forcing_data_resampled['precipitation_flux'] = precip_depth_hourly / 3600.0
                # Remove temporary depth variable
                if 'pptrate_depth' in forcing_data_resampled:
                    forcing_data_resampled = forcing_data_resampled.drop_vars('pptrate_depth')

            forcing_data = forcing_data_resampled
            self._forcing_time_step_size_override = 3600
            self.logger.info("Used mass-conserving resampling: sum for precipitation, mean for other variables")

        # NGEN requires forcing data beyond the configured end_time to complete the simulation
        # Extend end_time by 4 forcing timesteps to provide necessary lookahead data
        # (empirically determined - NGEN can access up to 3 timesteps beyond configured end_time)
        if inferred_step_seconds and inferred_step_seconds != 3600:
            forcing_timestep_seconds = 3600
        else:
            forcing_timestep_seconds = getattr(self, '_forcing_time_step_size_override', None) or self.forcing_time_step_size
        buffer_timesteps = 4  # Add extra buffer to be safe
        extended_end_time = end_time + pd.Timedelta(seconds=forcing_timestep_seconds * buffer_timesteps)
        self.logger.info(f"Extending forcing data from {end_time} to {extended_end_time} (NGEN requires {buffer_timesteps} timestep buffer)")

        forcing_data = fdp.subset_to_time_window(forcing_data, start_time, extended_end_time)

        # Pad forcing data if source doesn't extend to full buffer
        actual_end = pd.to_datetime(forcing_data.time.values[-1])
        if actual_end < extended_end_time:
            # Check if padding will overlap the actual simulation period (not just buffer)
            if actual_end < end_time:
                raise ValueError(
                    f"Forcing data ends at {actual_end} but simulation requires data until {end_time}. "
                    f"Padding within the simulation period would produce invalid results. "
                    f"Please provide forcing data that extends to at least {end_time}."
                )

            # Padding is only in the lookahead buffer zone (after simulation end)
            self.logger.warning(
                f"Source forcing ends at {actual_end}, padding buffer zone to {extended_end_time}. "
                f"Repeating last timestep for {buffer_timesteps} timestep NGEN lookahead buffer. "
                f"This padding is outside the simulation period ({start_time} to {end_time}) and should not affect results."
            )

            # Create additional timesteps by repeating last timestep values
            last_slice = forcing_data.isel(time=-1)
            padding_times = pd.date_range(
                start=actual_end + pd.Timedelta(seconds=forcing_timestep_seconds),
                end=extended_end_time,
                freq=f'{forcing_timestep_seconds}s'
            )
            padding_data = []
            for t in padding_times:
                padded = last_slice.copy()
                padded['time'] = t
                padding_data.append(padded)
            if padding_data:
                padding_ds = xr.concat(padding_data, dim='time')
                forcing_data = xr.concat([forcing_data, padding_ds], dim='time')
                self.logger.info(f"Added {len(padding_times)} padding timesteps in lookahead buffer")

        ngen_ds = self._create_ngen_forcing_dataset(forcing_data, catchment_ids)
        output_file = self.forcing_dir / "forcing.nc"
        # Ensure parent directory exists before saving
        output_file.parent.mkdir(parents=True, exist_ok=True)
        ngen_ds.to_netcdf(output_file, format='NETCDF4')
        self._write_csv_forcing_files(forcing_data, catchment_ids)
        return output_file

    def _decompose_wind_speed(self, forcing_data: xr.Dataset) -> xr.Dataset:
        """
        Decompose scalar wind speed into U and V components when needed.

        If forcing data contains 'wind_speed' (scalar wind speed) but lacks U and V
        components, this method creates them by assuming westerly wind direction.

        Args:
            forcing_data: Forcing dataset possibly containing 'wind_speed'

        Returns:
            Dataset with 'eastward_wind' and 'northward_wind' added if needed
        """
        has_u = 'eastward_wind' in forcing_data
        has_v = 'northward_wind' in forcing_data
        has_scalar = 'wind_speed' in forcing_data

        # If we already have both components, nothing to do
        if has_u and has_v:
            return forcing_data

        # If we have neither components nor scalar, can't help
        if not has_scalar:
            return forcing_data

        # Decompose scalar wind to U/V components
        # Assume westerly wind (from west, blowing east): U = windspd, V = 0
        # This is a common assumption when direction is unknown
        self.logger.warning(
            "Converting scalar wind speed to U/V components. "
            "Assuming westerly wind (U=wind_speed, V=0) since wind direction is not available. "
            "This approximation may affect PET and energy balance calculations."
        )

        if not has_u:
            forcing_data['eastward_wind'] = forcing_data['wind_speed'].copy()
            forcing_data['eastward_wind'].attrs['long_name'] = 'U-component of wind (assumed from scalar wind speed)'
            forcing_data['eastward_wind'].attrs['units'] = 'm/s'

        if not has_v:
            # Create V-component as zeros (westerly wind assumption)
            forcing_data['northward_wind'] = xr.zeros_like(forcing_data['wind_speed'])
            forcing_data['northward_wind'].attrs['long_name'] = 'V-component of wind (assumed zero for westerly wind)'
            forcing_data['northward_wind'].attrs['units'] = 'm/s'

        return forcing_data

    def _write_csv_forcing_files(self, forcing_data: xr.Dataset, catchment_ids: List[str]) -> Path:
        # Decompose scalar wind speed to U/V components if needed
        forcing_data = self._decompose_wind_speed(forcing_data)

        # Ensure forcing_dir exists before creating csv subdirectory
        from pathlib import Path
        self.forcing_dir: Path = Path(self.forcing_dir)  # Ensure it's a Path object

        # Ensure all parent directories exist with explicit mkdir calls
        current_path = self.forcing_dir
        while not current_path.exists() and current_path.parent != current_path:
            current_path = current_path.parent

        # Now create all needed directories from the top down
        for parent in reversed(list(self.forcing_dir.parents)):
            if not parent.exists():
                try:
                    parent.mkdir(exist_ok=True)
                except OSError:
                    pass  # Parent may have been created by another process

        self.forcing_dir.mkdir(parents=True, exist_ok=True)

        csv_dir = self.forcing_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)

        # Verify directory exists before proceeding
        if not csv_dir.exists():
            raise OSError(f"Failed to create directory: {csv_dir}")

        self.logger.info(f"CSV directory created: {csv_dir}, exists={csv_dir.exists()}, is_dir={csv_dir.is_dir()}")
        time_values = pd.to_datetime(forcing_data.time.values)

        # Variable mapping from ERA5/internal names to BMI standard names for NGEN
        # Use BMI standard naming that NGEN modules expect
        var_mapping = {
            'precipitation_flux': 'atmosphere_water__liquid_equivalent_precipitation_rate',
            'air_temperature': 'land_surface_air__temperature',
            'specific_humidity': 'atmosphere_air_water~vapor__specific_humidity',
            'surface_air_pressure': 'land_surface_air__pressure',
            'surface_downwelling_shortwave_flux': 'land_surface_radiation~incoming~shortwave__energy_flux',
            'surface_downwelling_longwave_flux': 'land_surface_radiation~incoming~longwave__energy_flux',
            'eastward_wind': 'land_surface_wind__x_component_of_velocity',
            'northward_wind': 'land_surface_wind__y_component_of_velocity',
        }

        for idx, cat_id in enumerate(catchment_ids):
            cols = {'time': time_values}
            for e_v, n_v in var_mapping.items():
                if e_v in forcing_data:
                    arr = forcing_data[e_v].values
                    cols[n_v] = arr[:, idx] if arr.ndim > 1 else arr

            df = pd.DataFrame(cols)

            # Precipitation unit handling depends on whether NOAH is enabled:
            # - NOAH-OWP expects mm/s (kg m-2 s-1), outputs QINSUR in mm/s to CFE
            # - CFE standalone expects mm/h for direct precipitation input
            # When NOAH is enabled, keep precip in mm/s; otherwise convert to mm/h for CFE
            precip_col = 'atmosphere_water__liquid_equivalent_precipitation_rate'
            if precip_col in df:
                if not self._include_noah:
                    # CFE standalone: convert mm/s → mm/h
                    df[precip_col] = df[precip_col] * 3600.0
                    self.logger.debug("Converted precipitation to mm/h for CFE standalone mode")
                else:
                    # NOAH enabled: keep in mm/s (NOAH's expected unit)
                    self.logger.debug("Keeping precipitation in mm/s for NOAH-OWP")

            # Note: Do NOT add AORC-style aliases (APCP_surface, precip_rate).
            # These map to the same CSDMS canonical name via WellKnownFields in ngen,
            # causing doubled precipitation vectors and ngen crashes.

            # Wind components are required for NGEN PET and NOAH modules
            # After decomposition, these should exist, but check just in case
            missing_wind_vars = []
            if 'land_surface_wind__x_component_of_velocity' not in df.columns:
                missing_wind_vars.append('eastward_wind (U-component of wind)')
            if 'land_surface_wind__y_component_of_velocity' not in df.columns:
                missing_wind_vars.append('northward_wind (V-component of wind)')

            if missing_wind_vars:
                raise ValueError(
                    f"Missing required wind components for NGEN: {', '.join(missing_wind_vars)}\n"
                    f"Wind data is required for PET and NOAH-OWP energy balance calculations.\n"
                    f"Available forcing variables: {list(forcing_data.data_vars)}\n"
                    f"Please ensure your forcing data includes wind components (U and V) or scalar windspd.\n"
                    f"Note: Scalar windspd is automatically decomposed to U/V assuming westerly wind."
                )

            df.to_csv(csv_dir / f"{cat_id}_forcing.csv", index=False)
        return csv_dir

    def _create_ngen_forcing_dataset(self, forcing_data: xr.Dataset, catchment_ids: List[str]) -> xr.Dataset:
        n_cats = len(catchment_ids)
        time_values = pd.to_datetime(forcing_data.time.values)
        time_ns = ((time_values.values - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 'ns')).astype(np.int64)
        ngen_ds = xr.Dataset()
        ngen_ds['ids'] = xr.DataArray(np.array(catchment_ids, dtype=object), dims=['catchment-id'])
        ngen_ds['Time'] = xr.DataArray(np.tile(time_ns, (n_cats, 1)).astype(np.float64), dims=['catchment-id', 'time'], attrs={'units': 'ns'})
        var_mapping = {'precipitation_flux': 'precip_rate', 'air_temperature': 'TMP_2maboveground', 'specific_humidity': 'SPFH_2maboveground', 'surface_air_pressure': 'PRES_surface', 'surface_downwelling_shortwave_flux': 'DSWRF_surface', 'surface_downwelling_longwave_flux': 'DLWRF_surface'}
        for e_v, n_v in var_mapping.items():
            if e_v in forcing_data:
                data = forcing_data[e_v].values.T
                if data.shape[0] == 1 and n_cats > 1: data = np.tile(data, (n_cats, 1))
                ngen_ds[n_v] = xr.DataArray(data.astype(np.float32), dims=['catchment-id', 'time'])
        return ngen_ds

    def generate_model_configs(self):
        """
        Generate configuration files for all enabled NGEN modules.

        Creates BMI configuration files for CFE, PET, Noah-OWP, and SLOTH
        modules based on catchment geometry and enabled module flags.
        Configurations are written to the setup directory.
        """
        catchment_gdf = gpd.read_file(self.get_catchment_path())
        config_gen = NgenConfigGenerator(self.config_dict, self.logger, self.setup_dir, catchment_gdf.crs)
        noah_et_fallback = getattr(self, '_noah_et_fallback', 'EVAPOTRANS')
        config_gen.set_module_availability(
            cfe=self._include_cfe, pet=self._include_pet, noah=self._include_noah,
            sloth=self._include_sloth, noah_et_fallback=noah_et_fallback,
            topmodel=self._include_topmodel, sacsma=self._include_sacsma,
            snow17=self._include_snow17,
        )
        config_gen.generate_all_configs(catchment_gdf, self.hru_id_col)

    def generate_realization_config(self, catchment_file: Path, nexus_file: Path, forcing_file: Path):
        """
        Generate the NGEN realization configuration file.

        Creates the main realization.json that defines the complete model
        configuration including forcing paths, module linkages, and output
        specifications required by the NGEN executable.

        Args:
            catchment_file: Path to catchment GeoJSON file.
            nexus_file: Path to nexus GeoJSON file.
            forcing_file: Path to forcing NetCDF file.
        """
        config_gen = NgenConfigGenerator(self.config_dict, self.logger, self.setup_dir, getattr(self, 'catchment_crs', None))
        noah_et_fallback = getattr(self, '_noah_et_fallback', 'EVAPOTRANS')
        config_gen.set_module_availability(
            cfe=self._include_cfe, pet=self._include_pet, noah=self._include_noah,
            sloth=self._include_sloth, noah_et_fallback=noah_et_fallback,
            topmodel=self._include_topmodel, sacsma=self._include_sacsma,
            snow17=self._include_snow17,
        )
        config_gen.generate_realization_config(forcing_file, self.project_dir, lib_paths=self._ngen_lib_paths)

        # Generate t-route config if routing is enabled
        run_troute = self._get_config_value(
            lambda: self.config.model.ngen.run_troute,
            default=True
        )
        if run_troute:
            self._generate_troute_config(nexus_file)

    def _generate_troute_config(self, nexus_file: Path):
        """
        Generate t-route configuration for NGEN routing.

        Creates troute_config.yaml with network topology and forcing parameters
        suitable for routing NGEN nexus outputs. Handles both lumped (single
        catchment) and distributed (multi-catchment) domains.

        Args:
            nexus_file: Path to nexus GeoJSON file for determining network size.
        """
        self.logger.info("Generating t-route configuration for NGEN routing")

        # Load nexus to determine if this is a lumped or distributed domain
        with open(nexus_file, 'r', encoding='utf-8') as f:
            nexus_data = json.load(f)
        num_nexuses = len(nexus_data.get('features', []))

        # For lumped domains (single nexus), t-route is optional but we still create config
        if num_nexuses == 1:
            self.logger.info("Lumped domain detected (single nexus). T-route will pass through single outlet.")

        # Get time parameters
        start_time = self.time_start or '2000-01-01 00:00'
        end_time = self.time_end or '2000-12-31 23:00'
        time_step_seconds = getattr(self, '_forcing_time_step_size_override', None) or self.forcing_time_step_size

        # Calculate number of timesteps
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        total_seconds = (end_dt - start_dt).total_seconds() + time_step_seconds
        nts = int(total_seconds / time_step_seconds)

        # Determine output directory for NGEN (where nex-*.csv files will be)
        ngen_output_dir = self.project_dir / 'simulations' / self.experiment_id / 'NGEN'
        troute_output_dir = ngen_output_dir / 'troute_output'

        # Check if we have a river network for distributed routing
        river_network_file = self.get_river_network_path()
        has_network = river_network_file.exists()

        # Create topology file for distributed domains
        topology_file = None
        if has_network and num_nexuses > 1:
            topology_file = self._create_troute_topology()

        # Build t-route config
        troute_config = {
            'log_parameters': {
                'showtiming': True,
                'log_level': 'INFO'
            },
            'network_topology_parameters': {
                'supernetwork_parameters': {
                    'title_string': f'NGEN routing for {self.domain_name}',
                    'columns': {
                        'key': 'id',
                        'downstream': 'toid',
                        'mainstem': 'mainstem'
                    }
                }
            },
            'compute_parameters': {
                'restart_parameters': {
                    'start_datetime': str(start_dt)
                },
                'forcing_parameters': {
                    'nts': nts,
                    'max_loop_size': 24,
                    'dt': time_step_seconds,
                    'qts_subdivisions': 1,
                    'qlat_input_folder': str(ngen_output_dir),
                    'qlat_file_pattern_filter': 'nex-*',
                    'binary_nexus_file_folder': None,
                    'coastal_boundary_input_file': None
                },
                'data_assimilation_parameters': {
                    'usgs_timeslices_folder': None,
                    'usace_timeslices_folder': None,
                    'reservoir_da': None
                }
            },
            'output_parameters': {
                'stream_output': {
                    'stream_output_directory': str(troute_output_dir),
                    'stream_output_time': 1,
                    'stream_output_type': '.csv',
                    'stream_output_internal_frequency': time_step_seconds
                }
            }
        }

        # Add topology file if available
        if topology_file:
            troute_config['network_topology_parameters']['supernetwork_parameters']['geo_file_path'] = str(topology_file)  # type: ignore[index]
        else:
            # For lumped domains or missing network, use nexus file as simple network
            troute_config['network_topology_parameters']['supernetwork_parameters']['geo_file_path'] = str(nexus_file)  # type: ignore[index]

        # Write config file
        troute_config_file = self.setup_dir / 'troute_config.yaml'
        with open(troute_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(troute_config, f, default_flow_style=False, sort_keys=False, indent=2)

        self.logger.info(f"T-route config created: {troute_config_file}")
        return troute_config_file

    def _create_troute_topology(self) -> Optional[Path]:
        """
        Create t-route network topology file from river network shapefile.

        Creates a GeoJSON flowlines file with all channel geometry parameters
        required by T-Route HYFeaturesNetwork for Muskingum-Cunge routing.
        Channel geometry is estimated using hydraulic geometry relationships
        based on drainage area.

        Returns:
            Path to topology GeoJSON file, or None if creation failed.
        """
        try:
            import numpy as np
        except ImportError:
            self.logger.warning("numpy not available, skipping topology file creation")
            return None

        self.logger.info("Creating t-route network topology file with channel geometry")
        river_network_file = self.get_river_network_path()

        if not river_network_file.exists():
            self.logger.warning(f"River network not found: {river_network_file}")
            return None

        try:
            river_gdf = gpd.read_file(river_network_file)
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Failed to read river network: {e}")
            return None

        seg_id_col = self._get_config_value(
            lambda: self.config.paths.river_network_segid, default='LINKNO'
        )
        downstream_col = self._get_config_value(
            lambda: self.config.paths.river_network_downsegid, default='DSLINKNO'
        )
        length_col = self._get_config_value(
            lambda: self.config.paths.river_network_length, default='Length'
        )
        slope_col = self._get_config_value(
            lambda: self.config.paths.river_network_slope, default='Slope'
        )

        # Get drainage area column (try common names)
        drainage_area_col = None
        for col_name in ['DSContArea', 'TotDASqKm', 'drainage_area', 'DAREA', 'AreaSqKm']:
            if col_name in river_gdf.columns:
                drainage_area_col = col_name
                break

        n_segments = len(river_gdf)

        # Create a new GeoDataFrame with hydrofabric-compatible column names
        # HYFeaturesNetwork expects: id, toid, lengthkm, slope, n, etc.
        flowlines_gdf = gpd.GeoDataFrame(geometry=river_gdf.geometry.copy())

        # Segment IDs - use 'wb-XX' format to match hydrofabric convention
        flowlines_gdf['id'] = [f'wb-{int(seg_id)}' for seg_id in river_gdf[seg_id_col].values]

        # Downstream IDs - use 'wb-XX' format, handle terminal nodes
        downstream_ids = river_gdf[downstream_col].values
        flowlines_gdf['toid'] = [
            f'wb-{int(ds_id)}' if ds_id > 0 else ''
            for ds_id in downstream_ids
        ]

        # Length in km (HYFeaturesNetwork expects km)
        if length_col in river_gdf.columns:
            lengths_m = river_gdf[length_col].values
        else:
            lengths_m = river_gdf.geometry.length.values
        flowlines_gdf['lengthkm'] = lengths_m / 1000.0

        # Slope
        if slope_col in river_gdf.columns:
            slopes = river_gdf[slope_col].values
        else:
            slopes = np.full(n_segments, 0.001)
        slopes = np.maximum(slopes, 0.0001)  # Minimum for numerical stability
        flowlines_gdf['So'] = slopes

        # Manning's n for channel
        flowlines_gdf['n'] = 0.035

        # === Channel Geometry Estimation ===
        # Get drainage area in km² for hydraulic geometry calculations
        if drainage_area_col:
            drainage_areas_m2 = river_gdf[drainage_area_col].values
            if np.median(drainage_areas_m2) > 10000:
                drainage_areas_km2 = drainage_areas_m2 / 1e6
            else:
                drainage_areas_km2 = drainage_areas_m2
            self.logger.debug(f"Using drainage area from column '{drainage_area_col}'")
        else:
            self.logger.warning("No drainage area column found. Using default estimates.")
            drainage_areas_km2 = np.full(n_segments, 100.0)

        drainage_areas_km2 = np.maximum(drainage_areas_km2, 1.0)

        # Bankfull width estimation: W = a * A^b
        width_coef = self._get_config_value(lambda: None, default=3.0, dict_key='TROUTE_WIDTH_COEF')
        width_exp = self._get_config_value(lambda: None, default=0.5, dict_key='TROUTE_WIDTH_EXP')
        bankfull_widths = width_coef * np.power(drainage_areas_km2, width_exp)
        bankfull_widths = np.clip(bankfull_widths, 2.0, 500.0)

        # Bankfull depth estimation: D = c * A^d
        depth_coef = self._get_config_value(lambda: None, default=0.3, dict_key='TROUTE_DEPTH_COEF')
        depth_exp = self._get_config_value(lambda: None, default=0.4, dict_key='TROUTE_DEPTH_EXP')
        bankfull_depths = depth_coef * np.power(drainage_areas_km2, depth_exp)
        bankfull_depths = np.clip(bankfull_depths, 0.3, 20.0)

        # Channel geometry columns
        flowlines_gdf['BtmWdth'] = 0.5 * bankfull_widths  # Bottom width
        flowlines_gdf['TopWdth'] = bankfull_widths  # Top width
        flowlines_gdf['TopWdthCC'] = 3.0 * bankfull_widths  # Compound channel width
        flowlines_gdf['nCC'] = 0.08  # Compound channel Manning's n
        flowlines_gdf['ChSlp'] = 0.05  # Channel side slope (1:20)

        # Cross-section area (trapezoidal)
        cross_section_areas = 0.5 * (flowlines_gdf['BtmWdth'] + flowlines_gdf['TopWdth']) * bankfull_depths
        flowlines_gdf['Cs'] = cross_section_areas

        # Muskingum parameters
        hydraulic_radius = bankfull_depths * 0.7
        velocities = (1.0 / 0.035) * np.power(hydraulic_radius, 2.0/3.0) * np.sqrt(slopes)
        velocities = np.clip(velocities, 0.1, 5.0)
        muskingum_k = lengths_m / velocities / 3600.0  # Convert to hours
        muskingum_k = np.clip(muskingum_k, 0.5, 100.0)
        flowlines_gdf['MusK'] = muskingum_k
        flowlines_gdf['MusX'] = 0.2

        # Additional columns expected by HYFeaturesNetwork
        flowlines_gdf['alt'] = 0.0  # Altitude (not used)
        flowlines_gdf['order'] = river_gdf.get('strmOrder', 1)  # Stream order
        flowlines_gdf['mainstem'] = 0  # Mainstem flag

        # Save as GeoJSON for standalone use
        geojson_file = self.setup_dir / 'troute_flowlines.geojson'
        flowlines_gdf.to_file(geojson_file, driver='GeoJSON')
        self.logger.info("Created " + str(len(flowlines_gdf)) + " records")

        # Create hydrofabric GeoPackage with separate flowpaths and flowpath_attributes layers
        # T-Route HYFeaturesNetwork expects these as separate layers that get merged
        gpkg_file = self.setup_dir / f'{self.domain_name}_hydrofabric_troute.gpkg'
        try:
            # Read existing catchments
            catchments_file = self.setup_dir / f'{self.domain_name}_catchments.geojson'
            nexus_file = self.setup_dir / 'nexus.geojson'

            if catchments_file.exists():
                divides_gdf = gpd.read_file(catchments_file)

                # Create flowpaths layer with geometry and routing identifiers only
                flowpaths_gdf = gpd.GeoDataFrame({
                    'id': flowlines_gdf['id'],
                    'toid': flowlines_gdf['toid'],
                }, geometry=flowlines_gdf.geometry, crs=flowlines_gdf.crs)

                # Create flowpath_attributes layer with all channel parameters (no geometry)
                flowpath_attrs_df = flowlines_gdf.drop(columns='geometry').copy()

                # Write flowpaths layer (with geometry)
                flowpaths_gdf.to_file(gpkg_file, layer='flowpaths', driver='GPKG')

                # Write flowpath_attributes layer (without geometry - use pandas to_sql)
                import sqlite3
                conn = sqlite3.connect(gpkg_file)
                flowpath_attrs_df.to_sql('flowpath_attributes', conn, if_exists='replace', index=False)
                conn.close()

                # Add divides layer to GeoPackage
                divides_gdf.to_file(gpkg_file, layer='divides', driver='GPKG', mode='a')

                # Add nexus layer if exists
                if nexus_file.exists():
                    nexus_gdf = gpd.read_file(nexus_file)
                    nexus_gdf.to_file(gpkg_file, layer='nexus', driver='GPKG', mode='a')

                self.logger.info(f"T-route hydrofabric GeoPackage created: {gpkg_file}")
                self.logger.debug("  Layers: flowpaths, flowpath_attributes, divides" +
                                  (", nexus" if nexus_file.exists() else ""))
            else:
                self.logger.warning(f"Catchments file not found: {catchments_file}. Using GeoJSON only.")
                gpkg_file = None

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Failed to create hydrofabric GeoPackage: {e}. Using GeoJSON only.")
            import traceback
            self.logger.debug(traceback.format_exc())
            gpkg_file = None

        # Return GeoPackage if available, otherwise GeoJSON
        topology_file = gpkg_file if gpkg_file and gpkg_file.exists() else geojson_file
        self.logger.info(f"T-route flowlines file created with channel geometry: {topology_file}")
        self.logger.debug(f"  Segments: {n_segments}")
        self.logger.debug(f"  Drainage area range: {drainage_areas_km2.min():.1f} - {drainage_areas_km2.max():.1f} km²")
        self.logger.debug(f"  Width range: {bankfull_widths.min():.1f} - {bankfull_widths.max():.1f} m")
        self.logger.debug(f"  Depth range: {bankfull_depths.min():.2f} - {bankfull_depths.max():.2f} m")
        return topology_file

    def get_river_network_path(self) -> Path:
        """Get path to river network shapefile."""
        river_path = self._get_config_value(
            lambda: self.config.paths.river_network_path, default='default'
        )
        if river_path == 'default':
            river_path = self.project_dir / 'shapefiles' / 'river_network'
        else:
            river_path = Path(river_path)

        river_name = self._get_config_value(
            lambda: self.config.paths.river_network_name, default='default'
        )
        if river_name == 'default':
            domain_method = self.domain_definition_method
            river_name = f"{self.domain_name}_riverNetwork_{domain_method}.shp"

        return river_path / river_name
