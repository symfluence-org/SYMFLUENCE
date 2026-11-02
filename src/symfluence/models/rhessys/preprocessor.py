# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
RHESSys Model Preprocessor

Handles preparation of RHESSys model inputs including:
- Climate forcing files (daily precipitation, temperature)
- Worldfile generation (domain structure)
- TEC file generation (temporal event control)
- Flow table generation (routing)

Sub-modules:
- worldfile_generator: Worldfile and header generation
- flow_table_generator: Flow table routing generation
- definitions_generator: Default parameter file generation
- fire_generator: WMFire fire spread input generation
"""
from __future__ import annotations

import logging
from typing import Tuple

import geopandas as gpd

from symfluence.core.modeling.base.base_preprocessor import BaseModelPreProcessor
from symfluence.core.registries import R
from symfluence.models.rhessys.climate_generator import RHESSysClimateGenerator
from symfluence.models.rhessys.definitions_generator import RHESSysDefinitionsGenerator
from symfluence.models.rhessys.fire_generator import RHESSysFireGenerator
from symfluence.models.rhessys.flow_table_generator import RHESSysFlowTableGenerator
from symfluence.models.rhessys.worldfile_generator import RHESSysWorldfileGenerator

logger = logging.getLogger(__name__)


@R.preprocessors.add("RHESSys")
class RHESSysPreProcessor(BaseModelPreProcessor):  # type: ignore[misc]
    """
    Prepares inputs for a RHESSys model run.

    RHESSys requires:
    - Climate station files with daily forcing (P, Tmax, Tmin)
    - Worldfile describing domain hierarchy (world > basin > hillslope > zone > patch > stratum)
    - TEC file with simulation dates and output events
    - Optional: Flow table for hillslope routing, fire grids for WMFire

    Sub-module generation is delegated to lazy-initialized generator classes:
    - worldfile_generator: RHESSysWorldfileGenerator
    - flow_table_generator: RHESSysFlowTableGenerator
    - definitions_generator: RHESSysDefinitionsGenerator
    - fire_generator: RHESSysFireGenerator
    """


    MODEL_NAME = "RHESSys"
    def __init__(self, config, logger):
        """
        Initialize the RHESSys preprocessor.

        Sets up RHESSys-specific directory structure and checks for optional
        WMFire (wildfire spread) module configuration.

        Args:
            config: Configuration dictionary or SymfluenceConfig object containing
                RHESSys settings, domain paths, and simulation parameters.
            logger: Logger instance for status messages and debugging.

        Note:
            Creates input directories under {project_dir}/settings/RHESSys/
            (worldfiles, tecfiles, routing, defs, fire) and climate data under
            {project_dir}/data/forcing/RHESSys_input/clim/.
        """
        super().__init__(config, logger)
        # Check for WMFire support (handles both wmfire and legacy vmfire config names)
        self.wmfire_enabled = self._check_wmfire_enabled()

        # Setup RHESSys-specific directories using base class paths
        # self.setup_dir = project_dir / "settings" / "RHESSys" (inherited)
        # self.forcing_dir = project_forcing_dir / "RHESSys_input" (inherited)
        self.worldfiles_dir = self.setup_dir / "worldfiles"
        self.tecfiles_dir = self.setup_dir / "tecfiles"
        self.climate_dir = self.forcing_dir / "clim"
        self.routing_dir = self.setup_dir / "routing"
        self.defs_dir = self.setup_dir / "defs"
        self.fire_dir = self.setup_dir / "fire"

        # Note: experiment_id and forcing_dataset are inherited as properties from ShapefileAccessMixin

        # Initialize worldfile initial conditions with configurable values
        # These can be overridden via config: RHESSYS_INIT_SAT_DEFICIT, RHESSYS_INIT_GW_STORAGE, etc.
        self._init_worldfile_conditions()

        # Lazy-init sub-module generators
        self._worldfile_gen = None
        self._flow_table_gen = None
        self._definitions_gen = None
        self._fire_gen = None

    # ------------------------------------------------------------------ #
    # Lazy-init properties for sub-module generators
    # ------------------------------------------------------------------ #

    @property
    def worldfile_generator(self) -> RHESSysWorldfileGenerator:
        """Lazy-initialized worldfile generator."""
        if self._worldfile_gen is None:
            self._worldfile_gen = RHESSysWorldfileGenerator(self)
        return self._worldfile_gen

    @property
    def flow_table_generator(self) -> RHESSysFlowTableGenerator:
        """Lazy-initialized flow table generator."""
        if self._flow_table_gen is None:
            self._flow_table_gen = RHESSysFlowTableGenerator(self)
        return self._flow_table_gen

    @property
    def definitions_generator(self) -> RHESSysDefinitionsGenerator:
        """Lazy-initialized definitions generator."""
        if self._definitions_gen is None:
            self._definitions_gen = RHESSysDefinitionsGenerator(self)
        return self._definitions_gen

    @property
    def fire_generator(self) -> RHESSysFireGenerator:
        """Lazy-initialized fire generator."""
        if self._fire_gen is None:
            self._fire_gen = RHESSysFireGenerator(self)
        return self._fire_gen

    # ------------------------------------------------------------------ #
    # Configuration and initialization
    # ------------------------------------------------------------------ #

    def _check_wmfire_enabled(self) -> bool:
        """Check if WMFire fire spread is enabled (supports both new and legacy config names)."""
        try:
            # Try new naming first
            if hasattr(self.config.model.rhessys, 'use_wmfire'):
                return self.config.model.rhessys.use_wmfire
            # Fall back to legacy vmfire naming
            if hasattr(self.config.model.rhessys, 'use_vmfire'):
                return self.config.model.rhessys.use_vmfire
        except AttributeError:
            pass
        return False

    def _init_worldfile_conditions(self) -> None:
        """
        Initialize worldfile state variables with configurable defaults.

        These values control the initial hydrological state at model start.
        For mountain catchments with snowmelt-dominated hydrology, starting
        near saturation (low sat_deficit) reduces spinup time significantly.

        Config options (all in meters):
            RHESSYS_INIT_SAT_DEFICIT: Initial saturation deficit [m]. Default 0.03.
                Lower = wetter soil. Range: 0.0 - 3.0
            RHESSYS_INIT_GW_STORAGE: Initial groundwater storage [m]. Default 0.1.
                Higher = more baseflow at start. Range: 0.0 - 2.0
            RHESSYS_INIT_RZ_STORAGE: Initial root zone storage [m]. Default 0.1.
            RHESSYS_INIT_UNSAT_STORAGE: Initial unsaturated zone storage [m]. Default 0.05.
        """
        # Get config values with sensible defaults for mountain catchments.
        # Start near saturation (sat_deficit=0.03m) so the model is in
        # a realistic hydrological state from the beginning. Starting dry
        # (e.g. sat_deficit=2.0) wastes years of spinup and degrades
        # calibration performance.
        try:
            self.init_sat_deficit = float(getattr(
                self.config.model.rhessys, 'init_sat_deficit', 0.03
            ))
        except (AttributeError, TypeError):
            self.init_sat_deficit = 0.03

        try:
            self.init_gw_storage = float(getattr(
                self.config.model.rhessys, 'init_gw_storage', 0.1
            ))
        except (AttributeError, TypeError):
            self.init_gw_storage = 0.1

        try:
            self.init_rz_storage = float(getattr(
                self.config.model.rhessys, 'init_rz_storage', 0.1
            ))
        except (AttributeError, TypeError):
            self.init_rz_storage = 0.1

        try:
            self.init_unsat_storage = float(getattr(
                self.config.model.rhessys, 'init_unsat_storage', 0.05
            ))
        except (AttributeError, TypeError):
            self.init_unsat_storage = 0.05

        logger.info(
            f"RHESSys worldfile initial conditions: "
            f"sat_deficit={self.init_sat_deficit}m, gw_storage={self.init_gw_storage}m, "
            f"rz_storage={self.init_rz_storage}m, unsat_storage={self.init_unsat_storage}m"
        )

    # ------------------------------------------------------------------ #
    # Main orchestration
    # ------------------------------------------------------------------ #

    def run_preprocessing(self) -> bool:
        """
        Run the complete RHESSys preprocessing workflow.

        Returns:
            bool: True if preprocessing succeeded, False otherwise
        """
        try:
            logger.info("Starting RHESSys preprocessing...")

            # Create directory structure
            self._create_directory_structure()

            # Generate climate files from forcing data
            self._generate_climate_files()

            # Generate worldfile (delegated to sub-module)
            self.worldfile_generator.generate_worldfile()

            # Generate TEC file
            self._generate_tec_file()

            # Generate flow table (delegated to sub-module)
            self.flow_table_generator.generate_flow_table()

            # Generate default files (delegated to sub-module)
            self.definitions_generator.generate_default_files()

            # Setup WMFire inputs if enabled (delegated to sub-module)
            if self.wmfire_enabled:
                self.fire_generator.setup_wmfire_inputs()

            logger.info("RHESSys preprocessing complete.")
            return True
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"RHESSys preprocessing failed: {e}", exc_info=True)
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _create_directory_structure(self):
        """Create RHESSys input directory structure."""
        self.create_directories(additional_dirs=[
            self.worldfiles_dir,
            self.tecfiles_dir,
            self.climate_dir,
            self.routing_dir,
            self.defs_dir,
            self.fire_dir,
        ])
        logger.info(f"Created RHESSys directories: settings at {self.setup_dir}, forcing at {self.forcing_dir}")

    # ------------------------------------------------------------------ #
    # Helper methods (used by sub-modules via self.pp)
    # ------------------------------------------------------------------ #

    def _get_utm_crs_from_bounds(self, gdf: gpd.GeoDataFrame) -> str:
        """Derive a UTM CRS from layer bounds (avoids centroid on geographic CRS)."""
        gdf_ll = gdf.to_crs("EPSG:4326") if gdf.crs is not None else gdf
        minx, miny, maxx, maxy = gdf_ll.total_bounds
        lon0 = (minx + maxx) / 2
        lat0 = (miny + maxy) / 2
        utm_zone = int((lon0 + 180) / 6) + 1
        hemisphere = 'north' if lat0 >= 0 else 'south'
        return f"EPSG:{32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone}"

    def _get_centroid_lon_lat(self, gdf: gpd.GeoDataFrame, utm_crs: str) -> Tuple[float, float]:
        """Compute centroid in projected CRS, then transform back to lon/lat."""
        gdf_ll = gdf.to_crs("EPSG:4326") if gdf.crs is not None else gdf
        gdf_proj = gdf_ll.to_crs(utm_crs)
        centroid_proj = gdf_proj.geometry.centroid.iloc[0]
        centroid_ll = gpd.GeoSeries([centroid_proj], crs=utm_crs).to_crs("EPSG:4326").iloc[0]
        return float(centroid_ll.x), float(centroid_ll.y)

    # ------------------------------------------------------------------ #
    # Climate file generation (delegates to RHESSysClimateGenerator)
    # ------------------------------------------------------------------ #

    def _generate_climate_files(self):
        """
        Generate RHESSys-compatible climate input files from forcing data.

        Delegates to RHESSysClimateGenerator for the actual file generation.
        """
        start_date, end_date = self._get_simulation_dates()

        generator = RHESSysClimateGenerator(
            config=self.config_dict,
            project_dir=self.project_dir,
            domain_name=self.domain_name,
            logger=self.logger
        )

        catchment_path = self.get_catchment_path() if hasattr(self, 'get_catchment_path') else None
        generator.generate_climate_files(start_date, end_date, catchment_path)

    # ------------------------------------------------------------------ #
    # TEC file generation (kept on orchestrator)
    # ------------------------------------------------------------------ #

    def _generate_tec_file(self):
        """
        Generate the RHESSys Temporal Event Control (TEC) file.

        The TEC file specifies simulation events and output timing.
        When WMFire is enabled, adds fire ignition event.
        """
        logger.info("Generating TEC file...")

        tec_file = self.tecfiles_dir / f"{self.domain_name}.tec"
        start_date, end_date = self._get_simulation_dates()

        # Build TEC content
        # Note: print_daily_on at hour 1, print_daily_growth_on at hour 2
        # Don't use print_daily_off - let output continue until simulation ends
        lines = [
            f"{start_date.year} {start_date.month} {start_date.day} 1 print_daily_on",
            f"{start_date.year} {start_date.month} {start_date.day} 2 print_daily_growth_on",
        ]

        # Add fire ignition event if WMFire is enabled
        if self.wmfire_enabled:
            fire_date = self._get_fire_ignition_date(start_date, end_date)
            if fire_date:
                # fire_grid_on triggers WMFire fire spread at the specified date
                lines.append(f"{fire_date.year} {fire_date.month} {fire_date.day} 1 fire_grid_on")
                logger.info(f"Added fire ignition event for {fire_date.strftime('%Y-%m-%d')}")

        content = "\n".join(lines) + "\n"
        tec_file.write_text(content, encoding='utf-8')
        logger.info(f"TEC file written: {tec_file}")

    def _get_fire_ignition_date(self, start_date, end_date):
        """
        Get the fire ignition date from config or calculate default.

        Returns:
            datetime: The ignition date, or None if fire should not be triggered.
        """
        from datetime import datetime, timedelta

        # Try to get ignition date from WMFire config
        try:
            wmfire_config = self.config.model.rhessys.wmfire
            if wmfire_config and hasattr(wmfire_config, 'ignition_date') and wmfire_config.ignition_date:
                # Parse date string (format: YYYY-MM-DD)
                ignition_date = datetime.strptime(wmfire_config.ignition_date, '%Y-%m-%d')
                logger.info(f"Using configured ignition date: {ignition_date.strftime('%Y-%m-%d')}")
                return ignition_date
        except (AttributeError, ValueError) as e:
            logger.debug(f"Could not get ignition date from config: {e}")

        # Default: use spinup end date + 1 day, or 30 days after start if no spinup
        try:
            spinup_period = self.config.experiment.spinup_period
            if spinup_period:
                # spinup_period is typically (start, end) tuple or string
                if isinstance(spinup_period, (list, tuple)) and len(spinup_period) >= 2:
                    spinup_end_str = spinup_period[1]
                    if isinstance(spinup_end_str, str):
                        spinup_end = datetime.strptime(spinup_end_str.strip(), '%Y-%m-%d')
                        ignition_date = spinup_end + timedelta(days=1)
                        logger.info(f"Using post-spinup ignition date: {ignition_date.strftime('%Y-%m-%d')}")
                        return ignition_date
        except (AttributeError, ValueError) as e:
            logger.debug(f"Could not parse spinup period: {e}")

        # Fallback: 30 days after simulation start
        ignition_date = start_date + timedelta(days=30)
        if ignition_date > end_date:
            ignition_date = start_date + timedelta(days=7)  # Use 7 days if simulation is short
        logger.info(f"Using default ignition date: {ignition_date.strftime('%Y-%m-%d')}")
        return ignition_date

    # ------------------------------------------------------------------ #
    # Alternative entry point
    # ------------------------------------------------------------------ #

    def preprocess(self, **kwargs):
        """
        Alternative entry point for preprocessing.
        """
        return self.run_preprocessing()
