# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GR model runner.

Handles model execution, state management, and output processing.
Supports both lumped and distributed spatial modes with optional mizuRoute routing.

Refactored to use the Unified Model Execution Framework.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import xarray as xr

from symfluence.core.constants import UnitConversion
from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler
from symfluence.core.modeling.base import BaseModelRunner
from symfluence.core.modeling.execution import SpatialOrchestrator
from symfluence.core.modeling.mixins import OutputConverterMixin, SpatialModeDetectionMixin
from symfluence.core.modeling.netcdf_utils import create_netcdf_encoding
from symfluence.core.modeling.spatial_modes import SpatialMode
from symfluence.core.registries import R

from ..mizuroute.mixins import MizuRouteConfigMixin
from .r_environment import (
    configure_r_dll_search,
    describe_rpy2_import_failure,
    ensure_airgr_available,
    r_path,
    rpy2_installed,
    run_r_script,
)

# Optional R/rpy2 support - only needed for GR models
# Broad exception handling is intentional here: rpy2 can raise RuntimeError, RRuntimeError,
# ImportError, or other exceptions when R is installed but broken (missing core packages,
# incompatible versions, etc.). We must catch all to provide graceful fallback.
# rpy2 prints noisy messages to stderr during R initialization (e.g. "Error importing in
# API mode", "Trying to import in ABI mode") — redirect stderr to suppress them.
# configure_r_dll_search() must run first: importing rpy2 starts the embedded
# interpreter, and on Windows that interpreter can only dyn.load() compiled R
# packages if R's own bin directory is already on PATH.
#
# The real exception is stashed in _RPY2_IMPORT_ERROR (via sys.exc_info so the
# allowlisted `except Exception:` line stays byte-for-byte identical). When rpy2
# is installed but this import fails — as it does when the embedded R cannot
# start on a given platform — __init__ surfaces that captured cause instead of
# the misleading "rpy2 is not installed" message.
_RPY2_IMPORT_ERROR: Optional[BaseException] = None
configure_r_dll_search()
try:
    import contextlib
    import io
    with contextlib.redirect_stderr(io.StringIO()):
        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr
    HAS_RPY2 = True
except Exception:  # noqa: BLE001 - Broad exception required for rpy2 import failures
    import sys as _sys
    _RPY2_IMPORT_ERROR = _sys.exc_info()[1]
    HAS_RPY2 = False
    robjects = None
    importr = None
    pandas2ri = None
    localconverter = None


@R.runners.add('GR', runner_method='run_gr')
class GRRunner(BaseModelRunner, SpatialOrchestrator, OutputConverterMixin, MizuRouteConfigMixin, SpatialModeDetectionMixin):  # type: ignore[misc]
    """
    Runner class for the GR family of models (initially GR4J).
    Handles model execution, state management, and output processing.
    Supports both lumped and distributed spatial modes.

    Uses the Unified Model Execution Framework for:

    - Subprocess execution (via ModelExecutor)
    - Spatial mode handling and routing (via SpatialOrchestrator)
    - Output format conversion (via OutputConverterMixin)

    Attributes:
        config (Dict[str, Any]): Configuration settings for GR models
        logger (Any): Logger object for recording run information
        project_dir (Path): Directory for the current project
        domain_name (str): Name of the domain being processed
    """

    MODEL_NAME = "GR"

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, reporting_manager: Optional[Any] = None, settings_dir: Optional[Path] = None):
        """
        Initialize the GR model runner.

        Sets up the GR (airGR/GR4J) execution environment including spatial mode
        detection, routing requirements check, and R/rpy2 validation.

        Args:
            config: Configuration dictionary or SymfluenceConfig object containing
                GR model settings, calibration parameters, and paths.
            logger: Logger instance for status messages and debugging output.
            reporting_manager: Optional reporting manager for experiment tracking
                and visualization.
            settings_dir: Optional override for GR settings directory path.

        Raises:
            ImportError: If R or rpy2 is not installed (required for GR models).
            ModelExecutionError: If the embedded R cannot load the airGR package.

        Note:
            GR models require R and the airGR package. airGR is verified up
            front rather than installed on demand: installing from CRAN
            mid-workflow is neither reproducible nor available offline.
        """
        # GR-specific: Check rpy2 dependency BEFORE calling super().
        if not HAS_RPY2:
            # rpy2 installed but its import failed => the embedded R could not
            # start. Surface the real, platform-specific cause (which the eager
            # module-level import swallowed) rather than telling the user to
            # install a package they already have.
            if _RPY2_IMPORT_ERROR is not None and rpy2_installed():
                raise ModelExecutionError(
                    describe_rpy2_import_failure(_RPY2_IMPORT_ERROR)
                ) from _RPY2_IMPORT_ERROR
            raise ImportError(
                "GR models require R and rpy2, but rpy2 could not be imported. "
                "The bootstrap installer attempts rpy2 by default; if it failed "
                "(e.g. R not on the system), other models continue to work but "
                "GR is unavailable until you install R and rpy2 manually: "
                "`pip install rpy2` (or `pip install -e \".[r]\"`). "
                "See https://rpy2.github.io/doc/latest/html/overview.html#installation"
            )

        # Fail fast on a broken R rather than several steps into the run.
        ensure_airgr_available(robjects)

        # Call base class
        super().__init__(config, logger, reporting_manager=reporting_manager)

        self.settings_dir = Path(settings_dir) if settings_dir else None

        # Instance variables for external parameters during calibration
        # These bypass the read-only config_dict property
        self._external_params: Optional[Dict[str, float]] = None
        self._skip_calibration: bool = False

        # Keep legacy attribute name for downstream GR code.
        self.output_path = self.output_dir

        # GR-specific configuration - determine spatial mode using mixin
        self.spatial_mode = self.detect_spatial_mode('GR')

        self.needs_routing = self._check_routing_requirements()

    def _setup_model_specific_paths(self) -> None:
        """Set up GR-specific paths."""
        # Catchment paths (use backward-compatible path resolution)
        catchment_file = self._find_basin_shapefile(required=True)
        self.catchment_path = catchment_file.parent
        self.catchment_name = catchment_file.name

        # GR setup and forcing paths
        if hasattr(self, 'settings_dir') and self.settings_dir:
            self.gr_setup_dir = self.settings_dir
        else:
            self.gr_setup_dir = self.project_dir / "settings" / "GR"

        self.forcing_gr_path = self.project_forcing_dir / 'GR_input'

    def _get_output_dir(self) -> Path:
        """GR uses output_path naming."""
        return self.get_experiment_output_dir()

    def run_gr(self, params: Optional[Dict[str, float]] = None) -> Optional[Path]:
        """
        Run the GR model.

        Args:
            params: Optional dictionary of parameters to use (skips internal calibration)

        Returns:
            Optional[Path]: Path to the output directory if successful, None otherwise
        """
        self.logger.debug(f"Starting GR model run in {self.spatial_mode} mode")

        # Store provided parameters in instance variables (not config_dict which is read-only)
        if params:
            self.logger.debug(f"Using external parameters for calibration: {params}")
            self._external_params = params
            self._skip_calibration = True

        with symfluence_error_handler(
            "GR model execution",
            self.logger,
            error_type=ModelExecutionError
        ):
            # Create output directory
            self.logger.debug(f"GR model output will be written to: {self.output_path}")
            self.output_path.mkdir(parents=True, exist_ok=True)

            # Execute GR model
            if self.spatial_mode == SpatialMode.LUMPED:
                success = self._execute_gr_lumped()
            else:  # distributed
                success = self._execute_gr_distributed()

            if success and self.needs_routing:
                self.logger.info("Running distributed routing with mizuRoute")
                success = self._run_distributed_routing()

            if success:
                self.logger.debug("GR model run completed successfully")

                # Calculate and log metrics for the run
                self._calculate_and_log_metrics()

                return self.output_path
            else:
                self.logger.error("GR model run failed")
                return None

    def _calculate_and_log_metrics(self) -> None:
        """Calculate and log performance metrics for the model run.

        Skipped during calibration (``_skip_calibration`` is True) because the
        calibration worker has its own metric calculation via
        ``GRStreamflowTarget``.
        """
        if self._skip_calibration:
            return

        try:
            from symfluence.models.gr.calibration.targets import GRStreamflowTarget

            self.logger.debug("Calculating performance metrics...")

            # Use GR-specific evaluator that knows about GR CSV output
            config = self.config if hasattr(self, 'config') and self.config else self.config_dict
            evaluator = GRStreamflowTarget(
                config,
                project_dir=self.project_dir,
                logger=self.logger
            )

            # Determine simulation directory
            # If routing was used, metrics should come from mizuRoute output
            if self.needs_routing:
                # Priority 1: Check config for specific mizuRoute output path
                mizu_output = self.mizu_experiment_output
                if mizu_output:
                    sim_dir = Path(mizu_output)
                else:
                    # Priority 2: Check for mizuRoute subdirectory in current output
                    sim_dir = self.output_path / 'mizuRoute'
                    if not sim_dir.exists():
                        # Priority 3: Use sibling to output_path (standard project structure)
                        sim_dir = self.output_path.parent / 'mizuRoute'

                if not sim_dir.exists():
                    self.logger.warning(f"MizuRoute simulation directory not found: {sim_dir}")
                    return
            else:
                sim_dir = self.output_path

            if not sim_dir.exists():
                self.logger.warning(f"Simulation directory not found for metrics: {sim_dir}")
                return

            self.logger.debug(f"Using simulation directory for metrics: {sim_dir}")

            # Evaluate
            metrics = evaluator.evaluate(sim_dir)

            if metrics and 'KGE' in metrics and not np.isnan(metrics['KGE']):
                kge_val = metrics['KGE']
                self.logger.info("=" * 40)
                self.logger.info(f"🏆 GR Model Performance ({self.spatial_mode})")
                self.logger.info(f"   KGE: {kge_val:.4f}")
                if 'NSE' in metrics and not np.isnan(metrics['NSE']):
                    self.logger.info(f"   NSE: {metrics['NSE']:.4f}")
                self.logger.info(f"   Output directory: {sim_dir}")
                self.logger.info("=" * 40)
            else:
                self.logger.warning("Could not calculate performance metrics (possibly missing observations or alignment failure)")
                if metrics:
                    self.logger.debug(f"Available metrics: {list(metrics.keys())}")

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Error calculating metrics: {e}", exc_info=True)
            self.logger.debug("Traceback: ", exc_info=True)

    def _load_calibrated_defaults(self) -> Dict[str, float]:
        """Load calibrated parameter values from GR_calib.Rdata if available.

        When running with external parameters that only cover a subset of the
        full CemaNeigeGR4J parameter vector (e.g. X1-X4 without CTG/Kf), this
        provides calibrated values for the missing parameters so they keep
        their optimized values instead of falling back to poor hardcoded
        defaults.

        Returns:
            Dictionary of calibrated parameter values (empty if unavailable).
        """
        if hasattr(self, '_calibrated_defaults_cache'):
            return self._calibrated_defaults_cache

        self._calibrated_defaults_cache: Dict[str, float] = {}

        try:
            rdata_path = self.output_path / 'GR_calib.Rdata'
            if not rdata_path.exists():
                # Try the standard experiment output dir
                rdata_path = self.get_experiment_output_dir() / 'GR_calib.Rdata'
            if not rdata_path.exists():
                return self._calibrated_defaults_cache

            robjects.r['load'](str(rdata_path))
            if 'OutputsCalib' in robjects.globalenv:
                param_final = list(robjects.globalenv['OutputsCalib'].rx2('ParamFinalR'))
                if len(param_final) == 8:
                    names = ['X1', 'X2', 'X3', 'X4', 'CTG', 'Kf', 'Gratio', 'Albedo_diff']
                elif len(param_final) == 6:
                    names = ['X1', 'X2', 'X3', 'X4', 'CTG', 'Kf']
                elif len(param_final) == 4:
                    names = ['X1', 'X2', 'X3', 'X4']
                else:
                    names = [f'P{i+1}' for i in range(len(param_final))]
                self._calibrated_defaults_cache = dict(zip(names, param_final))
                self.logger.debug(f"Loaded calibrated defaults for {len(names)} params from {rdata_path}")
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.debug(f"Could not load calibrated defaults: {e}", exc_info=True)

        return self._calibrated_defaults_cache

    def _forcing_corrections(self) -> Dict[str, float]:
        """Precipitation multiplier and temperature offset (identity unless calibrated)."""
        params = self._external_params or {}
        return {'PCORR': float(params.get('PCORR', 1.0)), 'TCORR': float(params.get('TCORR', 0.0))}

    def _glacier_enabled(self) -> bool:
        return bool(self._get_config_value(
            lambda: self.config.model.gr.glacier_enabled, default=False, dict_key='GR_GLACIER_ENABLED'
        ))

    def _execute_gr_lumped_glacier(self, params: Dict[str, float], n_layers: int,
                                   forcing: Dict[str, float]) -> bool:
        """Lumped CemaNeige -> degree-day ice melt -> GR4J (X2 as given).

        airGR has no glacier store, so CemaNeige and GR4J run as two calls and
        ice melt from the outline's share of each elevation layer is added to
        CemaNeige's liquid output before GR4J. See ``glacier.py``.
        """
        from .glacier import area_scale_series, compute_ice_melt, load_glacier_geometry

        gr_cfg = self.config.model.gr
        if not gr_cfg.glacier_outline:
            raise ModelExecutionError("GR_GLACIER_ENABLED requires GR_GLACIER_OUTLINE")
        dem_path = self.project_attributes_dir / 'elevation' / 'dem' / f"domain_{self.domain_name}_elv.tif"
        geom = load_glacier_geometry(str(dem_path), str(self.catchment_path / self.catchment_name),
                                     str(self._project_path(gr_cfg.glacier_outline)), n_layers)

        forcing_csv = self.forcing_gr_path / f"{self.domain_name}_input.csv"
        dates = pd.DatetimeIndex(pd.read_csv(forcing_csv, usecols=['time'])['time'].pipe(pd.to_datetime))
        run_start = pd.to_datetime(self.time_start).normalize()
        run_end = pd.to_datetime(self.time_end).normalize()
        run_idx = np.flatnonzero((dates >= run_start) & (dates <= run_end))
        if run_idx.size == 0:
            raise ModelExecutionError(f"GR forcing does not cover the run {run_start:%Y-%m-%d} to {run_end:%Y-%m-%d}")
        run_dates = dates[run_idx]

        cemaneige = [params.get(k, self._load_calibrated_defaults().get(k, d))
                     for k, d in (('CTG', 0.0), ('Kf', 3.69), ('Gratio', 0.1), ('Albedo_diff', 0.1))]
        r_dates = 'as.POSIXct(BasinObs$time, tz = "UTC")'
        r_vec = lambda v: f"c({', '.join(repr(float(x)) for x in v)})"  # noqa: E731
        run_r_script(robjects, f'''
            library(airGR)
            BasinObs <- read.csv({r_path(forcing_csv)})
            Ind_Run <- {int(run_idx[0]) + 1}L:{int(run_idx[-1]) + 1}L
            InputsCN <- CreateInputsModel(
                FUN_MOD = RunModel_CemaNeige, DatesR = {r_dates},
                Precip = BasinObs$pr * {forcing['PCORR']}, TempMean = BasinObs$temp + {forcing['TCORR']},
                ZInputs = {geom.z_mean}, HypsoData = {r_vec(geom.hypso)}, NLayers = {n_layers}, verbose = FALSE
            )
            RunOptionsCN <- CreateRunOptions(
                FUN_MOD = RunModel_CemaNeige, InputsModel = InputsCN, IndPeriod_WarmUp = 0L,
                IndPeriod_Run = Ind_Run, IsHyst = TRUE, warnings = FALSE, verbose = FALSE
            )
            OutputsCN <- RunModel_CemaNeige(InputsCN, RunOptionsCN, Param = {r_vec(cemaneige)})
            layer_matrix <- function(name) sapply(OutputsCN$CemaNeigeLayers, function(l) l[[name]])
            CN_swe <- layer_matrix("SnowPack"); CN_temp <- layer_matrix("Temp")
            CN_liquid <- rowMeans(layer_matrix("PliqAndMelt"))
            TRUE
        ''', "GR CemaNeige layers")

        def r_matrix(name: str) -> np.ndarray:
            # as.vector() flattens column-major (layer by layer), independent of
            # how the active rpy2 converter maps R matrices.
            return np.asarray(run_r_script(robjects, f'as.vector({name})', name), dtype=float).reshape(n_layers, -1).T

        swe, temp = r_matrix('CN_swe'), r_matrix('CN_temp')
        liquid = np.asarray(run_r_script(robjects, 'CN_liquid', 'CemaNeige liquid output'), dtype=float)

        area_file = self._project_path(gr_cfg.glacier_area_file) if gr_cfg.glacier_area_file else None
        mb_years = self._calibration_water_years()
        mb_target = gr_cfg.glacier_mb_target
        ice = compute_ice_melt(
            run_dates, swe, temp, geom.layer_fraction,
            area_scale_series(run_dates, geom.glacier_area_km2, area_file),
            ddf=params.get('ICE_DDF'),
            mb_target_mm=None if mb_target is None else 1000.0 * float(mb_target),
            mb_years=mb_years, ddf_bounds=gr_cfg.glacier_ice_ddf_bounds,
            snowfree_swe=gr_cfg.glacier_snowfree_swe,
        )
        if np.isfinite(ice.ddf_unclipped) and ice.ddf != ice.ddf_unclipped:
            self.logger.warning(f"Solved ice degree-day factor {ice.ddf_unclipped:.3g} clipped to {ice.ddf:.3g}; "
                                "the mass-balance target is not met")

        robjects.globalenv['GR_liquid'] = robjects.FloatVector(liquid + ice.ice_melt)
        x = [params.get(k, self._load_calibrated_defaults().get(k, d))
             for k, d in (('X1', 350.0), ('X2', 0.0), ('X3', 100.0), ('X4', 1.7))]
        run_r_script(robjects, f'''
            InputsGR <- CreateInputsModel(
                FUN_MOD = RunModel_GR4J, DatesR = {r_dates}[Ind_Run],
                Precip = GR_liquid, PotEvap = BasinObs$pet[Ind_Run], verbose = FALSE
            )
            RunOptionsGR <- CreateRunOptions(
                FUN_MOD = RunModel_GR4J, InputsModel = InputsGR, IndPeriod_WarmUp = 0L,
                IndPeriod_Run = seq_along(GR_liquid), warnings = FALSE, verbose = FALSE
            )
            OutputsModel <- RunModel_GR4J(InputsGR, RunOptionsGR, Param = {r_vec(x)})
            save(OutputsModel, file = {r_path(self.output_path / 'GR_results.Rdata')})
            TRUE
        ''', "GR4J with glacier melt")
        q = np.asarray(run_r_script(robjects, 'OutputsModel$Qsim', 'GR4J discharge'), dtype=float)

        pd.DataFrame({
            'datetime': run_dates.strftime('%Y-%m-%d %H:%M:%S'),
            'q_sim': q, 'ice_melt_mm': ice.ice_melt,
        }).to_csv(self.output_path / 'GR_results.csv', index=False)
        (self.output_path / 'GR_glacier_diagnostics.json').write_text(json.dumps({
            'ice_ddf': ice.ddf, 'ice_ddf_unclipped': ice.ddf_unclipped,
            'mb_target_mwe': mb_target, 'mb_years': list(mb_years),
            'glacier_area_km2_outline': geom.glacier_area_km2,
            'layer_glacier_fraction': geom.layer_fraction.tolist(),
            'forcing': forcing,
            'mass_balance_mwe': {str(k): v / 1000.0 for k, v in ice.mass_balance.items()},
        }, indent=1))
        return True

    def _project_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.project_dir / path

    def _calibration_water_years(self) -> list:
        """Water years lying fully inside the calibration period."""
        period = self._get_config_value(lambda: self.config.domain.calibration_period, default=None)
        if not period:
            raise ModelExecutionError("GR glacier mass-balance target needs a calibration period")
        start, end = (pd.to_datetime(p.strip()) for p in str(period).split(','))
        first_oct1 = pd.Timestamp(start.year, 10, 1)
        first = first_oct1.year + 1 if start <= first_oct1 else first_oct1.year + 2
        last_sep30 = pd.Timestamp(end.year, 9, 30)
        last = last_sep30.year if end >= last_sep30 else last_sep30.year - 1
        return list(range(first, last + 1))

    def _check_routing_requirements(self) -> bool:
        """Check if distributed routing is needed.

        Routing is enabled when:
        1. GR_ROUTING_INTEGRATION is explicitly set to 'mizuRoute', OR
        2. ROUTING_MODEL is 'mizuRoute' and spatial_mode is 'distributed'
        """
        # Check explicit GR routing integration setting
        routing_integration = self._get_config_value(
            lambda: self.config.model.gr.routing_integration if self.config.model and self.config.model.gr else None,
            'none'
        )

        # Check global routing model setting
        global_routing = self.routing_model

        # Enable routing if explicitly configured for GR
        if routing_integration and routing_integration.lower() == 'mizuroute':
            if self.spatial_mode == SpatialMode.DISTRIBUTED:
                self.logger.info("GR routing enabled via GR_ROUTING_INTEGRATION: mizuRoute")
                return True

        # Auto-enable routing if global routing model is mizuRoute and distributed mode
        if global_routing and global_routing.lower() == 'mizuroute':
            if self.spatial_mode == SpatialMode.DISTRIBUTED:
                self.logger.info("GR routing auto-enabled: ROUTING_MODEL=mizuRoute with distributed mode")
                return True

        return False

    def _execute_gr_distributed(self) -> bool:
        """
        Execute GR4J-CemaNeige in distributed mode for each HRU.

        Runs the GR4J model coupled with CemaNeige snow module separately
        for each hydrological response unit (HRU). Results are combined into
        a single NetCDF file compatible with mizuRoute routing.

        The workflow:
        1. Initialize R environment and load airGR package
        2. Load forcing data for all HRUs from NetCDF
        3. Extract DEM statistics for snow modeling (hypsometric curve)
        4. Loop through each HRU:
           - Extract HRU-specific forcing
           - Run GR4J-CemaNeige for that HRU
           - Collect results
        5. Combine all HRU results into mizuRoute-compatible format

        Returns:
            bool: True if all HRUs completed successfully, False otherwise.

        Note:
            Uses temporary directory isolation for parallel execution safety.
            Each HRU gets its own temporary CSV file to prevent race conditions.
        """
        self.logger.info("Running distributed GR4J workflow")

        # Create temp directory for this evaluation (worker isolation for parallel execution)
        temp_dir = self.output_path / '.gr_temp'
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Initialize R environment
            importr('base')
            ensure_airgr_available(robjects)

            # Load forcing data
            forcing_file = self.forcing_gr_path / f"{self.domain_name}_input_distributed.nc"
            ds = xr.open_dataset(forcing_file)

            n_hrus = len(ds.hru)
            self.logger.debug(f"Running GR4J for {n_hrus} HRUs")

            # Load DEM for hypsometric curve (use catchment-wide for now)
            dem_path = self.project_attributes_dir / 'elevation' / 'dem' / f"domain_{self.domain_name}_elv.tif"
            catchment = gpd.read_file(self.catchment_path / self.catchment_name)

            with rasterio.open(dem_path) as src:
                if catchment.crs != src.crs:
                    catchment = catchment.to_crs(src.crs)
                fill_value = src.nodata if src.nodata is not None else 0
                out_image, out_transform = rasterio.mask.mask(
                    src, catchment.geometry, crop=True, nodata=fill_value
                )
                masked_dem = out_image[0]
                masked_dem = masked_dem[masked_dem != fill_value]
                if masked_dem.size == 0:
                    raise ValueError(f"No valid DEM pixels after masking with catchment. Check DEM at {dem_path}")
                Hypso = np.percentile(masked_dem, np.arange(0, 101, 1))
                Zmean = np.mean(masked_dem)
                self.logger.debug(f"DEM: {masked_dem.size} valid pixels, "
                                  f"elev range [{masked_dem.min():.1f}, {masked_dem.max():.1f}] m, "
                                  f"mean={Zmean:.1f} m")

            # Get simulation periods
            time_start = pd.to_datetime(self.time_start)
            time_end = pd.to_datetime(self.time_end)
            spinup_start = pd.to_datetime(self.spinup_period.split(',')[0].strip()).strftime('%Y-%m-%d')
            spinup_end = pd.to_datetime(self.spinup_period.split(',')[1].strip()).strftime('%Y-%m-%d')
            run_start = time_start.strftime('%Y-%m-%d')
            run_end = time_end.strftime('%Y-%m-%d')

            # Store results for each HRU
            hru_results = []

            # Determine parameters (use instance variable, not read-only config_dict)
            external_params = self._external_params
            if external_params:
                # Use provided parameters, falling back to calibrated values then defaults
                calibrated_defaults = self._load_calibrated_defaults()
                x1 = external_params.get('X1', calibrated_defaults.get('X1', 257.24))
                x2 = external_params.get('X2', calibrated_defaults.get('X2', 1.012))
                x3 = external_params.get('X3', calibrated_defaults.get('X3', 88.23))
                x4 = external_params.get('X4', calibrated_defaults.get('X4', 2.208))
                ctg = external_params.get('CTG', calibrated_defaults.get('CTG', 0.0))
                kf = external_params.get('Kf', calibrated_defaults.get('Kf', 3.69))
                gratio = external_params.get('Gratio', calibrated_defaults.get('Gratio', 0.1))
                albedo_diff = external_params.get('Albedo_diff', calibrated_defaults.get('Albedo_diff', 0.1))
                param_str = f"c(X1={x1}, X2={x2}, X3={x3}, X4={x4}, CTG={ctg}, Kf={kf}, Gratio={gratio}, Albedo_diff={albedo_diff})"
            else:
                param_str = "c(X1=257.24, X2=1.012, X3=88.23, X4=2.208, CTG=0.0, Kf=3.69, Gratio=0.1, Albedo_diff=0.1)"

            # Loop through each HRU
            for hru_idx in range(n_hrus):
                hru_id = int(ds.hru_id.values[hru_idx]) if 'hru_id' in ds else hru_idx + 1
                self.logger.info(f"Processing HRU {hru_id} ({hru_idx + 1}/{n_hrus})")

                # Extract data for this HRU
                hru_data = ds.isel(hru=hru_idx)

                # Create temporary DataFrame for this HRU
                hru_df = pd.DataFrame({
                    'time': pd.to_datetime(hru_data.time.values).strftime('%Y-%m-%d'),
                    'pr': hru_data['pr'].values,
                    'temp': hru_data['temp'].values,
                    'pet': hru_data['pet'].values
                })

                # Save temporary CSV for R in isolated temp directory (prevents MPI race conditions)
                temp_csv = temp_dir / f"hru_{hru_id}_temp.csv"
                hru_df.to_csv(temp_csv, index=False)

                # Run GR4J for this HRU
                r_script = f'''
                    library(airGR)

                    # Load HRU data
                    BasinObs <- read.csv({r_path(temp_csv)})

                    # Preparation of InputsModel
                    InputsModel <- CreateInputsModel(
                        FUN_MOD = RunModel_CemaNeigeGR4J,
                        DatesR = as.POSIXct(BasinObs$time),
                        Precip = BasinObs$pr,
                        PotEvap = BasinObs$pet,
                        TempMean = BasinObs$temp,
                        HypsoData = c({', '.join(map(str, Hypso))}),
                        ZInputs = {Zmean}
                    )

                    # Parse dates and validate indices
                    date_vector <- format(as.Date(BasinObs$time), "%Y-%m-%d")

                    # Find indices with validation
                    Ind_Warm <- which(date_vector >= "{spinup_start}" & date_vector <= "{spinup_end}")
                    Ind_Run <- which(date_vector >= "{run_start}" & date_vector <= "{run_end}")

                    # Validate date ranges overlap with forcing data
                    if (length(Ind_Run) < 1) {{
                        forcing_range <- paste(min(as.Date(date_vector)), "to", max(as.Date(date_vector)))
                        stop(paste0("GR date mismatch for HRU {hru_id}: Requested {run_start} to {run_end} ",
                                   "but forcing data covers ", forcing_range))
                    }}

                    # Use parameters
                    Param <- {param_str}

                    # Preparation of RunOptions
                    RunOptions <- CreateRunOptions(
                        FUN_MOD = RunModel_CemaNeigeGR4J,
                        InputsModel = InputsModel,
                        IndPeriod_WarmUp = Ind_Warm,
                        IndPeriod_Run = Ind_Run,
                        IsHyst = TRUE
                    )

                    # Run model
                    OutputsModel <- RunModel_CemaNeigeGR4J(
                        InputsModel = InputsModel,
                        RunOptions = RunOptions,
                        Param = Param
                    )

                    # Extract results
                    data.frame(
                        date = format(OutputsModel$DatesR, "%Y-%m-%d"),
                        q_routed = OutputsModel$Qsim
                    )
                '''

                # Execute R script
                result_df = run_r_script(robjects, r_script, "distributed GR4J script")

                # Convert to pandas
                with localconverter(robjects.default_converter + pandas2ri.converter):
                    result_df = robjects.conversion.rpy2py(result_df)

                result_df['date'] = pd.to_datetime(result_df['date'])
                result_df['hru_id'] = hru_id

                hru_results.append(result_df)

                # Clean up temporary file
                temp_csv.unlink()

            # Combine all HRU results
            self.logger.info("Combining results from all HRUs")
            combined_results = pd.concat(hru_results, ignore_index=True)

            # Pivot to get time x HRU structure
            results_pivot = combined_results.pivot(index='date', columns='hru_id', values='q_routed')

            # Convert to xarray Dataset (mizuRoute format)
            self._save_distributed_results_for_routing(results_pivot, ds)

            # Verify output file was actually created
            output_nc = self.output_path / f"{self.domain_name}_{self.experiment_id}_runs_def.nc"
            if not output_nc.exists():
                self.logger.error(
                    f"GR model completed but output file was not created: {output_nc}. "
                    "Check that simulation dates overlap with forcing data."
                )
                return False

            return True

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error in distributed GR4J execution: {str(e)}", exc_info=True)
            import traceback
            self.logger.error(traceback.format_exc())
            return False

        finally:
            # Clean up temp directory if it exists
            if temp_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except Exception as cleanup_error:  # noqa: BLE001 — model execution resilience
                    self.logger.warning(f"Failed to clean up temp directory {temp_dir}: {cleanup_error}", exc_info=True)

    def _save_distributed_results_for_routing(self, results_df, forcing_ds):
        """
        Save distributed GR4J results in mizuRoute-compatible format.

        Creates a NetCDF file with structure matching mizuRoute expectations:
        dimensions (time, gru), gruId variable, and runoff in m/s.

        Args:
            results_df: DataFrame with DatetimeIndex and columns for each HRU ID,
                containing runoff values in mm/day.
            forcing_ds: Original forcing dataset used to extract HRU metadata.

        Output file structure:
        - Dimensions: (time, gru)
        - Variables:
          - gruId: Integer HRU identifiers
          - {routing_var}: Runoff converted from mm/day to m/s
        - Time coordinate in seconds since 1970-01-01

        File is written to: {output_path}/{domain_name}_{experiment_id}_runs_def.nc
        """
        self.logger.info("Saving distributed results in mizuRoute format")

        # Create time coordinate (seconds since 1970-01-01)
        time_values = results_df.index
        time_seconds = (time_values - pd.Timestamp('1970-01-01')).total_seconds().values

        # Get HRU IDs from columns
        hru_ids = results_df.columns.values.astype(int)
        n_hrus = len(hru_ids)

        # Create xarray Dataset with mizuRoute structure
        # Dimensions: (time, gru) - matching what mizuRoute expects
        ds_out = xr.Dataset(
            coords={
                'time': ('time', time_seconds),
                'gru': ('gru', np.arange(n_hrus))
            }
        )

        # Add gruId variable
        ds_out['gruId'] = xr.DataArray(
            hru_ids,
            dims=('gru',),
            attrs={
                'long_name': 'ID of grouped response unit',
                'units': '-'
            }
        )

        # Add streamflow data (convert from mm/day to m/s for mizuRoute)
        # 1 mm/day = 1 / (1000 * 86400) m/s
        routing_var = self.mizu_routing_var or 'q_routed'

        runoff_ms = results_df.values / (1000.0 * UnitConversion.SECONDS_PER_DAY)

        ds_out[routing_var] = xr.DataArray(
            runoff_ms,
            dims=('time', 'gru'),
            attrs={
                'long_name': 'GR4J runoff for mizuRoute routing',
                'units': 'm/s',
                'description': 'Runoff from distributed GR4J model (converted from mm/d)'
            }
        )

        # Add time attributes
        ds_out.time.attrs = {
            'units': 'seconds since 1970-01-01 00:00:00',
            'calendar': 'standard',
            'long_name': 'time'
        }

        # Add global attributes
        ds_out.attrs = {
            'model': 'GR4J-CemaNeige',
            'spatial_mode': 'distributed',
            'domain': self.domain_name,
            'experiment_id': self.experiment_id,
            'n_hrus': n_hrus,
            'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'description': 'Distributed GR4J simulation results for mizuRoute routing'
        }

        # Save to NetCDF
        output_file = self.output_path / f"{self.domain_name}_{self.experiment_id}_runs_def.nc"

        # Use standardized encoding utility
        encoding = create_netcdf_encoding(
            ds_out,
            compression=True,
            int_vars={'gruId': 'int32'}
        )

        ds_out.to_netcdf(output_file, encoding=encoding, format='NETCDF4')

        self.logger.info(f"Distributed results saved to: {output_file}")
        self.logger.info(f"Output dimensions: time={len(time_seconds)}, gru={n_hrus}")

    def _run_distributed_routing(self) -> bool:
        """Run mizuRoute routing for distributed GR4J output.

        Uses SpatialOrchestrator._run_mizuroute() for unified routing integration.
        """
        self.logger.info("Starting mizuRoute routing for distributed GR4J")

        # Update config for GR-mizuRoute integration
        self._setup_gr_mizuroute_config()

        # Check if control file already exists (to avoid overwriting process-specific ones)
        mizu_settings_dir = self.mizu_settings_path
        mizu_control = self.mizu_control_file or 'mizuRoute_control_GR.txt'

        create_control = True
        if mizu_settings_dir:
            control_path = Path(mizu_settings_dir) / mizu_control
            if control_path.exists():
                self.logger.debug(f"MizuRoute control file exists at {control_path}, will not overwrite")
                create_control = False

        # Use orchestrator method (creates control file and runs mizuRoute)
        spatial_config = self.get_spatial_config('GR')
        result = self._run_mizuroute(spatial_config, model_name='gr', create_control_file=create_control)

        return result is not None

    def _setup_gr_mizuroute_config(self):
        """Set up runtime state for GR-mizuRoute integration.

        No-op: the mizuroute preprocessor infers from_model from
        HYDROLOGICAL_MODEL when MIZU_FROM_MODEL is not explicitly set,
        and the control file name is handled by the fallback in
        _run_distributed_routing.
        """

    def _execute_gr_lumped(self):
        """
        Execute GR4J-CemaNeige in lumped mode for a single catchment.

        Runs the complete GR4J workflow including optional calibration using
        the Michel calibration algorithm. Uses airGR package via rpy2 for
        the actual model execution.

        The workflow:
        1. Load DEM and calculate hypsometric curve for snow modeling
        2. Initialize R environment and load airGR
        3. Create InputsModel with forcing data
        4. Run calibration (unless skip_calibration is True)
        5. Execute simulation with calibrated or provided parameters
        6. Save results to CSV and Rdata files

        Returns:
            bool: True if execution completed successfully, False otherwise.

        Output files:
        - GR_results.csv: Simulated streamflow timeseries
        - GR_results.Rdata: Full R model output object
        - GR_calib.Rdata: Calibration results (if calibration was run)
        - GRhydrology_plot.png: Hydrograph plot (if visualization enabled)
        """
        try:
            # Initialize R environment
            importr('base')
            ensure_airgr_available(robjects)
            skip_calibration = self._skip_calibration
            default_params = self._get_config_value(
                lambda: self.config.model.gr.default_params,
                default=[350, 0, 100, 1.7]
            )
            if len(default_params) != 4:
                raise ValueError("GR_DEFAULT_PARAMS must contain 4 values for X1, X2, X3, X4")

            # Read DEM
            dem_path = self.project_attributes_dir / 'elevation' / 'dem' / f"domain_{self.domain_name}_elv.tif"
            with rasterio.open(dem_path) as src:
                src.read(1)

            # Read catchment and get centroid
            catchment = gpd.read_file(self.catchment_path / self.catchment_name)

            # Mask DEM with catchment boundary
            with rasterio.open(dem_path) as src:
                if catchment.crs != src.crs:
                    catchment = catchment.to_crs(src.crs)
                # Use a known fill value so we can reliably filter masked pixels
                fill_value = src.nodata if src.nodata is not None else 0
                out_image, out_transform = rasterio.mask.mask(
                    src, catchment.geometry, crop=True, nodata=fill_value
                )
                masked_dem = out_image[0]
                masked_dem = masked_dem[masked_dem != fill_value]
                if masked_dem.size == 0:
                    raise ValueError(f"No valid DEM pixels after masking with catchment. Check DEM at {dem_path}")
                Hypso = np.percentile(masked_dem, np.arange(0, 101, 1))
                Zmean = np.mean(masked_dem)
                self.logger.debug(f"DEM: {masked_dem.size} valid pixels, "
                                  f"elev range [{masked_dem.min():.1f}, {masked_dem.max():.1f}] m, "
                                  f"mean={Zmean:.1f} m")

            time_start = pd.to_datetime(self.time_start)
            time_end = pd.to_datetime(self.time_end)

            spinup_start = pd.to_datetime(self.spinup_period.split(',')[0].strip()).strftime('%Y-%m-%d')
            spinup_end = pd.to_datetime(self.spinup_period.split(',')[1].strip()).strftime('%Y-%m-%d')
            calib_start = pd.to_datetime(self.calibration_period.split(',')[0].strip()).strftime('%Y-%m-%d')
            calib_end = pd.to_datetime(self.calibration_period.split(',')[1].strip()).strftime('%Y-%m-%d')
            run_start = time_start.strftime('%Y-%m-%d')
            run_end = time_end.strftime('%Y-%m-%d')

            self.logger.debug(f"Spinup period: {spinup_start} to {spinup_end}")
            self.logger.debug(f"Calibration period: {calib_start} to {calib_end}")
            self.logger.debug(f"Run period: {run_start} to {run_end}")

            # Determine parameters for R script
            # Use instance variable for external params (config_dict is read-only)
            # When external params only cover a subset (e.g. X1-X4 but not
            # CemaNeige), try to fill gaps from any previous calibration
            # (GR_calib.Rdata) so that un-optimized params keep their
            # calibrated values instead of falling back to poor defaults.
            external_params = self._external_params
            forcing = self._forcing_corrections()
            n_layers = self._get_config_value(lambda: self.config.model.gr.snow_layers, default=5)
            if self._glacier_enabled():
                if not external_params:
                    raise ModelExecutionError(
                        "GR glacier mode has no internal (Michel) calibration; "
                        "calibrate with the optimizer or pass parameters"
                    )
                return self._execute_gr_lumped_glacier(external_params, int(n_layers), forcing)
            if external_params:
                calibrated_defaults = self._load_calibrated_defaults()
                x1 = external_params.get('X1', calibrated_defaults.get('X1', default_params[0]))
                x2 = external_params.get('X2', calibrated_defaults.get('X2', default_params[1]))
                x3 = external_params.get('X3', calibrated_defaults.get('X3', default_params[2]))
                x4 = external_params.get('X4', calibrated_defaults.get('X4', default_params[3]))
                ctg = external_params.get('CTG', calibrated_defaults.get('CTG', 0.0))
                kf = external_params.get('Kf', calibrated_defaults.get('Kf', 3.69))
                gratio = external_params.get('Gratio', calibrated_defaults.get('Gratio', 0.1))
                albedo_diff = external_params.get('Albedo_diff', calibrated_defaults.get('Albedo_diff', 0.1))
                external_param_str = f"Param <- c(X1={x1}, X2={x2}, X3={x3}, X4={x4}, CTG={ctg}, Kf={kf}, Gratio={gratio}, Albedo_diff={albedo_diff})"
            else:
                external_param_str = f"Param <- c(X1={default_params[0]}, X2={default_params[1]}, X3={default_params[2]}, X4={default_params[3]}, CTG=0.0, Kf=3.69, Gratio=0.1, Albedo_diff=0.1)"

            # R script as a string with improved date handling
            r_script = f'''
                library(airGR)
                skip_calibration <- {"TRUE" if skip_calibration else "FALSE"}

                # Loading catchment data
                BasinObs <- read.csv({r_path(self.forcing_gr_path / f"{self.domain_name}_input.csv")})

                # Convert time column to POSIXct format
                BasinObs$time_posix <- as.POSIXct(BasinObs$time)

                # Create a safer date matching function that throws an error on failure
                find_date_indices <- function(start_date, end_date, date_vector, period_name) {{
                    date_vector_as_date <- as.Date(date_vector)
                    start_date_as_date <- as.Date(start_date)
                    end_date_as_date <- as.Date(end_date)

                    indices <- which(date_vector_as_date >= start_date_as_date &
                                    date_vector_as_date <= end_date_as_date)

                    if (length(indices) < 1) {{
                        forcing_range <- paste(min(date_vector_as_date), "to", max(date_vector_as_date))
                        requested_range <- paste(start_date, "to", end_date)
                        stop(paste0("GR date mismatch for ", period_name, " period: ",
                                   "Requested ", requested_range, " but forcing data covers ", forcing_range,
                                   ". Check SIMULATION_START/END and SPINUP/CALIBRATION periods in config."))
                    }}

                    return(indices)
                }}

                # Determine periods
                date_vector <- format(as.Date(BasinObs$time), "%Y-%m-%d")

                Ind_Warm <- find_date_indices("{spinup_start}", "{spinup_end}", date_vector, "spinup")
                Ind_Cal <- find_date_indices("{calib_start}", "{calib_end}", date_vector, "calibration")
                Ind_Run <- find_date_indices("{run_start}", "{run_end}", date_vector, "simulation")

                # Preparation of InputsModel object
                InputsModel <- CreateInputsModel(
                    FUN_MOD = RunModel_CemaNeigeGR4J,
                    DatesR = as.POSIXct(BasinObs$time),
                    Precip = BasinObs$pr * {forcing['PCORR']},
                    PotEvap = BasinObs$pet,
                    TempMean = BasinObs$temp + {forcing['TCORR']},
                    HypsoData = c({', '.join(map(str, Hypso))}),
                    ZInputs = {Zmean},
                    NLayers = {int(n_layers)}
                )

                # Preparation of RunOptions object
                RunOptions <- CreateRunOptions(
                    FUN_MOD = RunModel_CemaNeigeGR4J,
                    InputsModel = InputsModel,
                    IndPeriod_WarmUp = Ind_Warm,
                    IndPeriod_Run = Ind_Cal,
                    IsHyst = TRUE
                )

                # Calibration criterion
                InputsCrit <- CreateInputsCrit(
                    FUN_CRIT = ErrorCrit_{self.optimization_metric},
                    InputsModel = InputsModel,
                    RunOptions = RunOptions,
                    Obs = BasinObs$q_obs[Ind_Cal]
                )

                # Preparation of CalibOptions object
                CalibOptions <- CreateCalibOptions(
                    FUN_MOD = RunModel_CemaNeigeGR4J,
                    FUN_CALIB = Calibration_Michel,
                    IsHyst = TRUE
                )

                # Calibration (optional for tests)
                if (!skip_calibration) {{
                    OutputsCalib <- Calibration_Michel(
                        InputsModel = InputsModel,
                        RunOptions = RunOptions,
                        InputsCrit = InputsCrit,
                        CalibOptions = CalibOptions,
                        FUN_MOD = RunModel_CemaNeigeGR4J
                    )
                    save(OutputsCalib, file = {r_path(self.output_path / 'GR_calib.Rdata')})
                    Param <- OutputsCalib$ParamFinalR
                }} else {{
                    # Use provided parameters or defaults
                    {external_param_str}
                }}

                # Preparation of RunOptions for full simulation
                # Ind_Run already starts at time_start and includes spinup.
                # Do not replay that same warmup before the simulation start.
                RunOptions <- CreateRunOptions(
                    FUN_MOD = RunModel_CemaNeigeGR4J,
                    InputsModel = InputsModel,
                    IndPeriod_WarmUp = 0L,
                    IndPeriod_Run = Ind_Run,
                    IsHyst = TRUE
                )

                OutputsModel <- RunModel_CemaNeigeGR4J(
                    InputsModel = InputsModel,
                    RunOptions = RunOptions,
                    Param = Param
                )

                # Results preview
                if ({"TRUE" if self.reporting_manager and self.reporting_manager.visualize else "FALSE"}) {{
                    # Create plots directory
                    dir.create({r_path(self.project_dir / 'reporting' / 'results')}, recursive = TRUE, showWarnings = FALSE)
                    png({r_path(self.project_dir / 'reporting' / 'results' / 'GRhydrology_plot.png')}, height = 900, width = 900)
                    plot(OutputsModel, Qobs = BasinObs$q_obs[Ind_Run])
                    dev.off()
                }}

                # Save results
                save(OutputsModel, file = {r_path(self.output_path / 'GR_results.Rdata')})

                # Export to CSV for post-processing and metrics
                results_df <- data.frame(
                    datetime = format(OutputsModel$DatesR, "%Y-%m-%d %H:%M:%S"),
                    q_sim = OutputsModel$Qsim
                )
                write.csv(results_df, {r_path(self.output_path / 'GR_results.csv')}, row.names = FALSE)

                "GR model execution completed successfully"
            '''

            # Execute the R script
            run_r_script(robjects, r_script, "lumped GR4J script")
            self.logger.debug("R script executed successfully!")

            # Verify output file was actually created
            output_csv = self.output_path / 'GR_results.csv'
            if not output_csv.exists():
                self.logger.error(
                    f"GR model completed but output file was not created: {output_csv}. "
                    "Check that simulation dates overlap with forcing data."
                )
                return False

            return True

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"An error occurred: {str(e)}", exc_info=True)
            import traceback
            self.logger.error(traceback.format_exc())
            return False
