#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""Streamflow Evaluator.

This module provides streamflow (discharge) evaluation for hydrological calibration.
Handles extraction of simulated streamflow from both SUMMA and mizuRoute outputs,
automatic unit conversion (mass flux to volume flux), spatial aggregation for
distributed models, and observed data matching.
"""
from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, cast

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.constants import UnitConverter
from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.modeling.observation_paths import first_existing_path, streamflow_observation_candidates
from symfluence.core.modeling.output_file_locator import OutputFileLocator
from symfluence.core.path_resolver import find_basin_shapefile
from symfluence.core.registries import R

from .base import ModelEvaluator
from .streamflow_units import classify_streamflow_units

if TYPE_CHECKING:
    pass


@R.evaluators.add('STREAMFLOW')
class StreamflowEvaluator(ModelEvaluator):
    """Streamflow evaluator for calibrating hydrological models.

    This evaluator handles extraction and processing of streamflow (discharge)
    from hydrological model outputs for comparison with observations. It supports
    multiple model types (SUMMA, mizuRoute) and handles common data issues like
    unit conversions and spatial aggregation.

    Key Responsibilities:
        1. Locate streamflow output files from model simulations
        2. Detect output format (SUMMA vs mizuRoute) and extract appropriately
        3. Handle unit conversions: mass flux (kg m⁻² s⁻¹) → volume flux (m³ s⁻¹)
        4. Perform spatial aggregation for distributed/semi-distributed models
        5. Convert per-unit-area runoff to basin-scale discharge using catchment area
        6. Match observed streamflow data from calibration targets

    Unit Conversion Details:
        SUMMA outputs runoff in three possible representations:
        - Mass flux: kg m⁻² s⁻¹ (mass per unit area per unit time)
        - Volume flux: m s⁻¹ (depth per unit time)
        - Total runoff: basin-scale totals (mm/day)

        Conversion strategy:
        1. Detect units from NetCDF attributes and data magnitude
        2. If mass flux: divide by water density (1000 kg/m³) → volume flux
        3. Spatial aggregation: sum over HRU/GRU with area weighting
        4. Scale to basin discharge: multiply by catchment area (m²) → m³/s

    Supported Output Formats:
        - SUMMA timestep files (*.nc with variables like averageRoutedRunoff)
        - mizuRoute network output (*.nc with routed runoff by reach)
        - SUMMA+mizuRoute coupled (distributed runoff fed to routing)

    Catchment Area Resolution (Priority):
        1. Manual override: FIXED_CATCHMENT_AREA in config (highest priority)
        2. SUMMA attributes.nc: HRUarea sum (most reliable automated)
        3. Basin shapefile: GRU_area column (user-defined delineation)
        4. Catchment shapefile: geometry area (fallback)
        5. Default: 1 km² (last resort, triggers warning)

    Configuration Parameters:
        FIXED_CATCHMENT_AREA: Manual area override (m²), skips auto-detection
        RIVER_BASIN_SHP_AREA: Column name in river basin shapefile (default: GRU_area)
        CATCHMENT_SHP_AREA: Column name in catchment shapefile (default: HRU_area)
        OBSERVATIONS_PATH: Override to observed streamflow file path

    Common Issues & Solutions:
        Issue: Unit mismatch (extremely high/low values)
            Solution: Automatic detection via magnitude thresholding (>1e-6 m/s = mass flux)

        Issue: Spatial dimension (HRU vs GRU vs reach)
            Solution: Area-weighted aggregation when attributes available; fallback to first unit

        Issue: Missing catchment area
            Solution: Priority fallback system; default 1 km² with warning

        Issue: Different variable names across model versions
            Solution: Tries multiple common names (averageRoutedRunoff, basin__TotalRunoff, etc.)

    Attributes:
        Uses inherited from ModelEvaluator:
        - project_dir: Project root directory
        - config_dict: Configuration dictionary
        - logger: Logger instance
        - calibration_period: (start_date, end_date) tuple for evaluation
    """

    def __init__(self, config, project_dir: Optional[Path] = None, logger=None,
                 target: Optional[str] = None, *,
                 model_name: Optional[str] = None):
        """Initialise the evaluator.

        Args:
            target: Optional target override, as accepted by ``ModelEvaluator``.
            model_name: Model whose registered ``ModelResultExtractor`` and
                ``results/<exp>_results.csv`` column to use. Defaults to
                ``HYDROLOGICAL_MODEL`` when that names a single model.
        """
        super().__init__(config, project_dir, logger, target=target)
        self._model_name_override = model_name.upper() if model_name else None

    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Locate streamflow output files from simulation directory.

        Order: the generic OutputFileLocator search (mizuRoute/t-route first,
        then SUMMA-style NetCDF), then the patterns declared by the model's
        registered ``ModelResultExtractor``, then the experiment's
        ``results/<exp>_results.csv`` (only for the experiment's own
        simulation directory — see :meth:`_results_csv_for`).

        Args:
            sim_dir: Directory containing simulation outputs

        Returns:
            List[Path]: Paths to streamflow output files
        """
        if self._uses_native_wflow_discharge():
            # cryosphere.nc contains states, not outlet discharge. Do not let
            # the generic NetCDF locator select it ahead of the native CSV.
            output = sim_dir / 'output.csv'
            return [output] if output.is_file() else []
        locator = OutputFileLocator(self.logger)
        files = locator.find_streamflow_files(sim_dir)
        if files:
            return files
        files = self._extractor_streamflow_files(Path(sim_dir))
        if files:
            self.logger.debug(f"Streamflow files found via {self._result_model_name()} result extractor patterns")
            return files
        results_csv = self._results_csv_for(Path(sim_dir))
        return [results_csv] if results_csv is not None else []

    # ------------------------------------------------------------------
    # Model result extractor (R.result_extractors) support
    # ------------------------------------------------------------------

    def _result_model_name(self) -> Optional[str]:
        """Model whose result extractor / results column applies, if unambiguous."""
        if self._model_name_override:
            return self._model_name_override
        models = self._get_config_value(
            lambda: self.config.model.hydrological_model,
            default='', dict_key='HYDROLOGICAL_MODEL')
        if isinstance(models, (list, tuple)):
            names = [str(m).strip().upper() for m in models if str(m).strip()]
        else:
            names = [m.strip().upper() for m in str(models or '').split(',') if m.strip()]
        return names[0] if len(names) == 1 else None

    def _get_result_extractor(self) -> Optional[Any]:
        """Instantiate the registered result extractor for the model, if any."""
        model = self._result_model_name()
        if not model:
            return None
        try:
            extractor_cls = R.result_extractors.get(model)
        except (ImportError, AttributeError, KeyError) as e:
            self.logger.debug(f"Result extractor for {model} unavailable: {e}")
            return None
        if extractor_cls is None:
            return None
        try:
            extractor = extractor_cls(model)
        except TypeError:
            extractor = extractor_cls()
        try:
            patterns = extractor.get_output_file_patterns().get('streamflow')
        except (AttributeError, TypeError):
            return None
        return extractor if patterns else None

    def _extractor_streamflow_files(self, sim_dir: Path) -> List[Path]:
        """NetCDF files in ``sim_dir`` matching the extractor's streamflow patterns."""
        extractor = self._get_result_extractor()
        if extractor is None or not sim_dir.is_dir():
            return []
        found: List[Path] = []
        for pattern in extractor.get_output_file_patterns()['streamflow']:
            for path in sorted(sim_dir.glob(pattern)):
                if path.suffix == '.nc' and path.is_file() and path not in found:
                    found.append(path)
        return found

    def _extract_with_result_extractor(self, sim_files: List[Path]) -> Optional[pd.Series]:
        """Extract discharge through the model's registered result extractor.

        Used only where the result is verifiably discharge: a NetCDF variable
        from the extractor's streamflow names whose ``units`` attribute is
        volumetric, holding a single series (no unreduced spatial dimension —
        extractors reduce those by taking an arbitrary first element). Routed
        mizuRoute/t-route files are left to the routed-output branches, which
        own outlet/reach selection. Anything else returns None so the
        format-sniffing path runs as before.
        """
        extractor = self._get_result_extractor()
        if extractor is None:
            return None
        model = self._result_model_name()
        patterns = extractor.get_output_file_patterns()['streamflow']
        candidates = [f for f in sim_files if any(fnmatch.fnmatch(f.name, p.split('/')[-1]) for p in patterns)]
        for f in self._extractor_streamflow_files(sim_files[0].parent):
            if f not in candidates:
                candidates.append(f)
        var_names = extractor.get_variable_names('streamflow')
        for path in candidates:
            if path.suffix != '.nc' or self._is_mizuroute_output(path) or self._is_troute_output(path):
                continue
            try:
                with xr.open_dataset(path) as ds:
                    var_name = next((v for v in var_names if v in ds.variables), None)
                    if var_name is None:
                        continue
                    var = ds[var_name]
                    units = classify_streamflow_units(var.attrs.get('units'))
                    spatial = [d for d in var.dims if d != 'time' and var.sizes[d] > 1]
                if not units.is_volumetric:
                    self.logger.debug(
                        f"{model} extractor variable {var_name} has units "
                        f"{units.raw!r} (not volumetric); using format sniffing")
                    continue
                if spatial:
                    self.logger.debug(
                        f"{model} extractor variable {var_name} is distributed over "
                        f"{spatial}; using format sniffing")
                    continue
                series = extractor.extract_variable(path, 'streamflow')
            except (ValueError, KeyError, IndexError, OSError, ModelExecutionError) as e:
                self.logger.debug(f"{model} result extractor failed on {path.name}: {e}")
                continue
            if isinstance(series, pd.Series) and not series.empty:
                self.logger.info(
                    f"Streamflow extracted via {model} result extractor "
                    f"({path.name}:{var_name}, units {units.raw!r})")
                return series.astype(float) * units.factor
        return None

    # ------------------------------------------------------------------
    # results/<exp>_results.csv fallback
    # ------------------------------------------------------------------

    def _results_csv_path(self) -> Path:
        return self.project_dir / 'results' / f"{self.experiment_id}_results.csv"

    def _results_csv_for(self, sim_path: Path) -> Optional[Path]:
        """The results CSV, if it may stand in for output under ``sim_path``.

        The CSV holds the experiment's *own* run, so it is only offered for
        paths inside ``simulations/<experiment_id>``. A calibration worker's
        trial directory must never be scored against it — every trial would
        get the same stale metrics.
        """
        csv_path = self._results_csv_path()
        if not csv_path.is_file():
            return None
        if sim_path.resolve() == csv_path.resolve():
            return csv_path
        own_dir = (self.project_dir / 'simulations' / self.experiment_id).resolve()
        try:
            sim_path.resolve().relative_to(own_dir)
        except ValueError:
            return None
        return csv_path

    def _extract_results_csv_streamflow(self, sim_path: Path) -> Optional[pd.Series]:
        """Read ``<MODEL>_discharge_cms`` from ``results/<exp>_results.csv``."""
        model = self._result_model_name()
        csv_path = self._results_csv_for(sim_path)
        if model is None or csv_path is None:
            return None
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        except (OSError, ValueError, pd.errors.ParserError) as e:
            self.logger.debug(f"Could not read {csv_path}: {e}")
            return None
        for column in (f"{model}_discharge_cms", f"{model}_aggregated_discharge_cms"):
            if column in df.columns:
                series = pd.to_numeric(df[column], errors='coerce').dropna()
                if series.empty:
                    continue
                series.index = pd.DatetimeIndex(series.index)
                self.logger.warning(
                    f"Streamflow taken from {csv_path.name} column {column} "
                    "(no model output file could be read directly)")
                return series
        return None

    def _uses_native_wflow_discharge(self) -> bool:
        model = self._get_config_value(
            lambda: self.config.model.hydrological_model,
            default='', dict_key='HYDROLOGICAL_MODEL')
        spatial_mode = self._get_config_value(
            lambda: self.config.model.wflow.spatial_mode,
            dict_key='WFLOW_SPATIAL_MODE',
        )
        return str(model).upper() == 'WFLOW' and spatial_mode == 'distributed'

    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract streamflow data from simulation files with unit conversion.

        High-level extraction dispatch that:
        1. Detects output format (mizuRoute vs SUMMA)
        2. Calls appropriate extraction method
        3. Returns streamflow as pandas Series (m³/s)

        Data Processing:
        - Detects and corrects units (mass flux → volume flux)
        - Performs spatial aggregation for distributed models
        - Converts to basin-scale discharge

        Args:
            sim_files: List of simulation output NetCDF files
            **kwargs: Additional parameters (unused, for API consistency)

        Returns:
            pd.Series: Time series of streamflow (m³/s)

        Raises:
            Exception: If file cannot be read or no suitable variable found
        """
        sim_file = sim_files[0]
        try:
            if self._uses_native_wflow_discharge():
                # Native daily Q is already m3/s and is labelled by interval
                # end. Shift it to the same day as the calibration worker
                # (WflowResultExtractor); no area conversion or extra routing.
                df = pd.read_csv(sim_file, parse_dates=['time'])
                return pd.Series(df['Q'].values, index=pd.DatetimeIndex(df['time']) - pd.Timedelta(days=1))
            if sim_file.suffix == '.csv' and sim_file.resolve() == self._results_csv_path().resolve():
                csv_series = self._extract_results_csv_streamflow(sim_file)
                if csv_series is None:
                    raise ValueError(
                        f"No {self._result_model_name()}_discharge_cms column in {sim_file.name}")
                return csv_series
            # 1. The model's own registered result extractor, when it yields
            #    verifiably volumetric discharge.
            extracted = self._extract_with_result_extractor(sim_files)
            if extracted is not None:
                return extracted
            # 2. Format sniffing (mizuRoute, t-route, FUSE, VIC, SUMMA-style).
            try:
                return self._extract_by_format(sim_file)
            except (ValueError, KeyError, OSError) as sniff_error:
                # 3. The experiment's results CSV (<MODEL>_discharge_cms).
                csv_series = self._extract_results_csv_streamflow(sim_file)
                if csv_series is not None:
                    self.logger.info(f"Format sniffing failed ({sniff_error}); used results CSV fallback")
                    return csv_series
                raise
        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error extracting streamflow data from {sim_file}: {str(e)}")
            raise

    def _extract_by_format(self, sim_file: Path) -> pd.Series:
        """Dispatch on the detected output format (the pre-extractor path)."""
        if self._is_mizuroute_output(sim_file):
            return self._extract_mizuroute_streamflow(sim_file)
        if self._is_troute_output(sim_file):
            return self._extract_troute_streamflow(sim_file)
        if self._is_fuse_output(sim_file):
            return self._extract_fuse_streamflow(sim_file)
        if self._is_vic_output(sim_file):
            return self._extract_vic_streamflow(sim_file)
        self.logger.debug(f"No routed/FUSE/VIC signature in {sim_file.name}; using SUMMA-style extraction")
        return self._extract_summa_streamflow(sim_file)

    @staticmethod
    def _sum_volumetric_units(var: xr.DataArray, var_name: str) -> pd.Series:
        """Reduce an already-volumetric variable to one series.

        Singleton dimensions are squeezed; per-HRU/GRU discharge contributions
        (e.g. distributed LSTM ``averageRoutedRunoff`` in m3 s-1) are summed —
        never multiplied by an area.
        """
        for dim in [d for d in var.dims if d != 'time' and var.sizes[d] == 1]:
            var = var.squeeze(dim)
        unit_dims = [d for d in var.dims if d in ('hru', 'gru')]
        if unit_dims:
            var = var.sum(dim=unit_dims)
        other = [d for d in var.dims if d != 'time']
        if other:
            raise ValueError(
                f"Volumetric variable '{var_name}' has unreduced dimensions {other}; "
                "cannot pick a single discharge series")
        return cast(pd.Series, var.to_pandas())

    def _is_mizuroute_output(self, sim_file: Path) -> bool:
        """Detect if file contains mizuRoute routed output vs SUMMA direct output.

        Uses presence of mizuRoute-specific variables to distinguish formats:
        - mizuRoute has: IRFroutedRunoff, KWTroutedRunoff, reachID dimensions
        - SUMMA has: averageRoutedRunoff, basin__TotalRunoff, scalarTotalRunoff

        Args:
            sim_file: Path to NetCDF file

        Returns:
            bool: True if mizuRoute output format detected, False otherwise
        """
        try:
            with xr.open_dataset(sim_file) as ds:
                mizuroute_vars = ['IRFroutedRunoff', 'KWTroutedRunoff', 'dlayRunoff', 'reachID', 'seg']
                return any(var in ds.variables or var in ds.dims for var in mizuroute_vars)
        except (OSError, IOError, ValueError):
            return False

    def _is_troute_output(self, sim_file: Path) -> bool:
        """Detect if file contains t-route routed output.

        Uses presence of t-route-specific variables or filename patterns:
        - t-route has: flow, velocity, depth, feature_id/link dimensions
        - Filenames typically contain 'flowveldepth' or 'troute'

        Args:
            sim_file: Path to NetCDF file

        Returns:
            bool: True if t-route output format detected
        """
        # Check filename patterns first
        name_lower = sim_file.name.lower()
        if 'troute' in name_lower or 'flowveldepth' in name_lower or 'nex-troute' in name_lower:
            return True
        try:
            with xr.open_dataset(sim_file) as ds:
                troute_indicators = ['feature_id', 'flowveldepth', 'velocity', 'nudge']
                return any(var in ds.variables or var in ds.dims for var in troute_indicators)
        except (OSError, IOError, ValueError):
            return False

    def _is_vic_output(self, sim_file: Path) -> bool:
        """Detect if file contains VIC model output.

        Args:
            sim_file: Path to NetCDF file

        Returns:
            bool: True if VIC output format detected
        """
        try:
            with xr.open_dataset(sim_file) as ds:
                vic_vars = ['OUT_RUNOFF', 'OUT_BASEFLOW']
                return any(var in ds.variables for var in vic_vars)
        except (OSError, IOError, ValueError):
            return False

    def _is_fuse_output(self, sim_file: Path) -> bool:
        """Detect if file contains FUSE model output.

        Uses presence of FUSE-specific variables to distinguish from SUMMA output.

        Args:
            sim_file: Path to NetCDF file

        Returns:
            bool: True if FUSE output format detected
        """
        try:
            with xr.open_dataset(sim_file) as ds:
                fuse_vars = ['q_routed', 'q_instnt', 'total_discharge']
                return any(var in ds.variables for var in fuse_vars)
        except (OSError, IOError, ValueError):
            return False

    def _extract_fuse_streamflow(self, sim_file: Path) -> pd.Series:
        """Extract streamflow from FUSE model output.

        FUSE outputs routed discharge (q_routed) or instantaneous discharge
        (q_instnt) in mm/day. This method:
        1. Finds the appropriate streamflow variable
        2. For distributed output, identifies the outlet (highest mean discharge)
        3. Converts mm/day to m³/s using catchment area

        Args:
            sim_file: Path to FUSE output NetCDF

        Returns:
            pd.Series: Basin-scale streamflow time series (m³/s)

        Raises:
            ValueError: If no suitable streamflow variable found
        """
        with xr.open_dataset(sim_file) as ds:
            fuse_vars = ['q_routed', 'q_instnt', 'total_discharge', 'runoff']
            for var_name in fuse_vars:
                if var_name not in ds.variables:
                    continue

                var = ds[var_name]
                self.logger.debug(
                    f"Found FUSE streamflow variable {var_name}, "
                    f"dims={var.dims}, shape={var.shape}"
                )

                # Handle spatial dimensions (distributed FUSE output)
                spatial_dims = [d for d in var.dims if d not in ['time']]
                if spatial_dims:
                    # For distributed output, find outlet segment (highest mean Q)
                    # Squeeze singleton dimensions first (e.g., latitude=1)
                    for dim in spatial_dims:
                        if var.sizes[dim] == 1:
                            var = var.squeeze(dim)

                    # Re-check spatial dims after squeezing
                    spatial_dims = [d for d in var.dims if d not in ['time']]
                    if spatial_dims:
                        # Find outlet: segment with highest mean discharge
                        spatial_dim = spatial_dims[0]
                        seg_means = var.mean(dim='time').values
                        outlet_idx = np.argmax(seg_means)
                        self.logger.debug(
                            f"FUSE distributed output: using outlet index {outlet_idx} "
                            f"(mean Q={seg_means[outlet_idx]:.4f})"
                        )
                        var = var.isel({spatial_dim: outlet_idx})

                result = cast(pd.Series, var.to_pandas())

                units = ds[var_name].attrs.get('units', '').strip()
                parsed = classify_streamflow_units(units)
                if parsed.is_volumetric:
                    self.logger.debug(f"FUSE output already volumetric ({units!r})")
                    return result * parsed.factor
                if parsed.is_depth_rate:
                    # depth rate (e.g. mm/day) → m/s → × area → m³/s
                    catchment_area = self._get_catchment_area()
                    result = result * parsed.factor * catchment_area
                    self.logger.debug(
                        f"FUSE output converted from {units!r} to m³/s "
                        f"(area={catchment_area/1e6:.1f} km²)"
                    )
                    return result
                if 'mm' in units and 'timestep' in units:
                    # FUSE's native units string is 'mm timestep-1'; resolve the
                    # timestep length from the time coordinate so sub-daily
                    # output converts correctly too.
                    step_seconds = 86400.0
                    time_coord = ds.get('time')
                    if time_coord is not None and time_coord.sizes.get('time', 0) > 1:
                        import pandas as _pd
                        deltas = _pd.Series(time_coord.values).diff().dropna()
                        if not deltas.empty:
                            step_seconds = float(deltas.mode().iloc[0] / _pd.Timedelta(seconds=1))
                    catchment_area = self._get_catchment_area()
                    result = result * catchment_area / 1000.0 / step_seconds
                    self.logger.debug(
                        f"FUSE output converted from mm/timestep to m³/s "
                        f"(timestep={step_seconds:.0f} s, area={catchment_area/1e6:.1f} km²)"
                    )
                    return result
                if not units and var_name in ('q_routed', 'q_instnt'):
                    # FUSE's native discharge variables are mm/day by definition;
                    # tolerate a missing units attribute for them, but loudly.
                    self.logger.warning(
                        "FUSE variable %s has no units attribute; assuming the "
                        "FUSE-native mm/day convention", var_name,
                    )
                    catchment_area = self._get_catchment_area()
                    return result * catchment_area / 1000.0 / 86400.0
                from symfluence.core.exceptions import EvaluationError
                raise EvaluationError(
                    f"Unrecognized streamflow units {units!r} on FUSE variable "
                    f"'{var_name}' in {sim_file.name}. Expected m³/s or mm/day. "
                    "Set a correct 'units' attribute on the variable — guessing a "
                    "conversion here would silently rescale the simulated flow."
                )

            raise ValueError("No suitable streamflow variable found in FUSE output")

    def _extract_vic_streamflow(self, sim_file: Path) -> pd.Series:
        """Extract streamflow from VIC model output.

        VIC outputs surface runoff (OUT_RUNOFF) and baseflow (OUT_BASEFLOW)
        in mm per output timestep. This method:
        1. Sums runoff + baseflow
        2. Averages over spatial dimensions (for distributed runs)
        3. Converts mm/day to m³/s using catchment area

        Args:
            sim_file: Path to VIC output NetCDF

        Returns:
            pd.Series: Basin-scale streamflow time series (m³/s)

        Raises:
            ValueError: If no runoff variable found
        """
        with xr.open_dataset(sim_file) as ds:
            runoff = None
            baseflow = None

            for var in ['OUT_RUNOFF', 'RUNOFF']:
                if var in ds:
                    runoff = ds[var]
                    break
            for var in ['OUT_BASEFLOW', 'BASEFLOW']:
                if var in ds:
                    baseflow = ds[var]
                    break

            if runoff is None:
                raise ValueError("No runoff variable found in VIC output")

            # Average over spatial dimensions (handles both lumped and distributed)
            spatial_dims = [d for d in runoff.dims if d not in ['time']]
            total_q_mm = runoff.mean(dim=spatial_dims)
            if baseflow is not None:
                total_q_mm = total_q_mm + baseflow.mean(dim=spatial_dims)

            # Convert mm/day to m³/s: mm/day × area_m² / 1000 / 86400
            catchment_area = self._get_catchment_area()  # m²
            total_q_m3s = total_q_mm * catchment_area / 1000.0 / 86400.0

            self.logger.debug(
                f"VIC streamflow: mean={float(total_q_m3s.mean()):.3f} m³/s, "
                f"catchment area={catchment_area/1e6:.1f} km²"
            )

            return cast(pd.Series, total_q_m3s.to_pandas())

    def _extract_mizuroute_streamflow(self, sim_file: Path) -> pd.Series:
        """Extract streamflow from mizuRoute routed output.

        mizuRoute provides runoff values for each reach in the river network.
        This method:
        1. Tries multiple routed runoff variables (IRF, KWT, averaged)
        2. Identifies outlet reach/segment (highest mean discharge)
        3. Returns time series at outlet

        mizuRoute Output Variables:
        - IRFroutedRunoff: Impulse Response Function routing
        - KWTroutedRunoff: Kinematic Wave Theory routing
        - averageRoutedRunoff: Combined routing methods
        - dlayRunoff: Delayed runoff (older versions)

        Dimensions:
        - seg: River segment ID (older mizuRoute)
        - reachID: Reach ID (newer versions)

        Args:
            sim_file: Path to mizuRoute output NetCDF

        Returns:
            pd.Series: Time series of routed discharge (m³/s)

        Raises:
            ValueError: If no routed runoff variable found in file
        """
        with xr.open_dataset(sim_file) as ds:
            streamflow_vars = ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff', 'dlayRunoff']
            # One reach rule everywhere (core.modeling.utilities.reach_selection):
            # SIM_REACH_ID, else the topology outlet, else highest mean flow.
            from symfluence.core.modeling.utilities.reach_selection import (
                select_outlet_series,
                topology_candidates,
            )
            requested_reach = self._get_config_value(
                lambda: self.config.evaluation.sim_reach_id,
                default=None, dict_key='SIM_REACH_ID',
            )
            topology_name = self._get_config_value(
                lambda: self.config.model.mizuroute.topology,
                dict_key='SETTINGS_MIZU_TOPOLOGY',
            )
            for var_name in streamflow_vars:
                if var_name in ds.variables:
                    var = ds[var_name]
                    if not any(d in var.dims for d in ('seg', 'reachID')):
                        continue
                    return select_outlet_series(
                        ds, var, requested_reach,
                        topology_candidates(self.project_dir, topology_name, near=sim_file),
                        self.logger,
                    )
            raise ValueError("No suitable streamflow variable found in mizuRoute output")

    def _extract_troute_streamflow(self, sim_file: Path) -> pd.Series:
        """Extract streamflow from t-route routed output.

        t-route outputs flowveldepth files containing flow, velocity, and depth
        per feature (reach). This method:
        1. Tries multiple flow variable names
        2. Identifies outlet reach (highest mean discharge)
        3. Returns time series at outlet

        Args:
            sim_file: Path to t-route output NetCDF

        Returns:
            pd.Series: Time series of routed discharge (m3/s)

        Raises:
            ValueError: If no flow variable found in file
        """
        with xr.open_dataset(sim_file) as ds:
            flow_vars = ['flow', 'streamflow', 'q_lateral', 'discharge']
            for var_name in flow_vars:
                if var_name in ds.variables:
                    var = ds[var_name]
                    # Find spatial dimension (feature_id, link, or segment)
                    spatial_dim = None
                    for dim in ['feature_id', 'link', 'segment', 'reach']:
                        if dim in var.dims:
                            spatial_dim = dim
                            break
                    if spatial_dim:
                        segment_means = var.mean(dim='time').values
                        outlet_idx = np.argmax(segment_means)
                        result = cast(pd.Series, var.isel({spatial_dim: outlet_idx}).to_pandas())
                    else:
                        # Single-segment or already scalar
                        result = cast(pd.Series, var.to_pandas())
                    return result
            raise ValueError("No suitable flow variable found in t-route output")

    def _extract_summa_streamflow(self, sim_file: Path) -> pd.Series:
        """Extract streamflow from SUMMA model output with comprehensive processing.

        SUMMA outputs runoff at HRU/GRU level. This method:
        1. Finds streamflow variable (tries multiple common names)
        2. Detects and converts units (mass flux → volume flux)
        3. Performs spatial aggregation (area-weighted if possible)
        4. Scales to basin discharge using catchment area

        Unit Conversion Logic:
            SUMMA may output runoff in mass flux (kg m⁻² s⁻¹) incorrectly
            labeled as volume flux (m s⁻¹). Detection strategy:
            - Check units attribute for 'kg' (explicit mass flux)
            - Check data magnitude: if mean > 1e-6 m/s, likely mislabeled
              (realistic runoff typically 0.01-0.1 mm/day ≈ 10^-10 to 10^-9 m/s)
            - If mass flux detected: divide by 1000 (water density)

        Spatial Aggregation Strategy:
            If SUMMA attributes.nc exists with HRU/GRU areas:
            - Area-weighted sum: discharge = Σ(runoff_i × area_i)
            - Preserves spatial heterogeneity in calibration
            Fallback (if no attributes or error):
            - Select first HRU/GRU (triggers warning)
            - Not ideal for multi-unit basins

        Final Scaling:
            - Multiply per-unit runoff by catchment area (m²)
            - Result: basin-scale discharge (m³/s)

        SUMMA Variable Names (tried in order):
            - averageRoutedRunoff (preferred)
            - basin__TotalRunoff (newer versions)
            - scalarTotalRunoff (older versions)

        Args:
            sim_file: Path to SUMMA output NetCDF

        Returns:
            pd.Series: Basin-scale streamflow time series (m³/s)

        Raises:
            ValueError: If no suitable streamflow variable found
        """
        with xr.open_dataset(sim_file) as ds:
            # Check for pre-computed streamflow first (e.g., HBV, GR models output in m³/s)
            if 'streamflow' in ds.variables:
                var = ds['streamflow']
                units = var.attrs.get('units', 'unknown')
                self.logger.debug(f"Found pre-computed streamflow variable with units: '{units}'")
                # Already volumetric (any CF spelling): no area conversion.
                parsed = classify_streamflow_units(units)
                if parsed.is_volumetric:
                    return self._sum_volumetric_units(var, 'streamflow') * parsed.factor
                # Otherwise fall through to standard processing

            streamflow_vars = ['averageRoutedRunoff', 'basin__TotalRunoff', 'scalarTotalRunoff']
            for var_name in streamflow_vars:
                if var_name in ds.variables:
                    var = ds[var_name]

                    units = var.attrs.get('units', 'unknown')
                    self.logger.debug(f"Found streamflow variable {var_name} with units: '{units}'")

                    # Honour a volumetric units attribute (e.g. distributed LSTM
                    # writes per-GRU averageRoutedRunoff in m3 s-1): sum the
                    # contributions; the mass-flux heuristic and the area
                    # multiplication below apply to depth rates only.
                    parsed_units = classify_streamflow_units(units)
                    if parsed_units.is_volumetric:
                        self.logger.debug(f"{var_name} is volumetric ({units!r}); no area conversion")
                        return self._sum_volumetric_units(var, var_name) * parsed_units.factor
                    if parsed_units.is_depth_rate and 'kg' not in str(units) and parsed_units.factor != 1.0:
                        # e.g. mm/day or mm/h → m/s before the area conversion
                        self.logger.debug(f"Converting {var_name} from {units!r} to m/s")
                        var = var * parsed_units.factor
                        var.attrs['units'] = 'm s-1'

                    # Unit conversion: Mass flux (kg m-2 s-1) to Volume flux (m s-1)
                    # Uses centralized threshold from UnitConverter for consistency.
                    #
                    # Check for explicit 'kg' units OR unreasonably high values which indicates
                    # the value is likely mass flux but mislabeled as m/s.
                    #
                    # Physical reasoning for threshold (see UnitConverter.MASS_FLUX_THRESHOLD):
                    #   - 1e-6 m/s = 86.4 mm/day of runoff (extremely high)
                    #   - Typical mean runoff: 0.1-5 mm/day = 1e-9 to 6e-8 m/s
                    #   - Values exceeding threshold are almost certainly mass flux

                    is_mass_flux = False
                    if 'units' in var.attrs and 'kg' in var.attrs['units'] and 's-1' in var.attrs['units']:
                        is_mass_flux = True
                    elif float(var.mean().item()) > UnitConverter.MASS_FLUX_THRESHOLD:
                        self.logger.debug(
                            f"Variable {var_name} mean ({float(var.mean().item()):.2e}) exceeds "
                            f"threshold {UnitConverter.MASS_FLUX_THRESHOLD:.0e} m/s. "
                            "Assuming mislabeled mass flux."
                        )
                        is_mass_flux = True

                    if is_mass_flux:
                        self.logger.debug(
                            f"Converting {var_name} from mass flux to volume flux (dividing by 1000)"
                        )
                        var = var / 1000.0  # Divide by density of water

                    # Check if we need spatial aggregation
                    if len(var.shape) > 1 and any(d in var.dims for d in ['hru', 'gru']):
                        try:
                            # Try area-weighted aggregation first
                            attrs_file = self.project_dir / 'settings' / 'SUMMA' / 'attributes.nc'
                            if attrs_file.exists():
                                with xr.open_dataset(attrs_file) as attrs:
                                    # HRU areas, or GRU areas (HRUarea summed by
                                    # hru2gruId when nHRU != nGRU) -- shared rule.
                                    from symfluence.core.modeling.utilities.area_weighting import (
                                        area_weighted_discharge,
                                    )
                                    weighted_runoff = area_weighted_discharge(var, attrs)
                                    if weighted_runoff is not None:
                                        self.logger.debug(f"Performing area-weighted aggregation for {var_name}")
                                        return cast(pd.Series, weighted_runoff.to_pandas())

                        except Exception as e:  # noqa: BLE001 — must-not-raise contract
                            self.logger.warning(f"Failed to perform area-weighted aggregation: {e}")

                    # Fallback to selection (original logic)
                    if len(var.shape) > 1:
                        self.logger.warning(f"Using first spatial unit for {var_name} (potential error for multi-unit basins)")
                        if 'hru' in var.dims:
                            sim_data = cast(pd.Series, var.isel(hru=0).to_pandas())
                        elif 'gru' in var.dims:
                            sim_data = cast(pd.Series, var.isel(gru=0).to_pandas())
                        else:
                            non_time_dims = [dim for dim in var.dims if dim != 'time']
                            if non_time_dims:
                                sim_data = cast(pd.Series, var.isel({non_time_dims[0]: 0}).to_pandas())
                            else:
                                sim_data = cast(pd.Series, var.to_pandas())
                    else:
                        sim_data = cast(pd.Series, var.to_pandas())

                    catchment_area = self._get_catchment_area()
                    return sim_data * catchment_area
            raise ValueError("No suitable streamflow variable found in SUMMA output")

    # Catchment area validation bounds (m²)
    MIN_CATCHMENT_AREA = 1e3    # 0.001 km² - smallest reasonable catchment
    MAX_CATCHMENT_AREA = 1e12   # 1,000,000 km² - larger than any river basin

    @staticmethod
    def _ordered_area_shapefiles(directory: "Path") -> "List[Path]":
        """Deterministic, artifact-aware ordering of area shapefiles.

        Area detection used ``list(dir.glob("*.shp"))[0]``, whose order is
        filesystem-dependent — the same domain resolved
        ``..._riverBasins_lumped.shp`` (correct area) on one machine and a
        leftover ``..._riverBasins_lumped_temp.shp`` (a degenerate 2-feature
        scratch file, ~0.9 m²) on another, silently corrupting every
        discharge on the second. Order candidates instead of trusting glob:

        - drop obvious scratch artifacts (``*_temp.shp``, ``*.tmp.shp``);
        - sort the rest so the shortest, canonical name wins (``_lumped``
          before ``_lumped_temp``), ties broken lexically for stability.

        A caller that walks this list and validates each candidate's area is
        robust to a stray file that would otherwise be picked first — the key
        change from taking ``[0]`` blindly.
        """
        if not directory.exists():
            return []
        scratch = ("_temp", ".tmp")
        files = [
            f for f in directory.glob("*.shp")
            if not any(tok in f.stem.lower() for tok in scratch)
        ]
        # Include scratch files only as a last resort, never ahead of a real one.
        deferred = [
            f for f in directory.glob("*.shp")
            if any(tok in f.stem.lower() for tok in scratch)
        ]
        return sorted(files, key=lambda p: (len(p.stem), p.stem)) + \
            sorted(deferred, key=lambda p: (len(p.stem), p.stem))

    def _validate_catchment_area(
        self,
        area_m2: float,
        source: str
    ) -> Optional[float]:
        """Validate catchment area is within reasonable bounds.

        Checks that detected/calculated catchment area falls within
        physically plausible bounds.

        Validation Bounds:
            - Minimum: 1e3 m² (0.001 km²) - smallest reasonable catchment
            - Maximum: 1e12 m² (1,000,000 km²) - larger than Amazon basin

        Args:
            area_m2: Catchment area in square meters
            source: Description of where area came from (for logging)

        Returns:
            float if valid, None if invalid
        """
        if area_m2 <= 0:
            self.logger.warning(
                f"Invalid catchment area from {source}: {area_m2} m² (non-positive)"
            )
            return None

        if area_m2 < self.MIN_CATCHMENT_AREA:
            self.logger.warning(
                f"Invalid catchment area from {source}: {area_m2} m² "
                f"(< minimum {self.MIN_CATCHMENT_AREA} m²)"
            )
            return None

        if area_m2 > self.MAX_CATCHMENT_AREA:
            self.logger.warning(
                f"Invalid catchment area from {source}: {area_m2} m² "
                f"(> maximum {self.MAX_CATCHMENT_AREA} m²)"
            )
            return None

        return area_m2

    def _get_catchment_area(self) -> float:
        """Determine catchment area with multi-source fallback strategy.

        Catchment area is essential for converting per-unit-area runoff (m/s)
        to basin-scale discharge (m³/s): Q = runoff × area

        Resolution Priority (highest to lowest reliability):
        1. FIXED_CATCHMENT_AREA in config
           - User override, skips all detection
           - Use when auto-detection fails or for forcing specific values
        2. SUMMA attributes.nc: HRUarea sum
           - Most reliable: directly from model setup
           - Sum of all HRU areas (distributed or lumped)
           - Requires: settings/SUMMA/attributes.nc exists with HRUarea variable
        3. River basin shapefile: GRU_area column
           - User-defined basin delineation (e.g., from QGIS)
           - Searches shapefiles/river_basins/*.shp for GRU_area column
           - Fallback: calculate from geometry area (projects to UTM)
        4. Catchment shapefile: geometry area
           - Alternative delineation layer
           - Searches shapefiles/catchment/*.shp
           - Attempts HRU_area column first, then geometry
        5. Default: 1 km² (1e6 m²) or error if REQUIRE_EXPLICIT_CATCHMENT_AREA=True

        Validation:
            All areas validated via _validate_catchment_area() for bounds checking.

        Configuration:
            REQUIRE_EXPLICIT_CATCHMENT_AREA: If True, raise error when auto-detection fails
                                             (default: False)

        Returns:
            float: Catchment area in square meters (m²)

        Raises:
            ValueError: If auto-detection fails and REQUIRE_EXPLICIT_CATCHMENT_AREA=True
        """

        # Priority 0: Manual override from config
        fixed_area = self._get_config_value(
            lambda: None,
            default=None,
            dict_key='FIXED_CATCHMENT_AREA'
        )
        if fixed_area:
            validated = self._validate_catchment_area(float(fixed_area), 'config')
            if validated:
                self.logger.info(f"Using fixed catchment area from config: {validated} m²")
                return validated
            # Fall through to other methods if fixed area is invalid

        # Priority 1: SUMMA attributes file — model-specific, so only consult
        # it when SUMMA is actually the configured model (most reliable there;
        # other models must not read another model's settings files).
        active_model = str(self._get_config_value(
            lambda: self.config.model.hydrological_model,
            default='', dict_key='HYDROLOGICAL_MODEL'
        ) or '').upper()
        try:
            attrs_file = self.project_dir / 'settings' / 'SUMMA' / 'attributes.nc'
            if active_model == 'SUMMA' and attrs_file.exists():
                with xr.open_dataset(attrs_file) as attrs:
                    if 'HRUarea' in attrs.data_vars:
                        catchment_area_m2 = float(attrs['HRUarea'].values.sum())
                        validated = self._validate_catchment_area(
                            catchment_area_m2, 'SUMMA attributes.nc'
                        )
                        if validated:
                            self.logger.debug(
                                f"Using catchment area from SUMMA attributes: {validated:.0f} m²"
                            )
                            return validated

        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            self.logger.warning(f"Error reading SUMMA attributes file: {str(e)}")

        # Priority 2: Try basin shapefile
        try:
            import geopandas as gpd
            basin_path = self.project_dir / "shapefiles" / "river_basins"
            area_col = self._get_config_value(
                lambda: self.config.paths.river_basin_area,
                dict_key='RIVER_BASIN_SHP_AREA',
            )
            for basin_file in self._ordered_area_shapefiles(basin_path):
                gdf = gpd.read_file(basin_file)
                if area_col in gdf.columns:
                    total_area = gdf[area_col].sum()
                    validated = self._validate_catchment_area(
                        total_area, f'basin shapefile {basin_file.name} ({area_col})'
                    )
                    if validated:
                        self.logger.info(
                            f"Using catchment area from basin shapefile "
                            f"{basin_file.name}: {validated:.0f} m²"
                        )
                        return validated
                # Fallback: calculate from geometry
                if gdf.crs and gdf.crs.is_geographic:
                    # Use geopandas estimate_utm_crs() for automatic hemisphere detection
                    utm_crs = gdf.estimate_utm_crs()
                    gdf = gdf.to_crs(utm_crs)
                geom_area = gdf.geometry.area.sum()
                validated = self._validate_catchment_area(
                    geom_area, f'basin shapefile {basin_file.name} (geometry)'
                )
                if validated:
                    self.logger.info(
                        f"Using catchment area from basin geometry "
                        f"{basin_file.name}: {validated:.0f} m²"
                    )
                    return validated
                # This candidate did not yield a plausible area — try the next
                # one rather than giving up on the whole river_basins directory.
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            self.logger.warning(
                f"Could not calculate catchment area from basin shapefile: {str(e)}"
            )

        # Priority 3: Try catchment shapefile
        try:
            import geopandas as gpd
            # Resolve the nested discretized catchment (river_basins already
            # handled in priority 2).
            catchment_shp = find_basin_shapefile(
                self.project_dir / "shapefiles",
                self.domain_name,
                self.domain_definition_method,
                self.experiment_id,
                include_river_basins=False,
            )
            if catchment_shp is not None:
                gdf = gpd.read_file(catchment_shp)
                area_col = self._get_config_value(
                    lambda: self.config.paths.catchment_area,
                    dict_key='CATCHMENT_SHP_AREA',
                )
                if area_col in gdf.columns:
                    total_area = gdf[area_col].sum()
                    validated = self._validate_catchment_area(
                        total_area, f'catchment shapefile ({area_col})'
                    )
                    if validated:
                        self.logger.info(
                            f"Using catchment area from catchment shapefile: {validated:.0f} m²"
                        )
                        return validated
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            self.logger.warning(f"Error reading catchment shapefile: {str(e)}")

        # Fallback: Check if explicit area is required
        require_explicit = self._get_config_value(
            lambda: None,
            default=False,
            dict_key='REQUIRE_EXPLICIT_CATCHMENT_AREA'
        )

        if require_explicit:
            raise ValueError(
                "Catchment area auto-detection failed and REQUIRE_EXPLICIT_CATCHMENT_AREA=True. "
                "Set FIXED_CATCHMENT_AREA in config to specify the catchment area manually."
            )

        self.logger.warning(
            "Using default catchment area: 1,000,000 m² (1 km²). "
            "This may cause incorrect discharge calculations. "
            "Set FIXED_CATCHMENT_AREA in config or enable REQUIRE_EXPLICIT_CATCHMENT_AREA "
            "to enforce explicit specification."
        )
        return 1e6  # 1 km² fallback

    def get_observed_data_path(self) -> Path:
        """Get path to observed streamflow data file.

        Resolves observed streamflow from configuration or default location.

        Path Resolution:
        1. If OBSERVATIONS_PATH configured and not 'default': use it directly
        2. Otherwise: use default pattern:
           observations/streamflow/preprocessed/{domain_name}_streamflow_processed.csv

        The preprocessed CSV should contain:
        - Time index (datetime): observation timestamps
        - Streamflow column: discharge values (typically m³/s)
        - May have multiple columns; column selected via _get_observed_data_column()

        Returns:
            Path: Path to observed streamflow CSV file
        """
        # Check model-ready observations store first
        model_ready_obs = (
            self.project_dir / 'data' / 'model_ready' / 'observations'
            / f'{self.domain_name}_observations.nc'
        )
        if model_ready_obs.exists():
            return model_ready_obs

        obs_path = self._get_config_value(
            lambda: self.config.paths.observations_path,
            default=None,
            dict_key='OBSERVATIONS_PATH'
        )
        if obs_path == 'default' or not obs_path:
            return first_existing_path(
                streamflow_observation_candidates(self.project_dir, self.domain_name)
            )
        return Path(obs_path)

    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        """Find streamflow column in observed data CSV.

        Searches CSV columns for streamflow using heuristic pattern matching.
        This accommodates various naming conventions across different data sources.

        Matched Terms (case-insensitive):
        - 'flow': discharge, streamflow, river_flow
        - 'discharge': q_discharge, discharge
        - 'q_': q_mm, q_cms, q_observed (standardized naming)
        - 'streamflow': streamflow, daily_streamflow

        Args:
            columns: List of column names from observed data CSV

        Returns:
            Optional[str]: Matched column name if found, None otherwise

        Note:
            Returns first matching column found. If multiple candidates exist,
            consider manual specification in config or preprocessing step.
        """
        for col in columns:
            if any(term in col.lower() for term in ['flow', 'discharge', 'q_', 'streamflow']):
                return col
        return None

    def needs_routing(self) -> bool:
        """Determine if streamflow calibration requires mizuRoute routing.

        mizuRoute routing is needed when:
        1. Domain is semi-distributed or distributed (not lumped/point)
        2. Domain is lumped but with river network delineation

        Decision Logic:
            - If domain_method in [point, lumped]:
              * If routing_delineation == 'river_network': NEEDS ROUTING
              * Otherwise: DOESN'T NEED ROUTING
            - If domain_method in [semi-distributed, distributed]:
              * ALWAYS NEEDS ROUTING

        Why Routing?:
            - Lumped domains: SUMMA outputs lumped runoff to basin outlet
            - Distributed domains: runoff generated per HRU, must route through network
            - Semi-distributed: mixed; routing needed for proper streamflow timing

        Returns:
            bool: True if mizuRoute routing required, False if direct SUMMA sufficient

        Configuration:
            - domain.definition_method (from config): 'point', 'lumped', etc.
            - ROUTING_DELINEATION (from config): 'lumped' or 'river_network'
        """
        domain_method = self._get_config_value(
            lambda: self.config.domain.definition_method,
            default='lumped',
            dict_key='DOMAIN_DEFINITION_METHOD'
        )
        routing_delineation = self._get_config_value(
            lambda: self.config.domain.delineation.routing,
            dict_key='ROUTING_DELINEATION',
        )

        if domain_method not in ['point', 'lumped']:
            return True
        if domain_method == 'lumped' and routing_delineation == 'river_network':
            return True
        return False
