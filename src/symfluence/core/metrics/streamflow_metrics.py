# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Streamflow Metrics Utility

Shared utility for streamflow metric calculation across all optimization workers.
Consolidates duplicate implementations from GR, FUSE, SUMMA, MESH, HYPE, and NGEN workers.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd

from symfluence.core.constants import ModelDefaults, UnitConversion
from symfluence.core.metrics import kge, kge_prime, mae, nse, rmse
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.core.path_resolver import find_basin_shapefile

logger = logging.getLogger(__name__)


class StreamflowMetrics:
    """
    Shared utility for streamflow metric calculation across all workers.

    Provides standardized methods for:
    - Loading observation data
    - Getting catchment area from various sources (shapefile, NetCDF, text)
    - Converting runoff units to discharge
    - Aligning time series
    - Calculating performance metrics
    """

    def __init__(
        self,
        penalty_score: float = ModelDefaults.PENALTY_SCORE,
        box_cox_lambda: float = 0.2
    ):
        """
        Initialize StreamflowMetrics.

        Args:
            penalty_score: Value to return when metric calculation fails
            box_cox_lambda: Lambda for the Box-Cox flow transformation used by
                *_box_cox metrics. 0 falls back to the log transform
                (the Box-Cox lambda=0 limit).
        """
        self.penalty_score = penalty_score
        self.box_cox_lambda = box_cox_lambda

    def load_observations(
        self,
        config: Dict[str, Any],
        project_dir: Path,
        domain_name: str,
        resample_freq: Optional[str] = 'D',
        discharge_col: str = 'discharge_cms'
    ) -> Tuple[Optional[np.ndarray], Optional[pd.DatetimeIndex]]:
        """
        Load observation data from standard location.

        Args:
            config: Configuration dictionary
            project_dir: Project directory path
            domain_name: Domain name for path construction
            resample_freq: Resampling frequency (e.g., 'D' for daily, None for no resampling)
            discharge_col: Name of discharge column in CSV

        Returns:
            Tuple of (values, datetime_index), or (None, None) on error
        """
        try:
            # Strategy 0: model-ready observations store (grouped NetCDF).
            # Mirrors StreamflowEvaluator.get_observed_data_path(), which already
            # prefers the store — keeps the two streamflow loaders consistent.
            # Any miss falls through to the legacy preprocessed-CSV path below.
            store_path = (
                project_dir / 'data' / 'model_ready' / 'observations'
                / f'{domain_name}_observations.nc'
            )
            if store_path.exists():
                store_series = self._read_observations_store(store_path, domain_name, discharge_col)
                if store_series is not None and not store_series.empty:
                    if resample_freq:
                        store_series = store_series.resample(resample_freq).mean()
                    return np.asarray(store_series.values), cast(pd.DatetimeIndex, store_series.index)

            obs_file = resolve_config_value(config, lambda: config.paths.observations_path, 'default', 'OBSERVATIONS_PATH')
            if obs_file == 'default' or not obs_file:
                obs_file = resolve_data_subdir(project_dir, 'observations') / 'streamflow' / 'preprocessed' / f"{domain_name}_streamflow_processed.csv"
            else:
                obs_file = Path(obs_file)

            if not obs_file.exists():
                logger.error(f"Observations file not found: {obs_file}")
                return None, None

            # Read CSV with datetime index
            df_obs = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)

            # Get discharge column
            if discharge_col not in df_obs.columns:
                # Try to find a suitable column
                discharge_cols = [c for c in df_obs.columns if 'discharge' in c.lower() or 'flow' in c.lower()]
                if discharge_cols:
                    discharge_col = discharge_cols[0]
                    logger.debug(f"Using discharge column: {discharge_col}")
                else:
                    logger.error(f"No discharge column found in {obs_file}")
                    return None, None

            observed = df_obs[discharge_col]

            # Resample if requested
            if resample_freq:
                observed = observed.resample(resample_freq).mean()

            return np.asarray(observed.values), cast(pd.DatetimeIndex, observed.index)

        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"Error loading observations: {e}")
            return None, None

    def _read_observations_store(
        self,
        store_path: Path,
        domain_name: str,
        discharge_col: str = 'discharge_cms',
    ) -> Optional[pd.Series]:
        """Read the streamflow series from the model-ready observations store.

        Opens the ``streamflow`` group of the grouped NetCDF written by
        ``ObservationsNetCDFBuilder`` and returns a datetime-indexed Series in
        m3 s-1, or ``None`` when the group/variable is missing or unreadable.
        Never raises — callers fall back to the legacy CSV on ``None``.

        Args:
            store_path: Path to ``{domain}_observations.nc``.
            domain_name: Used to pick the matching gauge when multi-gauge.
            discharge_col: Preferred data-variable name (fuzzy fallback applied).
        """
        try:
            import xarray as xr

            with xr.open_dataset(store_path, group='streamflow') as ds:
                if not ds.data_vars:
                    return None

                # Pick the discharge variable (exact, then fuzzy — mirrors the CSV path).
                if discharge_col in ds.data_vars:
                    var = discharge_col
                else:
                    cands = [v for v in ds.data_vars
                             if 'discharge' in str(v).lower() or 'flow' in str(v).lower()]
                    if not cands:
                        return None
                    var = cands[0]

                da = ds[var]
                if 'time' not in da.dims:
                    return None

                # Collapse the spatial (gauge) dim to a single series: prefer the
                # gauge whose id matches the domain, else the first.
                spatial = [d for d in da.dims if d != 'time']
                if spatial:
                    dim = spatial[0]
                    idx = 0
                    id_var = next(
                        (c for c in ('gauge_id', f'{dim}_id', dim) if c in ds.variables),
                        None,
                    )
                    if id_var is not None:
                        ids = [str(x) for x in np.atleast_1d(ds[id_var].values)]
                        if domain_name in ids:
                            idx = ids.index(domain_name)
                    da = da.isel({dim: idx})

                # xarray decodes time -> datetime64 and masks _FillValue (-9999.0) to NaN.
                series = da.to_series().dropna()
                return series if not series.empty else None

        except Exception as e:  # noqa: BLE001 — must-not-raise; CSV fallback follows
            logger.debug(f"Could not read observations store {store_path}: {e}")
            return None

    def get_catchment_area(
        self,
        config: Dict[str, Any],
        project_dir: Path,
        domain_name: str,
        source: str = 'shapefile',
        settings_dir: Optional[Path] = None,
        default_area: float = 1.0
    ) -> float:
        """
        Get catchment area in km2 from various sources.

        FIXED_CATCHMENT_AREA (m²) always wins when set. The last-resort
        default is 1 km² and must stay identical to the
        StreamflowEvaluator fallback: trial metrics and the final
        evaluation convert units with this value, and any disagreement
        makes their scores incomparable.

        Args:
            config: Configuration dictionary
            project_dir: Project directory path
            domain_name: Domain name
            source: Source type: 'shapefile', 'netcdf', 'geodata'
            settings_dir: Settings directory (required for 'netcdf' and 'geodata' sources)
            default_area: Default area to return on error (km2)

        Returns:
            Catchment area in km2
        """
        try:
            fixed_area_m2 = config.get('FIXED_CATCHMENT_AREA')
            if fixed_area_m2:
                area_km2 = float(fixed_area_m2) / 1e6
                logger.debug(f"Using fixed catchment area from config: {area_km2:.2f} km2")
                return area_km2

            if source == 'shapefile':
                return self._get_area_from_shapefile(config, project_dir, domain_name, default_area)
            elif source == 'netcdf':
                return self._get_area_from_netcdf(settings_dir, domain_name, default_area)
            elif source == 'geodata':
                return self._get_area_from_geodata(settings_dir, default_area)
            else:
                logger.warning(f"Unknown area source: {source}. Using default {default_area} km2")
                return default_area
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.warning(f"Error getting catchment area from {source}: {e}. Using default {default_area} km2")
            return default_area

    @staticmethod
    def _projected_area_km2(gdf: Any) -> Optional[float]:
        """Area (km2) of a layer's geometry, projected off any geographic CRS."""
        try:
            projected = gdf if (gdf.crs is not None and not gdf.crs.is_geographic) \
                else gdf.to_crs(gdf.estimate_utm_crs())
            area = float(projected.geometry.area.sum()) / 1e6
            return area if area > 0 else None
        except (AttributeError, ValueError, TypeError):
            return None

    @staticmethod
    def _reconcile_with_geometry(
        column_km2: float,
        geometry_km2: Optional[float],
        path: Path,
    ) -> float:
        """Prefer the geometry when an area column disagrees with it wildly.

        An area column is only trustworthy if it was written in m². It is not
        always: the point delineator stored ``polygon.area`` straight off an
        EPSG:4326 geometry, so paradise_snotel_wa carried 0.0004 square
        degrees where consumers expected m². Divided by 1e6 that reads as
        4e-10 km² for a 3.39 km² domain, and discharge conversion silently
        fell back to the 1 km² default.

        Geometry is self-describing — it carries a CRS — so it is the better
        arbiter when the two disagree by more than an order of magnitude.
        A modest disagreement is left alone: HRU columns legitimately differ
        from the dissolved outline by small amounts.
        """
        if geometry_km2 is None or column_km2 <= 0:
            return column_km2 if column_km2 > 0 else (geometry_km2 or 0.0)

        ratio = column_km2 / geometry_km2
        if 0.1 < ratio < 10.0:
            return column_km2

        logger.warning(
            f"Area column in {path.name} gives {column_km2:.6g} km2 but its geometry "
            f"gives {geometry_km2:.6g} km2 ({ratio:.3g}x apart) — using the geometry. "
            f"A column written in the wrong units (square degrees rather than m2) "
            f"looks exactly like this."
        )
        return geometry_km2

    def _area_from_river_basin(
        self,
        project_dir: Path,
        domain_name: str,
    ) -> Optional[float]:
        """Catchment area (km2) from the delineated river-basin shapefile.

        Returns None when the shapefile is absent or carries no usable area,
        so callers fall through to the catchment/HRU lookup.
        """
        try:
            import geopandas as gpd

            basin_dir = project_dir / 'shapefiles' / 'river_basins'
            if not basin_dir.exists():
                return None

            candidates = sorted(basin_dir.glob('*.shp')) or sorted(basin_dir.glob('*.gpkg'))
            preferred = [p for p in candidates if domain_name in p.name]
            for path in (preferred or candidates):
                gdf = gpd.read_file(path)
                geometry_km2 = self._projected_area_km2(gdf)
                if 'GRU_area' in gdf.columns:
                    if 'GRU_ID' in gdf.columns and gdf['GRU_ID'].nunique() < len(gdf):
                        area_km2 = float(gdf.drop_duplicates('GRU_ID')['GRU_area'].sum()) / 1e6
                    else:
                        area_km2 = float(gdf['GRU_area'].sum()) / 1e6
                    area_km2 = self._reconcile_with_geometry(area_km2, geometry_km2, path)
                else:
                    area_km2 = geometry_km2 or 0.0
                if area_km2 > 0:
                    logger.debug(
                        f"Catchment area from river-basin shapefile {path.name}: "
                        f"{area_km2:.4f} km2"
                    )
                    return area_km2
        except (ImportError, OSError, RuntimeError, ValueError, KeyError,
                AttributeError, TypeError) as e:
            # Named rather than bare: a missing/unreadable layer, an absent
            # column or an unprojectable CRS should fall through to the
            # catchment lookup, not abort scoring.
            logger.debug(f"River-basin area lookup failed ({e}); using the catchment shapefile")
        return None

    def _get_area_from_shapefile(
        self,
        config: Dict[str, Any],
        project_dir: Path,
        domain_name: str,
        default_area: float
    ) -> float:
        """Get catchment area from shapefile/geopackage."""
        import geopandas as gpd

        # Priority 0: the river-basin shapefile written by define_domain.
        #
        # This is the delineated basin and the same source
        # StreamflowEvaluator._get_catchment_area() resolves, so preferring it
        # keeps the calibration objective and the final evaluation converting
        # mm/day -> m3/s with one area. They previously did not: the catchment
        # lookup below resolves an experiment-scoped HRU shapefile, and on the
        # P3 Bow-at-Banff domain 108 of 130 of those carried a stale
        # 2248.0606 km2 against the delineation's 2207.5038 km2 — a 1.837%
        # scaling on every simulated discharge, which showed up as FUSE
        # reporting a calibration KGE of 0.909638 for a run the final
        # evaluation scored at 0.904377.
        #
        # The delineation is also self-checking: its GRU_area column and its
        # recomputed geometry agree, which the stale HRU files do not.
        basin_area = self._area_from_river_basin(project_dir, domain_name)
        if basin_area is not None:
            return basin_area

        # Resolve via the shared finder: handles the nested experiment-scoped
        # layout, the GRUs/GRUS casing drift, cross-experiment shapefiles
        # (preprocessing often writes under run_1), and explicit
        # CATCHMENT_PATH / CATCHMENT_SHP_NAME overrides.
        catchment_file = find_basin_shapefile(
            project_dir / 'shapefiles',
            domain_name,
            resolve_config_value(config, lambda: config.domain.definition_method, 'lumped', 'DOMAIN_DEFINITION_METHOD'),
            resolve_config_value(config, lambda: config.domain.experiment_id, 'run_1', 'EXPERIMENT_ID'),
            explicit_path=resolve_config_value(config, lambda: config.paths.catchment_path, 'default', 'CATCHMENT_PATH'),
            explicit_name=resolve_config_value(config, lambda: config.paths.catchment_name, 'default', 'CATCHMENT_SHP_NAME'),
            logger=logger,
        )

        if catchment_file is None:
            logger.warning(
                f"Catchment shapefile not found for {domain_name}. "
                f"Using default area {default_area} km2. "
                f"This makes discharge unit conversions physically meaningless — "
                f"set FIXED_CATCHMENT_AREA in the config to the true area in m²."
            )
            return default_area

        gdf = gpd.read_file(catchment_file)

        # Priority 1: per-HRU area (m2) — sums to the catchment area regardless of
        # how the catchment is discretised (lumped, elevation bands, land class …).
        if 'HRU_area' in gdf.columns:
            area_km2 = gdf['HRU_area'].sum() / 1e6
            logger.debug(f"Catchment area from HRU_area: {area_km2:.2f} km2")
            return area_km2

        # Priority 2: GRU_area (m2) is the per-GRU basin area, repeated on every
        # HRU row when the catchment is HRU-discretised. Summing it across HRU
        # rows multiplies the basin area by the HRU count, so dedupe by GRU id.
        if 'GRU_area' in gdf.columns:
            if 'GRU_ID' in gdf.columns and gdf['GRU_ID'].nunique() < len(gdf):
                area_km2 = gdf.drop_duplicates('GRU_ID')['GRU_area'].sum() / 1e6
            else:
                area_km2 = gdf['GRU_area'].sum() / 1e6
            logger.debug(f"Catchment area from GRU_area: {area_km2:.2f} km2")
            return area_km2

        # Priority 3: Calculate from geometry
        if gdf.crs and not gdf.crs.is_geographic:
            area_m2 = gdf.geometry.area.sum()
        else:
            gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())
            area_m2 = gdf_utm.geometry.area.sum()

        area_km2 = area_m2 / 1e6
        logger.debug(f"Catchment area from geometry: {area_km2:.2f} km2")
        return area_km2

    def _get_area_from_netcdf(
        self,
        settings_dir: Optional[Path],
        domain_name: str,
        default_area: float
    ) -> float:
        """Get catchment area from SUMMA attributes.nc."""
        import xarray as xr

        if settings_dir is None:
            logger.warning("settings_dir required for netcdf area source")
            return default_area

        attrs_file = settings_dir / 'attributes.nc'
        if not attrs_file.exists():
            # Try SUMMA subdirectory
            attrs_file = settings_dir / 'SUMMA' / 'attributes.nc'

        if not attrs_file.exists():
            logger.warning(f"Attributes file not found: {attrs_file}")
            return default_area

        with xr.open_dataset(attrs_file) as attrs:
            if 'HRUarea' in attrs.data_vars:
                area_m2 = float(attrs['HRUarea'].values.sum())
                area_km2 = area_m2 / 1e6
                logger.debug(f"Catchment area from HRUarea: {area_km2:.2f} km2")
                return area_km2

        logger.warning("HRUarea variable not found in attributes.nc")
        return default_area

    def _get_area_from_geodata(
        self,
        settings_dir: Optional[Path],
        default_area: float
    ) -> float:
        """Get catchment area from HYPE GeoData.txt."""
        if settings_dir is None:
            logger.warning("settings_dir required for geodata area source")
            return default_area

        geodata_file = settings_dir / 'GeoData.txt'
        if not geodata_file.exists():
            logger.warning(f"GeoData.txt not found: {geodata_file}")
            return default_area

        geodata = pd.read_csv(geodata_file, sep='\t')
        if 'area' in geodata.columns:
            # HYPE area is in m2
            area_km2 = geodata['area'].sum() / 1e6
            logger.debug(f"Catchment area from GeoData.txt: {area_km2:.2f} km2")
            return area_km2

        logger.warning("'area' column not found in GeoData.txt")
        return default_area

    def convert_runoff_to_discharge(
        self,
        runoff: Union[np.ndarray, pd.Series],
        area_km2: float,
        input_units: str = 'mm/day'
    ) -> Union[np.ndarray, pd.Series]:
        """
        Convert runoff to discharge (m3/s).

        Args:
            runoff: Runoff values
            area_km2: Catchment area in km2
            input_units: Input units: 'mm/day', 'm/s', 'm3/s'

        Returns:
            Discharge in m3/s (same type as input)
        """
        if input_units == 'mm/day':
            # Q(m3/s) = Q(mm/day) * Area(km2) / 86.4
            return runoff * area_km2 / UnitConversion.MM_DAY_TO_CMS
        elif input_units == 'm/s':
            # Q(m3/s) = Q(m/s) * Area(m2) = Q(m/s) * Area(km2) * 1e6
            return runoff * area_km2 * 1e6
        elif input_units == 'm3/s':
            # Already in correct units
            return runoff
        else:
            logger.warning(f"Unknown input units: {input_units}. Assuming mm/day")
            return runoff * area_km2 / UnitConversion.MM_DAY_TO_CMS

    def align_timeseries(
        self,
        sim: pd.Series,
        obs: pd.Series,
        calibration_period: Optional[Tuple[str, str]] = None,
        min_overlap: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align simulation and observation time series.

        Args:
            sim: Simulation series with datetime index
            obs: Observation series with datetime index
            calibration_period: Optional (start_date, end_date) tuple for filtering
            min_overlap: Minimum number of overlapping points required

        Returns:
            Tuple of (obs_aligned, sim_aligned) as numpy arrays

        Raises:
            ValueError: If insufficient overlapping data
        """
        # Find common index
        common_idx = sim.index.intersection(obs.index)

        if len(common_idx) < min_overlap:
            raise ValueError(f"Insufficient overlapping data: {len(common_idx)} points (min: {min_overlap})")

        # Align to common index
        sim_aligned = sim.loc[common_idx]
        obs_aligned = obs.loc[common_idx]

        # Filter to calibration period if specified
        if calibration_period:
            start_date, end_date = calibration_period
            mask = (sim_aligned.index >= start_date) & (sim_aligned.index <= end_date)
            sim_aligned = sim_aligned[mask]
            obs_aligned = obs_aligned[mask]

            if len(sim_aligned) < min_overlap:
                raise ValueError(f"Insufficient data in calibration period: {len(sim_aligned)} points")

        # Remove NaN pairs
        valid_mask = ~(np.isnan(obs_aligned.values) | np.isnan(sim_aligned.values))
        obs_clean = obs_aligned.values[valid_mask]
        sim_clean = sim_aligned.values[valid_mask]

        if len(obs_clean) < min_overlap:
            raise ValueError(f"Insufficient valid data after NaN removal: {len(obs_clean)} points")

        return obs_clean, sim_clean

    def _transform_data(
        self,
        obs: np.ndarray,
        sim: np.ndarray,
        transform: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply a flow transformation to obs/sim arrays for transformed metrics.

        Args:
            obs: Observed values
            sim: Simulated values
            transform: Transformation type: 'log', 'inv', 'sqrt', 'box_cox'

        Returns:
            Tuple of (transformed_obs, transformed_sim)
        """
        epsilon = max(np.mean(obs) * 0.01, 1e-6) if np.mean(obs) > 0 else 1e-6

        if transform == 'log':
            obs_t = np.log(obs + epsilon)
            sim_t = np.log(sim + epsilon)
        elif transform == 'inv':
            obs_t = 1.0 / (obs + epsilon)
            sim_t = 1.0 / (sim + epsilon)
        elif transform == 'sqrt':
            obs_t = np.sqrt(obs + epsilon)
            sim_t = np.sqrt(sim + epsilon)
        elif transform in ('box_cox', 'box_cox_02'):
            # Box-Cox (default lambda=0.2 strongly emphasizes low flows);
            # lambda is configurable via BOX_COX_LAMBDA
            lam = self.box_cox_lambda
            if lam == 0:
                # Box-Cox lambda=0 limit is the log transform
                obs_t = np.log(obs + epsilon)
                sim_t = np.log(sim + epsilon)
            else:
                obs_t = ((obs + epsilon) ** lam - 1) / lam
                sim_t = ((sim + epsilon) ** lam - 1) / lam
        else:
            raise ValueError(f"Unknown transformation: {transform}")

        # Remove NaN/Inf from transformation
        valid = np.isfinite(obs_t) & np.isfinite(sim_t)
        return obs_t[valid], sim_t[valid]

    # Mapping from composite metric names to (base_metric, transform)
    TRANSFORMED_METRICS = {
        'kge_log':  ('kge', 'log'),
        'kge_inv':  ('kge', 'inv'),
        'kge_sqrt': ('kge', 'sqrt'),
        'kge_box_cox': ('kge', 'box_cox'),
        'nse_log':  ('nse', 'log'),
        'nse_sqrt': ('nse', 'sqrt'),
        'rmse_log': ('rmse', 'log'),
        # Case-insensitive aliases
        'KGE_LOG':  ('kge', 'log'),
        'KGE_INV':  ('kge', 'inv'),
        'KGE_SQRT': ('kge', 'sqrt'),
        'KGE_BOX_COX': ('kge', 'box_cox'),
        'NSE_LOG':  ('nse', 'log'),
        'NSE_SQRT': ('nse', 'sqrt'),
        'RMSE_LOG': ('rmse', 'log'),
    }

    def calculate_metrics(
        self,
        obs: np.ndarray,
        sim: np.ndarray,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate performance metrics with error handling.

        Supports both standard metrics and transformed variants:
        - Standard: 'kge', 'nse', 'rmse', 'mae', 'kge_prime'
        - Transformed: 'kge_log', 'kge_inv', 'kge_sqrt', 'nse_log', etc.
          These apply a flow transformation before computing the base metric,
          which emphasizes different parts of the hydrograph (e.g., low flows).

        Args:
            obs: Observed values (numpy array)
            sim: Simulated values (numpy array)
            metrics: List of metrics to calculate

        Returns:
            Dictionary of metric values, or penalty_score on error
        """
        if metrics is None:
            metrics = ['kge', 'nse']
        try:
            # Check for zero variance in simulation (common during spinup)
            if np.std(sim) == 0:
                logger.warning("Zero variance in simulation - returning penalty scores")
                return {m: -1.0 for m in metrics}  # Bad but not penalty

            result = {}
            for metric in metrics:
                try:
                    # Check if this is a transformed metric (e.g., kge_log, kge_inv)
                    if metric in self.TRANSFORMED_METRICS:
                        base_metric, transform = self.TRANSFORMED_METRICS[metric]
                        obs_t, sim_t = self._transform_data(obs, sim, transform)
                        if len(obs_t) == 0:
                            val = self.penalty_score
                        elif base_metric == 'kge':
                            val = kge(obs_t, sim_t, transfo=1)
                        elif base_metric == 'nse':
                            val = nse(obs_t, sim_t, transfo=1)
                        elif base_metric == 'rmse':
                            val = rmse(obs_t, sim_t, transfo=1)
                        else:
                            val = self.penalty_score
                    elif metric == 'kge':
                        val = kge(obs, sim, transfo=1)
                    elif metric == 'nse':
                        val = nse(obs, sim, transfo=1)
                    elif metric == 'rmse':
                        val = rmse(obs, sim, transfo=1)
                    elif metric == 'mae':
                        val = mae(obs, sim, transfo=1)
                    elif metric == 'kge_prime':
                        val = kge_prime(obs, sim, transfo=1)
                    else:
                        logger.warning(f"Unknown metric: {metric}")
                        val = self.penalty_score

                    # Handle NaN/Inf values
                    if pd.isna(val) or np.isinf(val):
                        val = self.penalty_score

                    result[metric] = float(val)

                except Exception as e:  # noqa: BLE001 — must-not-raise contract
                    logger.warning(f"Error calculating {metric}: {e}")
                    result[metric] = self.penalty_score

            return result

        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"Error in metric calculation: {e}")
            return {'kge': self.penalty_score}

    def evaluate_streamflow(
        self,
        sim_series: pd.Series,
        config: Dict[str, Any],
        project_dir: Path,
        domain_name: str,
        area_km2: Optional[float] = None,
        area_source: str = 'shapefile',
        settings_dir: Optional[Path] = None,
        input_units: str = 'mm/day',
        metrics: Optional[List[str]] = None,
        resample_freq: Optional[str] = 'D',
        calibration_period: Optional[Tuple[str, str]] = None
    ) -> Dict[str, float]:
        """
        Full evaluation pipeline: load obs, convert units, align, calculate metrics.

        Args:
            sim_series: Simulation time series (pandas Series with datetime index)
            config: Configuration dictionary
            project_dir: Project directory path
            domain_name: Domain name
            area_km2: Catchment area in km2 (if None, will be loaded from source)
            area_source: Source for area if not provided: 'shapefile', 'netcdf', 'geodata'
            settings_dir: Settings directory (for netcdf/geodata area sources)
            input_units: Units of simulation data: 'mm/day', 'm/s', 'm3/s'
            metrics: List of metrics to calculate
            resample_freq: Resampling frequency for observations
            calibration_period: Optional (start_date, end_date) for filtering

        Returns:
            Dictionary of metric values
        """
        if metrics is None:
            metrics = ['kge', 'nse']
        try:
            # Load observations
            obs_values, obs_index = self.load_observations(
                config, project_dir, domain_name, resample_freq
            )
            if obs_values is None:
                return {'kge': self.penalty_score}

            obs_series = pd.Series(obs_values, index=obs_index)

            # Get catchment area if not provided
            if area_km2 is None:
                area_km2 = self.get_catchment_area(
                    config, project_dir, domain_name, area_source, settings_dir
                )

            # Convert simulation to discharge if needed
            if input_units != 'm3/s':
                sim_discharge = self.convert_runoff_to_discharge(sim_series, area_km2, input_units)
            else:
                sim_discharge = sim_series

            # Ensure sim_discharge is a Series
            if not isinstance(sim_discharge, pd.Series):
                sim_discharge = pd.Series(sim_discharge, index=sim_series.index)

            # Align time series
            obs_aligned, sim_aligned = self.align_timeseries(
                sim_discharge, obs_series, calibration_period
            )

            # Calculate metrics
            return self.calculate_metrics(obs_aligned, sim_aligned, metrics)

        except ValueError as e:
            logger.warning(f"Alignment error: {e}")
            return {'kge': self.penalty_score}
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"Error in streamflow evaluation: {e}")
            return {'kge': self.penalty_score}


# Module-level instance for convenience
_streamflow_metrics = StreamflowMetrics()


def load_observations(
    config: Dict[str, Any],
    project_dir: Path,
    domain_name: str,
    resample_freq: Optional[str] = 'D'
) -> Tuple[Optional[np.ndarray], Optional[pd.DatetimeIndex]]:
    """Convenience function. See StreamflowMetrics.load_observations."""
    return _streamflow_metrics.load_observations(config, project_dir, domain_name, resample_freq)


def get_catchment_area(
    config: Dict[str, Any],
    project_dir: Path,
    domain_name: str,
    source: str = 'shapefile',
    settings_dir: Optional[Path] = None
) -> float:
    """Convenience function. See StreamflowMetrics.get_catchment_area."""
    return _streamflow_metrics.get_catchment_area(config, project_dir, domain_name, source, settings_dir)


def calculate_metrics(
    obs: np.ndarray,
    sim: np.ndarray,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """Convenience function. See StreamflowMetrics.calculate_metrics."""
    return _streamflow_metrics.calculate_metrics(obs, sim, metrics)
