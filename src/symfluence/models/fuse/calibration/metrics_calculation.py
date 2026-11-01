# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
FUSE Metrics Calculation

Standalone functions for calculating calibration metrics from FUSE output.
Handles observation loading, simulation reading (both direct FUSE and routed),
time alignment, and metric computation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from symfluence.core.constants import UnitConversion
from symfluence.core.metrics import StreamflowMetrics
from symfluence.core.mixins.config import resolve_config_value
from symfluence.models.fuse.calibration.file_manager import resolve_fuse_id

logger = logging.getLogger(__name__)


def load_observations(
    config: Dict[str, Any],
    project_dir: Path,
    log: Optional[logging.Logger] = None
) -> Optional[pd.Series]:
    """
    Load observed streamflow data.

    Args:
        config: Configuration dictionary
        project_dir: Project directory path
        log: Logger instance

    Returns:
        Daily-resampled observed discharge series, or None if not found

    Delegates to the shared ``StreamflowMetrics`` loader, which prefers the
    model-ready observations store and falls back to the legacy preprocessed
    CSV — keeping FUSE consistent with the other models' calibration loaders.
    """
    log = log or logger
    domain_name = resolve_config_value(config, lambda: config.domain.name, None, 'DOMAIN_NAME')

    values, index = StreamflowMetrics().load_observations(
        config, project_dir, domain_name, resample_freq='D',
    )
    if values is None or index is None:
        log.error("Observations not found (model-ready store or preprocessed CSV)")
        return None

    return pd.Series(values, index=index, name='discharge_cms')


def find_simulation_output(
    config: Dict[str, Any],
    output_dir: Path,
    log: Optional[logging.Logger] = None,
    mizuroute_dir: Optional[Path] = None,
    proc_id: int = 0,
    sim_dir: Optional[Path] = None,
) -> Tuple[Optional[Path], bool]:
    """
    Find the simulation output file, prioritizing routed output.

    Args:
        config: Configuration dictionary
        output_dir: Output directory
        log: Logger instance
        mizuroute_dir: Optional mizuRoute output directory
        proc_id: Process ID for parallel runs
        sim_dir: Optional simulation directory override

    Returns:
        Tuple of (sim_file_path, use_routed_output)
    """
    log = log or logger
    experiment_id = resolve_config_value(config, lambda: config.domain.experiment_id, None, 'EXPERIMENT_ID')

    # Check for routed output first
    if mizuroute_dir and Path(mizuroute_dir).exists():
        sim_file = _find_mizuroute_output(
            Path(mizuroute_dir), experiment_id, proc_id, log
        )
        if sim_file:
            return sim_file, True

    # Fall back to FUSE output
    sim_file = _find_fuse_output(config, output_dir, sim_dir, log)
    if sim_file:
        return sim_file, False

    return None, False


def _find_mizuroute_output(
    mizuroute_dir: Path,
    experiment_id: str,
    proc_id: int,
    log: logging.Logger
) -> Optional[Path]:
    """Find mizuRoute output file."""
    case_name = f"proc_{proc_id:02d}_{experiment_id}"

    # Try proc-specific pattern
    mizu_files = list(mizuroute_dir.glob(f"{case_name}.h.*.nc"))
    if mizu_files:
        mizu_files.sort(key=lambda f: f.stat().st_size, reverse=True)
        log.debug(f"Using mizuRoute output: {mizu_files[0]} (size: {mizu_files[0].stat().st_size} bytes)")
        return mizu_files[0]

    # Fallback to non-prefixed pattern
    mizu_files = list(mizuroute_dir.glob(f"{experiment_id}.h.*.nc"))
    if mizu_files:
        mizu_files.sort(key=lambda f: f.stat().st_size, reverse=True)
        log.debug(f"Using mizuRoute output: {mizu_files[0]} (size: {mizu_files[0].stat().st_size} bytes)")
        return mizu_files[0]

    # Older timestep naming
    old_pattern = mizuroute_dir / f"{experiment_id}_timestep.nc"
    if old_pattern.exists():
        log.debug(f"Using mizuRoute output: {old_pattern}")
        return old_pattern

    return None


def _find_fuse_output(
    config: Dict[str, Any],
    output_dir: Path,
    sim_dir: Optional[Path],
    log: logging.Logger
) -> Optional[Path]:
    """Find FUSE output file."""
    domain_name = resolve_config_value(config, lambda: config.domain.name, None, 'DOMAIN_NAME')
    experiment_id = resolve_config_value(config, lambda: config.domain.experiment_id, None, 'EXPERIMENT_ID')
    fuse_id = resolve_fuse_id(config, experiment_id)
    fuse_output_dir = Path(sim_dir) if sim_dir else output_dir

    candidates = [
        fuse_output_dir / f"{domain_name}_{fuse_id}_runs_def.nc",
        fuse_output_dir / f"{domain_name}_{fuse_id}_runs_best.nc",
        fuse_output_dir / f"{domain_name}_{fuse_id}_runs_pre.nc",
        fuse_output_dir.parent / f"{domain_name}_{fuse_id}_runs_def.nc",
        output_dir.parent / f"{domain_name}_{fuse_id}_runs_pre.nc",
    ]

    for cand in candidates:
        if cand.exists():
            return cand

    log.error(f"Simulation file not found. Searched: {[str(c) for c in candidates]}")
    return None


def read_routed_streamflow(
    sim_file_path: Path,
    config: Dict[str, Any],
    log: Optional[logging.Logger] = None
) -> Optional[pd.Series]:
    """
    Read routed streamflow from mizuRoute output.

    Args:
        sim_file_path: Path to mizuRoute output NetCDF
        config: Configuration dictionary
        log: Logger instance

    Returns:
        Daily-resampled simulated streamflow in m³/s, or None if failed
    """
    import xarray as xr

    log = log or logger

    try:
        ds = xr.open_dataset(sim_file_path, decode_times=True, decode_timedelta=True)
    except Exception as e:  # noqa: BLE001 — calibration resilience
        log.error(
            f"Cannot open routed output file {sim_file_path.name}: {e}. "
            f"File size: {sim_file_path.stat().st_size if sim_file_path.exists() else 'missing'} bytes"
        , exc_info=True)
        return None

    with ds:
        sim_reach_id = resolve_config_value(config, lambda: config.evaluation.sim_reach_id, None, 'SIM_REACH_ID')
        seg_idx = 0

        if sim_reach_id is not None and sim_reach_id != 'default':
            if 'reachID' in ds.variables:
                reach_ids = ds['reachID'].values
                matches = np.where(reach_ids == int(sim_reach_id))[0]
                if len(matches) > 0:
                    seg_idx = int(matches[0])
                    log.debug(f"Using segment index {seg_idx} for reach ID {sim_reach_id}")
                else:
                    log.warning(f"Reach ID {sim_reach_id} not found, using segment 0")
            else:
                log.warning("No reachID variable in mizuRoute output, using segment 0")

        if 'IRFroutedRunoff' in ds.variables:
            simulated = ds['IRFroutedRunoff'].isel(seg=seg_idx)
        elif 'dlayRunoff' in ds.variables:
            simulated = ds['dlayRunoff'].isel(seg=seg_idx)
        else:
            log.error(f"No routed runoff variable. Variables: {list(ds.variables.keys())}")
            return None

        simulated_streamflow = simulated.to_series()

    if not isinstance(simulated_streamflow.index, pd.DatetimeIndex):
        simulated_streamflow.index = pd.to_datetime(simulated_streamflow.index)

    return simulated_streamflow.resample('D').mean()


def read_fuse_streamflow(
    sim_file_path: Path,
    config: Dict[str, Any],
    project_dir: Path,
    streamflow_metrics: StreamflowMetrics,
    log: Optional[logging.Logger] = None
) -> Optional[pd.Series]:
    """
    Read streamflow directly from FUSE output.

    Handles lumped and distributed modes with appropriate unit conversion.

    Args:
        sim_file_path: Path to FUSE output NetCDF
        config: Configuration dictionary
        project_dir: Project directory
        streamflow_metrics: StreamflowMetrics utility instance
        log: Logger instance

    Returns:
        Daily simulated streamflow in m³/s, or None if failed
    """
    import xarray as xr

    log = log or logger
    domain_name = resolve_config_value(config, lambda: config.domain.name, None, 'DOMAIN_NAME')

    try:
        ds = xr.open_dataset(sim_file_path, decode_times=True, decode_timedelta=True)
    except Exception as e:  # noqa: BLE001 — calibration resilience
        log.error(
            f"Cannot open FUSE output file {sim_file_path.name}: {e}. "
            f"File size: {sim_file_path.stat().st_size if sim_file_path.exists() else 'missing'} bytes. "
            f"FUSE may have crashed silently (Fortran STOP returns exit code 0)."
        , exc_info=True)
        return None

    with ds:
        spatial_mode = resolve_config_value(config, lambda: config.model.fuse.spatial_mode, 'lumped', 'FUSE_SPATIAL_MODE')
        subcatchment_dim = resolve_config_value(config, lambda: config.model.fuse.subcatchment_dim, 'latitude', 'FUSE_SUBCATCHMENT_DIM')

        # Select runoff variable
        if 'q_routed' in ds.variables:
            runoff_var = ds['q_routed']
            var_name = 'q_routed'
        elif 'q_instnt' in ds.variables:
            runoff_var = ds['q_instnt']
            var_name = 'q_instnt'
        else:
            log.error(f"No runoff variable found. Variables: {list(ds.variables.keys())}")
            return None

        log.debug(f"FUSE output dimensions: {runoff_var.dims}, sizes: {dict(runoff_var.sizes)}")

        # Diagnostic: check for all-zero output
        raw_mean = float(runoff_var.mean())
        raw_max = float(runoff_var.max())
        if raw_mean < 1e-10 and raw_max < 1e-10:
            log.warning(
                f"FUSE output {var_name} is all zeros! Raw mean={raw_mean:.6f}, max={raw_max:.6f}. "
                f"This may indicate FUSE is not reading calibration parameters correctly."
            )

        has_param_set = 'param_set' in runoff_var.dims
        n_subcatchments = runoff_var.sizes.get(subcatchment_dim, 1)

        total_area_km2 = streamflow_metrics.get_catchment_area(config, project_dir, domain_name)

        if spatial_mode == 'distributed' and n_subcatchments > 1:
            simulated_streamflow = _read_distributed_fuse(
                runoff_var, subcatchment_dim, has_param_set,
                total_area_km2, n_subcatchments, log
            )
        else:
            simulated_streamflow = _read_lumped_fuse(
                runoff_var, has_param_set, total_area_km2, sim_file_path, log
            )

    if not isinstance(simulated_streamflow.index, pd.DatetimeIndex):
        try:
            simulated_streamflow.index = pd.to_datetime(simulated_streamflow.index)
            log.debug("Converted simulated streamflow index to DatetimeIndex")
        except Exception as e:  # noqa: BLE001 — calibration resilience
            log.error(f"Failed to convert time index to DatetimeIndex: {e}", exc_info=True)
            return None

    return simulated_streamflow


def _read_distributed_fuse(
    runoff_var: Any,
    subcatchment_dim: str,
    has_param_set: bool,
    total_area_km2: float,
    n_subcatchments: int,
    log: logging.Logger
) -> pd.Series:
    """Read and aggregate distributed FUSE output."""
    log.debug(f"Distributed mode: aggregating {n_subcatchments} subcatchments on '{subcatchment_dim}' dimension")

    isel_kwargs = {}
    if has_param_set:
        n_param_sets = runoff_var.sizes.get('param_set', 1)
        valid_param_set = 0
        for ps in range(n_param_sets):
            test_vals = runoff_var.isel(param_set=ps).values
            if not np.all(np.isnan(test_vals)):
                valid_param_set = ps
                break
        isel_kwargs['param_set'] = valid_param_set

    other_spatial_dim = 'latitude' if subcatchment_dim == 'longitude' else 'longitude'
    if other_spatial_dim in runoff_var.dims:
        isel_kwargs[other_spatial_dim] = 0

    runoff_selected = runoff_var.isel(**isel_kwargs) if isel_kwargs else runoff_var

    subcatchment_area = total_area_km2 / n_subcatchments
    simulated_cms = (
        runoff_selected * subcatchment_area / UnitConversion.MM_DAY_TO_CMS
    ).sum(dim=subcatchment_dim)

    simulated_streamflow = simulated_cms.to_pandas()
    log.debug(f"Aggregated distributed output: mean flow = {simulated_streamflow.mean():.2f} m³/s")
    return simulated_streamflow


def _read_lumped_fuse(
    runoff_var: Any,
    has_param_set: bool,
    area_km2: float,
    sim_file_path: Path,
    log: logging.Logger
) -> pd.Series:
    """Read lumped FUSE output and convert to m³/s."""
    isel_kwargs = {}
    if has_param_set:
        n_param_sets = runoff_var.sizes.get('param_set', 1)
        valid_param_set = 0
        for ps in range(n_param_sets):
            test_vals = runoff_var.isel(param_set=ps, latitude=0, longitude=0).values
            if not np.all(np.isnan(test_vals)):
                valid_param_set = ps
                break
        isel_kwargs['param_set'] = valid_param_set
        log.debug(f"Using param_set {valid_param_set} (has valid data)")

    if 'latitude' in runoff_var.dims:
        isel_kwargs['latitude'] = 0
    if 'longitude' in runoff_var.dims:
        isel_kwargs['longitude'] = 0

    simulated = runoff_var.isel(**isel_kwargs) if isel_kwargs else runoff_var
    simulated_streamflow = simulated.to_pandas()

    log.debug(
        f"DEBUG: raw sim mean={simulated_streamflow.mean():.4f} mm/day, "
        f"area={area_km2:.2f} km2, sim_file={sim_file_path}"
    )

    # Convert from mm/day to m³/s
    simulated_streamflow = simulated_streamflow * area_km2 / UnitConversion.MM_DAY_TO_CMS
    log.debug(f"DEBUG: converted sim mean={simulated_streamflow.mean():.4f} m3/s")
    return simulated_streamflow


def align_and_filter(
    observed: pd.Series,
    simulated: pd.Series,
    config: Dict[str, Any],
    log: Optional[logging.Logger] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align observed and simulated time series and filter to calibration period.

    Args:
        observed: Observed streamflow series
        simulated: Simulated streamflow series
        config: Configuration dictionary
        log: Logger instance

    Returns:
        Tuple of (obs_values, sim_values) as numpy arrays, or empty arrays if alignment fails
    """
    log = log or logger

    common_index = observed.index.intersection(simulated.index)
    if len(common_index) == 0:
        log.error("No overlapping time period")
        return np.array([]), np.array([])

    obs_aligned = observed.loc[common_index].dropna()
    sim_aligned = simulated.loc[common_index].dropna()

    # Filter to calibration period
    calib_period = resolve_config_value(config, lambda: config.domain.calibration_period, '', 'CALIBRATION_PERIOD')
    if calib_period and ',' in str(calib_period):
        try:
            calib_start, calib_end = [s.strip() for s in str(calib_period).split(',')]
            calib_start = pd.Timestamp(calib_start)
            calib_end = pd.Timestamp(calib_end)

            mask_obs = (obs_aligned.index >= calib_start) & (obs_aligned.index <= calib_end)
            mask_sim = (sim_aligned.index >= calib_start) & (sim_aligned.index <= calib_end)

            obs_aligned = obs_aligned[mask_obs]
            sim_aligned = sim_aligned[mask_sim]

            log.debug(f"Filtered to calibration period {calib_start} to {calib_end}: {len(obs_aligned)} points")
        except Exception as e:  # noqa: BLE001 — calibration resilience
            log.warning(f"Could not parse calibration period '{calib_period}': {e}", exc_info=True)

    common_index = obs_aligned.index.intersection(sim_aligned.index)
    return np.asarray(obs_aligned.loc[common_index].values), np.asarray(sim_aligned.loc[common_index].values)


def compute_metrics(
    obs_values: np.ndarray,
    sim_values: np.ndarray,
    config: Dict[str, Any],
    streamflow_metrics: StreamflowMetrics,
    log: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Compute streamflow metrics from aligned observation and simulation arrays.

    Args:
        obs_values: Observed values array
        sim_values: Simulated values array
        config: Configuration dictionary
        streamflow_metrics: StreamflowMetrics utility instance
        log: Logger instance

    Returns:
        Dictionary of metric names to values
    """
    log = log or logger

    metric_list = ['kge', 'nse', 'rmse', 'mae']

    # Add the configured single objective metric (e.g. KGE_LOG, KGE_BOX_COX)
    calibration_metric = config.get(
        'OPTIMIZATION_METRIC', config.get('CALIBRATION_METRIC', '')
    )
    if isinstance(calibration_metric, str):
        metric_lower = calibration_metric.lower()
        if metric_lower and metric_lower != 'composite' and metric_lower not in metric_list:
            metric_list.append(metric_lower)

    # Add composite metric components if configured
    composite_config = resolve_config_value(config, lambda: config.optimization.composite_metric, None, 'COMPOSITE_METRIC')
    if composite_config and isinstance(composite_config, dict):
        for metric_name in composite_config.keys():
            metric_lower = metric_name.lower()
            if metric_lower not in metric_list:
                metric_list.append(metric_lower)

    metrics = streamflow_metrics.calculate_metrics(
        obs_values, sim_values, metrics=metric_list
    )

    log.debug(
        f"FUSE metrics: KGE={metrics.get('kge', 'N/A')}, NSE={metrics.get('nse', 'N/A')}, "
        f"n_pts={len(obs_values)}, sim_mean={sim_values.mean():.2f}, obs_mean={obs_values.mean():.2f}"
    )

    if composite_config:
        extra = {k: f"{v:.4f}" for k, v in metrics.items() if k not in ('kge', 'nse', 'rmse', 'mae')}
        if extra:
            log.debug(f"FUSE composite components: {extra}")

    return metrics
