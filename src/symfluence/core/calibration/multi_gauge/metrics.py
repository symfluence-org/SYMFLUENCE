# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Model-agnostic multi-gauge calibration metrics base class.

This module provides the :class:`MultiGaugeMetrics` class and the
:func:`create_multi_gauge_config` helper for calculating performance
metrics across multiple stream gauges simultaneously, enabling
spatially distributed calibration of hydrological models.

Originally housed under the FUSE model package, these utilities are
model-agnostic and are used by multiple model backends (FUSE, HYPE, etc.).
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from symfluence.core import metrics as core_metrics


class MultiGaugeMetrics:
    """
    Calculate performance metrics across multiple stream gauges.

    This class reads simulated streamflow from mizuRoute output at
    multiple segment locations (corresponding to gauges) and calculates
    KGE (or other metrics) at each gauge, then aggregates them.
    """

    # Class-level cache for quality filter results — persists across instances
    # so that repeated DDS iterations don't re-compute and re-log the same filters
    _filter_cache: Optional[Tuple[tuple, List[int]]] = None
    _first_eval_logged: bool = False

    def __init__(
        self,
        gauge_segment_mapping_path: Path,
        obs_data_dir: Path,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize multi-gauge metrics calculator.

        Args:
            gauge_segment_mapping_path: Path to CSV mapping gauges to routing segments
            obs_data_dir: Path to directory containing observed streamflow data
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.gauge_segment_mapping_path = Path(gauge_segment_mapping_path)
        self.obs_data_dir = Path(obs_data_dir)

        # Load gauge-segment mapping
        self.gauge_mapping = self._load_gauge_mapping()

        # Cache for observed data
        self._obs_cache: Dict[int, pd.Series] = {}

        # mizuRoute output handle shared by all gauges of one evaluation
        self._sim_handle: Optional[Tuple[tuple, Tuple[Any, np.ndarray, str]]] = None
        self._sim_session_active = False

    def _load_gauge_mapping(self) -> pd.DataFrame:
        """Load the gauge-to-segment mapping from CSV."""
        if not self.gauge_segment_mapping_path.exists():
            raise FileNotFoundError(
                f"Gauge-segment mapping not found: {self.gauge_segment_mapping_path}"
            )

        df = pd.read_csv(self.gauge_segment_mapping_path)
        self.logger.debug(f"Loaded {len(df)} gauges from mapping file")
        return df

    def _load_observed_streamflow(
        self,
        gauge_id: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.Series]:
        """
        Load observed streamflow for a gauge.

        Args:
            gauge_id: Gauge ID from the mapping file
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering

        Returns:
            pandas Series with datetime index and streamflow in m³/s
        """
        # Check cache first
        if gauge_id in self._obs_cache:
            obs = self._obs_cache[gauge_id]
        else:
            # LamaH-Ice format: D_gauges/2_timeseries/daily/ID_{id}.csv
            obs_file = self.obs_data_dir / f"ID_{gauge_id}.csv"

            if not obs_file.exists():
                self.logger.warning(f"No observation file for gauge {gauge_id}: {obs_file}")
                return None

            try:
                # Read semicolon-delimited CSV
                df = pd.read_csv(obs_file, sep=';')

                # Create datetime index
                df['date'] = pd.to_datetime(
                    df[['YYYY', 'MM', 'DD']].rename(
                        columns={'YYYY': 'year', 'MM': 'month', 'DD': 'day'}
                    )
                )

                # Extract streamflow as Series
                obs = df.set_index('date')['qobs']

                # Cache for future use
                self._obs_cache[gauge_id] = obs

            except Exception as e:  # noqa: BLE001 — calibration resilience
                self.logger.warning(f"Error reading observations for gauge {gauge_id}: {e}", exc_info=True)
                return None

        # Filter by date range if specified
        if start_date or end_date:
            mask = pd.Series(True, index=obs.index)
            if start_date:
                mask &= obs.index >= pd.Timestamp(start_date)
            if end_date:
                mask &= obs.index <= pd.Timestamp(end_date)
            obs = obs[mask]

        return obs

    def _extract_simulated_at_segment(
        self,
        mizuroute_output_path: Path,
        segment_id: int,
        topology_path: Optional[Path] = None
    ) -> Optional[pd.Series]:
        """
        Extract simulated streamflow at a specific routing segment.

        Args:
            mizuroute_output_path: Path to mizuRoute output NetCDF
            segment_id: Routing segment ID to extract
            topology_path: Optional path to topology file for segment lookup

        Returns:
            pandas Series with datetime index and streamflow in m³/s
        """
        try:
            opened = self._open_simulation(mizuroute_output_path, topology_path)
            if opened is None:
                return None
            ds, seg_ids, runoff_var = opened
            try:
                # Find segment index
                seg_idx = np.where(seg_ids == segment_id)[0]
                if len(seg_idx) == 0:
                    self.logger.warning(f"Segment {segment_id} not found in mizuRoute output")
                    return None

                # Convert to pandas Series (avoid DataFrame return from generic stubs)
                sim = ds[runoff_var].isel(seg=seg_idx[0]).to_series()

                # Ensure datetime index
                if not isinstance(sim.index, pd.DatetimeIndex):
                    sim.index = pd.to_datetime(sim.index)

                # Resample to daily if needed (mizuRoute may output hourly)
                return sim.resample('D').mean()
            finally:
                if not self._sim_session_active:
                    self._close_simulation()

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error extracting simulated at segment {segment_id}: {e}", exc_info=True)
            return None

    def _open_simulation(
        self, mizuroute_output_path: Path, topology_path: Optional[Path]
    ) -> Optional[Tuple[Any, np.ndarray, str]]:
        """Open the mizuRoute output once and keep it for the evaluation.

        Returns ``(dataset, segment ids, routed-runoff variable)``. Inside
        :meth:`calculate_multi_gauge_metrics` the handle is reused for every
        gauge (it used to be reopened once per gauge per evaluation) and
        closed when the evaluation ends.
        """
        import xarray as xr

        key = (str(mizuroute_output_path), str(topology_path))
        if self._sim_handle is not None and self._sim_handle[0] == key:
            return self._sim_handle[1]
        self._close_simulation()

        ds = xr.open_dataset(mizuroute_output_path)
        opened: Optional[Tuple[Any, np.ndarray, str]] = None
        try:
            # Get segment indices
            seg_ids = None
            if 'segId' in ds.variables:
                seg_ids = ds['segId'].values
            elif 'reachID' in ds.variables:
                seg_ids = ds['reachID'].values
            elif topology_path and topology_path.exists():
                # Try to get from topology file
                with xr.open_dataset(topology_path) as topo_ds:
                    if 'segId' in topo_ds.variables:
                        seg_ids = topo_ds['segId'].values
                if seg_ids is None:
                    self.logger.error("Cannot find segment IDs")
            else:
                self.logger.error("Cannot find segment IDs in mizuRoute output")

            runoff_var = next(
                (v for v in ('IRFroutedRunoff', 'dlayRunoff', 'KWTroutedRunoff') if v in ds.variables),
                None,
            )
            if seg_ids is not None and runoff_var is None:
                self.logger.error(f"No routed runoff variable found in {mizuroute_output_path}")

            if seg_ids is not None and runoff_var is not None:
                opened = (ds, seg_ids, runoff_var)
        finally:
            if opened is None:
                ds.close()
        if opened is not None:
            self._sim_handle = (key, opened)
        return opened

    def _close_simulation(self) -> None:
        """Close the cached mizuRoute dataset, if any."""
        if self._sim_handle is not None:
            handle, self._sim_handle = self._sim_handle, None
            try:
                handle[1][0].close()
            except (OSError, RuntimeError, ValueError):
                pass

    def _calculate_kge(
        self,
        obs: np.ndarray,
        sim: np.ndarray
    ) -> float:
        """
        Calculate Kling-Gupta Efficiency.

        Args:
            obs: Observed values
            sim: Simulated values

        Returns:
            KGE value (higher is better, perfect = 1.0)
        """
        if len(obs) == 0 or len(sim) == 0:
            return -9999.0

        # Remove NaN values
        mask = ~(np.isnan(obs) | np.isnan(sim))
        obs = obs[mask]
        sim = sim[mask]

        if len(obs) < 10:  # Minimum data points
            return -9999.0

        # Shared KGE definition (symfluence.core.metrics). It is undefined
        # (NaN) for constant observations or a zero observed mean; those
        # gauges get the same failure sentinel as too-short records instead of
        # an arbitrary finite score.
        kge = core_metrics.kge(obs, sim)
        if not np.isfinite(kge):
            return -9999.0
        return float(kge)

    def _apply_quality_filters(
        self,
        gauge_ids: List[int],
        start_date: Optional[str],
        end_date: Optional[str],
        filter_config: Dict[str, Any]
    ) -> List[int]:
        """
        Filter gauges based on quality criteria. Results are cached so that
        filter details are only logged on the first call.

        Args:
            gauge_ids: List of gauge IDs to filter
            start_date: Start date for observation loading
            end_date: End date for observation loading
            filter_config: Dict with keys:
                - max_distance: Max distance_to_segment in degrees
                - min_obs_cv: Min coefficient of variation of observed flow
                - min_specific_q: Min specific discharge in mm/yr

        Returns:
            Filtered list of gauge IDs
        """
        max_distance = filter_config.get('max_distance')
        min_obs_cv = filter_config.get('min_obs_cv')
        min_specific_q = filter_config.get('min_specific_q')

        if not any([max_distance, min_obs_cv, min_specific_q]):
            return gauge_ids

        # Return cached result only for the same question: same gauges, same
        # filter thresholds, same period, same mapping and observations. Keyed
        # on the gauge ids alone, the first caller's answer was handed to every
        # later instance in the process (a different period, different
        # thresholds, another domain with overlapping ids).
        cache_key = (
            tuple(sorted(gauge_ids)),
            tuple(sorted((str(k), repr(v)) for k, v in filter_config.items())),
            str(start_date), str(end_date),
            str(self.gauge_segment_mapping_path), str(self.obs_data_dir),
        )
        if MultiGaugeMetrics._filter_cache is not None and MultiGaugeMetrics._filter_cache[0] == cache_key:
            return list(MultiGaugeMetrics._filter_cache[1])

        filtered = []

        for gauge_id in gauge_ids:
            gauge_row = self.gauge_mapping[self.gauge_mapping['id'] == gauge_id]
            if len(gauge_row) == 0:
                filtered.append(gauge_id)
                continue

            row = gauge_row.iloc[0]
            gauge_name = row.get('name', f'Gauge_{gauge_id}')

            # Distance filter
            if max_distance is not None and 'distance_to_segment' in row.index:
                dist = float(row['distance_to_segment'])
                if dist > max_distance:
                    self.logger.debug(
                        f"Quality filter: excluding gauge {gauge_id} ({gauge_name}) "
                        f"— distance {dist:.4f}° > {max_distance}°"
                    )
                    continue

            # Obs CV filter (low CV = nearly constant flow, e.g. spring-fed or glacier-buffered)
            if min_obs_cv is not None:
                obs = self._load_observed_streamflow(gauge_id, start_date, end_date)
                if obs is not None and len(obs.dropna()) > 30:
                    obs_clean = obs.dropna()
                    obs_mean = obs_clean.mean()
                    if obs_mean > 0:
                        cv = float(obs_clean.std() / obs_mean)
                        if cv < min_obs_cv:
                            self.logger.debug(
                                f"Quality filter: excluding gauge {gauge_id} ({gauge_name}) "
                                f"— obs CV {cv:.3f} < {min_obs_cv} (low variability)"
                            )
                            continue

            # Specific Q filter
            if min_specific_q is not None and 'area_calc' in row.index:
                area_km2 = float(row['area_calc'])
                if area_km2 > 0:
                    obs = self._load_observed_streamflow(gauge_id, start_date, end_date)
                    if obs is not None and len(obs.dropna()) > 30:
                        obs_mean_m3s = float(obs.dropna().mean())
                        specific_q = (obs_mean_m3s * 86400 * 365) / (area_km2 * 1e6) * 1000
                        if specific_q < min_specific_q:
                            self.logger.debug(
                                f"Quality filter: excluding gauge {gauge_id} ({gauge_name}) "
                                f"— specific Q {specific_q:.0f} mm/yr < {min_specific_q} (groundwater loss)"
                            )
                            continue

            filtered.append(gauge_id)

        n_removed = len(gauge_ids) - len(filtered)
        if n_removed > 0:
            self.logger.info(
                f"Quality filters removed {n_removed} gauges: {len(gauge_ids)} → {len(filtered)}"
            )

        # Cache the result at class level so it persists across instances
        MultiGaugeMetrics._filter_cache = (cache_key, list(filtered))

        return filtered

    def calculate_multi_gauge_metrics(
        self,
        mizuroute_output_path: Path,
        gauge_ids: Optional[List[int]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        topology_path: Optional[Path] = None,
        min_gauges: int = 5,
        aggregation: str = 'mean',
        weights: Optional[Dict[int, float]] = None,
        filter_config: Optional[Dict[str, Any]] = None,
        min_overlap_days: int = 10,
        kge_floor: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate KGE at multiple gauges and aggregate.

        Args:
            mizuroute_output_path: Path to mizuRoute output NetCDF
            gauge_ids: Optional list of gauge IDs to use (default: all)
            start_date: Optional start date for evaluation period
            end_date: Optional end date for evaluation period
            topology_path: Optional path to topology file
            min_gauges: Minimum number of gauges with valid data required
            aggregation: Aggregation method ('mean', 'median', 'weighted')
            weights: Optional weights for weighted aggregation (by gauge_id)
            filter_config: Optional quality filter configuration
            min_overlap_days: Minimum days of obs-sim overlap required per gauge
            kge_floor: Optional minimum KGE value; gauges below this floor are
                capped at the floor before aggregation. Prevents structurally
                unfittable gauges (regulated rivers, glaciers) from dominating
                the objective function. Recommended: -0.41 (KGE of mean predictor).

        Returns:
            Dictionary with aggregated metrics and per-gauge details
        """
        results: Dict[str, Any] = {
            'kge': -9999.0,
            'n_valid_gauges': 0,
            'n_total_gauges': 0,
            'per_gauge': {},
            'aggregation': aggregation
        }

        if not mizuroute_output_path.exists():
            self.logger.error(f"mizuRoute output not found: {mizuroute_output_path}")
            return results

        # Use all gauges if not specified
        if gauge_ids is None:
            gauge_ids = self.gauge_mapping['id'].tolist()

        # Apply quality filters if configured
        if filter_config:
            gauge_ids = self._apply_quality_filters(
                gauge_ids, start_date, end_date, filter_config
            )

        results['n_total_gauges'] = len(gauge_ids)

        kge_values: List[float] = []
        valid_weights: List[float] = []

        # One open of the simulation output for the whole evaluation.
        self._sim_session_active = True
        try:
            self._score_gauges(
                gauge_ids, mizuroute_output_path, topology_path, start_date, end_date,
                min_overlap_days, weights, results, kge_values, valid_weights,
            )
        finally:
            self._sim_session_active = False
            self._close_simulation()

        return self._aggregate_gauge_scores(
            results, kge_values, valid_weights, min_gauges, aggregation, kge_floor
        )

    def _score_gauges(
        self,
        gauge_ids: List[int],
        mizuroute_output_path: Path,
        topology_path: Optional[Path],
        start_date: Optional[str],
        end_date: Optional[str],
        min_overlap_days: int,
        weights: Optional[Dict[int, float]],
        results: Dict[str, Any],
        kge_values: List[float],
        valid_weights: List[float],
    ) -> None:
        """KGE at every gauge; fills ``results['per_gauge']`` and the two lists."""
        for gauge_id in gauge_ids:
            # Get segment ID for this gauge
            gauge_row = self.gauge_mapping[self.gauge_mapping['id'] == gauge_id]
            if len(gauge_row) == 0:
                self.logger.warning(f"Gauge {gauge_id} not in mapping")
                continue

            segment_id = int(gauge_row['nearest_segment'].iloc[0])
            gauge_name = gauge_row['name'].iloc[0] if 'name' in gauge_row.columns else f"Gauge_{gauge_id}"

            # Load observed data
            obs = self._load_observed_streamflow(gauge_id, start_date, end_date)
            if obs is None or len(obs) == 0:
                self.logger.debug(f"No observations for gauge {gauge_id}")
                results['per_gauge'][gauge_id] = {
                    'name': gauge_name,
                    'segment': segment_id,
                    'kge': None,
                    'reason': 'no_observations'
                }
                continue

            # Extract simulated at this segment
            sim = self._extract_simulated_at_segment(
                mizuroute_output_path, segment_id, topology_path
            )
            if sim is None or len(sim) == 0:
                self.logger.debug(f"No simulation for segment {segment_id}")
                results['per_gauge'][gauge_id] = {
                    'name': gauge_name,
                    'segment': segment_id,
                    'kge': None,
                    'reason': 'no_simulation'
                }
                continue

            # Align time series
            common_idx = obs.index.intersection(sim.index)
            if len(common_idx) < min_overlap_days:
                self.logger.debug(
                    f"Insufficient overlap for gauge {gauge_id}: "
                    f"{len(common_idx)} days < {min_overlap_days}"
                )
                results['per_gauge'][gauge_id] = {
                    'name': gauge_name,
                    'segment': segment_id,
                    'kge': None,
                    'reason': 'insufficient_overlap',
                    'n_overlap': len(common_idx)
                }
                continue

            obs_aligned = obs.loc[common_idx].values
            sim_aligned = sim.loc[common_idx].values

            # Calculate KGE
            kge = self._calculate_kge(obs_aligned, sim_aligned)

            # Suppress "Mean of empty slice" warning for gauges with missing data
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Mean of empty slice')
                results['per_gauge'][gauge_id] = {
                    'name': gauge_name,
                    'segment': segment_id,
                    'kge': kge,
                    'n_points': len(common_idx),
                    'obs_mean': float(np.nanmean(obs_aligned)),
                    'sim_mean': float(np.nanmean(sim_aligned))
                }

            if kge > -9998:  # Valid KGE
                kge_values.append(kge)
                if weights and gauge_id in weights:
                    valid_weights.append(weights[gauge_id])
                else:
                    # Default weight by catchment area if available
                    if 'area_calc' in gauge_row.columns:
                        valid_weights.append(gauge_row['area_calc'].iloc[0])
                    else:
                        valid_weights.append(1.0)

    def _aggregate_gauge_scores(
        self,
        results: Dict[str, Any],
        kge_values: List[float],
        valid_weights: List[float],
        min_gauges: int,
        aggregation: str,
        kge_floor: Optional[float],
    ) -> Dict[str, Any]:
        """Aggregate per-gauge KGE into the objective value."""
        results['n_valid_gauges'] = len(kge_values)

        # Log per-gauge diagnostic on first evaluation to help identify problem gauges
        if not MultiGaugeMetrics._first_eval_logged and kge_values:
            MultiGaugeMetrics._first_eval_logged = True
            # Sort gauges by KGE for diagnostic
            gauge_kge_pairs = [
                (gid, info.get('kge'), info.get('name', f'Gauge_{gid}'))
                for gid, info in results['per_gauge'].items()
                if info.get('kge') is not None and info['kge'] > -9998
            ]
            gauge_kge_pairs.sort(key=lambda x: x[1])

            # Log invalid gauges
            invalid = [
                (gid, info.get('reason', 'unknown'), info.get('name', f'Gauge_{gid}'))
                for gid, info in results['per_gauge'].items()
                if info.get('kge') is None
            ]
            if invalid:
                reasons: dict[str, int] = {}
                for _, reason, _ in invalid:
                    reasons[reason] = reasons.get(reason, 0) + 1
                self.logger.debug(
                    f"Multi-gauge diagnostic: {len(invalid)} invalid gauges — "
                    + ", ".join(f"{r}: {c}" for r, c in reasons.items())
                )

            # Log worst and best gauges
            if gauge_kge_pairs:
                n_worst = min(5, len(gauge_kge_pairs))
                n_best = min(5, len(gauge_kge_pairs))
                self.logger.debug(
                    f"Worst {n_worst} gauges: " +
                    ", ".join(f"{name}(ID={gid}): {kge:.3f}"
                             for gid, kge, name in gauge_kge_pairs[:n_worst])
                )
                self.logger.debug(
                    f"Best {n_best} gauges: " +
                    ", ".join(f"{name}(ID={gid}): {kge:.3f}"
                             for gid, kge, name in gauge_kge_pairs[-n_best:])
                )
                n_negative = sum(1 for _, k, _ in gauge_kge_pairs if k < 0)
                if n_negative > 0:
                    self.logger.debug(
                        f"  {n_negative}/{len(gauge_kge_pairs)} gauges have KGE < 0 "
                        f"(worse than climatology). Consider excluding or using KGE floor."
                    )

        # Check minimum gauges requirement
        if len(kge_values) < min_gauges:
            self.logger.warning(
                f"Only {len(kge_values)} valid gauges (minimum: {min_gauges})"
            )
            return results

        # Aggregate KGE values
        kge_array = np.array(kge_values)
        weight_array = np.array(valid_weights)

        # Store raw statistics before any floor is applied
        results['kge_std'] = float(np.std(kge_array))
        results['kge_min'] = float(np.min(kge_array))
        results['kge_max'] = float(np.max(kge_array))
        results['kge_median'] = float(np.median(kge_array))

        # Apply KGE floor if specified — prevents structurally unfittable gauges
        # (regulated rivers, glacier-fed basins) from dominating the objective function
        n_floored = 0
        if kge_floor is not None:
            n_floored = int(np.sum(kge_array < kge_floor))
            kge_array = np.maximum(kge_array, kge_floor)

        if aggregation == 'mean':
            results['kge'] = float(np.mean(kge_array))
        elif aggregation == 'median':
            results['kge'] = float(np.median(kge_array))
        elif aggregation == 'weighted':
            # A gauge without a usable weight (NaN / non-positive catchment
            # area) cannot take part in a weighted mean; one such gauge used
            # to turn the whole aggregate into NaN.
            weight_array = weight_array.astype(float)
            usable = np.isfinite(weight_array) & (weight_array > 0)
            if not usable.all():
                self.logger.warning(
                    f"Weighted multi-gauge aggregation: {int((~usable).sum())} of "
                    f"{len(weight_array)} gauges have no finite positive weight "
                    f"(catchment area) and are left out of the weighted mean"
                )
            results['n_weighted_gauges'] = int(usable.sum())
            if not usable.any():
                self.logger.error(
                    "Weighted multi-gauge aggregation: no gauge has a finite "
                    "positive weight; cannot aggregate"
                )
                return results
            normalised = weight_array[usable] / weight_array[usable].sum()
            results['kge'] = float(np.sum(kge_array[usable] * normalised))
        elif aggregation == 'min':
            # Worst gauge performance (conservative)
            results['kge'] = float(np.min(kge_array))
        else:
            results['kge'] = float(np.mean(kge_array))

        floor_info = f", {n_floored} floored at {kge_floor}" if n_floored > 0 else ""
        self.logger.info(
            f"Multi-gauge metrics: KGE={results['kge']:.4f} ({aggregation}) "
            f"from {len(kge_values)} gauges "
            f"(range: {results['kge_min']:.3f}-{results['kge_max']:.3f}{floor_info})"
        )

        return results

    def get_available_gauges(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_data_points: int = 365
    ) -> List[int]:
        """
        Get list of gauges with sufficient observation data.

        Args:
            start_date: Optional start date for data availability check
            end_date: Optional end date for data availability check
            min_data_points: Minimum number of data points required

        Returns:
            List of gauge IDs with sufficient data
        """
        available = []

        for _, row in self.gauge_mapping.iterrows():
            gauge_id = int(row['id'])
            obs = self._load_observed_streamflow(gauge_id, start_date, end_date)

            if obs is not None and len(obs.dropna()) >= min_data_points:
                available.append(gauge_id)

        self.logger.info(
            f"Found {len(available)} gauges with >= {min_data_points} data points"
        )
        return available


def create_multi_gauge_config(
    gauge_segment_mapping_path: str,
    obs_data_dir: str,
    gauge_ids: Optional[List[int]] = None,
    aggregation: str = 'mean',
    min_gauges: int = 5
) -> Dict[str, Any]:
    """
    Create configuration dictionary for multi-gauge calibration.

    Args:
        gauge_segment_mapping_path: Path to gauge-segment mapping CSV
        obs_data_dir: Path to observed streamflow directory
        gauge_ids: Optional list of specific gauge IDs to use
        aggregation: Aggregation method ('mean', 'median', 'weighted', 'min')
        min_gauges: Minimum number of valid gauges required

    Returns:
        Configuration dictionary to add to SYMFLUENCE config
    """
    return {
        'MULTI_GAUGE_CALIBRATION': True,
        'GAUGE_SEGMENT_MAPPING': gauge_segment_mapping_path,
        'MULTI_GAUGE_OBS_DIR': obs_data_dir,
        'MULTI_GAUGE_IDS': gauge_ids,
        'MULTI_GAUGE_AGGREGATION': aggregation,
        'MULTI_GAUGE_MIN_GAUGES': min_gauges
    }
