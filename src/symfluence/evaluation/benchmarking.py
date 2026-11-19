# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Hydrological model benchmarking using HydroBM library.

Provides benchmark calculation against simple models (mean flow, rainfall-runoff
ratios, etc.). Benchmarks are fitted on CALIBRATION_PERIOD and scored on the
configured calibration, EVALUATION_PERIOD and TEST_PERIOD windows -- the same
windows the model is scored on.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from hydrobm.calculate import calc_bm

from symfluence.core.mixins import ConfigMixin
from symfluence.core.mixins.project import ProjectContextMixin, resolve_data_subdir
from symfluence.core.path_resolver import PathResolverMixin
from symfluence.data.observation.paths import first_existing_path, streamflow_observation_candidates


class Benchmarker(ConfigMixin):
    def __init__(self, config: dict, logger):
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger
        self.project_dir = Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR')) / f"domain_{self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')}"
        self.evaluation_dir = self.project_dir / 'evaluation'
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)

    def load_preprocessed_data(self) -> pd.DataFrame:
        """Load preprocessed benchmark input data."""
        data_path = self.evaluation_dir / "benchmark_input_data.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Preprocessed data not found at {data_path}")

        data = pd.read_csv(data_path, index_col=0)
        data.index = pd.to_datetime(data.index)
        return data

    def _configured_masks(self, index: pd.DatetimeIndex) -> Tuple[Dict[str, np.ndarray], Optional[str]]:
        """Boolean masks for the configured calibration/evaluation/test periods.

        Benchmarks are scored on exactly the periods the model is scored on.
        Spin-up (SPINUP_PERIOD / deprecated EVALUATION_SPINUP_YEARS) is
        excluded from every mask. Without CALIBRATION_PERIOD the old
        median-year split is used, with a warning (returned as the second
        element) because it ignores the configured design.
        """
        from symfluence.core.modeling.utilities.periods import (
            parse_period,
            period_end_exclusive,
            spinup_window,
        )

        def _period(getter, key):
            value = self._get_config_value(getter, default='', dict_key=key)
            if not isinstance(value, str) or not value.strip():
                return None
            try:
                return parse_period(value, allow_extra=True)
            except (ValueError, TypeError):
                return None

        def _mask(period):
            return np.asarray((index >= period[0]) & (index < period_end_exclusive(period[1])))

        window = spinup_window(
            self._get_config_value(lambda: self.config.domain.spinup_period, default='', dict_key='SPINUP_PERIOD'),
            self._get_config_value(lambda: self.config.evaluation.spinup_years, default=0, dict_key='EVALUATION_SPINUP_YEARS'),
            self._get_config_value(lambda: self.config.domain.time_start, default=None, dict_key='EXPERIMENT_TIME_START'),
        )
        not_spinup = np.asarray(index >= window[1]) if window is not None else np.ones(len(index), dtype=bool)

        cal = _period(lambda: self.config.domain.calibration_period, 'CALIBRATION_PERIOD')
        val = _period(lambda: self.config.domain.evaluation_period, 'EVALUATION_PERIOD')
        test = _period(lambda: self.config.domain.test_period, 'TEST_PERIOD')

        masks: Dict[str, np.ndarray] = {}
        warning: Optional[str] = None
        if cal is not None:
            masks['cal'] = _mask(cal) & not_spinup
            if val is not None:
                masks['val'] = _mask(val) & not_spinup
            if test is not None:
                masks['test'] = _mask(test) & not_spinup
        else:
            years = sorted(index.year.unique())
            mid_year = years[len(years) // 2]
            warning = (
                "CALIBRATION_PERIOD is not set: benchmarks are split at the median year "
                f"({mid_year}) instead of the configured split-sample periods"
            )
            masks['cal'] = np.asarray(index.year < mid_year) & not_spinup
            masks['val'] = np.asarray(index.year >= mid_year) & not_spinup
        return masks, warning

    @staticmethod
    def _period_info(index: pd.DatetimeIndex, mask: np.ndarray) -> Dict[str, Any]:
        sel = index[mask]
        if len(sel) == 0:
            return {"start": None, "end": None, "points": 0, "years": []}
        return {
            "start": str(sel[0]),
            "end": str(sel[-1]),
            "points": int(mask.sum()),
            "years": [int(y) for y in sorted(sel.year.unique())],
        }

    def run_benchmarking(self) -> Dict[str, Any]:
        """Run hydrobm benchmarking on the configured split-sample periods.

        Benchmarks are fitted on CALIBRATION_PERIOD and scored on the
        calibration, EVALUATION_PERIOD and (optional) TEST_PERIOD windows, the
        same windows used to score the model.
        """
        self.logger.info("Starting hydrobm benchmarking")

        try:
            # Load and prepare data
            input_data = self.load_preprocessed_data()
            input_data = input_data.dropna()

            # Convert to xarray and create masks
            data = input_data.to_xarray()
            time_index = pd.DatetimeIndex(data.indexes[list(data.coords.keys())[0]])
            masks, split_warning = self._configured_masks(time_index)
            if split_warning:
                self.logger.warning(f"SPLIT-SAMPLE WARNING (benchmarks): {split_warning}")

            cal_mask = masks['cal']
            val_mask = masks.get('val')
            test_mask = masks.get('test')

            n_cal = int(cal_mask.sum())
            n_val = int(val_mask.sum()) if val_mask is not None else 0
            self.logger.info(
                f"Benchmark periods: calibration {n_cal} points, evaluation {n_val} points"
                + (f", test {int(test_mask.sum())} points" if test_mask is not None else "")
            )

            if n_cal < 30 or (val_mask is not None and n_val < 30):
                raise ValueError(f"Insufficient data in periods: cal={n_cal}, val={n_val}")

            # Define benchmarks
            benchmarks = [
                            # Streamflow benchmarks
                            "mean_flow",
                            "median_flow",
                            "annual_mean_flow",
                            "annual_median_flow",
                            "monthly_mean_flow",
                            "monthly_median_flow",
                            "daily_mean_flow",
                            "daily_median_flow",

                            # Long-term rainfall-runoff ratio benchmarks
                            "rainfall_runoff_ratio_to_all",
                            "rainfall_runoff_ratio_to_annual",
                            "rainfall_runoff_ratio_to_monthly",
                            "rainfall_runoff_ratio_to_daily",
                            "rainfall_runoff_ratio_to_timestep",

                            # Short-term rainfall-runoff ratio benchmarks
                            "monthly_rainfall_runoff_ratio_to_monthly",
                            "monthly_rainfall_runoff_ratio_to_daily",
                            "monthly_rainfall_runoff_ratio_to_timestep",

                            # Schaefli & Gupta (2007) benchmarks
                            "scaled_precipitation_benchmark",  # equivalent to "rainfall_runoff_ratio_to_daily"
                            "adjusted_precipitation_benchmark",
                            "adjusted_smoothed_precipitation_benchmark",
                        ]

            metrics = ['nse', 'kge', 'mse', 'rmse']

            # Run benchmarking
            self.logger.info("Running HydroBM calculations...")
            bm_kwargs = dict(
                precipitation="precipitation",
                streamflow="streamflow",
                benchmarks=benchmarks,
                metrics=metrics,
                calc_snowmelt=True,
                temperature="temperature",
                snowmelt_threshold=273.15,
                snowmelt_rate=3.0,
            )
            benchmark_flows, scores = calc_bm(
                data,
                cal_mask,
                val_mask=val_mask if val_mask is not None else [],
                **bm_kwargs,
            )
            scores_df = pd.DataFrame(scores)

            # Test period: same calibration fit, scored on the test window
            if test_mask is not None and int(test_mask.sum()) > 0:
                _, test_scores = calc_bm(data, cal_mask, val_mask=test_mask, **bm_kwargs)
                test_df = pd.DataFrame(test_scores)
                for col in [c for c in test_df.columns if c.endswith('_val')]:
                    scores_df[col[:-len('_val')] + '_test'] = test_df[col].values

            self._save_results(benchmark_flows, scores_df)

            # Create return dictionary with basic types
            result: Dict[str, Any] = {
                "benchmark_flows": benchmark_flows,
                "scores": scores_df,
                "cal_period": self._period_info(time_index, cal_mask),
                "val_period": (
                    self._period_info(time_index, val_mask) if val_mask is not None else None
                ),
            }
            if test_mask is not None:
                result["test_period"] = self._period_info(time_index, test_mask)
            if split_warning:
                result["period_warning"] = split_warning
            return result

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error during benchmark calculation: {e}")
            raise

    def _save_results(self, benchmark_flows: pd.DataFrame, scores: pd.DataFrame):
        """Save benchmark results to files with clear documentation."""
        try:
            flows_path = self.evaluation_dir / "benchmark_flows.csv"
            benchmark_flows.to_csv(flows_path)

            scores_path = self.evaluation_dir / "benchmark_scores.csv"
            scores.to_csv(scores_path)

            metadata = {
                "date_processed": datetime.now().isoformat(),
                "input_units": {
                    "temperature": "K",
                    "streamflow": "m³/s",
                    "precipitation": "mm/day"
                },
                "data_period": {
                    "start": str(benchmark_flows.index[0]),
                    "end": str(benchmark_flows.index[-1])
                },
                "benchmarks_used": [str(b) for b in scores.index],
                "metrics_used": [str(m) for m in scores.columns]
            }

            with open(self.evaluation_dir / "benchmark_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error saving benchmark results: {str(e)}")
            raise

class BenchmarkPreprocessor(PathResolverMixin, ProjectContextMixin):
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger

    def preprocess_benchmark_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Preprocess data for hydrobm benchmarking.
        """
        self.logger.info("Starting benchmark data preprocessing")

        # Load and process data
        streamflow_data = self._load_streamflow_data()
        forcing_data = self._load_forcing_data()
        merged_data = self._merge_data(streamflow_data, forcing_data)

        # Aggregate to daily timestep using a single resample pass
        daily_data = self._process_to_daily(merged_data)

        # Filter data for the experiment run period
        filtered_data = daily_data.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]

        # Validate and save data
        self._validate_data(filtered_data)
        output_path = self.project_dir / 'evaluation'
        output_path.mkdir(exist_ok=True)
        filtered_data.to_csv(output_path / "benchmark_input_data.csv")

        return filtered_data

    def _process_to_daily(self, data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data to daily values with correct units using a single resample."""
        daily_data = data.resample('D').agg({
            'temperature': 'mean',
            'streamflow': 'mean',
            'precipitation': 'sum'
        })
        return daily_data

    def _validate_data(self, data: pd.DataFrame):
        """Validate data ranges and consistency."""
        missing = data.isnull().sum()
        if missing.any():
            self.logger.warning(f"Missing values detected:\n{missing}")

        if (data['temperature'] < 200).any() or (data['temperature'] > 330).any():
            self.logger.warning("Temperature values outside physical range (200-330 K)")

        if (data['streamflow'] < 0).any():
            self.logger.warning("Negative streamflow values detected")

        if (data['precipitation'] < 0).any():
            self.logger.warning("Negative precipitation values detected")

        if (data['precipitation'] > 1000).any():
            self.logger.warning("Extremely high precipitation values (>1000 mm/day) detected")

        self.logger.info(f"Data statistics:\n{data.describe()}")

    def _load_streamflow_data(self) -> pd.DataFrame:
        """Load and basic process streamflow data."""
        domain_name = self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')
        streamflow_path = first_existing_path(
            streamflow_observation_candidates(self.project_dir, domain_name)
        )
        data = pd.read_csv(streamflow_path, parse_dates=['datetime'], index_col='datetime')
        # Ensure the index is sorted for a faster merge later on
        data.sort_index(inplace=True)
        return data.rename(columns={'discharge_cms': 'streamflow'})

    def _load_forcing_data(self) -> pd.DataFrame:
        """Load and process forcing data, returning hourly dataframe."""
        forcing_path = resolve_data_subdir(self.project_dir, 'forcing') / "basin_averaged_data"
        # Only this run's forcing: the store can hold several HRU sets of the domain (issue #339).
        files = self._select_run_forcing(sorted(forcing_path.glob("*.nc")))
        combined_ds = xr.open_mfdataset(files, combine='nested', concat_dim='time', data_vars='minimal', coords='minimal', compat='override').sortby('time')
        # Average across HRUs
        averaged_ds = combined_ds.mean(dim='hru')

        # Convert precipitation from kg m⁻² s⁻¹ (mm/s) to mm per timestep-hour.
        # Forcing data is hourly; _process_to_daily() sums 24 values to get mm/day.
        precip_data = averaged_ds['precipitation_flux'] * 3600  # mm/s -> mm/hr

        # Create DataFrame directly using to_series for better integration
        forcing_df = pd.DataFrame({
            'temperature': averaged_ds['air_temperature'].to_series(),
            'precipitation': precip_data.to_series()
        })
        forcing_df.sort_index(inplace=True)
        return forcing_df

    def _merge_data(self, streamflow_data: pd.DataFrame, forcing_data: pd.DataFrame) -> pd.DataFrame:
        """Merge streamflow and forcing data on timestamps using concatenation for efficiency."""
        merged_data = pd.concat([streamflow_data, forcing_data], axis=1, join='inner')

        # Verify data completeness
        expected_records = len(pd.date_range(merged_data.index.min(),
                                              merged_data.index.max(),
                                              freq='h'))
        if len(merged_data) != expected_records:
            self.logger.warning(f"Data gaps detected. Expected {expected_records} records, got {len(merged_data)}")

        return merged_data
