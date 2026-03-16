# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Hydrological model benchmarking using HydroBM library.

Provides benchmark calculation against simple models (mean flow, rainfall-runoff
ratios, etc.) with configurable calibration/validation period splitting.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import xarray as xr
from hydrobm.calculate import calc_bm

from symfluence.core.mixins import ConfigMixin
from symfluence.core.mixins.project import ProjectContextMixin, resolve_data_subdir
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

    def run_benchmarking(self) -> Dict[str, Any]:
        """Run hydrobm benchmarking using preprocessed data with year-based split."""
        self.logger.info("Starting hydrobm benchmarking")

        try:
            # Load and prepare data
            input_data = self.load_preprocessed_data()
            input_data = input_data.dropna()

            # Get unique years and find split point
            years = sorted(input_data.index.year.unique())
            mid_year = years[len(years) // 2]

            self.logger.info(f"Available years: {years}")
            self.logger.info(f"Split year: {mid_year}")

            # Split the data
            cal_data = input_data[input_data.index.year < mid_year]
            val_data = input_data[input_data.index.year >= mid_year]

            # Debug information
            self.logger.info("\nPeriod Information:")
            self.logger.info(f"Calibration: {cal_data.index[0]} to {cal_data.index[-1]} ({len(cal_data)} points)")
            self.logger.info(f"Validation: {val_data.index[0]} to {val_data.index[-1]} ({len(val_data)} points)")
            self.logger.info(f"Calibration years: {sorted(cal_data.index.year.unique())}")
            self.logger.info(f"Validation years: {sorted(val_data.index.year.unique())}")

            # Sample data
            self.logger.info("\nCalibration data sample:")
            self.logger.info(cal_data.head())
            self.logger.info("\nValidation data sample:")
            self.logger.info(val_data.head())

            # Convert to xarray and create masks
            data = input_data.to_xarray()

            # Create masks based on years
            cal_mask = data.indexes[list(data.coords.keys())[0]].year < mid_year
            val_mask = data.indexes[list(data.coords.keys())[0]].year >= mid_year

            # Verify the split
            n_cal = cal_mask.sum()
            n_val = val_mask.sum()

            if n_cal < 30 or n_val < 30:
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
            benchmark_flows, scores = calc_bm(
                data,
                cal_mask,
                val_mask=val_mask,
                precipitation="precipitation",
                streamflow="streamflow",
                benchmarks=benchmarks,
                metrics=metrics,
                calc_snowmelt=True,
                temperature="temperature",
                snowmelt_threshold=273.15,
                snowmelt_rate=3.0
            )

            scores_df = pd.DataFrame(scores)
            self._save_results(benchmark_flows, scores_df)

            # Create return dictionary with basic types
            return {
                "benchmark_flows": benchmark_flows,
                "scores": scores_df,
                "cal_period": {
                    "start": str(cal_data.index[0]),
                    "end": str(cal_data.index[-1]),
                    "points": int(n_cal),
                    "years": [int(year) for year in sorted(cal_data.index.year.unique())]
                },
                "val_period": {
                    "start": str(val_data.index[0]),
                    "end": str(val_data.index[-1]),
                    "points": int(n_val),
                    "years": [int(year) for year in sorted(val_data.index.year.unique())]
                }
            }

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

class BenchmarkPreprocessor(ProjectContextMixin):
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
        # Use open_mfdataset to load all netCDF files at once efficiently
        combined_ds = xr.open_mfdataset(list(forcing_path.glob("*.nc")), combine='nested', concat_dim='time', data_vars='minimal', coords='minimal', compat='override').sortby('time')
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
