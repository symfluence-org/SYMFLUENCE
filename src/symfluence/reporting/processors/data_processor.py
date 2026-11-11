# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Processor for loading and preparing data for visualization.

This module reads the standard simulation results file used by the plotters.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from symfluence.core.constants import ConfigKeys
from symfluence.core.mixins import ConfigMixin
from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.reporting.core.shapefile_helper import ShapefileHelper


class DataProcessor(ConfigMixin):
    """
    Loads the standard results file (simulations aligned with observations)
    for visualization.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the data processor.

        Args:
            config: SYMFLUENCE configuration dictionary
            logger: Logger instance
        """
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger
        self.project_dir = Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key=ConfigKeys.SYMFLUENCE_DATA_DIR)) / f"domain_{self._get_config_value(lambda: self.config.domain.name, dict_key=ConfigKeys.DOMAIN_NAME)}"
        self._shapefile_helper = ShapefileHelper(config, logger, self.project_dir)

    def read_results_file(self) -> pd.DataFrame:
        """
        Read simulation results and observed streamflow from standard results file.

        Returns:
            DataFrame containing aligned simulation and observation data.

        Raises:
            FileNotFoundError: If results file or observations file is missing.
            ValueError: If data is empty or invalid.
        """
        try:
            # Read simulation results
            results_file = self.project_dir / "results" / f"{self._get_config_value(lambda: self.config.domain.experiment_id, dict_key=ConfigKeys.EXPERIMENT_ID)}_results.csv"
            if not results_file.exists():
                raise FileNotFoundError(f"Results file not found: {results_file}")

            # Read the CSV file
            sim_df = pd.read_csv(results_file)

            # Check if the DataFrame is empty
            if sim_df.empty:
                self.logger.error("Results file is empty")
                raise ValueError("Results file contains no data")

            # Convert time column to datetime and set as index
            if 'time' in sim_df.columns:
                sim_df['time'] = pd.to_datetime(sim_df['time'])
                sim_df.set_index('time', inplace=True)
            elif 'datetime' in sim_df.columns:
                sim_df['datetime'] = pd.to_datetime(sim_df['datetime'])
                sim_df.set_index('datetime', inplace=True)
            elif 'Unnamed: 0' in sim_df.columns:
                # Handle case where dates are in 'Unnamed: 0' column
                sim_df['datetime'] = pd.to_datetime(sim_df['Unnamed: 0'])
                sim_df.set_index('datetime', inplace=True)
                sim_df.drop('Unnamed: 0', axis=1, errors='ignore')
            elif sim_df.index.name is None and isinstance(sim_df.index, pd.Index):
                # Try to convert the unnamed index to datetime
                try:
                    sim_df.index = pd.to_datetime(sim_df.index)
                except (ValueError, pd.errors.ParserError, OverflowError):
                    raise ValueError("Index cannot be converted to datetime format") from None
            else:
                raise ValueError("No time or datetime column found in results file")

            # Read observations
            obs_file_path = self._get_config_value(lambda: self.config.paths.observations_path, dict_key=ConfigKeys.OBSERVATIONS_PATH)
            if obs_file_path == 'default' or obs_file_path is None:
                obs_file_path = resolve_data_subdir(self.project_dir, 'observations') / 'streamflow' / 'preprocessed' / f"{self._get_config_value(lambda: self.config.domain.name, dict_key=ConfigKeys.DOMAIN_NAME)}_streamflow_processed.csv"
            else:
                obs_file_path = Path(obs_file_path)

            if not obs_file_path.exists():
                self.logger.warning(f"Observations file not found: {obs_file_path}")
                # If observations are missing, just return simulations
                return sim_df

            obs_df = pd.read_csv(obs_file_path)

            if obs_df.empty:
                self.logger.warning("Observations file is empty")
                return sim_df

            # Process observations
            obs_df['datetime'] = pd.to_datetime(obs_df['datetime'], format='mixed')
            obs_df.set_index('datetime', inplace=True)

            # Resample to daily if needed (assuming results are daily based on original logic)
            # Original logic resampled observations to daily. Let's keep that default but be mindful.
            obs_series = obs_df['discharge_cms'].resample('D').mean()

            # Combine into single dataframe
            results_df = sim_df.copy()
            results_df['Observed'] = obs_series

            # Skip first year (spin-up period) - preserving original logic
            if len(results_df.index) > 0:
                start_time = results_df.index[0]
                end_time = results_df.index[-1]
                duration_days = (end_time - start_time).days

                if duration_days > 365:
                    first_year = results_df.index[0].year
                    start_date = pd.Timestamp(year=first_year + 1, month=1, day=1)
                    results_df = results_df[results_df.index >= start_date]
                else:
                    self.logger.warning(f"Simulation duration ({duration_days} days) is short. Skipping first year removal to ensure data availability.")

            self.logger.info(f"Data period: {results_df.index[0]} to {results_df.index[-1]}")
            return results_df

        except (
            FileNotFoundError,
            OSError,
            KeyError,
            ValueError,
            TypeError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            UnicodeDecodeError,
            RuntimeError,
            AttributeError,
            IndexError,
        ) as e:
            self.logger.error(f"Error reading results: {str(e)}")
            raise
