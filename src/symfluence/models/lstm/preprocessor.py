# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
LSTM Model Preprocessor.

Handles data loading, cleaning, normalization, and tensor conversion for the LSTM model.
"""

import glob
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import xarray as xr
from sklearn.preprocessing import StandardScaler

from symfluence.models.base import BaseModelPreProcessor
from symfluence.models.registry import ModelRegistry
from symfluence.models.spatial_modes import SpatialMode

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


@ModelRegistry.register_preprocessor('LSTM')
class LSTMPreProcessor(BaseModelPreProcessor):
    """
    Handles data preprocessing for the LSTM model.

    Extends BaseModelPreProcessor to provide standard path resolution and
    configuration access while adding ML-specific functionality like data
    scaling and sequence creation.

    Attributes:
        config: SymfluenceConfig instance or dict
        logger: Logger instance
        project_dir: Path to project directory
        lookback: Number of time steps to look back
        device: PyTorch device for tensor allocation
        feature_scaler: StandardScaler for input features
        target_scaler: StandardScaler for target variables
        output_size: Number of output variables
        target_names: Names of target variables
        spatial_mode: 'lumped' or 'distributed' based on domain method
    """


    MODEL_NAME = "LSTM"
    def __init__(
        self,
        config: Union['SymfluenceConfig', Dict[str, Any]],
        logger: logging.Logger,
        project_dir: Optional[Path] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the LSTM preprocessor.

        Sets up data loading paths, scalers for normalization, and spatial
        mode detection for distributed vs lumped operation.

        Args:
            config: SymfluenceConfig instance or configuration dictionary
                containing LSTM hyperparameters (lookback window, use_snow flag)
                and domain settings.
            logger: Logger instance for status messages.
            project_dir: Optional path to project directory. If provided,
                overrides the path derived from config. Kept for backward
                compatibility with existing callers.
            device: Optional PyTorch device for tensor allocation (CPU or CUDA).
                Defaults to CPU if not provided.

        Note:
            The lookback window determines how many historical timesteps
            the LSTM uses for each prediction. Default is 30 timesteps.
        """
        # Call parent init for path resolution and config access
        super().__init__(config, logger)

        # Override project_dir if explicitly provided (backward compatibility)
        if project_dir is not None:
            self.project_dir = project_dir

        # ML-specific setup
        self.device = device if device is not None else torch.device('cpu')
        self.lookback = self._get_config_value(
            lambda: self.config.model.lstm.lookback,
            default=self._get_config_value(lambda: self.config.model.flash.lookback, default=30)
        )

        # Use inherited domain_definition_method from mixin
        self.spatial_mode = 'distributed' if self.domain_definition_method == 'delineate' else 'lumped'

        # Scalers for normalization
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.output_size = 1
        self.target_names = ['streamflow']

    def run_preprocessing(self) -> bool:
        """
        Run LSTM preprocessing.

        For ML models, data loading happens during training, so this method
        just logs the status and returns success.

        Returns:
            True indicating preprocessing is ready
        """
        self.logger.info("LSTM preprocessing - data loaded during training")
        return True

    def _prepare_forcing(self) -> None:
        """No-op for ML models - forcing handled in load_data()."""
        pass

    def _create_model_configs(self) -> None:
        """No-op for ML models - no config files needed."""
        pass

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load forcing, streamflow, and snow data from disk.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Forcing, streamflow, and snow dataframes.
        """
        self.logger.info("Loading data for LSTM model")

        # Load forcing data
        forcing_path = self.project_forcing_dir / 'basin_averaged_data'
        self.logger.info(f"Looking for forcing files in: {forcing_path}")

        # Check if directory exists
        if not forcing_path.exists():
            self.logger.error(f"Forcing path does not exist: {forcing_path}")
        else:
            self.logger.info(f"Directory exists. Contents: {list(forcing_path.glob('*'))}")

        forcing_files = glob.glob(str(forcing_path / '*.nc'))
        self.logger.info(f"Found forcing files: {forcing_files}")

        if not forcing_files:
            raise FileNotFoundError(f"No forcing files found in {forcing_path}")

        forcing_files.sort()
        datasets = [xr.open_dataset(file) for file in forcing_files]
        combined_ds = xr.concat(datasets, dim='time', data_vars='all')
        forcing_df = combined_ds.to_dataframe().reset_index()

        # Auto-rename legacy SUMMA-style variable names to CFIF if present
        from symfluence.data.preprocessing.cfif.variables import SUMMA_TO_CFIF_MAPPING
        legacy_col_renames = {k: v for k, v in SUMMA_TO_CFIF_MAPPING.items() if k in forcing_df.columns and v not in forcing_df.columns}
        if legacy_col_renames:
            forcing_df = forcing_df.rename(columns=legacy_col_renames)

        required_vars = ['hruId', 'time', 'precipitation_flux', 'surface_downwelling_shortwave_flux', 'surface_downwelling_longwave_flux', 'surface_air_pressure', 'air_temperature', 'specific_humidity', 'wind_speed']
        missing_vars = [var for var in required_vars if var not in forcing_df.columns]
        if missing_vars:
            raise ValueError(f"Missing required variables in forcing data: {missing_vars}")

        forcing_df['time'] = pd.to_datetime(forcing_df['time'])
        forcing_df = forcing_df.set_index(['time', 'hruId']).sort_index()

        # Load streamflow data
        streamflow_path = (
            self.project_observations_dir / 'streamflow' / 'preprocessed' /
            f"{self.domain_name}_streamflow_processed.csv"
        )

        if not streamflow_path.exists():
            # Fallback for legacy naming
            legacy_path = self.project_observations_dir / 'streamflow' / 'preprocessed' / "Bow_at_Banff_lumped_streamflow_processed.csv"
            if legacy_path.exists():
                streamflow_path = legacy_path
                self.logger.info(f"Using legacy streamflow path: {streamflow_path.name}")

        streamflow_df = pd.read_csv(streamflow_path, parse_dates=['datetime'], dayfirst=True)
        streamflow_df = streamflow_df.set_index('datetime').rename(columns={'discharge_cms': 'streamflow'})
        streamflow_df.index = pd.to_datetime(streamflow_df.index)

        # Load snow data
        snow_path = self.project_observations_dir / 'snow' / 'preprocessed'
        snow_files = glob.glob(str(snow_path / f"{self.domain_name}_filtered_snow_observations.csv"))

        if snow_files:
            snow_df = pd.concat([pd.read_csv(file, parse_dates=['datetime'], dayfirst=True) for file in snow_files])
            # Aggregate snow data across all stations
            snow_df = snow_df.groupby('datetime')['snw'].mean().reset_index()
            snow_df['datetime'] = pd.to_datetime(snow_df['datetime'])
            snow_df = snow_df.set_index('datetime')
        else:
            self.logger.warning(f"No snow observation files found in {snow_path}. Using empty DataFrame.")
            snow_df = pd.DataFrame() # Return empty DF if not found, to handle gracefully

        # Ensure all datasets cover the same time period
        if not snow_df.empty:
            start_date = max(
                forcing_df.index.get_level_values('time').min(),
                streamflow_df.index.min(),
                snow_df.index.min()
            )
            end_date = min(
                forcing_df.index.get_level_values('time').max(),
                streamflow_df.index.max(),
                snow_df.index.max()
            )
            snow_df = snow_df.loc[start_date:end_date]
            snow_df = snow_df.resample('h').interpolate(method='linear')
        else:
            start_date = max(
                forcing_df.index.get_level_values('time').min(),
                streamflow_df.index.min()
            )
            end_date = min(
                forcing_df.index.get_level_values('time').max(),
                streamflow_df.index.max()
            )

        forcing_df = forcing_df.loc[pd.IndexSlice[start_date:end_date, :], :]
        streamflow_df = streamflow_df.loc[start_date:end_date]

        self.logger.info(f"Loaded forcing data with shape: {forcing_df.shape}")
        self.logger.info(f"Loaded streamflow data with shape: {streamflow_df.shape}")
        if not snow_df.empty:
            self.logger.info(f"Loaded snow data with shape: {snow_df.shape}")

        return forcing_df, streamflow_df, snow_df

    def process_data(
        self,
        forcing_df: pd.DataFrame,
        streamflow_df: pd.DataFrame,
        snow_df: Optional[pd.DataFrame] = None,
        fit_scalers: bool = True,
        train_end_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, pd.DatetimeIndex, pd.DataFrame, List[int]]:
        """
        Preprocess data for LSTM model (clean, scale, sequence).

        Args:
            forcing_df: DataFrame containing forcing data.
            streamflow_df: DataFrame containing streamflow data.
            snow_df: Optional DataFrame containing snow data.
            fit_scalers: Whether to fit new scalers or use existing ones.
            train_end_idx: If provided and fit_scalers=True, scalers are fitted
                only on data up to this index to prevent data leakage from
                validation/test data. This should be the number of unique
                timesteps in the training set.

        Returns:
            Tuple containing:
                - X (torch.Tensor): Input sequences.
                - y (torch.Tensor): Target values.
                - common_dates (pd.DatetimeIndex): Dates corresponding to the data.
                - features_avg (pd.DataFrame): Averaged features dataframe.
                - hru_ids (List[int]): List of HRU IDs.
        """
        self.logger.info(f"Preprocessing data (fit_scalers={fit_scalers}, mode={self.spatial_mode})")

        # Align the data
        common_dates = forcing_df.index.get_level_values('time').unique().intersection(streamflow_df.index)
        if snow_df is not None and not snow_df.empty:
            common_dates = common_dates.intersection(snow_df.index)

        forcing_df = forcing_df.loc[pd.IndexSlice[common_dates, :], :]
        streamflow_df = streamflow_df.loc[common_dates]
        if snow_df is not None and not snow_df.empty:
            snow_df = snow_df.loc[common_dates]

        # Get HRU IDs
        hru_ids = forcing_df.index.get_level_values('hruId').unique().tolist()
        n_hrus = len(hru_ids)

        # Prepare features (forcing data)
        features = forcing_df.reset_index()
        feature_columns = features.columns.drop(
            ['time', 'hruId', 'hru', 'latitude', 'longitude']
            if 'time' in features.columns and 'hruId' in features.columns else []
        )

        if self.spatial_mode == SpatialMode.LUMPED:
            # Average features across all HRUs for each timestep
            features_to_scale = forcing_df.groupby('time')[feature_columns].mean()
            features_avg = features_to_scale.copy()
        else:
            # Distributed mode: scale features for all HRUs together
            # We treat all HRUs as samples for the same scaler
            features_to_scale = forcing_df[feature_columns]
            features_avg = forcing_df[feature_columns] # Keep indexed by [time, hruId]

        # Scale features
        if fit_scalers:
            if train_end_idx is not None:
                # Fit scaler only on training data to prevent data leakage
                self.feature_scaler.fit(features_to_scale[:train_end_idx])
            else:
                self.feature_scaler.fit(features_to_scale)
            scaled_features = self.feature_scaler.transform(features_to_scale)
        else:
            scaled_features = self.feature_scaler.transform(features_to_scale)

        scaled_features = np.clip(scaled_features, -10, 10)

        # Prepare targets (streamflow and optionally snow)
        if snow_df is not None and not snow_df.empty:
            targets_raw = pd.concat([streamflow_df['streamflow'], snow_df['snw']], axis=1)
            targets_raw.columns = ['streamflow', 'SWE']
            if fit_scalers:
                self.output_size = 2
                self.target_names = ['streamflow', 'SWE']
        else:
            targets_raw = pd.DataFrame(streamflow_df['streamflow'], columns=['streamflow'])
            if fit_scalers:
                self.output_size = 1
                self.target_names = ['streamflow']

        if self.spatial_mode == SpatialMode.DISTRIBUTED:
            # Repeat targets for each HRU if they are basin-wide
            # (In training through routing, we might want per-HRU targets if available,
            # but usually streamflow is only at outlet. If so, we can use 0 or dummy for internal HRUs
            # OR broadcast the outlet streamflow as a hint - but usually we route.)
            # For now, we broadcast to allow sequence creation, but training logic will handle routing.
            targets_to_scale = np.repeat(targets_raw.values, n_hrus, axis=0)
        else:
            targets_to_scale = targets_raw.values

        # Scale targets
        if fit_scalers:
            if train_end_idx is not None:
                # Fit scaler only on training data to prevent data leakage
                # For distributed mode, we need to account for n_hrus
                if self.spatial_mode == SpatialMode.DISTRIBUTED:
                    n_hrus = len(hru_ids)
                    train_samples = train_end_idx * n_hrus
                    self.target_scaler.fit(targets_to_scale[:train_samples])
                else:
                    self.target_scaler.fit(targets_to_scale[:train_end_idx])
            else:
                self.target_scaler.fit(targets_to_scale)
            scaled_targets = self.target_scaler.transform(targets_to_scale)
        else:
            scaled_targets = self.target_scaler.transform(targets_to_scale)

        scaled_targets = np.clip(scaled_targets, -10, 10)

        # Create sequences
        X, y = [], []

        if self.spatial_mode == SpatialMode.LUMPED:
            for i in range(len(scaled_features) - self.lookback):
                X.append(scaled_features[i:(i + self.lookback)])
                y.append(scaled_targets[i + self.lookback])

            X_tensor = torch.FloatTensor(np.array(X)).to(self.device)
            y_tensor = torch.FloatTensor(np.array(y)).to(self.device)
        else:
            # Distributed mode: Sequences are (timesteps, hrus, features)
            # Reshape scaled_features to (timesteps, hrus, n_features)
            n_timesteps = len(common_dates)
            n_features = len(feature_columns)
            feat_reshaped = scaled_features.reshape(n_timesteps, n_hrus, n_features)
            targ_reshaped = scaled_targets.reshape(n_timesteps, n_hrus, self.output_size)

            for i in range(n_timesteps - self.lookback):
                X.append(feat_reshaped[i:(i + self.lookback)]) # (lookback, hrus, features)
                y.append(targ_reshaped[i + self.lookback])     # (hrus, output_size)

            X_tensor = torch.FloatTensor(np.array(X)).to(self.device) # (B, T, N, F)
            y_tensor = torch.FloatTensor(np.array(y)).to(self.device) # (B, N, O)

        self.logger.info(f"Preprocessed data shape: X: {X_tensor.shape}, y: {y_tensor.shape}")
        return X_tensor, y_tensor, pd.DatetimeIndex(common_dates), features_avg, hru_ids

    def set_scalers(self, feature_scaler, target_scaler, output_size, target_names):
        """
        Set scalers and metadata from a saved checkpoint.

        Used when loading a pre-trained model to restore the exact normalization
        parameters used during training. This ensures predictions are consistent
        with the training data distribution.

        Args:
            feature_scaler: Fitted StandardScaler for input features.
            target_scaler: Fitted StandardScaler for target variables.
            output_size: Number of output variables (1 for streamflow only,
                2 if including SWE).
            target_names: List of target variable names matching output_size.
        """
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.output_size = output_size
        self.target_names = target_names
