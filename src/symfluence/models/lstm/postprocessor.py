# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
LSTM Model Postprocessor.

Handles result saving, visualization, and metric calculation for the LSTM model.
Migrated to extend StandardModelPostProcessor for extract_streamflow (Phase 1.4).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, cast

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.metrics import kge, kge_np, kge_prime, mae, nse, rmse
from symfluence.core.modeling.base import StandardModelPostProcessor
from symfluence.core.registries import R


@R.postprocessors.add('LSTM')
class LSTMPostProcessor(StandardModelPostProcessor):
    """
    Handles postprocessing for the LSTM model.

    Extends StandardModelPostProcessor for standard streamflow extraction,
    while keeping custom save_results() for LSTM's dual-mode output format.

    Attributes:
        model_name: "LSTM"
        output_file_pattern: "{experiment}_LSTM_output.nc"
        streamflow_variable: "predicted_streamflow"
        streamflow_unit: "cms"
    """

    # Model identification
    model_name = "LSTM"

    # NetCDF output configuration
    output_file_pattern = "{experiment}_LSTM_output.nc"
    streamflow_variable = "predicted_streamflow"
    streamflow_unit = "cms"

    def _get_model_name(self) -> str:
        return "LSTM"

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract streamflow from LSTM output.

        For LSTM, results are typically passed directly to save_results by the runner.
        This method is implemented to satisfy the abstract base class and can be used
        if we need to reload from the NetCDF file.

        Returns:
            Optional[Path]: Path to saved streamflow CSV, or None if extraction fails
        """
        try:
            output_file = self.sim_dir / f"{self.experiment_id}_LSTM_output.nc"
            if not output_file.exists():
                self.logger.warning(f"LSTM output file not found: {output_file}")
                return None

            ds = xr.open_dataset(output_file)
            if self.streamflow_variable in ds:
                streamflow = ds[self.streamflow_variable].to_series()
                ds.close()
                return self.save_streamflow_to_results(streamflow)

            ds.close()
            self.logger.warning(
                f"Variable '{self.streamflow_variable}' not found in {output_file}"
            )
            return None

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error extracting LSTM streamflow: {str(e)}", exc_info=True)
            return None

    def save_results(
        self,
        results: pd.DataFrame,
        use_snow: bool,
        hru_ids: Optional[List[int]] = None
    ):
        """
        Save LSTM model results to disk as NetCDF and CSV (for standardization).

        Handles both lumped and distributed modes with appropriate output formats:
        - Lumped mode: Simple time series output
        - Distributed mode: Multi-HRU output with gruId dimension for routing

        Args:
            results: Simulation results dataframe with predicted_streamflow column
            use_snow: Whether snow was simulated (adds scalarSWE variable)
            hru_ids: Optional list of HRU IDs for distributed mode

        Returns:
            Path to saved NetCDF file
        """
        self.logger.info("Saving LSTM model results")

        # Prepare the output directory
        self.sim_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.sim_dir / f"{self.experiment_id}_LSTM_output.nc"

        # Check if distributed
        is_distributed = hru_ids is not None and len(hru_ids) > 1

        if not is_distributed:
            # Handle case where results may have MultiIndex from distributed mode with single HRU
            if isinstance(results.index, pd.MultiIndex):
                # Flatten MultiIndex to just time
                results = results.reset_index(level='hruId', drop=True)
                if results.index.duplicated().any():
                    results = results[~results.index.duplicated(keep='first')]

            # Initialize dataset dictionary with streamflow
            data_vars = {
                "predicted_streamflow": (["time"], results['predicted_streamflow'].values)
            }

            # Add SWE if enabled in config
            if use_snow and 'predicted_SWE' in results.columns:
                data_vars["scalarSWE"] = (["time"], results['predicted_SWE'].values)

            # Get time coordinate - ensure it's a simple DatetimeIndex, not MultiIndex
            time_coord = results.index
            if hasattr(time_coord, 'name') and time_coord.name == 'time':
                time_coord = pd.DatetimeIndex(time_coord, name=None)

            # Create xarray Dataset
            ds = xr.Dataset(
                data_vars,
                coords={
                    "time": time_coord
                }
            )

            # Add attributes
            ds.predicted_streamflow.attrs['units'] = 'm3 s-1'
            ds.predicted_streamflow.attrs['long_name'] = 'Routed streamflow'

            # Save Streamflow to Standardized CSV (Triggers Plotting)
            self.save_streamflow_to_results(
                results['predicted_streamflow'],
                model_column_name=f"{self.model_name}_discharge_cms"
            )
        else:
            # Distributed mode: Unflatten results to (time, gru)
            # results index is [time, hruId]
            pivot_df = results.reset_index().pivot(
                index='time', columns='hruId', values='predicted_streamflow'
            )

            # Reorder columns to match hru_ids
            pivot_df = pivot_df[hru_ids]

            n_hrus = len(hru_ids)

            # Create Dataset compatible with mizuRoute/dRoute
            ds = xr.Dataset(
                coords={
                    'time': pivot_df.index,
                    'gru': np.arange(n_hrus)
                }
            )

            ds['gruId'] = (['gru'], np.array(hru_ids, dtype='int32'))

            # Use same var name as configured for routing if possible
            routing_var = self._get_config_value(
                lambda: self.config.model.mizuroute.routing_var,
                dict_key='SETTINGS_MIZU_ROUTING_VAR',
            )
            if routing_var == 'default':
                routing_var = 'averageRoutedRunoff'

            ds[routing_var] = (['time', 'gru'], pivot_df.values)
            ds[routing_var].attrs['units'] = 'm3 s-1'  # Default LSTM output is CMS
            ds[routing_var].attrs['long_name'] = 'LSTM runoff'

            # Add SWE if enabled
            if use_snow and 'predicted_SWE' in results.columns:
                swe_pivot = results.reset_index().pivot(
                    index='time', columns='hruId', values='predicted_SWE'
                )
                swe_pivot = swe_pivot[hru_ids]
                ds['scalarSWE'] = (['time', 'gru'], swe_pivot.values)
                ds.scalarSWE.attrs['units'] = 'mm'

            # Also save outlet prediction if available
            # Aggregate HRU runoff using simple summation
            # For proper routing, configure ROUTING_MODEL in config
            aggregated_discharge = pivot_df.sum(axis=1)
            self.save_streamflow_to_results(
                aggregated_discharge,
                model_column_name=f"{self.model_name}_aggregated_discharge_cms"
            )

        # Common attributes
        ds.attrs['model'] = 'LSTM'
        ds.attrs['experiment_id'] = self.experiment_id

        # Save as NetCDF
        ds.to_netcdf(output_file)
        self.logger.info(f"LSTM results saved to {output_file}")

        # Also produce the standard format file needed by some evaluators
        standard_file = self.sim_dir / f"{self.experiment_id}_timestep.nc"
        if not standard_file.exists():
            import shutil
            shutil.copy2(output_file, standard_file)

        return output_file

    def calculate_metrics(self, obs: pd.Series, sim: pd.Series) -> Dict[str, float]:
        """
        Calculate performance metrics for LSTM predictions.

        Computes standard hydrological performance metrics comparing observed
        and simulated streamflow. Handles missing data by dropping NaN values
        before calculation.

        Args:
            obs: Observed streamflow series (m^3/s)
            sim: Simulated streamflow series (m^3/s)

        Returns:
            dict: Dictionary of metric names to values including:
                - RMSE: Root Mean Square Error
                - KGE: Kling-Gupta Efficiency
                - KGEp: KGE' (modified KGE with bias term)
                - NSE: Nash-Sutcliffe Efficiency
                - MAE: Mean Absolute Error
                - KGEnp: Non-parametric KGE
        """
        # Ensure obs and sim have the same length and are aligned
        aligned_data = pd.concat([obs, sim], axis=1, keys=['obs', 'sim']).dropna()
        obs_vals = aligned_data['obs'].values
        sim_vals = aligned_data['sim'].values

        if len(obs_vals) == 0:
            self.logger.warning("No overlapping data for metric calculation")
            return {}

        return {
            'RMSE': rmse(obs_vals, sim_vals, transfo=1),
            'KGE': cast(float, kge(obs_vals, sim_vals, transfo=1)),
            'KGEp': cast(float, kge_prime(obs_vals, sim_vals, transfo=1)),
            'NSE': nse(obs_vals, sim_vals, transfo=1),
            'MAE': mae(obs_vals, sim_vals, transfo=1),
            'KGEnp': kge_np(obs_vals, sim_vals, transfo=1)
        }
