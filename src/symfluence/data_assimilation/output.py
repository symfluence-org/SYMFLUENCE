# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Data assimilation output writer.

Writes EnKF results to CF-1.6 compliant NetCDF files.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DAOutputManager:
    """Writes data assimilation results to NetCDF."""

    def write(
        self,
        output_path: Path,
        ensemble_predictions: np.ndarray,
        ensemble_means: np.ndarray,
        ensemble_stds: np.ndarray,
        observed_values: np.ndarray,
        analysis_times: Optional[List[int]] = None,
        time_index: Optional[np.ndarray] = None,
        global_attrs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write DA results to a NetCDF file.

        Args:
            output_path: Path to the output NetCDF file.
            ensemble_predictions: Per-member predictions (n_timesteps, n_members).
            ensemble_means: Ensemble mean (n_timesteps,).
            ensemble_stds: Ensemble std (n_timesteps,).
            observed_values: Observations (n_timesteps,).
            analysis_times: Timestep indices of analysis steps.
            time_index: Optional datetime index.
            global_attrs: Extra global attributes (e.g. catchment area and
                timestep used for the observation unit conversion).
        """
        import xarray as xr

        n_timesteps = len(ensemble_means)
        n_members = ensemble_predictions.shape[1] if ensemble_predictions.ndim > 1 else 0

        # Build time coordinate
        if time_index is not None:
            time_coord = time_index[:n_timesteps]
        else:
            time_coord = np.arange(n_timesteps)

        # Build dataset
        data_vars = {
            'ensemble_mean_streamflow': (['time'], ensemble_means),
            'ensemble_std_streamflow': (['time'], ensemble_stds),
            'observed_streamflow': (['time'], observed_values[:n_timesteps]),
        }

        # Innovation
        innovation = observed_values[:n_timesteps] - ensemble_means
        data_vars['innovation'] = (['time'], innovation)

        coords = {'time': time_coord}

        # Per-member predictions
        if n_members > 0:
            data_vars['ensemble_streamflow'] = (
                ['time', 'member'],
                ensemble_predictions[:n_timesteps],
            )
            coords['member'] = np.arange(n_members)

        # Analysis mask
        analysis_mask = np.zeros(n_timesteps, dtype=bool)
        if analysis_times:
            for t in analysis_times:
                if t < n_timesteps:
                    analysis_mask[t] = True
        data_vars['analysis_performed'] = (['time'], analysis_mask)

        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        # CF-1.6 attributes
        ds.attrs.update({
            'Conventions': 'CF-1.6',
            'title': 'SYMFLUENCE EnKF Data Assimilation Results',
            'method': 'Ensemble Kalman Filter',
            'n_members': n_members,
            'n_analyses': len(analysis_times) if analysis_times else 0,
        })
        if global_attrs:
            ds.attrs.update(global_attrs)

        ds['ensemble_mean_streamflow'].attrs = {
            'units': 'mm/timestep',
            'long_name': 'Ensemble mean streamflow prediction',
        }
        ds['ensemble_std_streamflow'].attrs = {
            'units': 'mm/timestep',
            'long_name': 'Ensemble standard deviation of streamflow',
        }
        ds['observed_streamflow'].attrs = {
            'units': 'mm/timestep',
            'long_name': 'Observed streamflow',
        }
        ds['innovation'].attrs = {
            'units': 'mm/timestep',
            'long_name': 'Innovation (observation minus predicted mean)',
        }
        if 'ensemble_streamflow' in ds:
            ds['ensemble_streamflow'].attrs = {
                'units': 'mm/timestep',
                'long_name': 'Per-member streamflow prediction',
            }

        # Write
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        encoding = {}
        for var in ds.data_vars:
            encoding[str(var)] = {'zlib': True, 'complevel': 4}

        ds.to_netcdf(output_path, encoding=encoding)
        logger.info("Wrote DA output: %s", output_path)
