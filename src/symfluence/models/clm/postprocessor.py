# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CLM Model Postprocessor

Extracts streamflow and other variables from CLM5 history output.
CLM outputs QRUNOFF in mm/s which is converted to m3/s for evaluation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import xarray as xr

from symfluence.core.modeling.base.standard_postprocessor import StandardModelPostProcessor
from symfluence.core.registries import R

logger = logging.getLogger(__name__)


@R.postprocessors.add("CLM")
class CLMPostProcessor(StandardModelPostProcessor):
    """
    Postprocesses CLM5 output.

    CLM history files (*.clm2.h0.*.nc) contain:
    - QRUNOFF: Total column runoff (mm/s)
    - QOVER: Surface runoff (mm/s)
    - QDRAI: Sub-surface drainage (mm/s)
    - QFLX_EVAP_TOT: Total evapotranspiration (mm/s)
    """

    def __init__(self, config, logger, reporting_manager=None):
        super().__init__(config, logger, reporting_manager=reporting_manager)

        self.domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            default='unknown',
            dict_key='DOMAIN_NAME',
        )

    def _get_model_name(self) -> str:
        return "CLM"

    def extract_streamflow(  # type: ignore[override]
        self,
        output_dir: Optional[Path] = None,
        **kwargs,
    ) -> Optional[Tuple[pd.Series, Dict]]:
        """
        Extract streamflow from CLM output.

        CLM outputs QRUNOFF in mm/s. Conversion to m3/s:
            Q_m3s = QRUNOFF_mm_s * catchment_area_m2 / 1000

        Returns:
            Tuple of (streamflow Series in m3/s, metadata dict)
        """
        if output_dir is None:
            output_dir = self.get_simulation_dir('CLM', self.config.domain.experiment_id)

        # Find history files
        hist_files = sorted(output_dir.glob("*.clm2.h0.*.nc"))
        if not hist_files:
            # Check subdirectories
            for subdir in ['run', 'hist', 'results']:
                sub = output_dir / subdir
                if sub.exists():
                    hist_files = sorted(sub.glob("*.clm2.h0.*.nc"))
                    if hist_files:
                        break

        if not hist_files:
            logger.warning(f"No CLM history files found in {output_dir}")
            return None

        logger.info(f"Processing {len(hist_files)} CLM history file(s)")

        # Open and concatenate
        ds = xr.open_mfdataset(hist_files, combine='by_coords')

        # Extract QRUNOFF (total runoff = surface + subsurface)
        if 'QRUNOFF' in ds:
            qrunoff = ds['QRUNOFF']
        elif 'QOVER' in ds and 'QDRAI' in ds:
            qrunoff = ds['QOVER'] + ds['QDRAI']
        else:
            logger.error("Neither QRUNOFF nor QOVER+QDRAI found in CLM output")
            ds.close()
            return None

        # Squeeze spatial dimensions for single-point
        while qrunoff.ndim > 1:
            last_dim = qrunoff.dims[-1]
            if last_dim != 'time':
                qrunoff = qrunoff.isel({last_dim: 0})
            else:
                break

        # Convert mm/s → m3/s using catchment area
        area_km2 = self._get_catchment_area()
        area_m2 = area_km2 * 1e6
        streamflow_m3s = qrunoff.values * area_m2 / 1000.0

        times = pd.to_datetime(ds['time'].values)
        ds.close()

        sim_series = pd.Series(streamflow_m3s, index=times, name='streamflow_m3s')

        metadata = {
            'units': 'm3/s',
            'source': 'CLM5 QRUNOFF',
            'conversion': 'QRUNOFF(mm/s) * area(m2) / 1000',
            'area_km2': area_km2,
            'n_files': len(hist_files),
        }

        logger.info(
            f"Extracted CLM streamflow: {len(sim_series)} timesteps, "
            f"mean={sim_series.mean():.3f} m3/s"
        )
        return sim_series, metadata

    def _get_catchment_area(self) -> float:
        """Get catchment area in km2 (config, else domain shapefile; raises if neither)."""
        area = self._get_config_value(
            lambda: self.config.domain.catchment_area,
            default=None,
            dict_key='CATCHMENT_AREA',
        )
        from symfluence.core.metrics.streamflow_metrics import require_catchment_area_km2
        return require_catchment_area_km2(
            self.config_dict, self.project_dir, self.domain_name, configured_km2=area,
        )
