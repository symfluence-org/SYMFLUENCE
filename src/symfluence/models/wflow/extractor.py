# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Wflow Result Extractor."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from symfluence.core.exceptions import ConfigurationError
from symfluence.core.modeling.base import ModelResultExtractor


def is_distributed_wflow(config) -> bool:
    """True when the run is native-routing distributed Wflow (WFLOW_SPATIAL_MODE)."""
    from symfluence.core.mixins.config import resolve_config_value
    if not config:
        return False
    return resolve_config_value(
        config, lambda: config.model.wflow.spatial_mode, None, 'WFLOW_SPATIAL_MODE'
    ) == 'distributed'


def to_period_start_labels(q: pd.Series) -> pd.Series:
    """Relabel distributed Wflow output from end-of-day to start-of-day.

    Native distributed Wflow writes daily values stamped at the end of the
    day they cover (00:00 of the next day). SYMFLUENCE observations are
    labelled by the day they describe, so every reader -- calibration worker,
    extractor and postprocessor -- shifts the labels back one day.
    """
    q = q.copy()
    q.index = pd.DatetimeIndex(q.index) - pd.Timedelta(days=1)
    return q


class WflowResultExtractor(ModelResultExtractor):
    """Wflow-specific result extraction."""

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        return {
            'streamflow': ['output.csv', 'output*.nc', '*output*.nc'],
            'et': ['output*.nc'],
            'snow': ['output*.nc'],
            'soil_moisture': ['output*.nc'],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        variable_mapping = {
            'streamflow': ['Q', 'Q_av', 'q_av', 'q_river'],
            'et': ['ActEvap', 'actevap', 'ET'],
            'snow': ['SnowWater', 'snow_water', 'SWE'],
            'soil_moisture': ['SatWaterDepth', 'satwaterdepth', 'USStore'],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(self, output_file: Path, variable_type: str, **kwargs) -> pd.Series:
        if output_file.suffix == '.csv':
            df = pd.read_csv(output_file, parse_dates=['time'])
            # New native-routing distributed output is daily, end-labelled.
            if 'WB_max' in df:
                return to_period_start_labels(pd.Series(df.Q.values, index=pd.DatetimeIndex(df.time)))
            raise ConfigurationError('Legacy lumped CSV requires its area-aware worker extractor')
        import xarray as xr
        var_names = self.get_variable_names(variable_type)
        try:
            ds = xr.open_dataset(output_file)
            found_var = None
            for var_name in var_names:
                if var_name in ds.data_vars:
                    found_var = var_name
                    break
            if found_var is None:
                available = list(ds.data_vars)
                ds.close()
                raise ValueError(
                    f"No suitable variable found in {output_file}. "
                    f"Tried: {var_names}. Available: {available}"
                )
            data = ds[found_var]
            spatial_dims = [d for d in data.dims if d not in ['time']]
            if spatial_dims:
                if variable_type == 'streamflow':
                    data = data.max(dim=spatial_dims)
                else:
                    data = data.mean(dim=spatial_dims)
            series = data.to_series()
            ds.close()
            return series
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Error reading Wflow output file {output_file}: {str(e)}") from e

    def extract_streamflow(self, output_dir: Path, catchment_area: Optional[float] = None, **kwargs) -> pd.Series:
        patterns = self.get_output_file_patterns()['streamflow']
        output_file = None
        for pattern in patterns:
            matches = list(output_dir.glob(pattern))
            if matches:
                output_file = matches[0]
                break
        if output_file is None:
            raise FileNotFoundError(f"No Wflow output file found in {output_dir}")
        return self.extract_variable(output_file, 'streamflow', **kwargs)

    def requires_unit_conversion(self, variable_type: str) -> bool:
        return False

    def get_spatial_aggregation_method(self, variable_type: str) -> Optional[str]:
        if variable_type == 'streamflow':
            return 'max'
        elif variable_type in ['et', 'precipitation']:
            return 'sum'
        return 'mean'
