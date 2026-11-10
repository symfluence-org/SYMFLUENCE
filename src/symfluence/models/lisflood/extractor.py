# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""LISFLOOD Result Extractor."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.modeling.base import ModelResultExtractor

#: LISFLOOD model step (DtSec) the preprocessor writes when
#: LISFLOOD_FORCING_TIMESTEP is unset (seconds).
DEFAULT_LISFLOOD_TIMESTEP_SECONDS = 86400


def resolve_lisflood_timestep_seconds(config) -> int:
    """Model step (s) -- the value the preprocessor writes as DtSec."""
    from symfluence.core.mixins.config import resolve_config_value
    return int(resolve_config_value(
        config, lambda: config.model.lisflood.forcing_timestep,
        DEFAULT_LISFLOOD_TIMESTEP_SECONDS, 'LISFLOOD_FORCING_TIMESTEP',
    ))


def read_tss_series(tss_path: Path, start, timestep_seconds: int) -> pd.Series:
    """Dated discharge series from a LISFLOOD/PCRaster ``.tss`` file.

    Shared by the calibration worker and the postprocessor. Data rows are
    lines whose fields are all numeric (``step value [value ...]``); header
    lines (title, column count, column names) are skipped. Step ``k`` is
    dated ``start + (k - 1) * timestep`` where ``start`` is the simulation
    start (midnight-aligned for daily or longer steps, as the preprocessor
    writes CalendarDayStart/StepStart) and ``timestep`` is the model DtSec.
    The last column is the (single-gauge) discharge.
    """
    steps, values = [], []
    for line in Path(tss_path).read_text().strip().split("\n"):
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            continue
        steps.append(int(nums[0]))
        values.append(nums[-1])
    if not values:
        raise ModelExecutionError(f"No data rows in TSS file {tss_path}")
    start_ts = pd.Timestamp(start)
    if timestep_seconds >= 86400:
        start_ts = start_ts.normalize()
    offsets = pd.to_timedelta([(k - 1) * timestep_seconds for k in steps], unit="s")
    return pd.Series(values, index=pd.DatetimeIndex(start_ts + offsets), name="discharge_cms")


class LisfloodResultExtractor(ModelResultExtractor):
    """LISFLOOD-specific result extraction."""

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        return {
            "streamflow": ["dis*.nc", "*discharge*.nc"],
            "et": ["eta*.nc", "*evaporation*.nc"],
            "snow": ["snow*.nc", "*SnowMelt*.nc"],
            "soil_moisture": ["*theta*.nc", "*SoilMoisture*.nc"],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        variable_mapping = {
            "streamflow": ["dis", "discharge", "Qsim", "chanq"],
            "et": ["eta", "ETA", "ActEvap", "TaInterception"],
            "snow": ["SnowMelt", "SnowCover", "snow_melt"],
            "soil_moisture": ["theta1", "theta2", "theta3", "TotalSoilMoisture"],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(self, output_file: Path, variable_type: str, **kwargs) -> pd.Series:
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
                    f"No suitable variable found in {output_file}. Tried: {var_names}. Available: {available}"
                )
            data = ds[found_var]
            spatial_dims = [d for d in data.dims if d not in ["time"]]
            if spatial_dims:
                if variable_type == "streamflow":
                    data = data.max(dim=spatial_dims)
                else:
                    data = data.mean(dim=spatial_dims)
            series = data.to_series()
            ds.close()
            return series
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Error reading LISFLOOD output file {output_file}: {str(e)}") from e

    def extract_streamflow(self, output_dir: Path, catchment_area: Optional[float] = None, **kwargs) -> pd.Series:
        patterns = self.get_output_file_patterns()["streamflow"]
        output_file = None
        for pattern in patterns:
            matches = list(output_dir.glob(pattern))
            if matches:
                output_file = matches[0]
                break
        if output_file is None:
            raise FileNotFoundError(f"No LISFLOOD output file found in {output_dir}")
        return self.extract_variable(output_file, "streamflow", **kwargs)

    def requires_unit_conversion(self, variable_type: str) -> bool:
        return False

    def get_spatial_aggregation_method(self, variable_type: str) -> Optional[str]:
        if variable_type == "streamflow":
            return "max"
        elif variable_type in ["et", "precipitation"]:
            return "sum"
        return "mean"
