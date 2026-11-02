# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Wflow model postprocessor."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from symfluence.core.modeling.base import StandardModelPostProcessor
from symfluence.core.registries import R


@R.postprocessors.add('WFLOW')
class WflowPostProcessor(StandardModelPostProcessor):
    """Postprocessor for the Wflow model.

    For lumped models, Wflow CSV output uses soil_surface_water__net_runoff_volume_flux
    in mm/dt which must be converted to m³/s. For distributed models, output is already
    in m³/s (river_water__volume_flow_rate).
    """

    model_name = "WFLOW"
    output_file_pattern = "output*"
    streamflow_variable = "Q"
    streamflow_unit = "cms"

    def _get_model_name(self) -> str:
        return "WFLOW"

    def _setup_model_specific_paths(self) -> None:
        self.wflow_output_dir = self.get_simulation_dir('WFLOW')
        catchment_file = self._get_catchment_file_path()
        self.catchment_path = catchment_file.parent
        self.catchment_name = catchment_file.name

    def _get_output_dir(self) -> Path:
        return self.get_simulation_dir('WFLOW')

    def _needs_unit_conversion(self) -> bool:
        """Check if the TOML output uses net_runoff_volume_flux (mm/dt units)."""
        config_file = self._get_config_value(
            lambda: None, 'wflow_sbm.toml', 'WFLOW_CONFIG_FILE',
        )
        settings_dir = self.project_dir / 'settings' / 'WFLOW'
        toml_path = settings_dir / config_file
        if toml_path.exists():
            content = toml_path.read_text()
            return 'net_runoff_volume_flux' in content
        return False

    def _convert_mm_dt_to_cms(self, series):
        """Convert mm per timestep to m³/s using basin area from staticmaps."""
        import xarray as xr
        staticmaps_name = self._get_config_value(
            lambda: None, 'wflow_staticmaps.nc', 'WFLOW_STATICMAPS_FILE',
        )
        staticmaps_path = self.project_dir / 'settings' / 'WFLOW' / staticmaps_name
        area = 0.0
        dt = 3600  # default hourly
        if staticmaps_path.exists():
            ds = xr.open_dataset(staticmaps_path)
            if 'wflow_cellarea' in ds:
                area = float(np.nansum(ds['wflow_cellarea'].values))
            ds.close()
        config_file = self._get_config_value(
            lambda: None, 'wflow_sbm.toml', 'WFLOW_CONFIG_FILE',
        )
        toml_path = self.project_dir / 'settings' / 'WFLOW' / config_file
        if toml_path.exists():
            import re
            content = toml_path.read_text()
            m = re.search(r'timestepsecs\s*=\s*(\d+)', content)
            if m:
                dt = int(m.group(1))
        if area > 0:
            self.logger.info(f"Converting Wflow output from mm/dt to m³/s (area={area:.0f} m², dt={dt}s)")
            return series * area / (dt * 1000.0)
        return series

    def extract_streamflow(self) -> Optional[Path]:
        import pandas as pd

        self.logger.info("Extracting streamflow from Wflow outputs")
        output_dir = self._get_output_dir()
        needs_conversion = self._needs_unit_conversion()

        # Try CSV first (primary output format from [output.csv] TOML section)
        csv_file = None
        for pattern in ['output.csv', 'output*.csv']:
            matches = list(output_dir.glob(pattern))
            if matches:
                csv_file = matches[0]
                break

        if csv_file is not None:
            try:
                df = pd.read_csv(csv_file, parse_dates=[0], index_col=0)
                q_col = None
                for col in ['Q', 'Q_av', 'q_av']:
                    if col in df.columns:
                        q_col = col
                        break
                if q_col is None and len(df.columns) == 1:
                    q_col = df.columns[0]
                if q_col is None:
                    self.logger.error(f"No Q column found in {csv_file}. Columns: {list(df.columns)}")
                    return None
                streamflow = df[q_col]
                if needs_conversion:
                    streamflow = self._convert_mm_dt_to_cms(streamflow)
                streamflow.index.name = 'datetime'
                if self.resample_frequency:
                    streamflow = streamflow.resample(self.resample_frequency).mean()
                return self.save_streamflow_to_results(streamflow, model_column_name='WFLOW_discharge_cms')
            except Exception as e:  # noqa: BLE001
                self.logger.warning(f"Failed to read CSV output: {e}, trying netCDF")

        # Fallback: try netCDF output
        nc_file = None
        for pattern in ['output*.nc', '*output*.nc']:
            matches = list(output_dir.glob(pattern))
            if matches:
                nc_file = matches[0]
                break
        if nc_file is None:
            self.logger.error(f"Wflow output not found in {output_dir}")
            return None
        try:
            import xarray as xr
            ds = xr.open_dataset(nc_file)
            q_var = None
            for var in ['Q', 'Q_av', 'q_av', 'q_river']:
                if var in ds.data_vars:
                    q_var = ds[var]
                    break
            if q_var is None:
                self.logger.error("No streamflow variable found in Wflow output")
                ds.close()
                return None
            spatial_dims = [d for d in q_var.dims if d not in ['time']]
            if spatial_dims:
                streamflow = q_var.max(dim=spatial_dims)
            else:
                streamflow = q_var
            streamflow = streamflow.to_series()
            ds.close()
            if needs_conversion:
                streamflow = self._convert_mm_dt_to_cms(streamflow)
            if self.resample_frequency:
                streamflow = streamflow.resample(self.resample_frequency).mean()
            return self.save_streamflow_to_results(streamflow, model_column_name='WFLOW_discharge_cms')
        except Exception as e:  # noqa: BLE001
            import traceback
            self.logger.error(f"Error extracting Wflow streamflow: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return None
