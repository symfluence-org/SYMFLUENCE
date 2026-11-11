# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
FEWS Pre-Adapter.

Orchestrates the pre-processing phase of a FEWS General Adapter run:
  1. Parse run_info.xml
  2. Load FEWS forcing data (PI-XML or NetCDF-CF)
  3. Apply ID mapping (FEWS -> SYMFLUENCE variable names)
  4. Generate / update SymfluenceConfig with FEWS overrides
  5. Import warm-start state files
  6. Write forcing to model input directory
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import xarray as xr

from .config import FEWSConfig
from .exceptions import FEWSAdapterError
from .id_map import IDMapper
from .netcdf_cf import read_fews_netcdf, write_fews_netcdf
from .pi_diagnostics import DiagnosticsCollector
from .pi_xml import read_pi_xml_timeseries
from .run_info import RunInfo, parse_run_info
from .state import import_states

logger = logging.getLogger(__name__)


def translated_forcing_path(run_info: RunInfo) -> Path:
    """Where the pre-adapter writes forcing translated to SYMFLUENCE names."""
    return run_info.work_dir / "symfluence" / "forcing.nc"


class FEWSPreAdapter:
    """Orchestrates the FEWS pre-adapter workflow.

    Args:
        run_info_path: Path to run_info.xml
        base_config_path: Optional path to base SYMFLUENCE config YAML
        data_format: Data exchange format (``pi-xml`` or ``netcdf-cf``)
        id_map_path: Optional path to YAML ID mapping file
        fews_config: Optional pre-built FEWSConfig (overrides auto-detection)
    """

    def __init__(
        self,
        run_info_path: Path,
        base_config_path: Optional[Path] = None,
        data_format: str = "netcdf-cf",
        id_map_path: Optional[str] = None,
        fews_config: Optional[FEWSConfig] = None,
    ) -> None:
        self.run_info_path = Path(run_info_path)
        self.base_config_path = base_config_path
        self.data_format = data_format
        self.id_map_path = id_map_path
        self._fews_config = fews_config
        # Set by run(): FEWS-derived config overrides and the translated forcing
        self.config_overrides: Dict[str, Any] = {}
        self.forcing_path: Optional[Path] = None

    def run(self, diag: Optional[DiagnosticsCollector] = None):
        """Execute the pre-adapter workflow.

        Args:
            diag: Optional diagnostics collector (created automatically if None)

        Returns:
            Tuple of (SymfluenceConfig, RunInfo)

        Raises:
            FEWSAdapterError: On failure
        """
        # 1. Parse run_info
        run_info = parse_run_info(self.run_info_path)
        logger.info("Parsed run_info: %s to %s", run_info.start_time, run_info.end_time)
        if diag:
            diag.info(f"Parsed run_info: {run_info.start_time} to {run_info.end_time}")

        # 2. Build FEWSConfig
        fews_cfg = self._fews_config or FEWSConfig(
            FEWS_WORK_DIR=str(run_info.work_dir),
            FEWS_DATA_FORMAT=self.data_format,
            FEWS_ID_MAP_FILE=self.id_map_path,
        )

        # 3. Build ID mapper
        mapper = IDMapper(fews_cfg)

        # 4. Load forcing data
        forcing_ds = self._load_forcing(run_info, fews_cfg)
        if diag:
            diag.info(f"Loaded forcing: {list(forcing_ds.data_vars)}")

        # 5. Apply ID mapping
        forcing_ds = mapper.rename_dataset_fews_to_sym(forcing_ds)
        logger.info("Mapped variables: %s", list(forcing_ds.data_vars))
        if diag:
            diag.info(f"Mapped to SYMFLUENCE names: {list(forcing_ds.data_vars)}")

        # 6. Import states
        if run_info.state_input_dir and fews_cfg.state_dir:
            # Use StateManager if model runner is available and state-capable
            model_runner = getattr(self, '_model_runner', None)
            if model_runner and hasattr(model_runner, 'supports_state') and model_runner.supports_state:
                from symfluence.core.modeling.state import StateManager
                StateManager.import_from_fews(run_info.state_input_dir, model_runner)
                if diag:
                    diag.info("Imported state files via StateManager")
            else:
                import_states(run_info.state_input_dir, Path(fews_cfg.state_dir))
                if diag:
                    diag.info("Imported state files")

        # 7. Write the translated forcing. It goes to the adapter's own
        #    directory, not back into FEWS's toModel/: there it would overwrite
        #    the file FEWS exported and be re-read as input (or mistaken for
        #    model output) on the next call.
        forcing_output = translated_forcing_path(run_info)
        write_fews_netcdf(forcing_ds, forcing_output)
        self.forcing_path = forcing_output
        logger.info("Wrote forcing to %s", forcing_output)
        if diag:
            diag.info(f"Wrote forcing to {forcing_output}")

        # 8. Build SymfluenceConfig
        self.config_overrides = run_info.to_config_overrides()
        config = self._build_config(run_info, fews_cfg)

        return config, run_info

    def _load_forcing(self, run_info: RunInfo, fews_cfg: FEWSConfig) -> xr.Dataset:
        """Load forcing data exported by FEWS.

        The files named in the run file (``inputTimeSeriesFile`` /
        ``inputNetcdfFile``) are used when present; otherwise every file of
        the configured format in the input directory.
        """
        input_dir = run_info.input_dir

        if fews_cfg.data_format == "pi-xml":
            xml_files = list(run_info.input_timeseries_files) or sorted(input_dir.glob("*.xml"))
            if not xml_files:
                raise FEWSAdapterError(f"No PI-XML files found in {input_dir}")
            datasets = [
                read_pi_xml_timeseries(f, missing_value=fews_cfg.missing_value)
                for f in xml_files
            ]
        else:
            nc_files = list(run_info.input_netcdf_files) or sorted(input_dir.glob("*.nc"))
            if not nc_files:
                raise FEWSAdapterError(f"No NetCDF files found in {input_dir}")
            datasets = [read_fews_netcdf(f) for f in nc_files]

        ds = datasets[0]
        for extra in datasets[1:]:
            ds = ds.merge(extra)
        return ds

    def _build_config(self, run_info: RunInfo, fews_cfg: FEWSConfig):
        """Build SymfluenceConfig from base config + FEWS overrides."""
        overrides = run_info.to_config_overrides()

        if self.base_config_path and Path(self.base_config_path).is_file():
            from symfluence.core.config.models import SymfluenceConfig
            return SymfluenceConfig.from_file(
                Path(self.base_config_path),
                overrides=overrides,
            )

        # Return overrides dict if no base config
        return overrides
