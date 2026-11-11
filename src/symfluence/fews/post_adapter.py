# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
FEWS Post-Adapter.

Orchestrates the post-processing phase of a FEWS General Adapter run:
  1. Load model output (the workflow's standardized results for the
     resolved configuration -- never a file from the FEWS exchange directories)
  2. Apply reverse ID mapping (SYMFLUENCE -> FEWS variable names)
  3. Write output in FEWS format (PI-XML or NetCDF-CF)
  4. Export state files
  5. Write diagnostics
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import xarray as xr

from .config import FEWSConfig
from .exceptions import FEWSAdapterError
from .id_map import IDMapper
from .netcdf_cf import write_fews_netcdf
from .pi_diagnostics import DiagnosticsCollector
from .pi_xml import write_pi_xml_timeseries
from .run_info import RunInfo, parse_run_info
from .state import export_states

logger = logging.getLogger(__name__)


class FEWSPostAdapter:
    """Orchestrates the FEWS post-adapter workflow.

    Args:
        run_info_path: Path to run_info.xml
        config_path: Path to SYMFLUENCE config YAML (for locating output);
            the run file's overrides (period, DOMAIN_NAME, ...) are applied
        config: Already-resolved SymfluenceConfig (takes precedence over
            ``config_path``)
        data_format: Data exchange format (``pi-xml`` or ``netcdf-cf``)
        id_map_path: Optional path to YAML ID mapping file
        fews_config: Optional pre-built FEWSConfig
    """

    def __init__(
        self,
        run_info_path: Path,
        config_path: Optional[Path] = None,
        data_format: str = "netcdf-cf",
        id_map_path: Optional[str] = None,
        fews_config: Optional[FEWSConfig] = None,
        config: Any = None,
    ) -> None:
        self.run_info_path = Path(run_info_path)
        self.config_path = config_path
        self._config = config
        self.data_format = data_format
        self.id_map_path = id_map_path
        self._fews_config = fews_config

    def run(self, diag: Optional[DiagnosticsCollector] = None) -> Path:
        """Execute the post-adapter workflow.

        Args:
            diag: Optional diagnostics collector

        Returns:
            Path to the FEWS output directory

        Raises:
            FEWSAdapterError: On failure
        """
        # 1. Parse run_info
        run_info = parse_run_info(self.run_info_path)
        logger.info("Post-adapter: output dir = %s", run_info.output_dir)
        if diag:
            diag.info(f"Post-adapter started, output dir: {run_info.output_dir}")

        # 2. Build FEWSConfig
        fews_cfg = self._fews_config or FEWSConfig(
            FEWS_WORK_DIR=str(run_info.work_dir),
            FEWS_DATA_FORMAT=self.data_format,
            FEWS_ID_MAP_FILE=self.id_map_path,
        )

        # 3. Build ID mapper
        mapper = IDMapper(fews_cfg)

        # 4. Load model output
        output_ds, source = self._load_model_output(run_info)
        if diag:
            diag.info(f"Loaded model output from {source}: {list(output_ds.data_vars)}")

        # 5. Apply reverse ID mapping
        output_ds = mapper.rename_dataset_sym_to_fews(output_ds)
        logger.info("Reverse-mapped variables: %s", list(output_ds.data_vars))
        if diag:
            diag.info(f"Mapped to FEWS names: {list(output_ds.data_vars)}")

        # 6. Write output
        output_dir = run_info.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        if fews_cfg.data_format == "pi-xml":
            output_path = (
                run_info.output_timeseries_files[0]
                if run_info.output_timeseries_files
                else output_dir / "timeseries.xml"
            )
            write_pi_xml_timeseries(
                output_ds,
                output_path,
                missing_value=fews_cfg.missing_value,
                location_id=self._location_id(run_info),
                time_zone=run_info.time_zone_hours,
            )
        else:
            output_path = (
                run_info.output_netcdf_files[0]
                if run_info.output_netcdf_files
                else output_dir / "output.nc"
            )
            write_fews_netcdf(output_ds, output_path)

        logger.info("Wrote FEWS output to %s", output_path)
        if diag:
            diag.info(f"Wrote output to {output_path}")

        # 7. Export states
        if run_info.state_output_dir and fews_cfg.state_dir:
            # Use StateManager if model runner is available and state-capable
            model_runner = getattr(self, '_model_runner', None)
            if model_runner and hasattr(model_runner, 'supports_state') and model_runner.supports_state:
                from symfluence.core.modeling.state import StateManager
                StateManager.export_to_fews(model_runner, run_info.state_output_dir)
                if diag:
                    diag.info("Exported state files via StateManager")
            else:
                export_states(Path(fews_cfg.state_dir), run_info.state_output_dir)
                if diag:
                    diag.info("Exported state files")

        return output_dir

    def _location_id(self, run_info: RunInfo) -> str:
        """FEWS locationId for exported series.

        The run-file property ``FEWS_LOCATION_ID`` (or ``LOCATION_ID``) wins;
        otherwise the SYMFLUENCE domain name.
        """
        for key in ("FEWS_LOCATION_ID", "LOCATION_ID"):
            if run_info.properties.get(key):
                return run_info.properties[key]
        config = self._config
        name = getattr(getattr(config, "domain", None), "name", None)
        return str(name) if name else "default"

    def _resolve_config(self, run_info: RunInfo):
        """The SYMFLUENCE config of this FEWS run (base config + run-file overrides)."""
        if self._config is not None and not isinstance(self._config, dict):
            return self._config
        if self.config_path is None or not Path(self.config_path).is_file():
            raise FEWSAdapterError(
                "A SYMFLUENCE configuration (--config) is required to locate the "
                "model output for export to FEWS"
                + (f"; not found: {self.config_path}" if self.config_path else ".")
            )
        from symfluence.core.config.models import SymfluenceConfig
        self._config = SymfluenceConfig.from_file(
            Path(self.config_path), overrides=run_info.to_config_overrides()
        )
        return self._config

    def _load_model_output(self, run_info: RunInfo) -> Tuple[xr.Dataset, Path]:
        """Load the simulated series the workflow produced for this configuration.

        Reads the standardized results table that ``postprocess_results``
        writes for every model (``<project>/results/<experiment_id>_results.csv``,
        columns ``<MODEL>_discharge_cms``), restricted to the FEWS run period.
        Nothing is ever taken from the FEWS exchange directories: the first
        ``*.nc`` found there is the forcing the pre-adapter wrote, and used to
        be exported back to FEWS as "model output".

        Returns:
            (dataset, source file). A single discharge column is named
            ``discharge``; several models keep their column names.

        Raises:
            FEWSAdapterError: If there is no results file or it holds no
                simulated values inside the run period.
        """
        import pandas as pd

        config = self._resolve_config(run_info)
        data_dir = getattr(config.system, "data_dir", None)
        if not data_dir:
            raise FEWSAdapterError("The SYMFLUENCE configuration has no data directory.")
        project_dir = Path(data_dir) / f"domain_{config.domain.name}"
        results_file = project_dir / "results" / f"{config.domain.experiment_id}_results.csv"
        if not results_file.is_file():
            raise FEWSAdapterError(
                f"No model output found: {results_file} does not exist. The model "
                "run (including the postprocess_results step) must complete before "
                "the FEWS post-adapter."
            )

        frame = pd.read_csv(results_file, index_col=0, parse_dates=True)
        frame = frame.select_dtypes("number")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise FEWSAdapterError(f"No model output found: {results_file} has no time index.")
        if frame.index.tz is not None:
            frame.index = frame.index.tz_convert("UTC").tz_localize(None)
        start = pd.Timestamp(run_info.start_time).tz_convert("UTC").tz_localize(None)
        end = pd.Timestamp(run_info.end_time).tz_convert("UTC").tz_localize(None)
        frame = frame.sort_index().loc[start:end].dropna(axis=1, how="all")
        if frame.empty or frame.shape[1] == 0:
            raise FEWSAdapterError(
                f"No model output found in {results_file} for the FEWS run period "
                f"{start} to {end}."
            )

        discharge_cols = [c for c in frame.columns if str(c).endswith("_discharge_cms")]
        rename = {discharge_cols[0]: "discharge"} if len(discharge_cols) == 1 else {}
        data_vars = {}
        for column in frame.columns:
            attrs = {"units": "m3/s"} if column in discharge_cols else {}
            if column in rename:
                attrs["source_column"] = str(column)
            data_vars[rename.get(column, str(column))] = xr.DataArray(
                frame[column].to_numpy(dtype=float), dims=["time"], attrs=attrs
            )
        ds = xr.Dataset(data_vars, coords={"time": frame.index.to_numpy()})
        logger.info("Loaded model output from %s", results_file)
        return ds, results_file
