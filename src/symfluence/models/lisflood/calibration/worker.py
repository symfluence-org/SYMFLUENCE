# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""LISFLOOD Worker."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd
import xarray as xr

from symfluence.core.calibration.workers.base_worker import BaseWorker
from symfluence.core.exceptions import ConfigurationError
from symfluence.core.metrics import StreamflowMetrics
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.modeling.utilities.periods import split_period
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R


@R.workers.add("LISFLOOD")
class LisfloodWorker(BaseWorker):
    """Worker for LISFLOOD model calibration."""

    _streamflow_metrics = StreamflowMetrics()

    def apply_parameters(self, params, settings_dir, **kwargs):
        try:
            config = kwargs.get("config", self.config) or {}
            domain_name = resolve_config_value(config, lambda: config.domain.name, "", 'DOMAIN_NAME')
            data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, ".", 'SYMFLUENCE_DATA_DIR'))
            original_settings = data_dir / f"domain_{domain_name}" / "settings" / "LISFLOOD"
            settings_dir = Path(settings_dir)

            # Copy maps from original to process-specific directory
            original_maps = original_settings / "maps"
            target_maps = settings_dir / "maps"
            if original_maps.exists() and original_maps != target_maps:
                if target_maps.exists():
                    shutil.rmtree(target_maps)
                shutil.copytree(original_maps, target_maps)

            # Copy and patch settings XML
            settings_file = resolve_config_value(config, lambda: config.model.lisflood.settings_file, "settings.xml", 'LISFLOOD_SETTINGS_FILE')
            original_xml = original_settings / settings_file
            target_xml = settings_dir / settings_file
            if original_xml.exists() and original_xml != target_xml:
                self._patch_settings_xml(
                    original_xml, target_xml, settings_dir, kwargs.get("output_dir") or kwargs.get("proc_output_dir")
                )

            from .parameter_manager import LisfloodParameterManager

            pm = LisfloodParameterManager(config, self.logger, settings_dir)
            return pm.update_model_files(params, settings_dir)
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Failed to apply LISFLOOD parameters: {e}")
            return False

    def _patch_settings_xml(self, source_xml, target_xml, settings_dir, output_dir):
        """Copy settings XML and patch paths for process isolation."""
        from xml.etree import ElementTree as ET  # nosec B405

        tree = ET.parse(source_xml)  # nosec B314
        root = tree.getroot()
        for textvar in root.iter("textvar"):
            name = textvar.get("name", "")
            if name == "PathMaps":
                textvar.set("value", str(settings_dir / "maps"))
            elif name == "PathOut" and output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                textvar.set("value", str(output_dir))
        tree.write(target_xml, encoding="unicode", xml_declaration=True)

    def run_model(self, config, settings_dir, output_dir, **kwargs):
        try:
            import os
            import sys

            override = kwargs.get("settings_file_override")
            if override:
                settings_path = Path(override)
            else:
                settings_file = resolve_config_value(config, lambda: config.model.lisflood.settings_file, "settings.xml", 'LISFLOOD_SETTINGS_FILE')
                settings_path = Path(settings_dir) / settings_file
            timeout = int(resolve_config_value(config, lambda: config.model.lisflood.timeout, 7200, 'LISFLOOD_TIMEOUT'))

            env = dict(os.environ)
            from symfluence.models.lisflood.runner import _find_pcraster_site_packages

            pcraster_site = _find_pcraster_site_packages()
            if pcraster_site:
                existing = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = f"{pcraster_site}:{existing}" if existing else pcraster_site
                pcraster_prefix = str(Path(pcraster_site).parent.parent.parent)
                env["CONDA_PREFIX"] = pcraster_prefix
            env.pop("PCRASTER_NR_WORKER_THREADS", None)

            result = run_subprocess(
                [sys.executable, "-c", f'from lisflood.main import main; main("{settings_path}")'],
                cwd=str(settings_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            if result.returncode != 0 and result.stderr:
                self.logger.warning(f"LISFLOOD stderr: {result.stderr[-500:]}")
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"LISFLOOD execution failed: {e}")
            return False

    def calculate_metrics(self, output_dir, config, **kwargs):
        try:
            sim = self._load_simulated_streamflow(Path(output_dir), config)
            if sim is None:
                self.logger.warning(f"No sim data found in {output_dir}")
                return {"KGE": -999.0, "NSE": -999.0}
            obs = self._load_observations(config)
            if obs is None:
                self.logger.warning("No observation data found")
                return {"KGE": -999.0, "NSE": -999.0}
            if not isinstance(sim.index, pd.DatetimeIndex):
                sim.index = pd.to_datetime(sim.index)
            if not isinstance(obs.index, pd.DatetimeIndex):
                obs.index = pd.to_datetime(obs.index)
            if len(sim) > 1:
                median_dt = pd.Series(sim.index).diff().median()
                if median_dt < pd.Timedelta(days=1):
                    sim = sim.resample("D").mean()
            # Explicit dict guard kept: a typed config scores the whole record.
            cal_period = (resolve_config_value(config, lambda: config.domain.calibration_period, "", 'CALIBRATION_PERIOD')
                          if isinstance(config, dict) else "")
            parts = split_period(cal_period)
            if parts:
                sim = sim.loc[parts[0]:parts[1]]  # type: ignore[misc]
                obs = obs.loc[parts[0]:parts[1]]  # type: ignore[misc]
            sim, obs = sim.align(obs, join="inner")
            common_idx = sim.dropna().index.intersection(obs.dropna().index)
            sim, obs = sim.loc[common_idx], obs.loc[common_idx]
            if len(sim) < 30:
                self.logger.warning(f"Insufficient data: n={len(sim)} < 30")
                return {"KGE": -999.0, "NSE": -999.0}
            metrics = self._streamflow_metrics.calculate_metrics(obs.values, sim.values)
            self.logger.info(f"Computed metrics: {metrics}")
            return metrics
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Metric calculation failed: {e}")
            return {"KGE": -999.0, "NSE": -999.0}

    def _load_simulated_streamflow(self, output_dir: Path, config=None):
        # Try NetCDF discharge output
        for pattern in ["dis*.nc", "*discharge*.nc"]:
            matches = list(output_dir.glob(pattern))
            if matches:
                ds = xr.open_dataset(matches[0])
                for var in ["dis", "discharge", "Qsim", "chanq"]:
                    if var in ds.data_vars:
                        q_var = ds[var]
                        spatial_dims = [d for d in q_var.dims if d not in ["time"]]
                        q = q_var.max(dim=spatial_dims).to_series() if spatial_dims else q_var.to_series()
                        ds.close()
                        return q
                ds.close()
        # Fallback: TSS file
        for pattern in ["dis*.tss", "*discharge*.tss"]:
            matches = list(output_dir.glob(pattern))
            if matches:
                return self._read_tss_file(matches[0], config)
        return None

    @staticmethod
    def _read_tss_file(tss_path: Path, config=None):
        """Dated TSS discharge (shared with the postprocessor)."""
        from symfluence.models.lisflood.extractor import (
            read_tss_series,
            resolve_lisflood_timestep_seconds,
        )
        start = resolve_config_value(config, lambda: config.domain.time_start, None, 'EXPERIMENT_TIME_START')
        if start is None:
            raise ConfigurationError("EXPERIMENT_TIME_START is required to date LISFLOOD .tss output")
        return read_tss_series(tss_path, start, resolve_lisflood_timestep_seconds(config))

    def _load_observations(self, config):
        try:
            domain_name = resolve_config_value(config, lambda: config.domain.name, "", 'DOMAIN_NAME')
            data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, ".", 'SYMFLUENCE_DATA_DIR'))
            obs_dir = data_dir / f"domain_{domain_name}" / "observations" / "streamflow" / "preprocessed"
            if not obs_dir.exists():
                return None
            obs_files = list(obs_dir.glob("*.csv"))
            if not obs_files:
                return None
            df = pd.read_csv(obs_files[0], parse_dates=True, index_col=0)
            return df["discharge_cms"] if "discharge_cms" in df.columns else df.iloc[:, 0]
        except Exception:  # noqa: BLE001
            return None
