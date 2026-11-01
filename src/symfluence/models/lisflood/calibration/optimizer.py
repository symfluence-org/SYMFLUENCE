# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""LISFLOOD Model Optimizer."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional
from xml.etree import ElementTree as ET  # nosec B405

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.registries import R

from .worker import LisfloodWorker  # noqa: F401


@R.optimizers.add("LISFLOOD")
class LisfloodModelOptimizer(BaseModelOptimizer):
    """LISFLOOD-specific optimizer using the unified BaseModelOptimizer framework."""

    def __init__(self, config, logger, optimization_settings_dir=None, reporting_manager=None):
        self.experiment_id = resolve_config_value(config, lambda: config.domain.experiment_id, None, 'EXPERIMENT_ID')
        self.data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, ".", 'SYMFLUENCE_DATA_DIR'))
        self.domain_name = resolve_config_value(config, lambda: config.domain.name, None, 'DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.lisflood_setup_dir = self.project_dir / "settings" / "LISFLOOD"
        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

    def _get_model_name(self) -> str:
        return "LISFLOOD"

    def _get_final_file_manager_path(self) -> Path:
        settings_file = self._get_config_value(
            lambda: self.config.model.lisflood.settings_file, default="settings.xml", dict_key="LISFLOOD_SETTINGS_FILE"
        )
        return self.lisflood_setup_dir / settings_file

    def _create_parameter_manager(self):
        from .parameter_manager import LisfloodParameterManager

        return LisfloodParameterManager(self.config, self.logger, self.lisflood_setup_dir)

    def _check_routing_needed(self) -> bool:
        return False

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run LISFLOOD with best params, directing output to output_dir."""
        settings_file = self._get_config_value(
            lambda: self.config.model.lisflood.settings_file, default="settings.xml", dict_key="LISFLOOD_SETTINGS_FILE"
        )
        src_xml = self.lisflood_setup_dir / settings_file

        # Copy settings XML to output dir and patch PathOut
        output_dir.mkdir(parents=True, exist_ok=True)
        final_xml = output_dir / settings_file
        shutil.copy2(src_xml, final_xml)

        tree = ET.parse(final_xml)  # nosec B314
        for textvar in tree.getroot().iter("textvar"):
            if textvar.get("name") == "PathOut":
                textvar.set("value", str(output_dir))
        tree.write(final_xml, encoding="unicode", xml_declaration=True)

        # Run with the patched settings
        return self.worker.run_model(
            self.config_dict,
            output_dir,
            output_dir,
            settings_file_override=str(final_xml),
        )

    def run_final_evaluation(self, best_params: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Run final evaluation with both cal and eval period metrics."""
        self.logger.info("=" * 60)
        self.logger.info("RUNNING FINAL EVALUATION")
        self.logger.info("=" * 60)
        self.logger.info("Running model with best parameters over full simulation period...")

        try:
            # Apply best params to the main settings XML
            from .parameter_manager import LisfloodParameterManager

            pm = LisfloodParameterManager(self.config, self.logger, self.lisflood_setup_dir)
            pm.update_model_files(best_params, self.lisflood_setup_dir)

            # Run model (output goes to the normal LISFLOOD output dir)
            final_output_dir = self.results_dir / "final_evaluation"
            final_output_dir.mkdir(parents=True, exist_ok=True)

            if not self._run_model_for_final_evaluation(final_output_dir):
                self.logger.error("LISFLOOD run failed during final evaluation")
                return None

            # Calculate metrics for calibration and evaluation periods
            from symfluence.core.metrics import StreamflowMetrics

            sim = self._load_final_discharge(final_output_dir)
            obs = self._load_observations()
            if sim is None or obs is None:
                self.logger.error("Could not load sim/obs for final evaluation")
                return None

            sm = StreamflowMetrics()
            cal_period = self._get_config_value(lambda: None, default="", dict_key="CALIBRATION_PERIOD")
            eval_period = self._get_config_value(lambda: None, default="", dict_key="EVALUATION_PERIOD")

            results = {"success": True, "best_params": best_params}

            for label, period_str in [("calibration", cal_period), ("evaluation", eval_period)]:
                if not period_str:
                    continue
                parts = [p.strip() for p in period_str.split(",")]
                if len(parts) != 2:
                    continue
                s = sim.loc[parts[0] : parts[1]]
                o = obs.loc[parts[0] : parts[1]]
                s, o = s.align(o, join="inner")
                common = s.dropna().index.intersection(o.dropna().index)
                s, o = s.loc[common], o.loc[common]
                if len(s) >= 30:
                    m = sm.calculate_metrics(o.values, s.values)
                    results[f"{label}_metrics"] = m
                    self.logger.info(
                        f"  {label.upper():12s}: KGE={m['kge']:.4f}  NSE={m['nse']:.4f}  "
                        f"Bias={m.get('pbias', 0):.1f}%  n={len(s)}"
                    )

            return results

        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Error in final evaluation: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            return None

    def _load_final_discharge(self, output_dir: Path):
        """Load discharge from LISFLOOD final evaluation output."""
        import pandas as pd
        import xarray as xr

        for pattern in ["dis*.nc", "*discharge*.nc"]:
            matches = list(output_dir.glob(pattern))
            if matches:
                ds = xr.open_dataset(matches[0])
                for var in ["dis", "discharge", "Qsim", "chanq"]:
                    if var in ds.data_vars:
                        q = ds[var]
                        spatial = [d for d in q.dims if d != "time"]
                        q = q.max(dim=spatial) if spatial else q
                        series = q.to_series()
                        ds.close()
                        if not isinstance(series.index, pd.DatetimeIndex):
                            series.index = pd.to_datetime(series.index)
                        return series.resample("D").mean()
                ds.close()

        for pattern in ["dis*.tss", "*discharge*.tss"]:
            matches = list(output_dir.glob(pattern))
            if matches:
                return self.worker._read_tss_file(matches[0])

        return None

    def _load_observations(self):
        """Load observed streamflow."""
        import pandas as pd

        obs_dir = self.project_dir / "observations" / "streamflow" / "preprocessed"
        if not obs_dir.exists():
            return None
        obs_files = list(obs_dir.glob("*.csv"))
        if not obs_files:
            return None
        df = pd.read_csv(obs_files[0], parse_dates=True, index_col=0)
        series = df["discharge_cms"] if "discharge_cms" in df.columns else df.iloc[:, 0]
        return series.resample("D").mean()
