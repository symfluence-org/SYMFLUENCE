"""SYMFLUENCE preprocessing, execution and DDS integration for Raven."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.calibration.parameters.base_parameter_manager import BaseParameterManager
from symfluence.core.calibration.workers.base_worker import BaseWorker
from symfluence.core.metrics import StreamflowMetrics
from symfluence.core.modeling.base import BaseModelRunner
from symfluence.core.modeling.base.base_preprocessor import BaseModelPreProcessor
from symfluence.models.utilities.elevation_forcing import daily_bands, upper_ice_fractions

from . import native


class RavenPreProcessor(BaseModelPreProcessor):
    MODEL_NAME = "RAVEN"

    def run_preprocessing(self):
        c = self.config
        s = c.model.raven
        bands, forcing = daily_bands(c.paths.forcing_path, s.hru_file, c.domain.time_start, c.domain.time_end)
        ice = upper_ice_fractions(bands.HRU_area, bands.elev_mean, s.glacier_fraction)
        native.setup(self.setup_dir, bands, forcing, ice, s.glacier_ice_mm)
        return True


class RavenRunner(BaseModelRunner):
    MODEL_NAME = "RAVEN"

    def _setup_model_specific_paths(self):
        self.settings_dir = self.project_dir / "settings/RAVEN"
        self.raven_exe = self.get_model_executable(
            install_path_key="RAVEN_INSTALL_PATH",
            default_install_subpath="installs/raven",
            default_exe_name="Raven",
            typed_exe_accessor=lambda: self.config.model.raven.exe,
            candidates=["bin", ""],
            must_exist=True,
        )

    def _build_run_command(self):
        out = self.project_dir / "simulations" / self.experiment_id / "RAVEN"
        out.mkdir(parents=True, exist_ok=True)
        return [str(self.raven_exe), str(self.settings_dir / "model"), "-o", str(out)]

    def _get_run_cwd(self):
        return self.settings_dir

    def _get_run_timeout(self):
        return self.config.model.raven.timeout


def discharge(output):
    p = Path(output) / "Hydrographs.csv"
    f = pd.read_csv(p)
    f.columns = f.columns.str.strip()
    col = next(c for c in f if "[m3/s]" in c and "observed" not in c.lower())
    # Raven reports interval-average discharge at the END of each interval.
    ix = pd.to_datetime(f["date"].astype(str) + " " + f["hour"].astype(str)) - pd.Timedelta(days=1)
    return pd.Series(f[col].values, index=ix).iloc[1:]


class RavenParameterManager(BaseParameterManager):
    def __init__(self, config, logger, settings_dir):
        super().__init__(config, logger, settings_dir)
        self.names = self._get_config_value(
            lambda: self.config.model.raven.params_to_calibrate,
            default=",".join(native.DEFAULTS),
            dict_key="RAVEN_PARAMS_TO_CALIBRATE",
        ).split(",")

    def _get_parameter_names(self):
        return self.names

    def _load_parameter_bounds(self):
        bounds = {k: dict(min=native.BOUNDS[k][0], max=native.BOUNDS[k][1]) for k in self.names}
        overrides = self._get_config_value(
            lambda: self.config.model.raven.param_bounds, default=None, dict_key="RAVEN_PARAM_BOUNDS"
        )
        if overrides:
            self._apply_config_bounds_override(bounds, {k: v for k, v in overrides.items() if k in bounds})
        return bounds

    def get_initial_parameters(self):
        return {k: native.DEFAULTS[k] for k in self.names}

    def update_model_files(self, params, settings_dir=None, **kwargs):
        native.parameters(settings_dir or self.settings_dir, params)
        return True


class RavenWorker(BaseWorker):
    def apply_parameters(self, params, settings_dir, **kwargs):
        cfg = kwargs.get("config", self.config)
        original = Path(cfg["SYMFLUENCE_DATA_DIR"]) / ("domain_" + cfg["DOMAIN_NAME"]) / "settings/RAVEN"
        target = Path(settings_dir)
        target.mkdir(parents=True, exist_ok=True)
        for f in original.glob("*"):
            if f.is_file() and f.parent != target:
                shutil.copy2(f, target / f.name)
        native.parameters(target, params)
        return True

    def run_model(self, config, settings_dir, output_dir, **kwargs):
        p = Path(config.get("RAVEN_INSTALL_PATH", "default"))
        exe = Path(config["SYMFLUENCE_DATA_DIR"]) / "installs/raven/bin/Raven" if str(p) == "default" else p / "Raven"
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "native.log").open("w") as log:
            result = subprocess.run(
                [str(exe), str(Path(settings_dir) / "model"), "-o", str(out)],
                cwd=settings_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=config.get("RAVEN_TIMEOUT", 600),
            )
        errors = out / "Raven_errors.txt"
        if result.returncode or (errors.exists() and "ERROR" in errors.read_text()):
            return False
        from .balance import check

        check(out, settings_dir)
        return (out / "Hydrographs.csv").exists()

    def calculate_metrics(self, output_dir, config, **kwargs):
        sim = discharge(output_dir)
        if not np.isfinite(sim).all():
            return {"KGE": -999.0}
        project = Path(config["SYMFLUENCE_DATA_DIR"]) / ("domain_" + config["DOMAIN_NAME"])
        metrics = StreamflowMetrics()
        values, index = metrics.load_observations(config, project, config["DOMAIN_NAME"], resample_freq=None)
        obs = pd.Series(values, index=index)
        start, end = [x.strip() for x in config["CALIBRATION_PERIOD"].split(",")]
        pair = pd.concat([obs.loc[start:end], sim.loc[start:end]], axis=1).dropna()
        if len(pair) < 365:
            return {"KGE": -999.0}
        result = metrics.calculate_metrics(pair.iloc[:, 0].values, pair.iloc[:, 1].values)
        if config.get("RAVEN_GLACIER_CONSTRAINTS", False):
            from .cryosphere import glacier_metrics

            result.update(glacier_metrics(
                output_dir, project / "settings/RAVEN/geometry.json",
                project / "data/observations/cryosphere/glacier_timeseries.csv",
                config["CALIBRATION_PERIOD"], config.get("RAVEN_GLACIER_MB_ERROR_SCALE", 1.0),
            ))
        return result


class RavenOptimizer(BaseModelOptimizer):
    def __init__(self, config, logger, optimization_settings_dir=None, reporting_manager=None):
        self.raven_setup_dir = Path(config.system.data_dir) / ("domain_" + config.domain.name) / "settings/RAVEN"
        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

    def _get_model_name(self):
        return "RAVEN"

    def _get_final_file_manager_path(self):
        return self.raven_setup_dir / "model.rvi"

    def _create_parameter_manager(self):
        return RavenParameterManager(self.config, self.logger, self.raven_setup_dir)

    def _check_routing_needed(self):
        return False

    def _run_model_for_final_evaluation(self, output_dir):
        return self.worker.run_model(self.config_dict, self.raven_setup_dir, output_dir)
