# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""PCR-GLOBWB calibration optimizer."""
from __future__ import annotations

import configparser
from pathlib import Path

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.registries import R


@R.optimizers.add('PCRGLOBWB')
class PCRGLOBWBModelOptimizer(BaseModelOptimizer):
    """Optimizer for PCR-GLOBWB model calibration."""

    def __init__(self, config, logger, optimization_settings_dir=None, reporting_manager=None):
        self.pcrglobwb_setup_dir = self.project_dir / 'settings' / 'PCRGLOBWB' if hasattr(self, 'project_dir') else None
        super().__init__(config, logger, optimization_settings_dir, reporting_manager)
        if self.pcrglobwb_setup_dir is None:
            data_dir = self._get_config_value(
                lambda: self.config.system.data_dir,
                default='.', dict_key='SYMFLUENCE_DATA_DIR',
            )
            domain = self._get_config_value(
                lambda: self.config.domain.name,
                default='', dict_key='DOMAIN_NAME',
            )
            self.pcrglobwb_setup_dir = Path(data_dir) / f'domain_{domain}' / 'settings' / 'PCRGLOBWB'

    def _get_model_name(self) -> str:
        return 'PCRGLOBWB'

    def _get_final_file_manager_path(self) -> Path:
        return self.pcrglobwb_setup_dir / 'setup.ini'

    def _create_parameter_manager(self):
        from .parameter_manager import PCRGLOBWBParameterManager
        return PCRGLOBWBParameterManager(self.config, self.logger, self.pcrglobwb_setup_dir)

    def _check_routing_needed(self) -> bool:
        return False

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        return self.worker.run_model(self.config_dict, self.pcrglobwb_setup_dir, output_dir)

    def _update_file_manager_for_final_run(self) -> None:
        """Update PCR-GLOBWB INI for full experiment period."""
        ini_path = self._get_final_file_manager_path()
        if not ini_path.exists():
            return

        sim_start = self._get_config_value(
            lambda: self.config.domain.time_start,
            dict_key='EXPERIMENT_TIME_START',
        )
        sim_end = self._get_config_value(
            lambda: self.config.domain.time_end,
            dict_key='EXPERIMENT_TIME_END',
        )
        if not sim_start or not sim_end:
            return

        import pandas as pd
        start = pd.to_datetime(sim_start).strftime('%Y-%m-%d')
        end = pd.to_datetime(sim_end).strftime('%Y-%m-%d')

        ini = configparser.ConfigParser()
        ini.optionxform = str
        ini.read(ini_path)
        if 'globalOptions' in ini:
            ini['globalOptions']['startTime'] = start
            ini['globalOptions']['endTime'] = end
        with open(ini_path, 'w') as f:
            ini.write(f)
        self.logger.info(f"Updated INI for full period: {start} to {end}")

    def _update_file_manager_output_path(self, output_dir: Path) -> None:
        """Update PCR-GLOBWB INI output directory."""
        ini_path = self._get_final_file_manager_path()
        if not ini_path.exists():
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        ini = configparser.ConfigParser()
        ini.optionxform = str
        ini.read(ini_path)
        if 'globalOptions' in ini:
            ini['globalOptions']['outputDir'] = str(output_dir)
        with open(ini_path, 'w') as f:
            ini.write(f)

    def _restore_file_manager_for_optimization(self) -> None:
        """Restore INI to calibration period."""
        ini_path = self._get_final_file_manager_path()
        if not ini_path.exists():
            return

        cal_period = self._get_config_value(
            lambda: self.config.domain.calibration_period,
            dict_key='CALIBRATION_PERIOD',
        )
        if not cal_period:
            return

        import pandas as pd
        parts = cal_period.split(',')
        if len(parts) != 2:
            return

        start = pd.to_datetime(parts[0].strip()).strftime('%Y-%m-%d')
        end = pd.to_datetime(parts[1].strip()).strftime('%Y-%m-%d')

        ini = configparser.ConfigParser()
        ini.optionxform = str
        ini.read(ini_path)
        if 'globalOptions' in ini:
            ini['globalOptions']['startTime'] = start
            ini['globalOptions']['endTime'] = end
        with open(ini_path, 'w') as f:
            ini.write(f)
