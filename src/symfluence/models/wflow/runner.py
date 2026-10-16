# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Wflow Model Runner."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from symfluence.core.modeling.base import BaseModelRunner
from symfluence.core.registries import R


@R.runners.add('WFLOW')
class WflowRunner(BaseModelRunner):
    """Runner for the Wflow model (wflow_cli binary)."""
    MODEL_NAME = "WFLOW"

    def run(self, **kwargs):
        from symfluence.core.exceptions import ModelExecutionError
        output = super().run(**kwargs)
        if output is None:
            raise ModelExecutionError('Wflow native execution or output verification failed')
        if self.config.model.wflow.spatial_mode == 'distributed':
            import tomllib

            import pandas as pd

            from .balance import check
            check(output)
            native = tomllib.loads((self.settings_dir / self.config.model.wflow.config_file).read_text())
            expected = pd.date_range(pd.Timestamp(native['time']['starttime']) + pd.Timedelta(days=1),
                                     pd.Timestamp(native['time']['endtime']), freq='D')
            actual = pd.DatetimeIndex(pd.read_csv(Path(output) / 'output.csv', usecols=['time']).time)
            if not actual.equals(expected):
                raise ModelExecutionError('Incomplete Wflow daily output coverage')
        return output

    def _setup_model_specific_paths(self) -> None:
        self.settings_dir = self.project_dir / "settings" / "WFLOW"
        self.wflow_exe = self.get_model_executable(
            install_path_key='WFLOW_INSTALL_PATH',
            default_install_subpath='installs/wflow',
            default_exe_name='wflow_cli',
            typed_exe_accessor=lambda: (
                self.config.model.wflow.exe
                if self.config.model and self.config.model.wflow
                else None
            ),
            candidates=['bin', ''],
            must_exist=True,
        )

    def _build_run_command(self) -> Optional[List[str]]:
        config_file_name = self._get_config_value(
            lambda: self.config.model.wflow.config_file,
            default='wflow_sbm.toml',
        )
        toml_path = self.settings_dir / config_file_name
        return [str(self.wflow_exe), str(toml_path)]

    def _get_run_cwd(self) -> Optional[Path]:
        return self.settings_dir

    def _get_run_timeout(self) -> int:
        return self._get_config_value(
            lambda: self.config.model.wflow.timeout,
            default=7200,
        )
